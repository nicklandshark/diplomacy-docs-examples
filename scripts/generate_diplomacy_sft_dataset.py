#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDORED_COOKBOOK_ROOT = REPO_ROOT / "vendor" / "tinker-cookbook"
if str(VENDORED_COOKBOOK_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDORED_COOKBOOK_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openai import AsyncOpenAI
from tinker_cookbook.tool_use.types import ToolInput

from tinker_training.diplomacy_adapter import (
    _build_system_prompt,
    _cleanup_episode_state,
    _get_rubric_for_environment,
    _initialize_episode_context,
    _new_score_semaphore,
    _tool_to_openai_schema,
)
from tinker_training.eval_utils import (
    DEFAULT_HELPER_API_KEY_ENV_VAR,
    DEFAULT_HELPER_BASE_URL,
    DEFAULT_HELPER_HTTP_REFERER,
    DEFAULT_HELPER_MODEL,
    DEFAULT_HELPER_X_TITLE,
    build_actor_runtime,
    build_runtime_policy,
    load_datum,
)
from tinker_training.prompt_family import (
    DEFAULT_PROMPT_FAMILY_DIR,
    PROMPT_FAMILY_ENVIRONMENTS,
    load_prompt_family_blocks,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TARGET_COUNTS = {
    "tool_accuracy": 8_000,
    "target_execution": 8_000,
    "supported_target": 8_000,
    "cooperative_press": 6_000,
    "full_press": 6_000,
}
MAX_TURNS = {
    "tool_accuracy": 14,
    "target_execution": 12,
    "supported_target": 14,
    "cooperative_press": 16,
    "full_press": 20,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate GPT-5.4 teacher traces for the Diplomacy SFT curriculum."
    )
    parser.add_argument("--model", default="gpt-5.4", help="Teacher model name.")
    parser.add_argument(
        "--reasoning-effort",
        default="high",
        choices=("low", "medium", "high"),
        help="Reasoning effort passed to the teacher model when supported.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Optional OpenAI-compatible base URL. Leave unset to use the default OpenAI API.",
    )
    parser.add_argument(
        "--api-key-env-var",
        default="OPENAI_API_KEY",
        help="Environment variable containing the OpenAI API key for the teacher model.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "data" / "diplomacy_sft"),
        help="Directory where the per-environment shards and merged dataset are written.",
    )
    parser.add_argument(
        "--prompt-family-dir",
        default=str(DEFAULT_PROMPT_FAMILY_DIR),
        help="Directory containing the stage-specific tracked prompts.",
    )
    parser.add_argument(
        "--background-actor-model",
        default=DEFAULT_HELPER_MODEL,
        help="Counterpart model used by the environment actors during teacher rollout generation.",
    )
    parser.add_argument(
        "--background-actor-base-url",
        default=DEFAULT_HELPER_BASE_URL,
        help="OpenAI-compatible base URL for the background actors.",
    )
    parser.add_argument(
        "--background-actor-api-key-env-var",
        default=DEFAULT_HELPER_API_KEY_ENV_VAR,
        help="API key env var used by the background actors.",
    )
    parser.add_argument(
        "--http-referer",
        default=DEFAULT_HELPER_HTTP_REFERER,
        help="HTTP-Referer used for the background actors.",
    )
    parser.add_argument(
        "--x-title",
        default=DEFAULT_HELPER_X_TITLE,
        help="X-Title used for the background actors.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Teacher sampling temperature.",
    )
    parser.add_argument(
        "--actor-max-turns",
        type=int,
        default=6,
        help="Maximum turns for the background actors.",
    )
    parser.add_argument(
        "--session-timeout-seconds",
        type=float,
        default=90.0,
        help="Per-session timeout used by the environment.",
    )
    parser.add_argument(
        "--max-candidates-per-seed",
        type=int,
        default=4,
        help="Maximum teacher attempts per seed before advancing to the next seed.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=21,
        help="First seed to try for each environment shard.",
    )
    parser.add_argument(
        "--environments",
        nargs="+",
        choices=PROMPT_FAMILY_ENVIRONMENTS,
        default=list(PROMPT_FAMILY_ENVIRONMENTS),
        help="Which environments to generate.",
    )
    for environment_kind, default_count in TARGET_COUNTS.items():
        parser.add_argument(
            f"--{environment_kind.replace('_', '-')}-count",
            type=int,
            default=default_count,
            help=f"Target successful trace count for {environment_kind}.",
        )
    return parser.parse_args()


def _target_count_for_environment(args: argparse.Namespace, environment_kind: str) -> int:
    return int(getattr(args, f"{environment_kind}_count"))


def _primary_success_value(environment_kind: str, metrics: dict[str, float]) -> float:
    if environment_kind == "tool_accuracy":
        return min(
            metrics.get("rubric/constraints_satisfied_metric", 0.0),
            metrics.get("rubric/complete_legal_submission_metric", 0.0),
            metrics.get("rubric/required_send_recall_metric", 0.0),
            metrics.get("rubric/required_read_recall_metric", 0.0),
            metrics.get("rubric/invalid_tool_budget_pass_metric", 0.0),
        )
    if environment_kind == "target_execution":
        return metrics.get("rubric/transition_target_without_press_metric", 0.0)
    return metrics.get("rubric/transition_target_satisfied_metric", 0.0)


async def run_teacher_candidate(
    *,
    client: AsyncOpenAI,
    model: str,
    reasoning_effort: str,
    environment_kind: str,
    seed: int,
    prompt_block: str,
    background_actor_model: str,
    background_actor_base_url: str,
    background_actor_api_key_env_var: str,
    http_referer: str,
    x_title: str,
    max_turns: int,
    actor_max_turns: int,
    session_timeout_seconds: float,
    temperature: float,
) -> dict[str, Any]:
    datum = load_datum(environment_kind, seed)
    actor_runtime = build_actor_runtime(
        actor_max_turns=actor_max_turns,
        session_timeout_seconds=session_timeout_seconds,
        background_actor_model=background_actor_model,
        background_actor_base_url=background_actor_base_url,
        background_actor_api_key_env_var=background_actor_api_key_env_var,
        http_referer=http_referer,
        x_title=x_title,
    )
    policy_config = build_runtime_policy(max_turns=max_turns)
    ctx = _initialize_episode_context(
        datum=datum,
        actor_runtime=actor_runtime,
        policy_config=policy_config,
    )
    state = ctx.state
    tracked_power = state["info"].get("tracked_power", state["info"].get("agent_name", ""))
    history: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": _build_system_prompt(
                agent_name=str(tracked_power),
                is_background_actor=False,
                default_idle_sleep_seconds=actor_runtime.default_idle_sleep_seconds,
                tracked_instruction_block=prompt_block,
            ),
        },
        *state["prompt"],
    ]
    tools = ctx.tool_executor.all_tools()
    tool_map = {tool_obj.name: tool_obj for tool_obj in tools}
    tool_schemas = [_tool_to_openai_schema(tool_obj) for tool_obj in tools]

    try:
        for turn_index in range(max_turns):
            request_kwargs: dict[str, Any] = {
                "model": model,
                "messages": history,
                "tools": tool_schemas,
                "temperature": temperature,
            }
            if reasoning_effort:
                request_kwargs["reasoning_effort"] = reasoning_effort
            response = await client.chat.completions.create(**request_kwargs)
            message = response.choices[0].message.model_dump(exclude_none=True)
            history.append(message)

            tool_calls = list(message.get("tool_calls") or [])
            should_stop = False
            for tool_call in tool_calls:
                tool_name = tool_call.get("function", {}).get("name", "")
                tool_obj = tool_map.get(tool_name)
                if tool_obj is None:
                    continue
                try:
                    arguments = json.loads(tool_call.get("function", {}).get("arguments", "") or "{}")
                    if not isinstance(arguments, dict):
                        arguments = {}
                except Exception:
                    arguments = {}
                tool_result = await tool_obj.run(
                    ToolInput(arguments=arguments, call_id=tool_call.get("id", ""))
                )
                history.extend(tool_result.messages)
                should_stop = should_stop or tool_result.should_stop

            reached_max_turns = turn_index + 1 >= max_turns
            done = should_stop or not tool_calls or reached_max_turns
            if done:
                if reached_max_turns and not state.get("stop_condition"):
                    state["stop_condition"] = "max_turns_reached"
                if not state.get("final_answer"):
                    text_parts = message.get("content", [])
                    if isinstance(text_parts, str):
                        text = text_parts
                    else:
                        rendered_parts: list[str] = []
                        for part in text_parts or []:
                            if isinstance(part, dict):
                                rendered_parts.append(str(part.get("text", "")))
                            else:
                                rendered_parts.append(str(part))
                        text = "".join(rendered_parts)
                    if text.strip():
                        state["final_answer"] = text.strip()
                state["completion"] = history[2:]
                rubric = _get_rubric_for_environment(environment_kind)
                await rubric.score_rollout(state, score_sem=_new_score_semaphore())
                break

        metrics = {
            key: float(value)
            for key, value in (state.get("metrics") or {}).items()
            if isinstance(value, (int, float))
        }
        return {
            "seed": seed,
            "status": "ok",
            "messages": history,
            "metrics": metrics,
            "reward": float(state.get("reward", 0.0) or 0.0),
            "trace_len": len(state.get("chatroom_tool_trace", [])),
            "rejected_tool_calls": float(metrics.get("rubric/rejected_tool_call_count_metric", 0.0)),
            "primary_success": _primary_success_value(environment_kind, metrics),
            "task_prompt": datum["info"]["task_prompt"],
            "tracked_power": datum["info"]["tracked_power"],
        }
    except Exception as exc:
        return {
            "seed": seed,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "primary_success": 0.0,
            "rejected_tool_calls": 99.0,
            "trace_len": 999,
        }
    finally:
        await _cleanup_episode_state(state)


def choose_best_candidate(environment_kind: str, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    successful = [candidate for candidate in candidates if candidate.get("primary_success", 0.0) >= 1.0]
    if not successful:
        return None
    return sorted(
        successful,
        key=lambda candidate: (
            -float(candidate.get("primary_success", 0.0)),
            float(candidate.get("rejected_tool_calls", 0.0)),
            int(candidate.get("trace_len", 9999)),
        ),
    )[0]


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


async def main_async(args: argparse.Namespace) -> int:
    api_key = os.environ.get(args.api_key_env_var, "").strip()
    if not api_key:
        raise ValueError(f"Environment variable {args.api_key_env_var} is not set.")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_blocks = load_prompt_family_blocks(args.prompt_family_dir)
    client = AsyncOpenAI(api_key=api_key, base_url=args.base_url, timeout=180)
    report: dict[str, Any] = {
        "teacher_model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "prompt_family_dir": str(Path(args.prompt_family_dir).expanduser().resolve()),
        "environments": {},
    }
    merged_path = output_dir / "merged.jsonl"
    if merged_path.exists():
        merged_path.unlink()

    try:
        for environment_kind in args.environments:
            shard_path = output_dir / f"{environment_kind}.jsonl"
            if shard_path.exists():
                shard_path.unlink()
            target_count = _target_count_for_environment(args, environment_kind)
            max_turns = MAX_TURNS[environment_kind]
            prompt_block = prompt_blocks[environment_kind]
            successes = 0
            attempts = 0
            seed = int(args.seed_start)
            env_report: dict[str, Any] = {
                "target_count": target_count,
                "successful_traces": 0,
                "attempted_rollouts": 0,
                "last_seed": seed - 1,
            }
            while successes < target_count:
                candidates = []
                for _ in range(args.max_candidates_per_seed):
                    attempts += 1
                    candidates.append(
                        await run_teacher_candidate(
                            client=client,
                            model=args.model,
                            reasoning_effort=args.reasoning_effort,
                            environment_kind=environment_kind,
                            seed=seed,
                            prompt_block=prompt_block,
                            background_actor_model=args.background_actor_model,
                            background_actor_base_url=args.background_actor_base_url,
                            background_actor_api_key_env_var=args.background_actor_api_key_env_var,
                            http_referer=args.http_referer,
                            x_title=args.x_title,
                            max_turns=max_turns,
                            actor_max_turns=args.actor_max_turns,
                            session_timeout_seconds=args.session_timeout_seconds,
                            temperature=args.temperature,
                        )
                    )
                best = choose_best_candidate(environment_kind, candidates)
                env_report["last_seed"] = seed
                if best is not None:
                    row = {
                        "messages": best["messages"],
                        "metadata": {
                            "environment_kind": environment_kind,
                            "seed": seed,
                            "teacher_model": args.model,
                            "reasoning_effort": args.reasoning_effort,
                            "reward": best["reward"],
                            "metrics": best["metrics"],
                            "tracked_power": best["tracked_power"],
                            "task_prompt": best["task_prompt"],
                        },
                    }
                    append_jsonl(shard_path, row)
                    append_jsonl(merged_path, row)
                    successes += 1
                seed += 1
                env_report["successful_traces"] = successes
                env_report["attempted_rollouts"] = attempts
                logger.info(
                    "[%s] %d/%d successful traces after %d rollout attempts",
                    environment_kind,
                    successes,
                    target_count,
                    attempts,
                )
            env_report["success_rate"] = successes / attempts if attempts else 0.0
            report["environments"][environment_kind] = env_report
        (output_dir / "generation_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        logger.info("Wrote SFT dataset shards to %s", output_dir)
        return 0
    finally:
        await client.close()


def main() -> int:
    args = parse_args()
    started_at = time.time()
    result = asyncio.run(main_async(args))
    logger.info("Finished in %.1fs", time.time() - started_at)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
