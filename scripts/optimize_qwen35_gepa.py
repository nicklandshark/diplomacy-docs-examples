#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import itertools
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Sequence

from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDORED_COOKBOOK_ROOT = REPO_ROOT / "vendor" / "tinker-cookbook"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(VENDORED_COOKBOOK_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDORED_COOKBOOK_ROOT))

from tinker_cookbook import tokenizer_utils
from tinker_cookbook.rl.types import SamplingRef
from tinker_training.diplomacy_adapter import (
    ActorRuntimeConfig,
    OpenRouterHeaders,
    RuntimePolicyConfig,
    build_actor_configs,
    get_default_renderer_name,
    get_default_tracked_instruction_block,
)
from tinker_training.diplomacy_gepa import (
    DEFAULT_HELPER_API_KEY_ENV_VAR,
    DEFAULT_HELPER_BASE_URL,
    DEFAULT_HELPER_HTTP_REFERER,
    DEFAULT_HELPER_MODEL,
    DEFAULT_HELPER_X_TITLE,
    DEFAULT_PER_SEED_TIMEOUT_SECONDS,
    DEFAULT_REFLECTION_MODEL,
    DEFAULT_REPORT_WORKERS,
    ExperimentPreset,
    SeedResult,
    TurnRecord,
    build_manual_review,
    build_pattern_summary,
    build_taxonomy_summary,
    classify_seed_result,
    get_gepa_train_pool,
    load_preset,
    load_seed_result,
    resolve_seed_pool,
    review_background_from_taxonomy,
    save_preset,
    save_seed_result,
    summarize_rows,
    write_json,
)
from tinker_training.rollout_backends import (
    ModalTrajectorySandboxRunner,
    TrajectoryRolloutRequest,
    deserialize_trajectory,
)

DEFAULT_MODEL = "Qwen/Qwen3.5-27B"
DEFAULT_ENVIRONMENT = "full_press"
DEFAULT_RUN_DIR = REPO_ROOT / ".tmp" / "gepa_long_run"
DEFAULT_APP_NAME = "diplomacy-gepa-long-run"
DEFAULT_LOG_ROOT = "~/tinker-runs/diplomacy-grpo"
DEFAULT_TEMPERATURE = 1.0
DEFAULT_REFLECTION_TIMEOUT_SECONDS = 60.0
TOOL_PATTERN = re.compile(r"<function=([^>]+)>")
JSON_TOOL_PATTERN = re.compile(r'"name"\s*:\s*"([^"]+)"')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Long-run GEPA and benchmark harness for Diplomacy full_press/tool_accuracy optimization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--phase",
        required=True,
        choices=("screen", "repair_screen", "optimize", "report", "compare", "train_smoke"),
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--environment", choices=("tool_accuracy", "full_press"), default=None)
    parser.add_argument("--preset-path", default=None)
    parser.add_argument("--preset-paths", nargs="*", default=None)
    parser.add_argument("--preset-a-path", default=None)
    parser.add_argument("--preset-b-path", default=None)
    parser.add_argument("--renderer-name", default=None)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--temperatures", default=None)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--max-turns-options", default=None)
    parser.add_argument("--actor-max-turns", type=int, default=6)
    parser.add_argument("--actor-max-turns-options", default=None)
    parser.add_argument("--session-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--helper-model", default=DEFAULT_HELPER_MODEL)
    parser.add_argument("--helper-base-url", default=DEFAULT_HELPER_BASE_URL)
    parser.add_argument("--helper-api-key-env-var", default=DEFAULT_HELPER_API_KEY_ENV_VAR)
    parser.add_argument("--helper-http-referer", default=DEFAULT_HELPER_HTTP_REFERER)
    parser.add_argument("--helper-x-title", default=DEFAULT_HELPER_X_TITLE)
    parser.add_argument("--prompt-candidate-paths", nargs="*", default=None)
    parser.add_argument("--screen-dir", default=None)
    parser.add_argument("--seed-pool", default="fullpress_screen")
    parser.add_argument("--report-workers", type=int, default=DEFAULT_REPORT_WORKERS)
    parser.add_argument("--per-seed-timeout-seconds", type=float, default=DEFAULT_PER_SEED_TIMEOUT_SECONDS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--tag", default=None)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--app-name", default=DEFAULT_APP_NAME)
    parser.add_argument("--round-index", type=int, default=1)
    parser.add_argument("--max-metric-calls", type=int, default=128)
    parser.add_argument("--reflection-lm", default=DEFAULT_REFLECTION_MODEL)
    parser.add_argument("--reflection-timeout-seconds", type=float, default=DEFAULT_REFLECTION_TIMEOUT_SECONDS)
    parser.add_argument("--reflection-minibatch-size", type=int, default=8)
    parser.add_argument("--train-seed-pool", default=None)
    parser.add_argument("--val-seed-pool", default="gepa_val")
    parser.add_argument("--confirm-seed-pool", default="confirm")
    parser.add_argument("--log-root", default=DEFAULT_LOG_ROOT)
    parser.add_argument("--train-smoke-run-name", default=None)
    return parser.parse_args()


def parse_int_options(raw: str | None, default: Sequence[int]) -> list[int]:
    if raw is None:
        return list(default)
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_float_options(raw: str | None, default: Sequence[float]) -> list[float]:
    if raw is None:
        return list(default)
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def load_datum(environment: str, seed: int) -> dict[str, Any]:
    if environment == "tool_accuracy":
        from data_generator import build_tool_accuracy_dataset

        return build_tool_accuracy_dataset(num_sessions=1, seed=seed).to_list()[0]

    from data_generator import build_full_press_dataset

    return build_full_press_dataset(num_sessions=1, seed=seed).to_list()[0]


def build_actor_runtime(preset: ExperimentPreset) -> ActorRuntimeConfig:
    return ActorRuntimeConfig(
        actor_configs=build_actor_configs(
            base_url=preset.helper_base_url,
            api_key_env_var=preset.helper_api_key_env_var,
            model=preset.helper_model,
        ),
        actor_max_turns=preset.actor_max_turns,
        session_timeout_seconds=preset.session_timeout_seconds,
        default_idle_sleep_seconds=0.5,
        openrouter_headers=OpenRouterHeaders(
            http_referer=preset.helper_http_referer,
            x_title=preset.helper_x_title,
        ),
    )


def build_default_preset(args: argparse.Namespace) -> ExperimentPreset:
    environment = args.environment or DEFAULT_ENVIRONMENT
    renderer_name = args.renderer_name or get_default_renderer_name(
        args.model_name,
        disable_thinking=args.disable_thinking,
    )
    tracked_instruction_block = get_default_tracked_instruction_block(default_idle_sleep_seconds=0.5)
    return ExperimentPreset(
        model_name=args.model_name,
        environment=environment,
        renderer_name=renderer_name,
        disable_thinking=args.disable_thinking,
        temperature=args.temperature,
        max_turns=args.max_turns,
        actor_max_turns=args.actor_max_turns,
        session_timeout_seconds=args.session_timeout_seconds,
        max_tokens=args.max_tokens,
        helper_model=args.helper_model,
        helper_base_url=args.helper_base_url,
        helper_api_key_env_var=args.helper_api_key_env_var,
        helper_http_referer=args.helper_http_referer,
        helper_x_title=args.helper_x_title,
        tracked_instruction_block=tracked_instruction_block,
    )


def read_prompt_candidate(path: Path | None, preset: ExperimentPreset) -> str:
    if path is None:
        return preset.tracked_instruction_block or get_default_tracked_instruction_block(
            default_idle_sleep_seconds=0.5
        )
    return path.read_text().strip()


def slugify(value: str) -> str:
    lowered = value.lower()
    lowered = re.sub(r"[^a-z0-9]+", "-", lowered)
    return lowered.strip("-") or "value"


def prompt_fingerprint(prompt: str | None) -> str:
    if not prompt:
        return "baseline"
    digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()
    return digest[:8]


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def record_metric_call(
    *,
    metric_dir: Path,
    call_index: int,
    candidate: str,
    row: SeedResult,
) -> None:
    candidate_hash = prompt_fingerprint(candidate)
    candidate_dir = metric_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = candidate_dir / f"{candidate_hash}.txt"
    if not candidate_path.exists():
        candidate_path.write_text(candidate.strip() + "\n", encoding="utf-8")
    row_payload = {
        "call_index": call_index,
        "candidate_hash": candidate_hash,
        "seed_result": row.to_json(),
    }
    rows_dir = metric_dir / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    write_json(rows_dir / f"{call_index:05d}-{candidate_hash}-seed_{row.seed:04d}.json", row_payload)
    append_jsonl(
        metric_dir / "metric_calls.jsonl",
        {
            "call_index": call_index,
            "candidate_hash": candidate_hash,
            "seed": row.seed,
            "status": row.status,
            "score": row.score,
            "reward": row.reward,
            "dominant_failure": row.dominant_failure,
        },
    )


def load_metric_cache(metric_dir: Path) -> tuple[int, dict[tuple[str, int], SeedResult]]:
    rows_dir = metric_dir / "rows"
    if not rows_dir.exists():
        return 0, {}
    cache: dict[tuple[str, int], SeedResult] = {}
    count = 0
    for artifact_path in sorted(rows_dir.glob("*.json")):
        payload = json.loads(artifact_path.read_text())
        candidate_hash = str(payload["candidate_hash"])
        row = SeedResult.from_json(payload["seed_result"])
        if is_transient_seed_error(row):
            continue
        cache[(candidate_hash, row.seed)] = row
        count += 1
    return count, cache


def helper_api_key_from_env(env_var: str) -> str:
    api_key = os.environ.get(env_var)
    if api_key:
        return api_key
    raise RuntimeError(f"{env_var} must be set.")


def is_transient_seed_error(row: SeedResult) -> bool:
    if row.status != "error":
        return False
    failure_kind = (row.failure_kind or "").lower()
    failure_message = (row.failure_message or "").lower()
    return failure_kind == "conflicterror" and "app is stopped or disabled" in failure_message


def build_openai_compatible_lm(
    *,
    model_name: str,
    base_url: str,
    api_key_env_var: str,
    http_referer: str,
    x_title: str,
    timeout_seconds: float,
):
    client = OpenAI(
        api_key=helper_api_key_from_env(api_key_env_var),
        base_url=base_url,
        timeout=timeout_seconds,
        default_headers={
            "HTTP-Referer": http_referer,
            "X-Title": x_title,
        },
    )

    def _lm(prompt: str | list[dict[str, Any]]) -> str:
        if isinstance(prompt, str):
            messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        else:
            messages = prompt
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
        )
        message = completion.choices[0].message
        content = message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_chunks: list[str] = []
            for block in content:
                if hasattr(block, "text") and getattr(block, "text"):
                    text_chunks.append(str(block.text))
                elif isinstance(block, dict) and block.get("text"):
                    text_chunks.append(str(block["text"]))
            return "".join(text_chunks)
        return str(content or "")

    return _lm


def resolve_reflection_lm(
    *,
    reflection_lm: Any,
    helper_base_url: str,
    helper_api_key_env_var: str,
    helper_http_referer: str,
    helper_x_title: str,
    timeout_seconds: float,
) -> Any:
    if not isinstance(reflection_lm, str):
        return reflection_lm
    return build_openai_compatible_lm(
        model_name=reflection_lm,
        base_url=helper_base_url,
        api_key_env_var=helper_api_key_env_var,
        http_referer=helper_http_referer,
        x_title=helper_x_title,
        timeout_seconds=timeout_seconds,
    )


def screen_preset_slug(preset: ExperimentPreset) -> str:
    return (
        f"{slugify(preset.renderer_name)}"
        f"-t{preset.temperature}"
        f"-m{preset.max_turns}"
        f"-a{preset.actor_max_turns}"
        f"-p{prompt_fingerprint(preset.tracked_instruction_block)}"
    )


class ModalPoolEvaluator:
    def __init__(
        self,
        *,
        preset: ExperimentPreset,
        app_name: str,
        per_seed_timeout_seconds: float,
    ) -> None:
        self.preset = preset
        self.runner = ModalTrajectorySandboxRunner(
            app_name=app_name,
            timeout_seconds=max(1, math.ceil(per_seed_timeout_seconds)),
        )
        self.tokenizer = tokenizer_utils.get_tokenizer(preset.model_name)

    async def __aenter__(self) -> "ModalPoolEvaluator":
        await self.runner.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.runner.aclose()

    async def evaluate_pool(
        self,
        *,
        seeds: Sequence[int],
        output_dir: Path,
        report_workers: int,
        resume: bool,
        label: str,
    ) -> list[SeedResult]:
        output_dir.mkdir(parents=True, exist_ok=True)
        rows_dir = output_dir / "rows"
        rows_dir.mkdir(parents=True, exist_ok=True)
        total = len(seeds)
        completed = 0
        completed_lock = asyncio.Lock()
        semaphore = asyncio.Semaphore(max(1, report_workers))
        results: list[SeedResult | None] = [None] * total

        async def _run_index(index: int, seed: int) -> None:
            nonlocal completed
            async with semaphore:
                row = await self.evaluate_seed(
                    seed=seed,
                    artifact_path=rows_dir / f"seed_{seed:04d}.json",
                    resume=resume,
                )
            results[index] = row
            async with completed_lock:
                completed += 1
                print(
                    f"[{label}] {completed}/{total} seed={seed} status={row.status} "
                    f"reward={row.reward:.2f} failure={row.dominant_failure}",
                    flush=True,
                )

        await asyncio.gather(*(_run_index(index, seed) for index, seed in enumerate(seeds)))
        return [row for row in results if row is not None]

    async def evaluate_seed(
        self,
        *,
        seed: int,
        artifact_path: Path,
        resume: bool,
    ) -> SeedResult:
        if resume and artifact_path.exists():
            cached_row = load_seed_result(artifact_path)
            if not is_transient_seed_error(cached_row):
                return cached_row

        started_at = time.time()
        request = TrajectoryRolloutRequest(
            datum=load_datum(self.preset.environment, seed),
            environment_kind=self.preset.environment,
            model_name=self.preset.model_name,
            renderer_name=self.preset.renderer_name,
            actor_runtime=build_actor_runtime(self.preset),
            policy_config=RuntimePolicyConfig(max_turns=self.preset.max_turns),
            sampling_ref=SamplingRef(
                base_model=self.preset.model_name,
                base_url=os.environ.get("TINKER_BASE_URL"),
            ),
            max_tokens=self.preset.max_tokens,
            temperature=self.preset.temperature,
            trajectory_index=0,
            group_id=f"{self.preset.environment}:{self.preset.model_name}:{seed}",
            enable_logging=False,
            tracked_instruction_block=self.preset.tracked_instruction_block,
        )

        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                result = await self.runner.run_trajectory(request)
                break
            except Exception as exc:  # pragma: no cover - exercised through SeedResult path tests
                last_exc = exc
                failure_row = SeedResult(
                    seed=seed,
                    status="error",
                    execution_backend="tinker_modal",
                    score=-1.0,
                    reward=-1.0,
                    gate=0.0,
                    transition_target=0.0,
                    relevant_submission=0.0,
                    rejected_tool_calls=0.0,
                    wait_count=0,
                    read_conversation_count=0,
                    compact_order_error=False,
                    has_think_close=False,
                    turns=0,
                    max_turns=self.preset.max_turns,
                    wall_time_seconds=time.time() - started_at,
                    failure_kind=type(exc).__name__,
                    failure_message=str(exc),
                    artifact_path=str(artifact_path),
                )
                if attempt == 0 and is_transient_seed_error(failure_row):
                    await self.runner.aclose()
                    await self.runner.start()
                    continue
                row = replace(failure_row, dominant_failure=classify_seed_result(failure_row))
                save_seed_result(artifact_path, row)
                return row
        else:  # pragma: no cover - defensive, loop always returns or breaks
            raise RuntimeError(f"Unreachable retry loop state for seed {seed}: {last_exc}")

        row = self._result_to_seed_result(
            seed=seed,
            result=result,
            wall_time_seconds=time.time() - started_at,
            artifact_path=artifact_path,
        )
        save_seed_result(artifact_path, row)
        return row

    def _result_to_seed_result(
        self,
        *,
        seed: int,
        result: Any,
        wall_time_seconds: float,
        artifact_path: Path,
    ) -> SeedResult:
        if result.failure_kind is not None or result.trajectory is None:
            row = SeedResult(
                seed=seed,
                status="error",
                execution_backend="tinker_modal",
                score=-1.0,
                reward=-1.0,
                gate=0.0,
                transition_target=0.0,
                relevant_submission=0.0,
                rejected_tool_calls=0.0,
                wait_count=0,
                read_conversation_count=0,
                compact_order_error=False,
                has_think_close=False,
                turns=0,
                max_turns=self.preset.max_turns,
                wall_time_seconds=wall_time_seconds,
                failure_kind=result.failure_kind,
                failure_message=result.failure_message,
                artifact_path=str(artifact_path),
            )
            return replace(row, dominant_failure=classify_seed_result(row))

        trajectory = deserialize_trajectory(result.trajectory)
        merged_metrics: dict[str, float] = {}
        turn_records: list[TurnRecord] = []
        decoded_actions: list[str] = []
        observation_snippets: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0
        for index, transition in enumerate(trajectory.transitions):
            prompt_tokens += int(getattr(transition.ob, "length", 0) or 0)
            completion_tokens += len(transition.ac.tokens)
            action_text = self.tokenizer.decode(transition.ac.tokens, skip_special_tokens=False)
            decoded_actions.append(action_text)
            tool_names = extract_tool_names(action_text)
            observation_excerpt = render_observation_excerpt(transition.ob, self.tokenizer)[:500]
            observation_snippets.append(observation_excerpt.lower())
            metrics = {
                key: float(value)
                for key, value in transition.metrics.items()
                if isinstance(value, (int, float))
            }
            merged_metrics.update(metrics)
            turn_records.append(
                TurnRecord(
                    turn_index=index,
                    tools=tool_names,
                    action_text=action_text,
                    observation_excerpt=observation_excerpt,
                    reward=float(transition.reward),
                    episode_done=bool(transition.episode_done),
                    metrics=metrics,
                )
            )

        merged_metrics.update(
            {
                key: float(value)
                for key, value in result.metrics.items()
                if isinstance(value, (int, float))
            }
        )
        reward = sum(float(transition.reward) for transition in trajectory.transitions) + float(
            result.final_reward
        )
        wait_count = sum(record.tools.count("wait") for record in turn_records)
        read_conversation_count = sum(
            record.tools.count("read_conversation") for record in turn_records
        )
        compact_order_error = any(
            "does not belong to" in snippet or "invalid order" in snippet for snippet in observation_snippets
        )
        has_think_close = any("</think>" in action_text for action_text in decoded_actions)
        stop_condition = infer_stop_condition(
            turn_records=turn_records,
            merged_metrics=merged_metrics,
            max_turns=self.preset.max_turns,
        )
        gate, transition_target, relevant_submission = extract_objective_metrics(
            merged_metrics=merged_metrics,
            environment=self.preset.environment,
        )
        rejected_tool_calls = float(
            merged_metrics.get(
                "rubric/rejected_tool_call_count_metric",
                merged_metrics.get("rollout/rejected_tool_calls", 0.0),
            )
        )
        score = (
            reward
            + 0.10 * transition_target
            + 0.05 * relevant_submission
            - 0.05 * rejected_tool_calls
            - 0.02 * max(0, wait_count - 1)
            - (0.05 if compact_order_error else 0.0)
            - (0.05 if has_think_close and self.preset.environment == "full_press" else 0.0)
        )
        row = SeedResult(
            seed=seed,
            status="ok",
            execution_backend="tinker_modal",
            score=score,
            reward=reward,
            gate=gate,
            transition_target=transition_target,
            relevant_submission=relevant_submission,
            rejected_tool_calls=rejected_tool_calls,
            wait_count=wait_count,
            read_conversation_count=read_conversation_count,
            compact_order_error=compact_order_error,
            has_think_close=has_think_close,
            turns=len(turn_records),
            max_turns=self.preset.max_turns,
            wall_time_seconds=wall_time_seconds,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            stop_condition=stop_condition,
            last_metrics=merged_metrics,
            turn_records=turn_records,
            artifact_path=str(artifact_path),
        )
        return replace(row, dominant_failure=classify_seed_result(row))


def extract_tool_names(action_text: str) -> list[str]:
    if not action_text:
        return []
    xml_names = TOOL_PATTERN.findall(action_text)
    if xml_names:
        return xml_names
    names: list[str] = []
    for tool_call in re.findall(r"<tool_call>(.*?)</tool_call>", action_text, flags=re.DOTALL):
        names.extend(JSON_TOOL_PATTERN.findall(tool_call))
    return names


def model_input_text(model_input: Any) -> str:
    if hasattr(model_input, "model_dump"):
        return json.dumps(model_input.model_dump(mode="python"), ensure_ascii=False)
    return str(model_input)


def render_observation_excerpt(model_input: Any, tokenizer: Any) -> str:
    if not hasattr(model_input, "model_dump"):
        return str(model_input)
    payload = model_input.model_dump(mode="python")
    chunks = payload.get("chunks", [])
    rendered_chunks: list[str] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            rendered_chunks.append(str(chunk))
            continue
        tokens = chunk.get("tokens")
        if isinstance(tokens, list) and tokens:
            try:
                rendered_chunks.append(tokenizer.decode(tokens, skip_special_tokens=False))
                continue
            except Exception:
                pass
        if "text" in chunk:
            rendered_chunks.append(str(chunk["text"]))
            continue
        if "content" in chunk:
            rendered_chunks.append(str(chunk["content"]))
            continue
        rendered_chunks.append(json.dumps(chunk, ensure_ascii=False))
    text = "".join(rendered_chunks).strip()
    if not text:
        return json.dumps(payload, ensure_ascii=False)
    # Keep the tail because the latest tool result is far more useful than the
    # repeated system/tool schema preamble at the front of the prompt.
    return text[-2000:]


def infer_stop_condition(
    *,
    turn_records: Sequence[TurnRecord],
    merged_metrics: dict[str, float],
    max_turns: int,
) -> str | None:
    if merged_metrics.get("tool/finish_called", 0.0) > 0.0:
        return "finish_called"
    if merged_metrics.get("episode/max_turns_reached", 0.0) > 0.0:
        return "max_turns_reached"
    if turn_records and turn_records[-1].episode_done:
        return "episode_done"
    if len(turn_records) >= max_turns:
        return "max_turns_reached"
    return None


def extract_objective_metrics(
    *,
    merged_metrics: dict[str, float],
    environment: str,
) -> tuple[float, float, float]:
    if environment == "full_press":
        return (
            float(merged_metrics.get("rubric/full_press_gate_metric", 0.0)),
            float(merged_metrics.get("rubric/transition_target_satisfied_metric", 0.0)),
            float(merged_metrics.get("rubric/relevant_actor_submission_metric", 0.0)),
        )
    constraints = float(merged_metrics.get("rubric/constraints_satisfied_metric", 0.0))
    complete_legal = float(merged_metrics.get("rubric/complete_legal_submission_metric", 0.0))
    send_recall = float(merged_metrics.get("rubric/required_send_recall_metric", 0.0))
    read_recall = float(merged_metrics.get("rubric/required_read_recall_metric", 0.0))
    invalid_tool_budget = float(merged_metrics.get("rubric/invalid_tool_budget_pass_metric", 0.0))
    gate = min(constraints, complete_legal, send_recall, read_recall, invalid_tool_budget)
    transition_target = complete_legal
    relevant_submission = send_recall
    return gate, transition_target, relevant_submission


async def evaluate_preset(
    *,
    preset: ExperimentPreset,
    seeds: Sequence[int],
    output_dir: Path,
    app_name: str,
    report_workers: int,
    per_seed_timeout_seconds: float,
    resume: bool,
    label: str,
) -> tuple[list[SeedResult], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    rows = load_completed_rows(output_dir=output_dir, seeds=seeds) if resume else None
    if rows is None:
        async with ModalPoolEvaluator(
            preset=preset,
            app_name=app_name,
            per_seed_timeout_seconds=per_seed_timeout_seconds,
        ) as evaluator:
            rows = await evaluator.evaluate_pool(
                seeds=seeds,
                output_dir=output_dir,
                report_workers=report_workers,
                resume=resume,
                label=label,
            )
    summary = summarize_rows(rows)
    taxonomy = build_taxonomy_summary(rows)
    patterns = build_pattern_summary(rows)
    review = build_manual_review(rows)
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "taxonomy.json", taxonomy)
    write_json(output_dir / "patterns.json", patterns)
    write_json(output_dir / "review.json", review)
    save_preset(output_dir / "preset.json", preset)
    return rows, summary, taxonomy, patterns, review


def load_completed_rows(*, output_dir: Path, seeds: Sequence[int]) -> list[SeedResult] | None:
    rows_dir = output_dir / "rows"
    if not rows_dir.exists():
        return None
    rows: list[SeedResult] = []
    for seed in seeds:
        artifact_path = rows_dir / f"seed_{seed:04d}.json"
        if not artifact_path.exists():
            return None
        rows.append(load_seed_result(artifact_path))
    return rows


def make_screen_presets(args: argparse.Namespace) -> list[ExperimentPreset]:
    base_presets = (
        [load_preset(Path(path)) for path in args.preset_paths]
        if args.preset_paths
        else [build_default_preset(args)]
    )
    prompt_candidates: list[Path | None] = [None]
    if args.prompt_candidate_paths:
        prompt_candidates.extend(Path(path) for path in args.prompt_candidate_paths)
    temperatures = parse_float_options(args.temperatures, [args.temperature])
    max_turn_options = parse_int_options(args.max_turns_options, [args.max_turns])
    actor_max_turn_options = parse_int_options(args.actor_max_turns_options, [args.actor_max_turns])
    renderer_names = (
        [part.strip() for part in args.renderer_name.split(",") if part.strip()]
        if args.renderer_name and "," in args.renderer_name
        else [args.renderer_name] if args.renderer_name else []
    )
    rendered_presets: list[ExperimentPreset] = []
    seen: set[str] = set()
    for base_preset in base_presets:
        renderer_candidates = renderer_names or [base_preset.renderer_name]
        for renderer_name, temperature, max_turns, actor_max_turns, prompt_path in itertools.product(
            renderer_candidates,
            temperatures,
            max_turn_options,
            actor_max_turn_options,
            prompt_candidates,
        ):
            candidate = replace(
                base_preset,
                environment=args.environment or base_preset.environment,
                renderer_name=renderer_name,
                disable_thinking="disable_thinking" in renderer_name,
                temperature=temperature,
                max_turns=max_turns,
                actor_max_turns=actor_max_turns,
                tracked_instruction_block=read_prompt_candidate(prompt_path, base_preset),
            )
            key = json.dumps(candidate.to_json(), sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            rendered_presets.append(candidate)
    return rendered_presets


def rank_screen_results(summary: dict[str, Any], environment: str) -> tuple[float, ...]:
    if environment == "tool_accuracy":
        return (
            -float(summary.get("parse_or_markup_failure_rate", 0.0)),
            -float(summary.get("rejected_tool_call_rate", 0.0)),
            float(summary.get("mean_reward", 0.0)),
            -float(summary.get("mean_wall_time_seconds", 0.0)),
        )
    return (
        float(summary.get("gate_pass_rate", 0.0)),
        float(summary.get("mean_reward", 0.0)),
        -float(summary.get("wait_loop_rate", 0.0)),
        -float(summary.get("mean_wall_time_seconds", 0.0)),
    )


def compare_summaries(
    *,
    baseline_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
) -> dict[str, float]:
    numeric_keys = set(baseline_summary) | set(candidate_summary)
    deltas: dict[str, float] = {}
    for key in numeric_keys:
        baseline_value = baseline_summary.get(key)
        candidate_value = candidate_summary.get(key)
        if isinstance(baseline_value, (int, float)) or isinstance(candidate_value, (int, float)):
            deltas[key] = float(candidate_value or 0.0) - float(baseline_value or 0.0)
    return deltas


def decide_promotion(
    *,
    baseline_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
) -> dict[str, Any]:
    reward_delta = float(candidate_summary.get("mean_reward", 0.0)) - float(
        baseline_summary.get("mean_reward", 0.0)
    )
    gate_delta = float(candidate_summary.get("gate_pass_rate", 0.0)) - float(
        baseline_summary.get("gate_pass_rate", 0.0)
    )
    transition_delta = float(candidate_summary.get("transition_target_rate", 0.0)) - float(
        baseline_summary.get("transition_target_rate", 0.0)
    )
    rejected_delta = float(candidate_summary.get("mean_rejected_tool_calls", 0.0)) - float(
        baseline_summary.get("mean_rejected_tool_calls", 0.0)
    )
    wait_delta = float(candidate_summary.get("mean_wait_count", 0.0)) - float(
        baseline_summary.get("mean_wait_count", 0.0)
    )
    promoted = (
        reward_delta >= 0.05 or gate_delta >= 0.05 or transition_delta >= 0.05
    ) and not (rejected_delta > 0.0 and wait_delta > 0.0)
    return {
        "promoted": promoted,
        "reward_delta": reward_delta,
        "gate_delta": gate_delta,
        "transition_delta": transition_delta,
        "rejected_tool_call_delta": rejected_delta,
        "wait_count_delta": wait_delta,
    }


def resolve_round_seed_pools(args: argparse.Namespace) -> tuple[tuple[str, list[int]], tuple[str, list[int]], tuple[str, list[int]]]:
    train_pool = (
        resolve_seed_pool(args.train_seed_pool)
        if args.train_seed_pool
        else get_gepa_train_pool(args.round_index)
    )
    val_pool = resolve_seed_pool(args.val_seed_pool)
    confirm_pool = resolve_seed_pool(args.confirm_seed_pool)
    return train_pool, val_pool, confirm_pool


def ensure_required_envs() -> None:
    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY must be set.")
    if not os.environ.get(DEFAULT_HELPER_API_KEY_ENV_VAR):
        raise RuntimeError(f"{DEFAULT_HELPER_API_KEY_ENV_VAR} must be set.")


def run_screen_phase(args: argparse.Namespace) -> None:
    ensure_required_envs()
    _, seeds = resolve_seed_pool(args.seed_pool)
    run_dir = Path(args.run_dir)
    presets = make_screen_presets(args)
    screen_environment = presets[0].environment if presets else (args.environment or DEFAULT_ENVIRONMENT)
    screen_dir = run_dir / "screen" / (args.tag or slugify(f"{args.model_name}-{screen_environment}-{args.seed_pool}"))
    screen_dir.mkdir(parents=True, exist_ok=True)
    ranked_entries: list[dict[str, Any]] = []
    for index, preset in enumerate(presets, start=1):
        config_dir = screen_dir / f"{index:03d}-{screen_preset_slug(preset)}"
        label = f"screen:{index}/{len(presets)}"
        try:
            _, summary, taxonomy, patterns, _ = asyncio.run(
                evaluate_preset(
                    preset=preset,
                    seeds=seeds,
                    output_dir=config_dir,
                    app_name=f"{args.app_name}-screen",
                    report_workers=args.report_workers,
                    per_seed_timeout_seconds=args.per_seed_timeout_seconds,
                    resume=args.resume,
                    label=label,
                )
            )
        except KeyboardInterrupt:
            print(f"[{label}] interrupted; partial artifacts remain in {config_dir}", flush=True)
            raise
        ranked_entries.append(
            {
                "preset_path": str(config_dir / "preset.json"),
                "summary_path": str(config_dir / "summary.json"),
                "taxonomy_path": str(config_dir / "taxonomy.json"),
                "patterns_path": str(config_dir / "patterns.json"),
                "rank_key": rank_screen_results(summary, preset.environment),
                "summary": summary,
                "taxonomy": taxonomy,
                "patterns": patterns,
            }
        )
    ranked_entries.sort(key=lambda entry: entry["rank_key"], reverse=True)
    for rank, entry in enumerate(ranked_entries[: args.top_k], start=1):
        preset = load_preset(Path(entry["preset_path"]))
        save_preset(screen_dir / f"top_{rank}_preset.json", preset)
    write_json(screen_dir / "ranked_results.json", ranked_entries)
    print(json.dumps(ranked_entries[: args.top_k], indent=2))


def repair_saved_screen(
    *,
    screen_dir: Path,
    seeds: Sequence[int],
    top_k: int,
) -> list[dict[str, Any]]:
    if not screen_dir.exists():
        raise RuntimeError(f"Screen directory does not exist: {screen_dir}")
    ranked_entries: list[dict[str, Any]] = []
    for config_dir in sorted(path for path in screen_dir.iterdir() if path.is_dir()):
        preset_path = config_dir / "preset.json"
        if not preset_path.exists():
            continue
        rows = load_completed_rows(output_dir=config_dir, seeds=seeds)
        if rows is None:
            continue
        preset = load_preset(preset_path)
        summary = summarize_rows(rows)
        taxonomy = build_taxonomy_summary(rows)
        patterns = build_pattern_summary(rows)
        review = build_manual_review(rows)
        write_json(config_dir / "summary.json", summary)
        write_json(config_dir / "taxonomy.json", taxonomy)
        write_json(config_dir / "patterns.json", patterns)
        write_json(config_dir / "review.json", review)
        ranked_entries.append(
            {
                "preset_path": str(preset_path),
                "summary_path": str(config_dir / "summary.json"),
                "taxonomy_path": str(config_dir / "taxonomy.json"),
                "patterns_path": str(config_dir / "patterns.json"),
                "review_path": str(config_dir / "review.json"),
                "rank_key": rank_screen_results(summary, preset.environment),
                "summary": summary,
                "taxonomy": taxonomy,
                "patterns": patterns,
                "review": review,
            }
        )
    ranked_entries.sort(key=lambda entry: entry["rank_key"], reverse=True)
    for rank, entry in enumerate(ranked_entries[:top_k], start=1):
        preset = load_preset(Path(entry["preset_path"]))
        save_preset(screen_dir / f"top_{rank}_preset.json", preset)
    write_json(screen_dir / "ranked_results.json", ranked_entries)
    return ranked_entries


def run_repair_screen_phase(args: argparse.Namespace) -> None:
    _, seeds = resolve_seed_pool(args.seed_pool)
    screen_environment = args.environment or DEFAULT_ENVIRONMENT
    screen_dir = (
        Path(args.screen_dir)
        if args.screen_dir
        else Path(args.run_dir)
        / "screen"
        / (args.tag or slugify(f"{args.model_name}-{screen_environment}-{args.seed_pool}"))
    )
    ranked_entries = repair_saved_screen(
        screen_dir=screen_dir,
        seeds=seeds,
        top_k=args.top_k,
    )
    print(json.dumps(ranked_entries[: args.top_k], indent=2))


class GEPAPromptEvaluator:
    def __init__(
        self,
        *,
        preset: ExperimentPreset,
        app_name: str,
        per_seed_timeout_seconds: float,
        metric_dir: Path | None = None,
    ) -> None:
        self.preset = preset
        self.metric_dir = metric_dir
        self._call_index = 0
        self._metric_cache: dict[tuple[str, int], SeedResult] = {}
        if self.metric_dir is not None:
            self._call_index, self._metric_cache = load_metric_cache(self.metric_dir)
        self._async_runner = asyncio.Runner()
        self._evaluator = ModalPoolEvaluator(
            preset=preset,
            app_name=app_name,
            per_seed_timeout_seconds=per_seed_timeout_seconds,
        )
        self._runner_started = False

    def _ensure_runner_started(self) -> None:
        if self._runner_started:
            return
        self._async_runner.run(self._evaluator.runner.start())
        self._runner_started = True

    def close(self) -> None:
        try:
            if self._runner_started:
                self._async_runner.run(self._evaluator.runner.aclose())
        finally:
            self._async_runner.close()

    def __call__(self, candidate: str, example: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        seed = int(example["seed"])
        candidate_hash = prompt_fingerprint(candidate)
        cached_row = self._metric_cache.get((candidate_hash, seed))
        if cached_row is not None:
            return cached_row.score, cached_row.to_json()
        self._ensure_runner_started()
        row = self._async_runner.run(self._evaluate_candidate(candidate, seed))
        if self.metric_dir is not None:
            self._call_index += 1
            record_metric_call(
                metric_dir=self.metric_dir,
                call_index=self._call_index,
                candidate=candidate,
                row=row,
            )
            self._metric_cache[(candidate_hash, seed)] = row
        return row.score, row.to_json()

    async def _evaluate_candidate(self, candidate: str, seed: int) -> SeedResult:
        preset = replace(self.preset, tracked_instruction_block=candidate)
        started_at = time.time()
        request = TrajectoryRolloutRequest(
            datum=load_datum(preset.environment, seed),
            environment_kind=preset.environment,
            model_name=preset.model_name,
            renderer_name=preset.renderer_name,
            actor_runtime=build_actor_runtime(preset),
            policy_config=RuntimePolicyConfig(max_turns=preset.max_turns),
            sampling_ref=SamplingRef(
                base_model=preset.model_name,
                base_url=os.environ.get("TINKER_BASE_URL"),
            ),
            max_tokens=preset.max_tokens,
            temperature=preset.temperature,
            trajectory_index=0,
            group_id=f"gepa:{preset.model_name}:{seed}",
            enable_logging=False,
            tracked_instruction_block=candidate,
        )
        try:
            result = await self._evaluator.runner.run_trajectory(request)
        except Exception as exc:
            row = SeedResult(
                seed=seed,
                status="error",
                execution_backend="tinker_modal",
                score=-1.0,
                reward=-1.0,
                gate=0.0,
                transition_target=0.0,
                relevant_submission=0.0,
                rejected_tool_calls=0.0,
                wait_count=0,
                read_conversation_count=0,
                compact_order_error=False,
                has_think_close=False,
                turns=0,
                max_turns=preset.max_turns,
                wall_time_seconds=time.time() - started_at,
                failure_kind=type(exc).__name__,
                failure_message=str(exc),
            )
            return replace(row, dominant_failure=classify_seed_result(row))
        return self._evaluator._result_to_seed_result(
            seed=seed,
            result=result,
            wall_time_seconds=time.time() - started_at,
            artifact_path=Path("/tmp") / f"gepa-seed-{seed}.json",
        )


def run_optimize_phase(args: argparse.Namespace) -> None:
    ensure_required_envs()
    try:
        import gepa.optimize_anything as oa
        from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything
    except ImportError as exc:
        raise RuntimeError(
            "The optimize phase requires the 'gepa' package. Add gepa[full] to the project environment first."
        ) from exc

    preset = load_preset(Path(args.preset_path)) if args.preset_path else build_default_preset(args)
    train_pool, val_pool, confirm_pool = resolve_round_seed_pools(args)
    run_dir = Path(args.run_dir)
    round_dir = run_dir / "optimize" / f"round_{args.round_index:02d}"
    round_dir.mkdir(parents=True, exist_ok=True)
    decision_path = round_dir / "decision.json"
    if args.resume and decision_path.exists():
        print(decision_path.read_text())
        return

    seed_candidate = preset.tracked_instruction_block or get_default_tracked_instruction_block(
        default_idle_sleep_seconds=0.5
    )
    baseline_taxonomy_path = run_dir / "latest_taxonomy.json"
    latest_patterns_path = run_dir / "latest_patterns.json"
    baseline_taxonomy = (
        json.loads(baseline_taxonomy_path.read_text()) if baseline_taxonomy_path.exists() else None
    )
    latest_patterns = (
        json.loads(latest_patterns_path.read_text()) if latest_patterns_path.exists() else None
    )
    evaluator = GEPAPromptEvaluator(
        preset=preset,
        app_name=f"{args.app_name}-optimize",
        per_seed_timeout_seconds=args.per_seed_timeout_seconds,
        metric_dir=round_dir / "metric_calls",
    )
    try:
        reflection_lm = resolve_reflection_lm(
            reflection_lm=args.reflection_lm,
            helper_base_url=preset.helper_base_url,
            helper_api_key_env_var=preset.helper_api_key_env_var,
            helper_http_referer=preset.helper_http_referer,
            helper_x_title=preset.helper_x_title,
            timeout_seconds=args.reflection_timeout_seconds,
        )
        background = review_background_from_taxonomy(
            baseline_taxonomy=baseline_taxonomy,
            pattern_summary=latest_patterns,
            extra_lines=[
                "Known 27B failure modes to fix if they still appear:",
                "- compact order formatting like 'A MUN-BOH' instead of 'A MUN - BOH'",
                "- passive wait/read_conversation loops",
                "- stray '</think>' markup before tool calls",
                "- finishing too late",
                "- missing best-effort submission after no counterpart reply",
            ],
        )
        result = optimize_anything(
            seed_candidate=seed_candidate,
            evaluator=evaluator,
            dataset=[{"seed": seed} for seed in train_pool[1]],
            valset=[{"seed": seed} for seed in val_pool[1]],
            objective=(
                f"Improve {preset.model_name} on Diplomacy {preset.environment}. "
                "The tracked policy should use tools directly, avoid waiting loops, coordinate efficiently, "
                "format orders legally, and finish on time."
            ),
            background=background,
            config=GEPAConfig(
                engine=EngineConfig(
                    run_dir=str(round_dir / "gepa_run"),
                    seed=args.round_index,
                    display_progress_bar=True,
                    parallel=False,
                    max_metric_calls=args.max_metric_calls,
                    capture_stdio=False,
                ),
                reflection=ReflectionConfig(
                    reflection_lm=reflection_lm,
                    reflection_minibatch_size=args.reflection_minibatch_size,
                ),
            ),
        )
    finally:
        evaluator.close()

    best_candidate = result.best_candidate
    if not isinstance(best_candidate, str):
        raise TypeError(f"Expected GEPA to return a string candidate, got {type(best_candidate)!r}")

    baseline_preset = replace(preset, tracked_instruction_block=seed_candidate)
    candidate_preset = replace(preset, tracked_instruction_block=best_candidate)
    _, baseline_val_summary, baseline_val_taxonomy, baseline_val_patterns, _ = asyncio.run(
        evaluate_preset(
            preset=baseline_preset,
            seeds=val_pool[1],
            output_dir=round_dir / "baseline_val",
            app_name=f"{args.app_name}-val",
            report_workers=args.report_workers,
            per_seed_timeout_seconds=args.per_seed_timeout_seconds,
            resume=args.resume,
            label=f"round{args.round_index}:baseline_val",
        )
    )
    _, candidate_train_summary, _, candidate_train_patterns, _ = asyncio.run(
        evaluate_preset(
            preset=candidate_preset,
            seeds=train_pool[1],
            output_dir=round_dir / "candidate_train",
            app_name=f"{args.app_name}-train",
            report_workers=args.report_workers,
            per_seed_timeout_seconds=args.per_seed_timeout_seconds,
            resume=args.resume,
            label=f"round{args.round_index}:candidate_train",
        )
    )
    _, candidate_val_summary, candidate_val_taxonomy, candidate_val_patterns, _ = asyncio.run(
        evaluate_preset(
            preset=candidate_preset,
            seeds=val_pool[1],
            output_dir=round_dir / "candidate_val",
            app_name=f"{args.app_name}-val",
            report_workers=args.report_workers,
            per_seed_timeout_seconds=args.per_seed_timeout_seconds,
            resume=args.resume,
            label=f"round{args.round_index}:candidate_val",
        )
    )
    decision = decide_promotion(
        baseline_summary=baseline_val_summary,
        candidate_summary=candidate_val_summary,
    )
    confirm_summary: dict[str, Any] | None = None
    confirm_taxonomy: dict[str, Any] | None = None
    confirm_patterns: dict[str, Any] | None = None
    if decision["promoted"]:
        _, confirm_summary, confirm_taxonomy, confirm_patterns, _ = asyncio.run(
            evaluate_preset(
                preset=candidate_preset,
                seeds=confirm_pool[1],
                output_dir=round_dir / "confirm",
                app_name=f"{args.app_name}-confirm",
                report_workers=args.report_workers,
                per_seed_timeout_seconds=args.per_seed_timeout_seconds,
                resume=args.resume,
                label=f"round{args.round_index}:confirm",
            )
        )
        decision["confirmed"] = True
        decision["confirm_summary_path"] = str(round_dir / "confirm" / "summary.json")
        write_json(run_dir / "latest_taxonomy.json", confirm_taxonomy or candidate_val_taxonomy)
        write_json(run_dir / "latest_patterns.json", confirm_patterns or candidate_val_patterns)
        save_preset(run_dir / "latest_promoted_preset.json", candidate_preset)
    else:
        decision["confirmed"] = False
        write_json(run_dir / "latest_taxonomy.json", candidate_val_taxonomy)
        write_json(run_dir / "latest_patterns.json", candidate_val_patterns)

    (round_dir / "seed_candidate.txt").write_text(seed_candidate + "\n")
    (round_dir / "best_candidate.txt").write_text(best_candidate + "\n")
    save_preset(round_dir / "baseline_preset.json", baseline_preset)
    save_preset(round_dir / "candidate_preset.json", candidate_preset)
    write_json(
        round_dir / "round_summary.json",
        {
            "round_index": args.round_index,
            "train_pool": {"name": train_pool[0], "seeds": train_pool[1]},
            "val_pool": {"name": val_pool[0], "seeds": val_pool[1]},
            "confirm_pool": {"name": confirm_pool[0], "seeds": confirm_pool[1]},
            "baseline_val_summary": baseline_val_summary,
            "candidate_train_summary": candidate_train_summary,
            "candidate_val_summary": candidate_val_summary,
            "baseline_val_taxonomy": baseline_val_taxonomy,
            "baseline_val_patterns": baseline_val_patterns,
            "candidate_train_patterns": candidate_train_patterns,
            "candidate_val_taxonomy": candidate_val_taxonomy,
            "candidate_val_patterns": candidate_val_patterns,
            "confirm_summary": confirm_summary,
            "confirm_taxonomy": confirm_taxonomy,
            "confirm_patterns": confirm_patterns,
            "deltas": compare_summaries(
                baseline_summary=baseline_val_summary,
                candidate_summary=candidate_val_summary,
            ),
            "decision": decision,
            "reflection_lm": args.reflection_lm,
            "reflection_transport": {
                "helper_base_url": preset.helper_base_url,
                "helper_api_key_env_var": preset.helper_api_key_env_var,
            },
            "max_metric_calls": args.max_metric_calls,
        },
    )
    write_json(decision_path, decision)
    print(json.dumps({"decision": decision, "round_dir": str(round_dir)}, indent=2))


def run_report_phase(args: argparse.Namespace) -> None:
    ensure_required_envs()
    preset = load_preset(Path(args.preset_path)) if args.preset_path else build_default_preset(args)
    run_dir = Path(args.run_dir)
    _, seeds = resolve_seed_pool(args.seed_pool)
    report_dir = run_dir / "report" / (args.tag or slugify(f"{preset.model_name}-{args.seed_pool}"))
    _, summary, taxonomy, patterns, review = asyncio.run(
        evaluate_preset(
            preset=preset,
            seeds=seeds,
            output_dir=report_dir,
            app_name=f"{args.app_name}-report",
            report_workers=args.report_workers,
            per_seed_timeout_seconds=args.per_seed_timeout_seconds,
            resume=args.resume,
            label="report",
        )
    )
    write_json(report_dir / "report.json", {"summary": summary, "taxonomy": taxonomy, "patterns": patterns, "review": review})
    write_json(run_dir / "latest_taxonomy.json", taxonomy)
    write_json(run_dir / "latest_patterns.json", patterns)
    write_json(run_dir / "latest_manual_review.json", review)
    print(json.dumps({"summary": summary, "taxonomy": taxonomy, "patterns": patterns}, indent=2))


def run_compare_phase(args: argparse.Namespace) -> None:
    ensure_required_envs()
    if not args.preset_a_path or not args.preset_b_path:
        raise RuntimeError("--preset-a-path and --preset-b-path are required for compare.")
    preset_a = load_preset(Path(args.preset_a_path))
    preset_b = load_preset(Path(args.preset_b_path))
    _, seeds = resolve_seed_pool(args.seed_pool)
    compare_dir = Path(args.run_dir) / "compare" / (args.tag or slugify(f"{preset_a.model_name}-vs-{preset_b.model_name}-{args.seed_pool}"))
    _, summary_a, taxonomy_a, patterns_a, review_a = asyncio.run(
        evaluate_preset(
            preset=preset_a,
            seeds=seeds,
            output_dir=compare_dir / "candidate_a",
            app_name=f"{args.app_name}-compare-a",
            report_workers=args.report_workers,
            per_seed_timeout_seconds=args.per_seed_timeout_seconds,
            resume=args.resume,
            label="compare:a",
        )
    )
    _, summary_b, taxonomy_b, patterns_b, review_b = asyncio.run(
        evaluate_preset(
            preset=preset_b,
            seeds=seeds,
            output_dir=compare_dir / "candidate_b",
            app_name=f"{args.app_name}-compare-b",
            report_workers=args.report_workers,
            per_seed_timeout_seconds=args.per_seed_timeout_seconds,
            resume=args.resume,
            label="compare:b",
        )
    )
    comparison = {
        "candidate_a": {"preset_path": args.preset_a_path, "summary": summary_a, "taxonomy": taxonomy_a, "patterns": patterns_a, "review": review_a},
        "candidate_b": {"preset_path": args.preset_b_path, "summary": summary_b, "taxonomy": taxonomy_b, "patterns": patterns_b, "review": review_b},
        "delta_b_minus_a": compare_summaries(baseline_summary=summary_a, candidate_summary=summary_b),
    }
    write_json(compare_dir / "comparison.json", comparison)
    print(json.dumps(comparison, indent=2))


def run_train_smoke_phase(args: argparse.Namespace) -> None:
    preset = load_preset(Path(args.preset_path)) if args.preset_path else build_default_preset(args)
    train_dir = Path(args.run_dir) / "train_smoke" / (args.tag or slugify(preset.model_name))
    train_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = train_dir / "tracked_instruction_block.txt"
    prompt_path.write_text((preset.tracked_instruction_block or "").strip() + "\n")
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/train_tinker_grpo_curriculum.py",
        "--model-name",
        preset.model_name,
        "--renderer-name",
        preset.renderer_name,
        "--log-root",
        args.log_root,
        "--run-name",
        args.train_smoke_run_name or f"{slugify(preset.model_name)}-gepa-smoke",
        "--openrouter-model",
        preset.helper_model,
        "--openrouter-base-url",
        preset.helper_base_url,
        "--openrouter-api-key-env-var",
        preset.helper_api_key_env_var,
        "--http-referer",
        preset.helper_http_referer,
        "--x-title",
        preset.helper_x_title,
        "--actor-max-turns",
        str(preset.actor_max_turns),
        "--session-timeout-seconds",
        str(preset.session_timeout_seconds),
        "--stage1-train-examples",
        "10",
        "--stage1-eval-examples",
        "2",
        "--stage1-batch-size",
        "1",
        "--stage1-group-size",
        "1",
        "--stage1-max-turns",
        str(preset.max_turns),
        "--stage1-max-tokens",
        str(preset.max_tokens),
        "--stage2-train-examples",
        "10",
        "--stage2-eval-examples",
        "2",
        "--stage2-batch-size",
        "1",
        "--stage2-group-size",
        "1",
        "--stage2-max-turns",
        str(max(preset.max_turns, 12)),
        "--stage2-max-tokens",
        str(max(preset.max_tokens, 384)),
        "--tracked-instruction-block-path",
        str(prompt_path),
    ]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> None:
    args = parse_args()
    if args.phase == "screen":
        run_screen_phase(args)
        return
    if args.phase == "repair_screen":
        run_repair_screen_phase(args)
        return
    if args.phase == "optimize":
        run_optimize_phase(args)
        return
    if args.phase == "report":
        run_report_phase(args)
        return
    if args.phase == "compare":
        run_compare_phase(args)
        return
    if args.phase == "train_smoke":
        run_train_smoke_phase(args)
        return
    raise RuntimeError(f"Unsupported phase: {args.phase}")


if __name__ == "__main__":
    main()
