#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from openai import AsyncOpenAI

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDORED_COOKBOOK_ROOT = REPO_ROOT / "vendor" / "tinker-cookbook"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(VENDORED_COOKBOOK_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDORED_COOKBOOK_ROOT))

from data_generator import build_full_press_dataset, build_tool_accuracy_dataset
from tinker_cookbook import renderers
from tinker_cookbook.rl.types import SamplingRef
from tinker_cookbook.tool_use.types import ToolInput
from tinker_training.diplomacy_adapter import (
    ActorRuntimeConfig,
    OpenRouterHeaders,
    RuntimePolicyConfig,
    _build_system_prompt,
    _cleanup_episode_state,
    _get_rubric_for_environment,
    _initialize_episode_context,
    _new_score_semaphore,
    _tool_to_openai_schema,
    build_actor_configs,
    get_default_renderer_name,
)
from tinker_training.rollout_backends import (
    ModalTrajectorySandboxRunner,
    TrajectoryRolloutRequest,
    deserialize_trajectory,
)

EnvironmentKind = Literal["tool_accuracy", "full_press"]
ExecutionBackend = Literal["openrouter", "tinker_modal"]
RootBackend = Literal["auto", "openrouter", "tinker"]


@dataclass(frozen=True)
class ModelSpec:
    lineup_name: str
    openrouter_model: str | None
    prefill_per_million: float
    sample_per_million: float
    note: str = ""
    tinker_model_name: str | None = None
    tinker_renderer_name: str | None = None


@dataclass
class BenchmarkResult:
    lineup_name: str
    openrouter_model: str | None
    note: str
    environment: str
    seed: int
    execution_backend: ExecutionBackend
    status: str
    wall_time_seconds: float
    reward: float | None = None
    trace_len: int | None = None
    turns: int | None = None
    stop_condition: str | None = None
    root_prompt_tokens: int = 0
    root_completion_tokens: int = 0
    estimated_root_cost_usd: float | None = None
    metrics: dict[str, float] | None = None
    error: str | None = None
    fallback_reason: str | None = None


MODEL_SPECS: list[ModelSpec] = [
    ModelSpec(
        "Qwen/Qwen3-4B-Instruct-2507",
        "qwen/qwen3-4b:free",
        0.07,
        0.22,
        "closest available OpenRouter variant",
        tinker_model_name="Qwen/Qwen3-4B-Instruct-2507",
        tinker_renderer_name="qwen3_instruct",
    ),
    ModelSpec("Qwen/Qwen3-8B", "qwen/qwen3-8b", 0.13, 0.40),
    ModelSpec(
        "Qwen/Qwen3-8B-Base",
        None,
        0.13,
        0.40,
        "no OpenRouter base-model match found; pricing inferred from Qwen/Qwen3-8B",
    ),
    ModelSpec("Qwen/Qwen3-30B-A3B", "qwen/qwen3-30b-a3b", 0.12, 0.30),
    ModelSpec(
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "qwen/qwen3-30b-a3b-instruct-2507",
        0.12,
        0.30,
        "pricing inferred from Qwen/Qwen3-30B-A3B",
        tinker_model_name="Qwen/Qwen3-30B-A3B-Instruct-2507",
        tinker_renderer_name="qwen3_instruct",
    ),
    ModelSpec(
        "Qwen/Qwen3-30B-A3B-Base",
        None,
        0.12,
        0.30,
        "no OpenRouter base-model match found; pricing inferred from Qwen/Qwen3-30B-A3B",
    ),
    ModelSpec("Qwen/Qwen3-VL-30B-A3B-Instruct", "qwen/qwen3-vl-30b-a3b-instruct", 0.18, 0.44),
    ModelSpec("Qwen/Qwen3-32B", "qwen/qwen3-32b", 0.49, 1.47),
    ModelSpec(
        "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "qwen/qwen3-235b-a22b",
        0.68,
        1.70,
        "closest available OpenRouter id",
    ),
    ModelSpec("Qwen/Qwen3-VL-235B-A22B-Instruct", "qwen/qwen3-vl-235b-a22b-instruct", 1.02, 2.56),
    ModelSpec("Qwen/Qwen3.5-397B-A17B", "qwen/qwen3.5-397b-a17b", 2.00, 5.00),
    ModelSpec("Qwen/Qwen3.5-35B-A3B", "qwen/qwen3.5-35b-a3b", 0.36, 0.89),
    ModelSpec("Qwen/Qwen3.5-27B", "qwen/qwen3.5-27b", 1.24, 3.73),
    ModelSpec("Qwen/Qwen3.5-4B", None, 0.22, 0.67, "no OpenRouter match found"),
    ModelSpec(
        "meta-llama/Llama-3.2-1B",
        "meta-llama/llama-3.2-1b-instruct",
        0.03,
        0.09,
        "OpenRouter only exposes instruct",
        tinker_model_name="meta-llama/Llama-3.2-1B-Instruct",
        tinker_renderer_name="llama3",
    ),
    ModelSpec(
        "meta-llama/Llama-3.2-3B",
        "meta-llama/llama-3.2-3b-instruct",
        0.06,
        0.18,
        "OpenRouter only exposes instruct",
        tinker_model_name="meta-llama/Llama-3.2-3B-Instruct",
        tinker_renderer_name="llama3",
    ),
    ModelSpec(
        "meta-llama/Llama-3.1-8B",
        "meta-llama/llama-3.1-8b-instruct",
        0.13,
        0.40,
        "OpenRouter only exposes instruct",
        tinker_model_name="meta-llama/Llama-3.1-8B-Instruct",
        tinker_renderer_name="llama3",
    ),
    ModelSpec(
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/llama-3.1-8b-instruct",
        0.13,
        0.40,
        "pricing inferred from meta-llama/Llama-3.1-8B",
        tinker_model_name="meta-llama/Llama-3.1-8B-Instruct",
        tinker_renderer_name="llama3",
    ),
    ModelSpec(
        "meta-llama/Llama-3.1-70B",
        "meta-llama/llama-3.1-70b-instruct",
        1.05,
        3.16,
        "OpenRouter only exposes instruct",
        tinker_model_name="meta-llama/Llama-3.1-70B-Instruct",
        tinker_renderer_name="llama3",
    ),
    ModelSpec(
        "meta-llama/Llama-3.3-70B-Instruct",
        "meta-llama/llama-3.3-70b-instruct",
        1.05,
        3.16,
        "pricing inferred from meta-llama/Llama-3.1-70B",
        tinker_model_name="meta-llama/Llama-3.3-70B-Instruct",
        tinker_renderer_name="llama3",
    ),
    ModelSpec("deepseek-ai/DeepSeek-V3.1", "deepseek/deepseek-chat-v3.1", 1.13, 2.81, "using chat variant"),
    ModelSpec("deepseek-ai/DeepSeek-V3.1-Base", None, 1.13, 2.81, "no OpenRouter base-model match found"),
    ModelSpec("openai/gpt-oss-120b", "openai/gpt-oss-120b", 0.18, 0.44),
    ModelSpec("openai/gpt-oss-20b", "openai/gpt-oss-20b", 0.12, 0.30),
    ModelSpec("moonshotai/Kimi-K2-Thinking", "moonshotai/kimi-k2-thinking", 0.98, 2.44),
    ModelSpec("moonshotai/Kimi-K2.5", "moonshotai/kimi-k2.5", 1.47, 3.66),
    ModelSpec("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "nvidia/nemotron-3-nano-30b-a3b", 0.13, 0.33),
    ModelSpec("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16", "nvidia/nemotron-3-super-120b-a12b", 0.38, 0.96),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the Tinker lineup on Diplomacy tasks, using OpenRouter first and "
            "optionally falling back to a Modal-backed Tinker sampling client."
        )
    )
    parser.add_argument("--environment", choices=["tool_accuracy", "full_press"], default="tool_accuracy")
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--max-turns", type=int, default=16)
    parser.add_argument("--actor-max-turns", type=int, default=12)
    parser.add_argument("--session-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--model-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--verbose-turns", action="store_true")
    parser.add_argument(
        "--background-actor-model",
        default="openai/gpt-5.4-mini",
        help="OpenRouter model used for the fixed counterpart actors.",
    )
    parser.add_argument(
        "--root-backend",
        choices=("auto", "openrouter", "tinker"),
        default="auto",
        help=(
            "Backend for the root policy. 'auto' tries OpenRouter first and can fall back to "
            "Tinker on Modal. 'tinker' runs the root policy directly through Tinker on Modal."
        ),
    )
    parser.add_argument(
        "--fallback-provider",
        choices=("none", "tinker"),
        default="tinker",
        help="Fallback provider to use when OpenRouter cannot complete a model benchmark.",
    )
    parser.add_argument(
        "--modal-app-name",
        default="diplomacy-lineup-bench",
        help="Modal app name used for fallback rollouts.",
    )
    parser.add_argument("--modal-cpu", type=float, default=2.0)
    parser.add_argument("--modal-memory-mb", type=int, default=4096)
    parser.add_argument(
        "--output-dir",
        default=".tmp/openrouter_lineup_bench",
        help="Directory where JSON results and Pareto plot are written.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        help="Optional subset of lineup names to benchmark.",
    )
    return parser.parse_args()


def load_datum(environment: EnvironmentKind, seed: int) -> dict[str, Any]:
    if environment == "tool_accuracy":
        return build_tool_accuracy_dataset(num_sessions=1, seed=seed).to_list()[0]
    return build_full_press_dataset(num_sessions=1, seed=seed).to_list()[0]


def build_actor_runtime(
    *,
    actor_max_turns: int,
    session_timeout_seconds: float,
    background_actor_model: str,
) -> ActorRuntimeConfig:
    return ActorRuntimeConfig(
        actor_configs=build_actor_configs(model=background_actor_model),
        actor_max_turns=actor_max_turns,
        session_timeout_seconds=session_timeout_seconds,
        default_idle_sleep_seconds=0.5,
        openrouter_headers=OpenRouterHeaders(
            http_referer="https://local.codex",
            x_title="diplomacy-lineup-bench",
        ),
    )


def estimate_cost_usd(spec: ModelSpec, *, prompt_tokens: int, completion_tokens: int) -> float:
    return (
        prompt_tokens * spec.prefill_per_million + completion_tokens * spec.sample_per_million
    ) / 1_000_000.0


def merge_transition_metrics(backend_metrics: dict[str, float | int], trajectory: Any) -> dict[str, float]:
    merged: dict[str, float] = {}
    for transition in trajectory.transitions:
        for key, value in transition.metrics.items():
            if not isinstance(value, (int, float)):
                continue
            numeric_value = float(value)
            if key.startswith("tool/") or key.startswith("episode/"):
                merged[key] = merged.get(key, 0.0) + numeric_value
            else:
                merged[key] = numeric_value
    for key, value in backend_metrics.items():
        if isinstance(value, (int, float)):
            merged[key] = float(value)
    return merged


async def run_single_openrouter_benchmark(
    *,
    spec: ModelSpec,
    environment: EnvironmentKind,
    seed: int,
    max_turns: int,
    actor_max_turns: int,
    session_timeout_seconds: float,
    temperature: float,
    background_actor_model: str,
    verbose_turns: bool,
) -> BenchmarkResult:
    started_at = time.time()
    if not spec.openrouter_model:
        return BenchmarkResult(
            lineup_name=spec.lineup_name,
            openrouter_model=None,
            note=spec.note,
            environment=environment,
            seed=seed,
            execution_backend="openrouter",
            status="skipped",
            wall_time_seconds=0.0,
            error="No OpenRouter model match configured.",
        )

    datum = load_datum(environment, seed)
    actor_runtime = build_actor_runtime(
        actor_max_turns=actor_max_turns,
        session_timeout_seconds=session_timeout_seconds,
        background_actor_model=background_actor_model,
    )
    policy_config = RuntimePolicyConfig(max_turns=max_turns)
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
            ),
        },
        *state["prompt"],
    ]
    tools = ctx.tool_executor.all_tools()
    tool_map = {tool_obj.name: tool_obj for tool_obj in tools}
    tool_schemas = [_tool_to_openai_schema(tool_obj) for tool_obj in tools]
    client = AsyncOpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://local.codex",
            "X-Title": "diplomacy-lineup-bench",
        },
        timeout=180,
    )

    prompt_tokens = 0
    completion_tokens = 0
    assistant_turns = 0
    try:
        for turn_index in range(max_turns):
            if verbose_turns:
                print(f"    turn {turn_index}", flush=True)
            response = await client.chat.completions.create(
                model=spec.openrouter_model,
                messages=history,
                tools=tool_schemas,
                temperature=temperature,
            )
            usage = response.usage
            if usage is not None:
                prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
                completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
            message = response.choices[0].message.model_dump(exclude_none=True)
            history.append(message)
            assistant_turns += 1

            tool_calls = list(message.get("tool_calls") or [])
            if verbose_turns:
                print(f"    tool_calls={len(tool_calls)}", flush=True)
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
                    text = renderers.format_content_as_string(text_parts) if text_parts else ""
                    if text.strip():
                        state["final_answer"] = text.strip()
                state["completion"] = history[2:]
                rubric = _get_rubric_for_environment(environment)
                await rubric.score_rollout(state, score_sem=_new_score_semaphore())
                break

        metrics = {
            key: float(value)
            for key, value in (state.get("metrics") or {}).items()
            if isinstance(value, (int, float))
        }
        return BenchmarkResult(
            lineup_name=spec.lineup_name,
            openrouter_model=spec.openrouter_model,
            note=spec.note,
            environment=environment,
            seed=seed,
            execution_backend="openrouter",
            status="ok",
            wall_time_seconds=time.time() - started_at,
            reward=float(state.get("reward")) if state.get("reward") is not None else None,
            trace_len=len(state.get("chatroom_tool_trace", [])),
            turns=assistant_turns,
            stop_condition=str(state.get("stop_condition")) if state.get("stop_condition") else None,
            root_prompt_tokens=prompt_tokens,
            root_completion_tokens=completion_tokens,
            estimated_root_cost_usd=estimate_cost_usd(
                spec,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            ),
            metrics=metrics,
        )
    except Exception as exc:
        return BenchmarkResult(
            lineup_name=spec.lineup_name,
            openrouter_model=spec.openrouter_model,
            note=spec.note,
            environment=environment,
            seed=seed,
            execution_backend="openrouter",
            status="error",
            wall_time_seconds=time.time() - started_at,
            error=f"{type(exc).__name__}: {exc}",
            root_prompt_tokens=prompt_tokens,
            root_completion_tokens=completion_tokens,
        )
    finally:
        await client.close()
        await _cleanup_episode_state(state)


async def run_single_tinker_modal_benchmark(
    *,
    spec: ModelSpec,
    environment: EnvironmentKind,
    seed: int,
    max_turns: int,
    actor_max_turns: int,
    session_timeout_seconds: float,
    temperature: float,
    background_actor_model: str,
    runner: ModalTrajectorySandboxRunner,
    fallback_reason: str | None = None,
) -> BenchmarkResult:
    started_at = time.time()
    tinker_model_name = spec.tinker_model_name or spec.lineup_name
    actor_runtime = build_actor_runtime(
        actor_max_turns=actor_max_turns,
        session_timeout_seconds=session_timeout_seconds,
        background_actor_model=background_actor_model,
    )
    policy_config = RuntimePolicyConfig(max_turns=max_turns)
    renderer_name = spec.tinker_renderer_name or get_default_renderer_name(
        tinker_model_name,
        disable_thinking=False,
    )
    request = TrajectoryRolloutRequest(
        datum=load_datum(environment, seed),
        environment_kind=environment,
        model_name=tinker_model_name,
        renderer_name=renderer_name,
        actor_runtime=actor_runtime,
        policy_config=policy_config,
        sampling_ref=SamplingRef(
            base_model=tinker_model_name,
            base_url=os.environ.get("TINKER_BASE_URL"),
        ),
        max_tokens=512,
        temperature=temperature,
        trajectory_index=0,
        group_id=f"bench:{spec.lineup_name}:{seed}",
        enable_logging=False,
    )

    try:
        result = await runner.run_trajectory(request)
        if result.failure_kind is not None or result.trajectory is None:
            return BenchmarkResult(
                lineup_name=spec.lineup_name,
                openrouter_model=spec.openrouter_model,
                note=spec.note,
                environment=environment,
                seed=seed,
                execution_backend="tinker_modal",
                status="error",
                wall_time_seconds=time.time() - started_at,
                error=f"{result.failure_kind or 'unknown'}: {result.failure_message or ''}".strip(),
                fallback_reason=fallback_reason,
            )

        trajectory = deserialize_trajectory(result.trajectory)
        prompt_tokens = sum(int(transition.ob.length) for transition in trajectory.transitions)
        completion_tokens = sum(len(transition.ac.tokens) for transition in trajectory.transitions)
        merged_metrics = merge_transition_metrics(result.metrics, trajectory)
        reward = sum(float(transition.reward) for transition in trajectory.transitions) + float(
            result.final_reward
        )
        return BenchmarkResult(
            lineup_name=spec.lineup_name,
            openrouter_model=spec.openrouter_model,
            note=spec.note,
            environment=environment,
            seed=seed,
            execution_backend="tinker_modal",
            status="ok",
            wall_time_seconds=time.time() - started_at,
            reward=reward,
            trace_len=len(trajectory.transitions),
            turns=len(trajectory.transitions),
            root_prompt_tokens=prompt_tokens,
            root_completion_tokens=completion_tokens,
            estimated_root_cost_usd=estimate_cost_usd(
                spec,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            ),
            metrics=merged_metrics,
            fallback_reason=fallback_reason,
        )
    except Exception as exc:
        return BenchmarkResult(
            lineup_name=spec.lineup_name,
            openrouter_model=spec.openrouter_model,
            note=spec.note,
            environment=environment,
            seed=seed,
            execution_backend="tinker_modal",
            status="error",
            wall_time_seconds=time.time() - started_at,
            error=f"{type(exc).__name__}: {exc}",
            fallback_reason=fallback_reason,
        )


def pareto_front(results: list[BenchmarkResult]) -> list[BenchmarkResult]:
    ranked = sorted(
        [
            result
            for result in results
            if result.status == "ok"
            and result.reward is not None
            and result.estimated_root_cost_usd is not None
            and math.isfinite(result.estimated_root_cost_usd)
        ],
        key=lambda item: (item.estimated_root_cost_usd, -(item.reward or 0.0)),
    )
    frontier: list[BenchmarkResult] = []
    best_reward = -float("inf")
    for result in ranked:
        reward = float(result.reward or 0.0)
        if reward > best_reward:
            frontier.append(result)
            best_reward = reward
    return frontier


def write_pareto_plot(results: list[BenchmarkResult], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    valid = [
        result
        for result in results
        if result.status == "ok"
        and result.reward is not None
        and result.estimated_root_cost_usd is not None
    ]
    if not valid:
        return
    frontier = pareto_front(valid)
    plt.figure(figsize=(12, 8))

    by_backend: dict[str, list[BenchmarkResult]] = {}
    for item in valid:
        by_backend.setdefault(item.execution_backend, []).append(item)
    for backend_name, items in by_backend.items():
        plt.scatter(
            [item.estimated_root_cost_usd for item in items],
            [item.reward for item in items],
            alpha=0.7,
            label=backend_name,
        )
        for item in items:
            plt.annotate(
                item.lineup_name.split("/")[-1],
                (item.estimated_root_cost_usd, item.reward),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8,
            )

    if frontier:
        plt.plot(
            [item.estimated_root_cost_usd for item in frontier],
            [item.reward for item in frontier],
            color="red",
            linewidth=2,
            label="Pareto frontier",
        )
    plt.xlabel("Estimated root-policy inference cost per rollout (USD)")
    plt.ylabel(f"{valid[0].environment} reward")
    plt.title("Diplomacy lineup: cost vs performance")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


async def maybe_fallback_to_tinker(
    *,
    result: BenchmarkResult,
    spec: ModelSpec,
    args: argparse.Namespace,
    runner: ModalTrajectorySandboxRunner | None,
) -> BenchmarkResult:
    if args.fallback_provider != "tinker" or result.status == "ok":
        return result
    if runner is None:
        return result
    if not os.environ.get("TINKER_API_KEY", "").strip():
        result.error = (
            f"{result.error or 'OpenRouter failed.'} Tinker fallback requested but TINKER_API_KEY is not set."
        )
        return result
    print("  falling back to Tinker sampling client via Modal", flush=True)
    return await run_single_tinker_modal_benchmark(
        spec=spec,
        environment=args.environment,
        seed=args.seed,
        max_turns=args.max_turns,
        actor_max_turns=args.actor_max_turns,
        session_timeout_seconds=args.session_timeout_seconds,
        temperature=args.temperature,
        background_actor_model=args.background_actor_model,
        runner=runner,
        fallback_reason=result.error or result.status,
    )


async def amain(args: argparse.Namespace) -> int:
    if not os.environ.get("OPENROUTER_API_KEY", "").strip():
        raise SystemExit("OPENROUTER_API_KEY is not set.")
    if (
        args.root_backend == "tinker" or args.fallback_provider == "tinker"
    ) and not os.environ.get("TINKER_API_KEY", "").strip():
        raise SystemExit("TINKER_API_KEY is not set.")
    requested = set(args.models or [])
    specs = [spec for spec in MODEL_SPECS if not requested or spec.lineup_name in requested]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{args.environment}_seed{args.seed}_results.json"
    plot_path = output_dir / f"{args.environment}_seed{args.seed}_pareto.png"

    runner: ModalTrajectorySandboxRunner | None = None
    if args.fallback_provider == "tinker" or args.root_backend == "tinker":
        runner = ModalTrajectorySandboxRunner(
            app_name=args.modal_app_name,
            timeout_seconds=max(1, int(math.ceil(args.model_timeout_seconds))),
            cpu=args.modal_cpu,
            memory_mb=args.modal_memory_mb,
        )

    results: list[BenchmarkResult] = []
    try:
        for index, spec in enumerate(specs, start=1):
            print(f"[{index}/{len(specs)}] {spec.lineup_name} -> {spec.openrouter_model or 'SKIP'}", flush=True)
            if args.root_backend == "tinker":
                assert runner is not None
                try:
                    result = await asyncio.wait_for(
                        run_single_tinker_modal_benchmark(
                            spec=spec,
                            environment=args.environment,
                            seed=args.seed,
                            max_turns=args.max_turns,
                            actor_max_turns=args.actor_max_turns,
                            session_timeout_seconds=args.session_timeout_seconds,
                            temperature=args.temperature,
                            background_actor_model=args.background_actor_model,
                            runner=runner,
                            fallback_reason=None,
                        ),
                        timeout=args.model_timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    result = BenchmarkResult(
                        lineup_name=spec.lineup_name,
                        openrouter_model=spec.openrouter_model,
                        note=spec.note,
                        environment=args.environment,
                        seed=args.seed,
                        execution_backend="tinker_modal",
                        status="timeout",
                        wall_time_seconds=args.model_timeout_seconds,
                        error=f"Timed out after {args.model_timeout_seconds:.1f}s",
                    )
            else:
                try:
                    result = await asyncio.wait_for(
                        run_single_openrouter_benchmark(
                            spec=spec,
                            environment=args.environment,
                            seed=args.seed,
                            max_turns=args.max_turns,
                            actor_max_turns=args.actor_max_turns,
                            session_timeout_seconds=args.session_timeout_seconds,
                            temperature=args.temperature,
                            background_actor_model=args.background_actor_model,
                            verbose_turns=args.verbose_turns,
                        ),
                        timeout=args.model_timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    result = BenchmarkResult(
                        lineup_name=spec.lineup_name,
                        openrouter_model=spec.openrouter_model,
                        note=spec.note,
                        environment=args.environment,
                        seed=args.seed,
                        execution_backend="openrouter",
                        status="timeout",
                        wall_time_seconds=args.model_timeout_seconds,
                        error=f"Timed out after {args.model_timeout_seconds:.1f}s",
                    )

                if args.root_backend == "auto":
                    result = await maybe_fallback_to_tinker(
                        result=result,
                        spec=spec,
                        args=args,
                        runner=runner,
                    )
            print(json.dumps(asdict(result), indent=2), flush=True)
            results.append(result)
            json_path.write_text(json.dumps([asdict(item) for item in results], indent=2))
            write_pareto_plot(results, plot_path)
    finally:
        if runner is not None:
            await runner.aclose()

    print(f"Wrote results to {json_path}")
    print(f"Wrote Pareto plot to {plot_path}")
    return 0


def main() -> int:
    args = parse_args()
    return asyncio.run(amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
