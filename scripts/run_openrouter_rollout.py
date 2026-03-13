#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from rich.console import Console
from rich.syntax import Syntax
from rich.text import Text

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("aiohttp").setLevel(logging.WARNING)
logging.getLogger("aiohttp.access").setLevel(logging.WARNING)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import environments.full_press as full_press
import environments.tool_accuracy as tool_accuracy


DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "google/gemini-3-flash-preview"
DEFAULT_OUTPUT_DIR = ".tmp/rollouts"
STANDARD_POWERS = [
    "AUSTRIA",
    "ENGLAND",
    "FRANCE",
    "GERMANY",
    "ITALY",
    "RUSSIA",
    "TURKEY",
]



@dataclass
class RolloutSummary:
    environment: str
    seed: int
    reward: float | None
    final_answer: str
    stop_condition: str | None
    rejected_tool_calls: int
    tracked_power: str
    relevant_powers: list[str]
    actor_statuses: dict[str, str]
    order_constraints: list[dict[str, Any]]
    transition_target: dict[str, Any] | None
    trace_len: int
    output_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run live Diplomacy RLM rollouts against an OpenAI-compatible API such as OpenRouter."
    )
    parser.add_argument(
        "--environment",
        choices=["tool_accuracy", "full_press"],
        required=True,
        help="Which Diplomacy environment to run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=21,
        help="Base dataset seed for the first rollout.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of sequential rollouts to run, incrementing the seed each time.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model name to send to the API.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--api-key-env-var",
        default="OPENROUTER_API_KEY",
        help="Environment variable that stores the API key.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where rollout JSON files and the manifest are written.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature passed to the model.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=32,
        help="Root rollout max turns.",
    )
    parser.add_argument(
        "--actor-max-turns",
        type=int,
        default=24,
        help="Relevant actor rollout max turns.",
    )
    parser.add_argument(
        "--sub-tool-max-turns",
        type=int,
        default=5,
        help="Max tool turns per sub-LLM batch item.",
    )
    parser.add_argument(
        "--global-session-timeout-seconds",
        type=float,
        default=90.0,
        help="Per-session timeout for waiting on actors and messages.",
    )
    parser.add_argument(
        "--default-idle-sleep-seconds",
        type=float,
        default=0.5,
        help="Suggested poll sleep interval surfaced to the model.",
    )
    parser.add_argument(
        "--code-execution-timeout",
        type=int,
        default=60,
        help="Sandbox code execution timeout in seconds.",
    )
    parser.add_argument(
        "--http-referer",
        default="https://local.codex",
        help="Optional HTTP-Referer header for OpenRouter or compatible providers.",
    )
    parser.add_argument(
        "--x-title",
        default="diplomacy-rl",
        help="Optional X-Title header for OpenRouter or compatible providers.",
    )
    return parser.parse_args()


def build_actor_configs(base_url: str, api_key_env_var: str, model: str) -> dict[str, dict[str, Any]]:
    return {
        power: {
            "api_base_url": base_url,
            "api_key_var": api_key_env_var,
            "model": model,
            "timeout": 180.0,
        }
        for power in STANDARD_POWERS
    }


def build_env(args: argparse.Namespace, seed: int):
    actor_configs = build_actor_configs(args.base_url, args.api_key_env_var, args.model)
    common_kwargs = {
        "actor_configs": actor_configs,
        "num_train_examples": 1,
        "seed": seed,
        "max_turns": args.max_turns,
        "sub_tool_max_turns": args.sub_tool_max_turns,
        "actor_max_turns": args.actor_max_turns,
        "global_session_timeout_seconds": args.global_session_timeout_seconds,
        "default_idle_sleep_seconds": args.default_idle_sleep_seconds,
        "code_execution_timeout": args.code_execution_timeout,
    }
    if args.environment == "tool_accuracy":
        return tool_accuracy.load_environment(**common_kwargs)
    return full_press.load_environment(**common_kwargs)


POWER_COLORS: dict[str, str] = {
    "AUSTRIA": "red",
    "ENGLAND": "blue",
    "FRANCE": "cyan",
    "GERMANY": "bright_black",
    "ITALY": "green",
    "RUSSIA": "magenta",
    "TURKEY": "yellow",
}


def _patch_execute_code(env, label: str = "root", color: str = "bold cyan") -> None:
    """Patch env._execute_code to print code as it's executed."""
    original = env._execute_code
    repl_lang = env.repl_language  # "bash" or "python"

    async def patched(code: str, state) -> dict[str, Any]:
        turn = state.get("repl_call_count", 0) + 1

        # Header
        header = Text(f"\n── {label} ", style=color)
        header.append(f"turn {turn} ", style=f"{color} dim")
        header.append(f"[{repl_lang}] ", style="bold yellow")
        header.append("─" * max(1, 50 - len(f"── {label} turn {turn} [{repl_lang}] ")), style="dim")
        console.print(header)

        # Code
        console.print(Syntax(code.strip(), repl_lang, theme="monokai", line_numbers=False, word_wrap=True))

        # Execute
        result = await original(code, state)

        # Output
        status = result.get("status", "")
        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")
        output = (stdout + "\n" + stderr).strip()
        status_style = "green" if status == "success" else "red"
        console.print(Text(f"  [{status}]", style=status_style), end=" ")

        if output:
            lines = output.splitlines()
            if len(lines) > 20:
                output_text = "\n".join(lines[:20]) + f"\n... ({len(lines) - 20} more lines)"
            else:
                output_text = "\n".join(lines)
            console.print(Text(output_text, style="dim"))
        else:
            console.print(Text("(no output)", style="dim"))

        return result

    env._execute_code = patched


def _patch_actor_rollouts(env) -> None:
    """Patch _run_actor_rollout to also instrument actor envs."""
    original_run = env._run_actor_rollout

    async def patched_run(actor_name: str, state):
        # Patch _build_actor_env to instrument the actor env
        original_build = env._build_actor_env

        def patched_build():
            actor_env = original_build()
            color = POWER_COLORS.get(actor_name, "white")
            _patch_execute_code(actor_env, label=actor_name, color=f"bold {color}")
            return actor_env

        env._build_actor_env = patched_build
        try:
            return await original_run(actor_name, state)
        finally:
            env._build_actor_env = original_build

    env._run_actor_rollout = patched_run


def print_summary(state: dict[str, Any]) -> None:
    """Print final rollout summary."""
    info = state.get("info", {})
    if not isinstance(info, dict):
        info = {}

    console.print()
    console.print(Text("═" * 60, style="bold"))
    console.print(Text("  ROLLOUT COMPLETE", style="bold green"))
    console.print(Text("═" * 60, style="bold"))

    tracked = info.get("tracked_power", "?")
    reward = state.get("reward", "?")
    reward_color = "green" if (reward or 0) > 0 else "red"
    console.print(f"  Power:               [bold cyan]{tracked}[/bold cyan]")
    console.print(f"  Reward:              [bold {reward_color}]{reward}[/bold {reward_color}]")
    console.print(f"  Final answer:        [bold]{state.get('final_answer', '')}[/bold]")
    console.print(f"  Stop condition:      {state.get('stop_condition', '')}")
    rejected = state.get("rejected_tool_calls", 0)
    console.print(f"  Rejected tool calls: [{'red' if rejected else 'green'}]{rejected}[/]")
    console.print()


def compact_trace(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in trace:
        compact.append(
            {
                "tool_name": item.get("tool_name"),
                "success": item.get("success"),
                "participants": item.get("participants"),
                "error": item.get("error"),
                "result_preview": item.get("result_preview"),
            }
        )
    return compact


async def run_once(args: argparse.Namespace, *, seed: int, run_index: int, output_dir: Path) -> RolloutSummary:
    log.info("Run %d | seed=%d | env=%s | model=%s", run_index, seed, args.environment, args.model)
    log.info("Building environment...")
    env = build_env(args, seed)
    # Patch to stream code execution to terminal
    _patch_execute_code(env, label="ROOT", color="bold cyan")
    _patch_actor_rollouts(env)

    log.info("Dataset ready (%d rows). Connecting to %s ...", len(env.dataset), args.base_url)
    client = AsyncOpenAI(
        api_key=os.environ[args.api_key_env_var],
        base_url=args.base_url,
        default_headers={
            "HTTP-Referer": args.http_referer,
            "X-Title": args.x_title,
        },
        timeout=180,
    )
    try:
        row = env.dataset[0]
        log.info("Starting rollout (max_turns=%d) ...", args.max_turns)
        state = await env.rollout(row, client, args.model, {"temperature": args.temperature})
        log.info("Rollout complete. Scoring...")
        await env.rubric.score_rollout(state)
        log.info("Scoring done.")
        trace = [item for item in state.get("chatroom_tool_trace", []) if isinstance(item, dict)]
        print_summary(state)
        info = state.get("info", {})
        if not isinstance(info, dict):
            info = {}
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        filename = f"{args.environment}_seed{seed}_run{run_index}_{timestamp}.json"
        output_path = output_dir / filename
        payload = {
            "environment": args.environment,
            "seed": seed,
            "model": args.model,
            "base_url": args.base_url,
            "reward": state.get("reward"),
            "final_answer": state.get("final_answer"),
            "stop_condition": state.get("stop_condition"),
            "rejected_tool_calls": state.get("rejected_tool_calls"),
            "tracked_power": info.get("tracked_power"),
            "relevant_powers": info.get("relevant_powers"),
            "order_constraints": info.get("order_constraints"),
            "transition_target": info.get("transition_target"),
            "task_config": info.get("task_config"),
            "actor_statuses": state.get("actor_statuses"),
            "trace_overview": compact_trace(trace),
            "chatroom_tool_trace": trace,
            "completion": state.get("completion"),
        }
        output_path.write_text(json.dumps(payload, indent=2, default=str))
        log.info("Saved rollout to %s", output_path)
        return RolloutSummary(
            environment=args.environment,
            seed=seed,
            reward=payload["reward"],
            final_answer=str(payload["final_answer"] or ""),
            stop_condition=payload["stop_condition"],
            rejected_tool_calls=int(payload["rejected_tool_calls"] or 0),
            tracked_power=str(payload["tracked_power"] or ""),
            relevant_powers=[
                str(item) for item in payload.get("relevant_powers", []) if isinstance(item, str)
            ],
            actor_statuses={
                str(key): str(value)
                for key, value in (payload.get("actor_statuses") or {}).items()
                if isinstance(key, str)
            },
            order_constraints=[
                dict(item) for item in payload.get("order_constraints", []) if isinstance(item, dict)
            ],
            transition_target=dict(payload["transition_target"]) if isinstance(payload["transition_target"], dict) else None,
            trace_len=len(trace),
            output_path=str(output_path),
        )
    finally:
        await client.close()
        with contextlib.suppress(Exception):
            await env._teardown()


async def amain(args: argparse.Namespace) -> int:
    api_key = os.environ.get(args.api_key_env_var, "").strip()
    if not api_key:
        raise SystemExit(
            f"Environment variable {args.api_key_env_var} is not set. "
            "Export it first, for example: source ~/.bashrc"
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[RolloutSummary] = []
    for run_index in range(args.runs):
        seed = args.seed + run_index
        summary = await run_once(args, seed=seed, run_index=run_index, output_dir=output_dir)
        summaries.append(summary)
        print(json.dumps(asdict(summary), indent=2))
    manifest_path = output_dir / f"{args.environment}_manifest_seed{args.seed}_runs{args.runs}.json"
    manifest_path.write_text(json.dumps([asdict(summary) for summary in summaries], indent=2))
    print(f"Wrote manifest to {manifest_path}")
    return 0


def main() -> int:
    args = parse_args()
    return asyncio.run(amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
