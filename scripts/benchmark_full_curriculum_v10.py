#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDORED_COOKBOOK_ROOT = REPO_ROOT / "vendor" / "tinker-cookbook"
if str(VENDORED_COOKBOOK_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDORED_COOKBOOK_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tinker_cookbook import tokenizer_utils

from scripts.train_tinker_grpo_curriculum import build_full_v1_stage_specs
from tinker_training.curriculum import (
    CurriculumConfig,
    DEFAULT_OPENROUTER_API_KEY_ENV_VAR,
    DEFAULT_OPENROUTER_BASE_URL,
    ModalRolloutConfig,
    StageSpec,
    run_curriculum,
)
from tinker_training.diplomacy_adapter import get_default_renderer_name
from tinker_training.diplomacy_gepa import (
    build_manual_review,
    build_pattern_summary,
    build_taxonomy_summary,
    summarize_rows,
)
from tinker_training.eval_utils import (
    DEFAULT_HELPER_MODEL,
    build_actor_runtime,
    build_runtime_policy,
    build_sampling_ref,
    load_datum,
    seed_result_from_tinker_result,
    write_json,
)
from tinker_training.prompt_family import DEFAULT_PROMPT_FAMILY_DIR, load_prompt_family_blocks
from tinker_training.rollout_backends import ModalTrajectorySandboxRunner, TrajectoryRolloutRequest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ENVIRONMENTS = (
    "tool_accuracy",
    "target_execution",
    "supported_target",
    "cooperative_press",
    "full_press",
)
MAX_TURNS = {
    "tool_accuracy": 14,
    "target_execution": 12,
    "supported_target": 14,
    "cooperative_press": 16,
    "full_press": 20,
}
PRIMARY_METRIC = {
    "tool_accuracy": "constraints_satisfied_rate",
    "target_execution": "transition_target_rate",
    "supported_target": "gate_pass_rate",
    "cooperative_press": "gate_pass_rate",
    "full_press": "gate_pass_rate",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the GPT-5.4-teacher SFT plus five-stage RL curriculum pipeline."
    )
    parser.add_argument(
        "--phase",
        choices=("screen", "sft", "isolated_smoke", "curriculum_smoke", "all"),
        default="all",
        help="Which benchmark phase to run.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / ".tmp" / "full_curriculum_bench"),
        help="Directory for benchmark outputs.",
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen3-30B-A3B-Instruct-2507",
        help="Student model evaluated in every condition.",
    )
    parser.add_argument(
        "--renderer-name",
        default=None,
        help="Optional explicit renderer override. Leave unset to use the recommended renderer.",
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="Root policy temperature.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Global RL learning rate.")
    parser.add_argument("--sft-learning-rate", type=float, default=1e-5, help="Global SFT learning rate.")
    parser.add_argument("--lora-rank", type=int, default=32, help="Shared LoRA rank.")
    parser.add_argument("--screen-seed-start", type=int, default=21, help="First zero-shot screen seed.")
    parser.add_argument("--screen-seed-end", type=int, default=84, help="Last zero-shot screen seed.")
    parser.add_argument("--prompt-family-dir", default=str(DEFAULT_PROMPT_FAMILY_DIR))
    parser.add_argument("--helper-model", default=DEFAULT_HELPER_MODEL)
    parser.add_argument("--actor-max-turns", type=int, default=6)
    parser.add_argument("--session-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--modal-app-name", default="diplomacy-curriculum-bench")
    parser.add_argument("--modal-timeout-seconds", type=int, default=900)
    parser.add_argument("--modal-cpu", type=float, default=2.0)
    parser.add_argument("--modal-memory-mb", type=int, default=4096)
    parser.add_argument("--log-root", default=str(REPO_ROOT / ".tmp" / "curriculum_smokes"))
    parser.add_argument("--wandb-project", default="diplomacy-grpo")
    parser.add_argument("--openrouter-model", default=DEFAULT_HELPER_MODEL)
    parser.add_argument("--openrouter-base-url", default=DEFAULT_OPENROUTER_BASE_URL)
    parser.add_argument("--openrouter-api-key-env-var", default=DEFAULT_OPENROUTER_API_KEY_ENV_VAR)
    parser.add_argument("--http-referer", default="https://local.codex")
    parser.add_argument("--x-title", default="diplomacy-full-curriculum-bench")
    parser.add_argument(
        "--sft-output-dir",
        default=str(REPO_ROOT / "data" / "diplomacy_sft"),
        help="Directory for generated SFT data shards.",
    )
    parser.add_argument(
        "--sft-run-root",
        default=str(REPO_ROOT / ".tmp" / "diplomacy_sft_runs"),
        help="Directory for SFT training runs.",
    )
    parser.add_argument(
        "--sft-checkpoint-path",
        default=None,
        help="Optional existing SFT checkpoint to reuse instead of running the SFT phase.",
    )
    return parser.parse_args()


def _screen_seeds(args: argparse.Namespace) -> list[int]:
    return list(range(args.screen_seed_start, args.screen_seed_end + 1))


def _condition_sampling_ref(args: argparse.Namespace, checkpoint_path: str | None) -> Any:
    return build_sampling_ref(
        model_name=args.model_name,
        checkpoint_path=checkpoint_path,
        base_url=os.environ.get("TINKER_BASE_URL"),
    )


def _screen_prompt(prompt_blocks: dict[str, str], environment_kind: str, prompt_mode: str) -> str:
    if prompt_mode == "raw_full_press_prompt":
        return prompt_blocks["full_press"]
    return prompt_blocks[environment_kind]


def _condition_output_dir(base_dir: Path, condition_name: str, environment_kind: str, prompt_mode: str) -> Path:
    return base_dir / condition_name / environment_kind / prompt_mode


async def evaluate_screen_condition(
    *,
    args: argparse.Namespace,
    condition_name: str,
    checkpoint_path: str | None,
    prompt_blocks: dict[str, str],
    output_dir: Path,
) -> dict[str, Any]:
    tokenizer = tokenizer_utils.get_tokenizer(args.model_name)
    actor_runtime = build_actor_runtime(
        actor_max_turns=args.actor_max_turns,
        session_timeout_seconds=args.session_timeout_seconds,
        background_actor_model=args.helper_model,
        background_actor_base_url=args.openrouter_base_url,
        background_actor_api_key_env_var=args.openrouter_api_key_env_var,
        http_referer=args.http_referer,
        x_title=args.x_title,
    )
    sampling_ref = _condition_sampling_ref(args, checkpoint_path)
    runner = ModalTrajectorySandboxRunner(
        app_name=args.modal_app_name,
        timeout_seconds=args.modal_timeout_seconds,
        cpu=args.modal_cpu,
        memory_mb=args.modal_memory_mb,
        single_use_containers=True,
        max_inputs=1,
        retries=0,
    )
    await runner.start()
    try:
        report: dict[str, Any] = {}
        for environment_kind in ENVIRONMENTS:
            report[environment_kind] = {}
            for prompt_mode in ("raw_full_press_prompt", "environment_specific_prompt"):
                env_output_dir = _condition_output_dir(output_dir, condition_name, environment_kind, prompt_mode)
                rows_dir = env_output_dir / "rows"
                rows_dir.mkdir(parents=True, exist_ok=True)
                rows = []
                tracked_instruction_block = _screen_prompt(prompt_blocks, environment_kind, prompt_mode)
                for seed in _screen_seeds(args):
                    artifact_path = rows_dir / f"seed_{seed:04d}.json"
                    started_at = time.time()
                    request = TrajectoryRolloutRequest(
                        datum=load_datum(environment_kind, seed),
                        environment_kind=environment_kind,
                        model_name=args.model_name,
                        renderer_name=args.renderer_name
                        or get_default_renderer_name(args.model_name, disable_thinking=False),
                        actor_runtime=actor_runtime,
                        policy_config=build_runtime_policy(MAX_TURNS[environment_kind]),
                        sampling_ref=sampling_ref,
                        max_tokens=512,
                        temperature=args.temperature,
                        trajectory_index=0,
                        group_id=f"curriculum-bench:{condition_name}:{environment_kind}:{prompt_mode}:{seed}",
                        enable_logging=False,
                        tracked_instruction_block=tracked_instruction_block,
                    )
                    result = await runner.run_trajectory(request)
                    row = seed_result_from_tinker_result(
                        environment=environment_kind,
                        seed=seed,
                        result=result,
                        tokenizer=tokenizer,
                        max_turns=MAX_TURNS[environment_kind],
                        wall_time_seconds=time.time() - started_at,
                        artifact_path=artifact_path,
                    )
                    artifact_path.write_text(json.dumps(row.to_json(), indent=2, sort_keys=True) + "\n")
                    rows.append(row)
                summary = summarize_rows(rows)
                taxonomy = build_taxonomy_summary(rows)
                patterns = build_pattern_summary(rows)
                review = build_manual_review(rows)
                write_json(env_output_dir / "summary.json", summary)
                write_json(env_output_dir / "taxonomy.json", taxonomy)
                write_json(env_output_dir / "patterns.json", patterns)
                write_json(env_output_dir / "review_samples.json", review)
                report[environment_kind][prompt_mode] = {
                    "summary": summary,
                    "taxonomy": taxonomy,
                    "patterns": patterns,
                    "review_samples": review,
                }
        return report
    finally:
        await runner.aclose()


def _stage_train_examples_for_smoke(stage: StageSpec) -> int:
    return stage.batch_size * 10


def _smoke_stage(stage: StageSpec) -> StageSpec:
    return replace(
        stage,
        num_train_examples=_stage_train_examples_for_smoke(stage),
        num_eval_examples=2,
        group_size=1,
    )


def _build_curriculum_config(
    *,
    args: argparse.Namespace,
    prompt_blocks: dict[str, str],
    stages: tuple[StageSpec, ...],
    run_name: str,
    initial_checkpoint_path: str | None = None,
) -> CurriculumConfig:
    renderer_name = args.renderer_name or get_default_renderer_name(
        args.model_name,
        disable_thinking=False,
    )
    return CurriculumConfig(
        model_name=args.model_name,
        renderer_name=renderer_name,
        log_root=args.log_root,
        run_name=run_name,
        wandb_project=args.wandb_project,
        openrouter_model=args.openrouter_model,
        openrouter_base_url=args.openrouter_base_url,
        openrouter_api_key_env_var=args.openrouter_api_key_env_var,
        http_referer=args.http_referer,
        x_title=args.x_title,
        actor_max_turns=args.actor_max_turns,
        session_timeout_seconds=args.session_timeout_seconds,
        default_idle_sleep_seconds=0.5,
        max_message_length=2000,
        save_every=5,
        eval_every=2,
        num_groups_to_log=1,
        rollout_json_export=True,
        stages=stages,
        modal_rollout=ModalRolloutConfig(
            app_name=args.modal_app_name,
            timeout_seconds=args.modal_timeout_seconds,
            cpu=args.modal_cpu,
            memory_mb=args.modal_memory_mb,
        ),
        initial_checkpoint_path=initial_checkpoint_path,
        tracked_instruction_block=prompt_blocks["full_press"],
    )


def _latest_stage_metrics(stage_log_dir: Path) -> dict[str, Any]:
    metrics_path = stage_log_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return {}
    last_payload: dict[str, Any] = {}
    for line in metrics_path.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            last_payload = payload
    return last_payload


def _first_nonzero_batch(stage_log_dir: Path) -> int | None:
    metrics_path = stage_log_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return None
    for line in metrics_path.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        metrics = payload.get("metrics", payload)
        if any(
            float(metrics.get(metric, 0.0)) > 0.0
            for metric in (
                "rubric/transition_target_without_press_metric",
                "rubric/transition_target_satisfied_metric",
                "rubric/full_press_gate_metric",
                "rubric/constraints_satisfied_metric",
            )
        ):
            return int(payload.get("step", payload.get("batch", 0)))
    return None


async def run_isolated_smokes(
    *,
    args: argparse.Namespace,
    prompt_blocks: dict[str, str],
    initial_checkpoint_path: str | None,
    label: str,
) -> dict[str, Any]:
    stage_specs = tuple(_smoke_stage(stage) for stage in build_full_v1_stage_specs(
        prompt_blocks=prompt_blocks,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
    ))
    report: dict[str, Any] = {}
    for stage in stage_specs:
        run_name = f"{label}-{stage.name}"
        config = _build_curriculum_config(
            args=args,
            prompt_blocks=prompt_blocks,
            stages=(stage,),
            run_name=run_name,
            initial_checkpoint_path=initial_checkpoint_path,
        )
        manifest_path = await run_curriculum(config)
        manifest = json.loads(manifest_path.read_text())
        stage_log_dir = Path(manifest["stages"][0]["log_dir"])
        report[stage.name] = {
            "manifest_path": str(manifest_path),
            "status": manifest["status"],
            "first_nonzero_success_batch": _first_nonzero_batch(stage_log_dir),
            "last_metrics": _latest_stage_metrics(stage_log_dir),
        }
    return report


async def run_curriculum_smoke(
    *,
    args: argparse.Namespace,
    prompt_blocks: dict[str, str],
    initial_checkpoint_path: str | None,
    label: str,
) -> dict[str, Any]:
    stage_specs = tuple(
        _smoke_stage(stage)
        for stage in build_full_v1_stage_specs(
            prompt_blocks=prompt_blocks,
            learning_rate=args.learning_rate,
            lora_rank=args.lora_rank,
        )
    )
    config = _build_curriculum_config(
        args=args,
        prompt_blocks=prompt_blocks,
        stages=stage_specs,
        run_name=f"{label}-full-v1",
        initial_checkpoint_path=initial_checkpoint_path,
    )
    manifest_path = await run_curriculum(config)
    manifest = json.loads(manifest_path.read_text())
    per_stage = {}
    for stage_entry in manifest.get("stages", []):
        stage_log_dir = Path(stage_entry["log_dir"])
        per_stage[stage_entry["stage"]] = {
            "status": stage_entry["status"],
            "first_nonzero_success_batch": _first_nonzero_batch(stage_log_dir),
            "last_metrics": _latest_stage_metrics(stage_log_dir),
        }
    return {
        "manifest_path": str(manifest_path),
        "manifest": manifest,
        "per_stage": per_stage,
    }


def run_sft_phase(args: argparse.Namespace, output_dir: Path) -> str:
    dataset_dir = Path(args.sft_output_dir).expanduser().resolve()
    train_root = Path(args.sft_run_root).expanduser().resolve()
    generate_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "generate_diplomacy_sft_dataset.py"),
        "--output-dir",
        str(dataset_dir),
        "--prompt-family-dir",
        args.prompt_family_dir,
    ]
    train_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train_diplomacy_sft.py"),
        "--model-name",
        args.model_name,
        "--dataset-path",
        str(dataset_dir / "merged.jsonl"),
        "--log-root",
        str(train_root),
        "--learning-rate",
        str(args.sft_learning_rate),
        "--lora-rank",
        str(args.lora_rank),
    ]
    subprocess.run(generate_cmd, check=True, cwd=str(REPO_ROOT))
    subprocess.run(train_cmd, check=True, cwd=str(REPO_ROOT))
    latest_runs = sorted(train_root.glob("*/sft_manifest.json"))
    if not latest_runs:
        raise FileNotFoundError(f"No SFT manifest found under {train_root}")
    manifest_path = latest_runs[-1]
    manifest = json.loads(manifest_path.read_text())
    write_json(output_dir / "sft_report.json", {
        "dataset_dir": str(dataset_dir),
        "manifest_path": str(manifest_path),
        "manifest": manifest,
    })
    checkpoint = manifest.get("checkpoint") or {}
    checkpoint_path = checkpoint.get("state_path")
    if not checkpoint_path:
        raise ValueError(f"SFT manifest at {manifest_path} does not contain a final checkpoint path.")
    return str(checkpoint_path)


def _environment_primary_value(environment_kind: str, summary: dict[str, Any]) -> float:
    if environment_kind == "tool_accuracy":
        return float(summary.get("gate_pass_rate", 0.0))
    if environment_kind == "target_execution":
        return float(summary.get("transition_target_rate", 0.0))
    return float(summary.get("gate_pass_rate", 0.0))


def _compare_prompt_family(report: dict[str, Any]) -> dict[str, Any]:
    comparisons = {}
    improved = 0
    for environment_kind, payload in report.items():
        raw_summary = payload["raw_full_press_prompt"]["summary"]
        env_summary = payload["environment_specific_prompt"]["summary"]
        raw_value = _environment_primary_value(environment_kind, raw_summary)
        env_value = _environment_primary_value(environment_kind, env_summary)
        if env_value >= raw_value:
            improved += 1
        comparisons[environment_kind] = {
            "raw_primary_metric": raw_value,
            "environment_specific_primary_metric": env_value,
            "primary_metric_name": PRIMARY_METRIC[environment_kind],
        }
    return {"improved_or_matched_environments": improved, "comparisons": comparisons}


def write_zero_shot_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = ["# Zero-shot report", ""]
    for condition_name, condition_report in report.items():
        lines.append(f"## {condition_name}")
        comparison = _compare_prompt_family(condition_report)
        lines.append(
            f"- stage-specific prompt matched or beat raw full-press prompt in {comparison['improved_or_matched_environments']}/5 environments"
        )
        for environment_kind, payload in condition_report.items():
            raw_summary = payload["raw_full_press_prompt"]["summary"]
            env_summary = payload["environment_specific_prompt"]["summary"]
            lines.append(
                f"- `{environment_kind}`: raw `{_environment_primary_value(environment_kind, raw_summary):.3f}` vs env-specific `{_environment_primary_value(environment_kind, env_summary):.3f}`"
            )
        lines.append("")
    path.write_text("\n".join(lines))


def final_recommendation(
    *,
    base_curriculum: dict[str, Any] | None,
    sft_curriculum: dict[str, Any] | None,
) -> dict[str, Any]:
    recommendation = {
        "recommended_default": "legacy_two_stage",
        "reason": "The new full_v1 stack has not been benchmarked yet.",
    }
    if not base_curriculum or not sft_curriculum:
        return recommendation
    base_stage5 = base_curriculum.get("per_stage", {}).get("stage5_full_press", {})
    sft_stage5 = sft_curriculum.get("per_stage", {}).get("stage5_full_press", {})
    base_metrics = base_stage5.get("last_metrics", {}).get("metrics", base_stage5.get("last_metrics", {}))
    sft_metrics = sft_stage5.get("last_metrics", {}).get("metrics", sft_stage5.get("last_metrics", {}))
    base_gate = float(base_metrics.get("rubric/full_press_gate_metric", 0.0))
    sft_gate = float(sft_metrics.get("rubric/full_press_gate_metric", 0.0))
    if sft_gate >= base_gate:
        recommendation = {
            "recommended_default": "full_v1",
            "reason": "SFT+RL matched or beat the base RL curriculum on the stage-5 full-press gate metric.",
            "stage5_gate_base_plus_rl": base_gate,
            "stage5_gate_sft_plus_rl": sft_gate,
        }
    else:
        recommendation = {
            "recommended_default": "legacy_two_stage",
            "reason": "SFT+RL did not beat the base RL curriculum on the stage-5 full-press gate metric.",
            "stage5_gate_base_plus_rl": base_gate,
            "stage5_gate_sft_plus_rl": sft_gate,
        }
    return recommendation


async def main_async(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_blocks = load_prompt_family_blocks(args.prompt_family_dir)
    zero_shot_report: dict[str, Any] = {}
    base_curriculum_report: dict[str, Any] | None = None
    sft_curriculum_report: dict[str, Any] | None = None

    if args.phase in {"screen", "all"}:
        zero_shot_report["base_zero_shot"] = await evaluate_screen_condition(
            args=args,
            condition_name="base_zero_shot",
            checkpoint_path=None,
            prompt_blocks=prompt_blocks,
            output_dir=output_dir / "screen",
        )
        if args.sft_checkpoint_path:
            zero_shot_report["sft_zero_shot"] = await evaluate_screen_condition(
                args=args,
                condition_name="sft_zero_shot",
                checkpoint_path=args.sft_checkpoint_path,
                prompt_blocks=prompt_blocks,
                output_dir=output_dir / "screen",
            )
        write_json(output_dir / "zero_shot_report.json", zero_shot_report)
        write_zero_shot_markdown(output_dir / "zero_shot_report.md", zero_shot_report)

    sft_checkpoint_path = args.sft_checkpoint_path
    if args.phase in {"sft", "all"} and not sft_checkpoint_path:
        sft_checkpoint_path = run_sft_phase(args, output_dir)

    if args.phase in {"isolated_smoke", "all"}:
        isolated_report = {
            "base_plus_rl": await run_isolated_smokes(
                args=args,
                prompt_blocks=prompt_blocks,
                initial_checkpoint_path=None,
                label="base-plus-rl",
            )
        }
        if sft_checkpoint_path:
            isolated_report["sft_plus_rl"] = await run_isolated_smokes(
                args=args,
                prompt_blocks=prompt_blocks,
                initial_checkpoint_path=sft_checkpoint_path,
                label="sft-plus-rl",
            )
        write_json(output_dir / "isolated_smoke_report.json", isolated_report)

    if args.phase in {"curriculum_smoke", "all"}:
        base_curriculum_report = await run_curriculum_smoke(
            args=args,
            prompt_blocks=prompt_blocks,
            initial_checkpoint_path=None,
            label="base-plus-rl",
        )
        curriculum_report = {"base_plus_rl": base_curriculum_report}
        if sft_checkpoint_path:
            sft_curriculum_report = await run_curriculum_smoke(
                args=args,
                prompt_blocks=prompt_blocks,
                initial_checkpoint_path=sft_checkpoint_path,
                label="sft-plus-rl",
            )
            curriculum_report["sft_plus_rl"] = sft_curriculum_report
        write_json(output_dir / "curriculum_smoke_report.json", curriculum_report)

    write_json(
        output_dir / "final_recommendation.json",
        final_recommendation(
            base_curriculum=base_curriculum_report,
            sft_curriculum=sft_curriculum_report,
        ),
    )
    return 0


def main() -> int:
    args = parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
