#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDORED_COOKBOOK_ROOT = REPO_ROOT / "vendor" / "tinker-cookbook"
if str(VENDORED_COOKBOOK_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDORED_COOKBOOK_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tinker_training.curriculum import (
    CurriculumConfig,
    DEFAULT_LOG_ROOT,
    DEFAULT_MODEL_NAME,
    DEFAULT_OPENROUTER_API_KEY_ENV_VAR,
    DEFAULT_OPENROUTER_BASE_URL,
    DEFAULT_OPENROUTER_MODEL,
    DEFAULT_WANDB_PROJECT,
    ModalRolloutConfig,
    StageSpec,
    run_curriculum,
)
from tinker_training.diplomacy_adapter import get_default_renderer_name
from tinker_training.prompt_family import DEFAULT_PROMPT_FAMILY_DIR, load_prompt_family_blocks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CURRICULUM_PRESETS = ("legacy_two_stage", "full_v1")
FULL_V1_STAGE_DEFAULTS: tuple[dict[str, Any], ...] = (
    {
        "name": "stage1_tool_accuracy",
        "environment_kind": "tool_accuracy",
        "num_train_examples": 64,
        "num_eval_examples": 8,
        "batch_size": 16,
        "group_size": 4,
        "max_tokens": 256,
        "max_turns": 14,
        "max_trajectory_tokens": 8192,
        "train_seed": 21,
        "eval_seed": 10_021,
    },
    {
        "name": "stage2_target_execution",
        "environment_kind": "target_execution",
        "num_train_examples": 64,
        "num_eval_examples": 8,
        "batch_size": 16,
        "group_size": 4,
        "max_tokens": 320,
        "max_turns": 12,
        "max_trajectory_tokens": 8192,
        "train_seed": 37,
        "eval_seed": 10_037,
    },
    {
        "name": "stage3_supported_target",
        "environment_kind": "supported_target",
        "num_train_examples": 64,
        "num_eval_examples": 8,
        "batch_size": 16,
        "group_size": 4,
        "max_tokens": 320,
        "max_turns": 14,
        "max_trajectory_tokens": 8192,
        "train_seed": 53,
        "eval_seed": 10_053,
    },
    {
        "name": "stage4_cooperative_press",
        "environment_kind": "cooperative_press",
        "num_train_examples": 48,
        "num_eval_examples": 8,
        "batch_size": 12,
        "group_size": 4,
        "max_tokens": 384,
        "max_turns": 16,
        "max_trajectory_tokens": 12_288,
        "train_seed": 69,
        "eval_seed": 10_069,
    },
    {
        "name": "stage5_full_press",
        "environment_kind": "full_press",
        "num_train_examples": 48,
        "num_eval_examples": 8,
        "batch_size": 12,
        "group_size": 4,
        "max_tokens": 384,
        "max_turns": 20,
        "max_trajectory_tokens": 12_288,
        "train_seed": 85,
        "eval_seed": 10_085,
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Tinker GRPO curriculum for Diplomacy with Modal-isolated rollouts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--curriculum-preset",
        choices=CURRICULUM_PRESETS,
        default="legacy_two_stage",
        help="Which curriculum preset to run.",
    )
    parser.add_argument(
        "--prompt-family-dir",
        default=str(DEFAULT_PROMPT_FAMILY_DIR),
        help="Directory containing the stage-specific tracked-policy prompts for `full_v1`.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Base model to fine-tune with Tinker.",
    )
    parser.add_argument(
        "--renderer-name",
        default=None,
        help="Optional explicit renderer override. Leave unset to use the recommended renderer for the model.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Use the thinking-enabled recommended renderer when the model supports it.",
    )
    parser.add_argument(
        "--log-root",
        default=DEFAULT_LOG_ROOT,
        help="Root directory that will contain the curriculum manifest and per-stage logs.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional fixed run name. Leave unset to generate one from the model name and timestamp.",
    )
    parser.add_argument(
        "--wandb-project",
        default=DEFAULT_WANDB_PROJECT,
        help="Weights & Biases project for the single curriculum training run.",
    )
    parser.add_argument(
        "--initial-checkpoint-path",
        default=None,
        help="Optional checkpoint to warm-start from when no local curriculum state exists yet.",
    )
    parser.add_argument(
        "--tracked-instruction-block-path",
        default=None,
        help="Optional path to a tracked-policy instruction block override used as the global fallback prompt.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Global learning rate used by the `full_v1` curriculum preset.",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=32,
        help="Global LoRA rank used by the `full_v1` curriculum preset.",
    )

    parser.add_argument("--modal-app-name", default="diplomacy-grpo-rollouts", help="Modal app name used for remote rollout workers.")
    parser.add_argument("--modal-timeout-seconds", type=int, default=900, help="Per-trajectory Modal timeout.")
    parser.add_argument("--modal-cpu", type=float, default=2.0, help="vCPU request for each Modal rollout worker.")
    parser.add_argument("--modal-memory-mb", type=int, default=4096, help="Memory request for each Modal rollout worker in MB.")

    parser.add_argument(
        "--openrouter-model",
        default=DEFAULT_OPENROUTER_MODEL,
        help="Model used for the non-trained counterpart powers.",
    )
    parser.add_argument(
        "--openrouter-base-url",
        default=DEFAULT_OPENROUTER_BASE_URL,
        help="OpenRouter-compatible base URL for the counterpart actors.",
    )
    parser.add_argument(
        "--openrouter-api-key-env-var",
        default=DEFAULT_OPENROUTER_API_KEY_ENV_VAR,
        help="Environment variable that stores the OpenRouter API key.",
    )
    parser.add_argument("--http-referer", default="https://local.codex", help="HTTP-Referer header sent to OpenRouter.")
    parser.add_argument("--x-title", default="diplomacy-grpo", help="X-Title header sent to OpenRouter.")
    parser.add_argument("--actor-max-turns", type=int, default=18, help="Maximum turns allowed for each background actor trajectory.")
    parser.add_argument("--session-timeout-seconds", type=float, default=90.0, help="Per-environment wall clock timeout.")
    parser.add_argument("--default-idle-sleep-seconds", type=float, default=0.5, help="Default polling interval used by wait().")
    parser.add_argument("--max-message-length", type=int, default=2000, help="Maximum press message length accepted by the tool executor.")
    parser.add_argument("--save-every", type=int, default=10, help="Save a resumable state checkpoint every N global batches.")
    parser.add_argument("--eval-every", type=int, default=5, help="Run evaluation every N global batches.")
    parser.add_argument("--num-groups-to-log", type=int, default=2, help="How many rollout groups per batch get rich HTML/logtree output.")
    parser.add_argument("--disable-rollout-json-export", action="store_true", help="Disable JSONL rollout summary export.")

    parser.add_argument("--stage1-train-examples", type=int, default=64, help="Number of training sessions sampled for stage 1 (`tool_accuracy`).")
    parser.add_argument("--stage1-eval-examples", type=int, default=8, help="Number of evaluation sessions sampled for stage 1.")
    parser.add_argument("--stage1-batch-size", type=int, default=16, help="Number of prompt groups per optimizer batch in stage 1.")
    parser.add_argument("--stage1-group-size", type=int, default=4, help="Number of trajectories sampled per prompt group in stage 1.")
    parser.add_argument("--stage1-max-tokens", type=int, default=256, help="Maximum model output tokens per trajectory for stage 1.")
    parser.add_argument("--stage1-max-turns", type=int, default=14, help="Maximum environment turns per trajectory for stage 1.")
    parser.add_argument("--stage1-max-trajectory-tokens", type=int, default=8192, help="Hard cap on full trajectory token budget for stage 1.")
    parser.add_argument("--stage1-train-seed", type=int, default=21, help="Training dataset seed for stage 1.")
    parser.add_argument("--stage1-eval-seed", type=int, default=10_021, help="Evaluation dataset seed for stage 1.")
    parser.add_argument("--stage1-learning-rate", type=float, default=3e-5, help="Learning rate used while stage 1 is active.")
    parser.add_argument("--stage1-lora-rank", type=int, default=32, help="LoRA rank. Must match stage 2 because the curriculum now uses one Tinker run.")

    parser.add_argument("--stage2-train-examples", type=int, default=48, help="Number of training sessions sampled for stage 2 (`full_press`).")
    parser.add_argument("--stage2-eval-examples", type=int, default=8, help="Number of evaluation sessions sampled for stage 2.")
    parser.add_argument("--stage2-batch-size", type=int, default=12, help="Number of prompt groups per optimizer batch in stage 2.")
    parser.add_argument("--stage2-group-size", type=int, default=4, help="Number of trajectories sampled per prompt group in stage 2.")
    parser.add_argument("--stage2-max-tokens", type=int, default=384, help="Maximum model output tokens per trajectory for stage 2.")
    parser.add_argument("--stage2-max-turns", type=int, default=20, help="Maximum environment turns per trajectory for stage 2.")
    parser.add_argument("--stage2-max-trajectory-tokens", type=int, default=12288, help="Hard cap on full trajectory token budget for stage 2.")
    parser.add_argument("--stage2-train-seed", type=int, default=37, help="Training dataset seed for stage 2.")
    parser.add_argument("--stage2-eval-seed", type=int, default=10_037, help="Evaluation dataset seed for stage 2.")
    parser.add_argument("--stage2-learning-rate", type=float, default=2e-5, help="Learning rate used while stage 2 is active.")
    parser.add_argument("--stage2-lora-rank", type=int, default=32, help="LoRA rank. Must match stage 1 because the curriculum now uses one Tinker run.")
    return parser.parse_args()


def build_stages(args: argparse.Namespace) -> tuple[StageSpec, ...]:
    curriculum_preset = getattr(args, "curriculum_preset", "legacy_two_stage")
    prompt_family_dir = getattr(args, "prompt_family_dir", str(DEFAULT_PROMPT_FAMILY_DIR))
    learning_rate = getattr(args, "learning_rate", 2e-5)
    lora_rank = getattr(args, "lora_rank", 32)
    if curriculum_preset == "full_v1":
        prompt_blocks = load_prompt_family_blocks(prompt_family_dir)
        return build_full_v1_stage_specs(
            prompt_blocks=prompt_blocks,
            learning_rate=learning_rate,
            lora_rank=lora_rank,
        )

    return (
        StageSpec(
            name="stage1_tool_accuracy",
            environment_kind="tool_accuracy",
            num_train_examples=args.stage1_train_examples,
            num_eval_examples=args.stage1_eval_examples,
            batch_size=args.stage1_batch_size,
            group_size=args.stage1_group_size,
            max_tokens=args.stage1_max_tokens,
            max_turns=args.stage1_max_turns,
            max_trajectory_tokens=args.stage1_max_trajectory_tokens,
            train_seed=args.stage1_train_seed,
            eval_seed=args.stage1_eval_seed,
            learning_rate=args.stage1_learning_rate,
            lora_rank=args.stage1_lora_rank,
        ),
        StageSpec(
            name="stage2_full_press",
            environment_kind="full_press",
            num_train_examples=args.stage2_train_examples,
            num_eval_examples=args.stage2_eval_examples,
            batch_size=args.stage2_batch_size,
            group_size=args.stage2_group_size,
            max_tokens=args.stage2_max_tokens,
            max_turns=args.stage2_max_turns,
            max_trajectory_tokens=args.stage2_max_trajectory_tokens,
            train_seed=args.stage2_train_seed,
            eval_seed=args.stage2_eval_seed,
            learning_rate=args.stage2_learning_rate,
            lora_rank=args.stage2_lora_rank,
        ),
    )


def build_full_v1_stage_specs(
    *,
    prompt_blocks: dict[str, str],
    learning_rate: float,
    lora_rank: int,
) -> tuple[StageSpec, ...]:
    return tuple(
        StageSpec(
            learning_rate=learning_rate,
            lora_rank=lora_rank,
            tracked_instruction_block=prompt_blocks[stage_defaults["environment_kind"]],
            **stage_defaults,
        )
        for stage_defaults in FULL_V1_STAGE_DEFAULTS
    )


def build_config(args: argparse.Namespace) -> CurriculumConfig:
    renderer_name = args.renderer_name or get_default_renderer_name(
        args.model_name,
        disable_thinking=not args.enable_thinking,
    )
    tracked_instruction_block = None
    if args.tracked_instruction_block_path:
        tracked_instruction_block = Path(args.tracked_instruction_block_path).read_text().strip()
    return CurriculumConfig(
        model_name=args.model_name,
        renderer_name=renderer_name,
        log_root=args.log_root,
        run_name=args.run_name,
        wandb_project=args.wandb_project,
        openrouter_model=args.openrouter_model,
        openrouter_base_url=args.openrouter_base_url,
        openrouter_api_key_env_var=args.openrouter_api_key_env_var,
        http_referer=args.http_referer,
        x_title=args.x_title,
        actor_max_turns=args.actor_max_turns,
        session_timeout_seconds=args.session_timeout_seconds,
        default_idle_sleep_seconds=args.default_idle_sleep_seconds,
        max_message_length=args.max_message_length,
        save_every=args.save_every,
        eval_every=args.eval_every,
        num_groups_to_log=args.num_groups_to_log,
        rollout_json_export=not args.disable_rollout_json_export,
        stages=build_stages(args),
        modal_rollout=ModalRolloutConfig(
            app_name=args.modal_app_name,
            timeout_seconds=args.modal_timeout_seconds,
            cpu=args.modal_cpu,
            memory_mb=args.modal_memory_mb,
        ),
        initial_checkpoint_path=args.initial_checkpoint_path,
        tracked_instruction_block=tracked_instruction_block,
    )


def main() -> int:
    args = parse_args()
    config = build_config(args)
    manifest_path = asyncio.run(run_curriculum(config))
    logger.info("Curriculum manifest written to %s", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
