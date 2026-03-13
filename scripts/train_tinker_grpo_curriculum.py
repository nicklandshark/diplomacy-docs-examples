#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDORED_COOKBOOK_ROOT = REPO_ROOT / "vendor" / "tinker-cookbook"
if str(VENDORED_COOKBOOK_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDORED_COOKBOOK_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tinker_training.curriculum import (
    CurriculumConfig,
    ModalRolloutConfig,
    StageSpec,
    run_curriculum,
)
from tinker_training.diplomacy_adapter import get_default_renderer_name

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Tinker GRPO curriculum for Diplomacy with optional Modal-isolated rollouts."
    )
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--renderer-name", default=None)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--log-root", default="~/tinker-runs/diplomacy-grpo")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--wandb-project", default="diplomacy-grpo")
    parser.add_argument("--initial-checkpoint-path", default=None)

    parser.add_argument("--rollout-backend", choices=["local", "modal"], default="modal")
    parser.add_argument("--modal-app-name", default="diplomacy-grpo-rollouts")
    parser.add_argument("--modal-timeout-seconds", type=int, default=900)
    parser.add_argument("--modal-cpu", type=float, default=2.0)
    parser.add_argument("--modal-memory-mb", type=int, default=4096)
    parser.add_argument(
        "--disable-local-fallback-on-infra-failure",
        action="store_true",
        help="Disable the one-shot local fallback when a Modal transport failure occurs.",
    )

    parser.add_argument("--openrouter-model", default="google/gemini-3-flash-preview")
    parser.add_argument("--openrouter-base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--openrouter-api-key-env-var", default="OPENROUTER_API_KEY")
    parser.add_argument("--http-referer", default="https://local.codex")
    parser.add_argument("--x-title", default="diplomacy-grpo")
    parser.add_argument("--actor-max-turns", type=int, default=18)
    parser.add_argument("--session-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--default-idle-sleep-seconds", type=float, default=0.5)
    parser.add_argument("--max-message-length", type=int, default=2000)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--num-groups-to-log", type=int, default=2)
    parser.add_argument("--disable-rollout-json-export", action="store_true")

    parser.add_argument("--stage1-train-examples", type=int, default=128)
    parser.add_argument("--stage1-eval-examples", type=int, default=16)
    parser.add_argument("--stage1-batch-size", type=int, default=16)
    parser.add_argument("--stage1-group-size", type=int, default=4)
    parser.add_argument("--stage1-max-tokens", type=int, default=256)
    parser.add_argument("--stage1-max-turns", type=int, default=14)
    parser.add_argument("--stage1-max-trajectory-tokens", type=int, default=8192)
    parser.add_argument("--stage1-train-seed", type=int, default=21)
    parser.add_argument("--stage1-eval-seed", type=int, default=10_021)
    parser.add_argument("--stage1-learning-rate", type=float, default=3e-5)
    parser.add_argument("--stage1-lora-rank", type=int, default=32)

    parser.add_argument("--stage2-train-examples", type=int, default=96)
    parser.add_argument("--stage2-eval-examples", type=int, default=16)
    parser.add_argument("--stage2-batch-size", type=int, default=12)
    parser.add_argument("--stage2-group-size", type=int, default=4)
    parser.add_argument("--stage2-max-tokens", type=int, default=384)
    parser.add_argument("--stage2-max-turns", type=int, default=20)
    parser.add_argument("--stage2-max-trajectory-tokens", type=int, default=12288)
    parser.add_argument("--stage2-train-seed", type=int, default=37)
    parser.add_argument("--stage2-eval-seed", type=int, default=10_037)
    parser.add_argument("--stage2-learning-rate", type=float, default=2e-5)
    parser.add_argument("--stage2-lora-rank", type=int, default=32)
    return parser.parse_args()


def build_stages(args: argparse.Namespace) -> tuple[StageSpec, ...]:
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


def build_config(args: argparse.Namespace) -> CurriculumConfig:
    renderer_name = args.renderer_name or get_default_renderer_name(
        args.model_name,
        disable_thinking=not args.enable_thinking,
    )
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
        rollout_backend=args.rollout_backend,
        modal_rollout=ModalRolloutConfig(
            app_name=args.modal_app_name,
            timeout_seconds=args.modal_timeout_seconds,
            cpu=args.modal_cpu,
            memory_mb=args.modal_memory_mb,
            local_fallback_on_infra_failure=not args.disable_local_fallback_on_infra_failure,
        ),
        initial_checkpoint_path=args.initial_checkpoint_path,
    )


def main() -> int:
    args = parse_args()
    config = build_config(args)
    manifest_path = asyncio.run(run_curriculum(config))
    logger.info("Curriculum manifest written to %s", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
