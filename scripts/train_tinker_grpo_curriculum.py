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
from tinker_training.hybrid_schedule import (
    HYBRID_BATCH_SIZE,
    HYBRID_EVAL_EXAMPLES_PER_ENVIRONMENT,
    HYBRID_EVAL_SEED_BASE,
    HYBRID_GROUP_SIZE,
    HYBRID_MAX_TOKENS,
    HYBRID_TOTAL_BATCHES_DEFAULT,
    HYBRID_TRAIN_SEED,
)
from tinker_training.prompt_family import DEFAULT_PROMPT_FAMILY_DIR, load_prompt_family_blocks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Tinker GRPO curriculum for Diplomacy with Modal-isolated rollouts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--prompt-family-dir",
        default=str(DEFAULT_PROMPT_FAMILY_DIR),
        help="Directory containing the stage-specific tracked-policy prompts for the hybrid curriculum.",
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
        "--hybrid-total-batches",
        type=int,
        default=HYBRID_TOTAL_BATCHES_DEFAULT,
        help="Total number of hybrid curriculum batches to train.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Global learning rate used by the hybrid curriculum.",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=32,
        help="Global LoRA rank used by the hybrid curriculum.",
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
    return parser.parse_args()


def build_stages(args: argparse.Namespace) -> tuple[StageSpec, ...]:
    return (
        StageSpec(
            name="hybrid_v1",
            environment_kind="hybrid",
            num_train_examples=max(1, int(args.hybrid_total_batches)) * HYBRID_BATCH_SIZE,
            num_eval_examples=HYBRID_EVAL_EXAMPLES_PER_ENVIRONMENT,
            batch_size=HYBRID_BATCH_SIZE,
            group_size=HYBRID_GROUP_SIZE,
            max_tokens=HYBRID_MAX_TOKENS,
            max_turns=20,
            max_trajectory_tokens=12_288,
            train_seed=HYBRID_TRAIN_SEED,
            eval_seed=HYBRID_EVAL_SEED_BASE,
            learning_rate=args.learning_rate,
            lora_rank=args.lora_rank,
        ),
    )


def build_config(args: argparse.Namespace) -> CurriculumConfig:
    renderer_name = args.renderer_name or get_default_renderer_name(
        args.model_name,
        disable_thinking=not args.enable_thinking,
    )
    prompt_blocks = load_prompt_family_blocks(args.prompt_family_dir)
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
        prompt_blocks=prompt_blocks,
        hybrid_total_batches=max(1, int(args.hybrid_total_batches)),
        modal_rollout=ModalRolloutConfig(
            app_name=args.modal_app_name,
            timeout_seconds=args.modal_timeout_seconds,
            cpu=args.modal_cpu,
            memory_mb=args.modal_memory_mb,
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
