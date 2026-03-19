#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDORED_COOKBOOK_ROOT = REPO_ROOT / "vendor" / "tinker-cookbook"
if str(VENDORED_COOKBOOK_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDORED_COOKBOOK_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tinker_cookbook import checkpoint_utils
from tinker_cookbook.supervised import train as supervised_train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

from tinker_training.diplomacy_adapter import get_default_renderer_name

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_STUDENT_MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run supervised fine-tuning for the Diplomacy SFT dataset.")
    parser.add_argument("--model-name", default=DEFAULT_STUDENT_MODEL_NAME, help="Student model to fine-tune.")
    parser.add_argument(
        "--renderer-name",
        default=None,
        help="Optional explicit renderer override. Leave unset to use the recommended renderer.",
    )
    parser.add_argument(
        "--dataset-path",
        default=str(REPO_ROOT / "data" / "diplomacy_sft" / "merged.jsonl"),
        help="JSONL file containing chat-style SFT examples.",
    )
    parser.add_argument(
        "--log-root",
        default=str(REPO_ROOT / ".tmp" / "diplomacy_sft_runs"),
        help="Directory where the SFT run logs and manifest are written.",
    )
    parser.add_argument("--run-name", default=None, help="Optional fixed run name.")
    parser.add_argument(
        "--load-checkpoint-path",
        default=None,
        help="Optional checkpoint to warm-start the SFT run from.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Global SFT learning rate.")
    parser.add_argument("--num-epochs", type=int, default=1, help="Number of SFT epochs.")
    parser.add_argument("--lora-rank", type=int, default=32, help="LoRA rank for the SFT run.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for SFT.")
    parser.add_argument(
        "--max-length",
        type=int,
        default=16384,
        help="Maximum token length for rendered examples.",
    )
    parser.add_argument("--save-every", type=int, default=20, help="Save checkpoint cadence.")
    parser.add_argument("--eval-every", type=int, default=20, help="Evaluation cadence.")
    parser.add_argument(
        "--test-size",
        type=int,
        default=256,
        help="Held-out example count for the built-in NLL evaluator.",
    )
    parser.add_argument("--shuffle-seed", type=int, default=0, help="Dataset shuffle seed.")
    parser.add_argument("--base-url", default=None, help="Optional Tinker base URL override.")
    parser.add_argument("--wandb-project", default="diplomacy-sft", help="W&B project name.")
    parser.add_argument("--wandb-name", default=None, help="Optional W&B run name.")
    return parser.parse_args()


def resolve_log_path(args: argparse.Namespace) -> Path:
    if args.run_name:
        run_name = args.run_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_suffix = args.model_name.split("/", 1)[-1].replace("/", "-")
        run_name = f"{model_suffix}-diplomacy-sft-{timestamp}"
    return Path(args.log_root).expanduser().resolve() / run_name


def build_config(args: argparse.Namespace, *, log_path: Path) -> supervised_train.Config:
    renderer_name = args.renderer_name or get_default_renderer_name(
        args.model_name,
        disable_thinking=True,
    )
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=args.model_name,
        renderer_name=renderer_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    dataset_builder = FromConversationFileBuilder(
        common_config=common_config,
        file_path=str(Path(args.dataset_path).expanduser().resolve()),
        test_size=args.test_size,
        shuffle_seed=args.shuffle_seed,
    )
    wandb_name = args.wandb_name or log_path.name
    return supervised_train.Config(
        log_path=str(log_path),
        model_name=args.model_name,
        load_checkpoint_path=args.load_checkpoint_path,
        renderer_name=renderer_name,
        dataset_builder=dataset_builder,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        base_url=args.base_url,
        wandb_project=args.wandb_project,
        wandb_name=wandb_name,
        lora_rank=args.lora_rank,
        save_every=args.save_every,
        eval_every=args.eval_every,
    )


def write_manifest(*, args: argparse.Namespace, log_path: Path) -> Path:
    checkpoint = checkpoint_utils.get_last_checkpoint(str(log_path), required_key="state_path")
    manifest = {
        "model_name": args.model_name,
        "dataset_path": str(Path(args.dataset_path).expanduser().resolve()),
        "log_path": str(log_path),
        "checkpoint": checkpoint,
        "renderer_name": args.renderer_name,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "lora_rank": args.lora_rank,
    }
    manifest_path = log_path / "sft_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def main() -> int:
    args = parse_args()
    log_path = resolve_log_path(args)
    log_path.mkdir(parents=True, exist_ok=True)
    config = build_config(args, log_path=log_path)
    asyncio.run(supervised_train.main(config))
    manifest_path = write_manifest(args=args, log_path=log_path)
    logger.info("SFT manifest written to %s", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
