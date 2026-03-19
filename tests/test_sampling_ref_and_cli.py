from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from scripts.train_tinker_grpo_curriculum import build_config, parse_args
from tinker_cookbook.rl import train as rl_train


class _DummyDatasetBuilder:
    async def __call__(self):
        raise AssertionError("Dataset builder should not be used in this unit test")


class _FakeSamplerResult:
    def __init__(self, path: str) -> None:
        self.path = path


class _FakeFuture:
    def __init__(self, result) -> None:
        self._result = result

    async def result_async(self):
        return self._result


class _FakeTrainingClient:
    def __init__(self) -> None:
        self.saved_names: list[str] = []
        self.created_paths: list[str] = []

    async def save_weights_for_sampler_async(self, name: str, ttl_seconds=None):
        self.saved_names.append(name)
        return _FakeFuture(_FakeSamplerResult(f"sampler://{name}"))

    def create_sampling_client(self, sampler_path: str):
        self.created_paths.append(sampler_path)
        return {"sampler_path": sampler_path}


def test_save_checkpoint_returns_sampling_ref() -> None:
    cfg = rl_train.Config(
        learning_rate=1e-5,
        dataset_builder=_DummyDatasetBuilder(),
        model_name="Qwen/Qwen3.5-27B",
        max_tokens=32,
        log_path="/tmp/diplomacy-test",
        base_url="https://tinker.example",
    )
    fake_training_client = _FakeTrainingClient()

    sampling_client, sampling_ref, metrics = asyncio.run(
        rl_train.save_checkpoint_and_get_sampling_client(
            cfg,
            fake_training_client,
            i_batch=1,
            log_path="/tmp/diplomacy-test",
            save_every=0,
        )
    )

    assert sampling_client == {"sampler_path": "sampler://000001"}
    assert sampling_ref.sampler_path == "sampler://000001"
    assert sampling_ref.base_url == "https://tinker.example"
    assert metrics["sampling/saved_sampler"] == 1.0


def test_cli_builds_modal_curriculum_config() -> None:
    args = argparse.Namespace(
        model_name="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        renderer_name=None,
        enable_thinking=False,
        log_root="~/tinker-runs/diplomacy-grpo",
        run_name="test-run",
        wandb_project="diplomacy-grpo",
        prompt_family_dir="prompts/gepa-full-press",
        initial_checkpoint_path=None,
        hybrid_total_batches=24,
        learning_rate=2e-5,
        lora_rank=32,
        modal_app_name="rollout-app",
        modal_timeout_seconds=321,
        modal_cpu=3.0,
        modal_memory_mb=8192,
        openrouter_model="google/gemini-3-flash-preview",
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_api_key_env_var="OPENROUTER_API_KEY",
        http_referer="https://local.codex",
        x_title="diplomacy-grpo",
        actor_max_turns=18,
        session_timeout_seconds=90.0,
        default_idle_sleep_seconds=0.5,
        max_message_length=2000,
        save_every=10,
        eval_every=5,
        num_groups_to_log=2,
        disable_rollout_json_export=False,
    )

    config = build_config(args)
    assert config.modal_rollout.app_name == "rollout-app"
    assert config.modal_rollout.timeout_seconds == 321
    assert config.modal_rollout.cpu == 3.0
    assert config.renderer_name == "qwen3_disable_thinking"
    assert len(config.stages) == 1
    assert config.stages[0].name == "hybrid_v1"
    assert config.stages[0].environment_kind == "hybrid"
    assert config.hybrid_total_batches == 24
    assert config.stages[0].num_train_examples == 24 * 16
    assert config.prompt_blocks is not None
    assert config.prompt_blocks["full_press"]

def test_cli_default_rollout_counts_are_reduced(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train_tinker_grpo_curriculum.py"])

    args = parse_args()

    assert args.hybrid_total_batches == 80
    assert args.learning_rate == 2e-5
    assert args.lora_rank == 32
