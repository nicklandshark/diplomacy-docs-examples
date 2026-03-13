from __future__ import annotations

import argparse
import asyncio

from scripts.train_tinker_grpo_curriculum import build_config
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
        model_name="Qwen/Qwen3.5-27B",
        renderer_name=None,
        enable_thinking=False,
        log_root="~/tinker-runs/diplomacy-grpo",
        run_name="test-run",
        wandb_project="diplomacy-grpo",
        initial_checkpoint_path=None,
        rollout_backend="modal",
        modal_app_name="rollout-app",
        modal_timeout_seconds=321,
        modal_cpu=3.0,
        modal_memory_mb=8192,
        disable_local_fallback_on_infra_failure=False,
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
        stage1_train_examples=128,
        stage1_eval_examples=16,
        stage1_batch_size=16,
        stage1_group_size=4,
        stage1_max_tokens=256,
        stage1_max_turns=14,
        stage1_max_trajectory_tokens=8192,
        stage1_train_seed=21,
        stage1_eval_seed=10021,
        stage1_learning_rate=3e-5,
        stage1_lora_rank=32,
        stage2_train_examples=96,
        stage2_eval_examples=16,
        stage2_batch_size=12,
        stage2_group_size=4,
        stage2_max_tokens=384,
        stage2_max_turns=20,
        stage2_max_trajectory_tokens=12288,
        stage2_train_seed=37,
        stage2_eval_seed=10037,
        stage2_learning_rate=2e-5,
        stage2_lora_rank=32,
    )

    config = build_config(args)
    assert config.rollout_backend == "modal"
    assert config.modal_rollout.app_name == "rollout-app"
    assert config.modal_rollout.timeout_seconds == 321
    assert config.modal_rollout.cpu == 3.0
    assert len(config.stages) == 2
    assert config.stages[0].name == "stage1_tool_accuracy"
    assert config.stages[1].name == "stage2_full_press"
