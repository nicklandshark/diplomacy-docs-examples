from __future__ import annotations

import asyncio
import json
from pathlib import Path

from tinker_cookbook.rl import train as rl_train
from tinker_training import curriculum


class _FakeDataset:
    def __init__(self, num_batches: int) -> None:
        self.num_batches = num_batches

    def get_batch(self, index: int):
        assert 0 <= index < self.num_batches
        return []

    def __len__(self) -> int:
        return self.num_batches


class _FakeBuilder:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeOffsetDataset:
    def __init__(self) -> None:
        self._batches = [
            [_FakeBuilder("batch-0")],
            [_FakeBuilder("batch-1")],
        ]

    def get_batch(self, index: int):
        return self._batches[index]

    def __len__(self) -> int:
        return len(self._batches)


class _FakeDatasetBuilder:
    def __init__(self, num_batches: int) -> None:
        self.num_batches = num_batches

    async def __call__(self):
        return _FakeDataset(self.num_batches), None


class _FakeRootLogger:
    def __init__(self) -> None:
        self.metrics: list[tuple[dict[str, float], int | None]] = []

    def log_metrics(self, metrics, step: int | None = None) -> None:
        self.metrics.append((dict(metrics), step))

    def get_logger_url(self) -> None:
        return None

    def close(self) -> None:
        return None

    def sync(self) -> None:
        return None


class _FakeTrainingClient:
    def get_tokenizer(self):
        return object()


class _FakeSamplerSaveResult:
    def __init__(self, path: str) -> None:
        self.path = path


class _FakeSamplerSaveFuture:
    def __init__(self, path: str) -> None:
        self._result = _FakeSamplerSaveResult(path)

    async def result_async(self) -> _FakeSamplerSaveResult:
        return self._result


class _FakeBootstrapTrainingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int | None]] = []
        self.created_sampler_paths: list[str] = []

    async def save_weights_for_sampler_async(
        self,
        name: str,
        ttl_seconds: int | None = None,
    ) -> _FakeSamplerSaveFuture:
        self.calls.append((name, ttl_seconds))
        return _FakeSamplerSaveFuture(f"sampler://{name}")

    def create_sampling_client(self, sampler_path: str):
        self.created_sampler_paths.append(sampler_path)
        return {"sampler_path": sampler_path}


class _FakeRolloutRunner:
    backend_name = "fake"

    async def start(self) -> None:
        return None

    async def aclose(self) -> None:
        return None

    def summary(self) -> dict[str, int]:
        return {"call_count": 0}

    def snapshot_metrics(self) -> dict[str, float]:
        return {"rollout/runner_call_count": 0.0}


def _make_train_config(log_path: Path) -> rl_train.Config:
    return rl_train.Config(
        learning_rate=3e-5,
        dataset_builder=_FakeDatasetBuilder(1),
        model_name="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        max_tokens=32,
        log_path=str(log_path),
        eval_every=0,
        save_every=1,
        renderer_name="qwen3_disable_thinking",
        wandb_project=None,
        lora_rank=8,
        num_groups_to_log=0,
        rollout_json_export=False,
    )


def test_offset_rl_dataset_translates_global_batch_indices() -> None:
    dataset = curriculum.OffsetRLDataset(_FakeOffsetDataset(), batch_offset=3)

    assert len(dataset) == 5
    assert [builder.name for builder in dataset.get_batch(3)] == ["batch-0"]
    assert [builder.name for builder in dataset.get_batch(4)] == ["batch-1"]


def test_validate_curriculum_rejects_mismatched_lora_ranks() -> None:
    config = curriculum.CurriculumConfig(
        model_name="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        renderer_name="qwen3_disable_thinking",
        log_root="~/runs",
        wandb_project="demo",
        openrouter_model="google/gemini-3-flash-preview",
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_api_key_env_var="OPENROUTER_API_KEY",
        http_referer="https://local.codex",
        x_title="demo",
        actor_max_turns=2,
        session_timeout_seconds=10.0,
        default_idle_sleep_seconds=0.1,
        max_message_length=2000,
        save_every=10,
        eval_every=0,
        num_groups_to_log=0,
        rollout_json_export=False,
        stages=(
            curriculum.StageSpec(
                name="stage1_tool_accuracy",
                environment_kind="tool_accuracy",
                num_train_examples=4,
                num_eval_examples=0,
                batch_size=2,
                group_size=2,
                max_tokens=32,
                max_turns=4,
                max_trajectory_tokens=None,
                train_seed=1,
                eval_seed=101,
                learning_rate=3e-5,
                lora_rank=16,
            ),
            curriculum.StageSpec(
                name="stage2_full_press",
                environment_kind="full_press",
                num_train_examples=6,
                num_eval_examples=0,
                batch_size=3,
                group_size=2,
                max_tokens=64,
                max_turns=6,
                max_trajectory_tokens=None,
                train_seed=2,
                eval_seed=102,
                learning_rate=2e-5,
                lora_rank=32,
            ),
        ),
    )

    try:
        curriculum.validate_curriculum(config)
    except ValueError as exc:
        assert "same LoRA rank" in str(exc)
    else:
        raise AssertionError("Expected validation to fail for mismatched LoRA ranks")


def test_curriculum_reuses_one_training_client_across_both_stages(tmp_path, monkeypatch) -> None:
    shared_training_client = _FakeTrainingClient()
    root_logger = _FakeRootLogger()
    training_calls: list[dict[str, object]] = []

    def fake_build_stage_train_config(
        *,
        stage_runtime: curriculum.StageRuntime,
        config: curriculum.CurriculumConfig,
        actor_runtime,
        rollout_runner,
        rollout_runner_id: str,
    ) -> rl_train.Config:
        del actor_runtime, rollout_runner, rollout_runner_id
        return rl_train.Config(
            learning_rate=stage_runtime.stage.learning_rate,
            dataset_builder=_FakeDatasetBuilder(stage_runtime.num_batches),
            model_name=config.model_name,
            max_tokens=stage_runtime.stage.max_tokens,
            log_path=str(stage_runtime.stage_log_dir),
            eval_every=config.eval_every,
            save_every=config.save_every,
            renderer_name=config.renderer_name,
            wandb_project=None,
            lora_rank=stage_runtime.stage.lora_rank,
            num_groups_to_log=0,
            rollout_json_export=False,
        )

    async def fake_training_func(
        *,
        start_batch: int,
        end_batch: int,
        num_batches: int,
        cfg: rl_train.Config,
        training_client,
        kl_reference_client,
        evaluators,
        dataset,
        ml_logger,
        tokenizer,
    ) -> None:
        del kl_reference_client, evaluators, dataset, tokenizer
        training_calls.append(
            {
                "training_client": training_client,
                "start_batch": start_batch,
                "end_batch": end_batch,
                "num_batches": num_batches,
                "log_path": cfg.log_path,
            }
        )
        ml_logger.log_metrics({"train/call_count": 1.0}, step=start_batch)

    async def fake_save_checkpoint_async(
        *,
        training_client,
        name: str,
        log_path: str,
        loop_state: dict[str, object],
        kind: str = "state",
        ttl_seconds=None,
    ) -> dict[str, str]:
        del training_client, ttl_seconds
        log_dir = Path(log_path)
        log_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, str] = {}
        if kind in ("state", "both"):
            paths["state_path"] = f"state://{name}"
        if kind in ("sampler", "both"):
            paths["sampler_path"] = f"sampler://{name}"
        with (log_dir / "checkpoints.jsonl").open("a") as handle:
            handle.write(json.dumps({"name": name, **loop_state, **paths}) + "\n")
        return paths

    async def fake_create_training_client(
        *,
        config: curriculum.CurriculumConfig,
        service_client,
        user_metadata: dict[str, str],
        resume_state_path: str | None,
        lora_rank: int,
    ):
        del config, service_client, user_metadata, resume_state_path, lora_rank
        return shared_training_client

    monkeypatch.setattr(curriculum, "build_actor_runtime", lambda config: object())
    monkeypatch.setattr(curriculum, "create_rollout_runner", lambda config: _FakeRolloutRunner())
    monkeypatch.setattr(curriculum, "build_stage_train_config", fake_build_stage_train_config)
    monkeypatch.setattr(curriculum, "_select_training_function", lambda cfg: fake_training_func)
    monkeypatch.setattr(curriculum, "create_training_client", fake_create_training_client)
    monkeypatch.setattr(curriculum.ml_log, "setup_logging", lambda **kwargs: root_logger)
    monkeypatch.setattr(curriculum.checkpoint_utils, "save_checkpoint_async", fake_save_checkpoint_async)

    class _DummyServiceClient:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

    monkeypatch.setattr(curriculum.tinker, "ServiceClient", _DummyServiceClient)

    config = curriculum.CurriculumConfig(
        model_name="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        renderer_name="qwen3_disable_thinking",
        log_root=str(tmp_path),
        wandb_project="demo",
        openrouter_model="google/gemini-3-flash-preview",
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_api_key_env_var="OPENROUTER_API_KEY",
        http_referer="https://local.codex",
        x_title="demo",
        actor_max_turns=2,
        session_timeout_seconds=10.0,
        default_idle_sleep_seconds=0.1,
        max_message_length=2000,
        save_every=10,
        eval_every=0,
        num_groups_to_log=0,
        rollout_json_export=False,
        stages=(
            curriculum.StageSpec(
                name="stage1_tool_accuracy",
                environment_kind="tool_accuracy",
                num_train_examples=4,
                num_eval_examples=0,
                batch_size=2,
                group_size=2,
                max_tokens=32,
                max_turns=4,
                max_trajectory_tokens=None,
                train_seed=1,
                eval_seed=101,
                learning_rate=3e-5,
                lora_rank=16,
            ),
            curriculum.StageSpec(
                name="stage2_full_press",
                environment_kind="full_press",
                num_train_examples=6,
                num_eval_examples=0,
                batch_size=3,
                group_size=2,
                max_tokens=64,
                max_turns=6,
                max_trajectory_tokens=None,
                train_seed=2,
                eval_seed=102,
                learning_rate=2e-5,
                lora_rank=16,
            ),
        ),
        run_name="curriculum-unit-test",
    )

    manifest_path = asyncio.run(curriculum.run_curriculum(config))
    manifest = json.loads(manifest_path.read_text())

    assert len(training_calls) == 2
    assert training_calls[0]["training_client"] is shared_training_client
    assert training_calls[1]["training_client"] is shared_training_client
    assert training_calls[0]["start_batch"] == 0
    assert training_calls[0]["end_batch"] == 2
    assert training_calls[1]["start_batch"] == 2
    assert training_calls[1]["end_batch"] == 4
    assert manifest["single_tinker_training_run"] is True
    assert manifest["status"] == "completed"
    assert [stage["stage"] for stage in manifest["stages"]] == [
        "stage1_tool_accuracy",
        "stage2_full_press",
    ]
    assert all(stage["status"] == "completed" for stage in manifest["stages"])
    assert manifest["stages"][-1]["state_path"] == "state://final"


async def test_bootstrap_sampling_sampler_path_reuses_matching_checkpoint(
    tmp_path,
    monkeypatch,
) -> None:
    cfg = _make_train_config(tmp_path / "stage2_full_press")
    existing_sampler_ckpt = {
        "name": "000002-stage2_full_press-bootstrap-123",
        "batch": 2,
        "sampler_path": "sampler://existing",
    }

    monkeypatch.setattr(
        rl_train.checkpoint_utils,
        "get_last_checkpoint",
        lambda log_dir, required_key="sampler_path": existing_sampler_ckpt,
    )

    async def fail_save_checkpoint_async(**kwargs):
        del kwargs
        raise AssertionError("bootstrap checkpoint should have been reused")

    monkeypatch.setattr(
        rl_train.checkpoint_utils,
        "save_checkpoint_async",
        fail_save_checkpoint_async,
    )

    sampler_path = await rl_train._bootstrap_sampling_sampler_path(
        cfg=cfg,
        training_client=object(),
        start_batch=2,
    )

    assert sampler_path == "sampler://existing"


async def test_bootstrap_sampling_sampler_path_uses_unique_stage_local_name(
    tmp_path,
    monkeypatch,
) -> None:
    cfg = _make_train_config(tmp_path / "stage2_full_press")
    training_client = _FakeBootstrapTrainingClient()

    monkeypatch.setattr(
        rl_train.checkpoint_utils,
        "get_last_checkpoint",
        lambda log_dir, required_key="sampler_path": None,
    )

    sampler_path = await rl_train._bootstrap_sampling_sampler_path(
        cfg=cfg,
        training_client=training_client,
        start_batch=2,
    )

    assert len(training_client.calls) == 1
    checkpoint_name, ttl_seconds = training_client.calls[0]
    assert sampler_path == f"sampler://{checkpoint_name}"
    assert checkpoint_name.startswith("000002-stage2_full_press-bootstrap-")
    assert checkpoint_name != "000002"
    checkpoint_log = json.loads((tmp_path / "stage2_full_press" / "checkpoints.jsonl").read_text())
    assert checkpoint_log["name"] == checkpoint_name
    assert checkpoint_log["batch"] == 2
    assert checkpoint_log["sampler_path"] == sampler_path
    assert ttl_seconds == cfg.ttl_seconds


async def test_do_sync_training_reuses_existing_sampler_checkpoint_on_stage_resume(
    tmp_path,
    monkeypatch,
) -> None:
    cfg = _make_train_config(tmp_path / "stage2_full_press")
    training_client = _FakeBootstrapTrainingClient()
    existing_sampler_ckpt = {
        "name": "000002-stage2_full_press-bootstrap-123",
        "batch": 2,
        "sampler_path": "sampler://existing",
    }

    monkeypatch.setattr(
        rl_train.checkpoint_utils,
        "get_last_checkpoint",
        lambda log_dir, required_key="sampler_path": existing_sampler_ckpt,
    )

    await rl_train.do_sync_training(
        start_batch=2,
        end_batch=2,
        num_batches=5,
        cfg=cfg,
        training_client=training_client,
        kl_reference_client=None,
        evaluators=[],
        dataset=_FakeDataset(0),
        ml_logger=_FakeRootLogger(),
        tokenizer=object(),
    )

    assert training_client.calls == []
    assert training_client.created_sampler_paths == ["sampler://existing"]


def test_seed_stage_sampler_checkpoint_from_previous_stage(tmp_path) -> None:
    previous_stage_runtime = curriculum.StageRuntime(
        stage=curriculum.StageSpec(
            name="stage1_tool_accuracy",
            environment_kind="tool_accuracy",
            num_train_examples=2,
            num_eval_examples=0,
            batch_size=1,
            group_size=1,
            max_tokens=32,
            max_turns=4,
            max_trajectory_tokens=None,
            train_seed=1,
            eval_seed=101,
            learning_rate=3e-5,
            lora_rank=8,
        ),
        stage_log_dir=tmp_path / "stage1_tool_accuracy",
        global_start_batch=0,
        global_end_batch=2,
    )
    stage_runtime = curriculum.StageRuntime(
        stage=curriculum.StageSpec(
            name="stage2_full_press",
            environment_kind="full_press",
            num_train_examples=3,
            num_eval_examples=0,
            batch_size=1,
            group_size=1,
            max_tokens=64,
            max_turns=4,
            max_trajectory_tokens=None,
            train_seed=2,
            eval_seed=102,
            learning_rate=2e-5,
            lora_rank=8,
        ),
        stage_log_dir=tmp_path / "stage2_full_press",
        global_start_batch=2,
        global_end_batch=5,
    )

    previous_stage_runtime.stage_log_dir.mkdir(parents=True, exist_ok=True)
    (previous_stage_runtime.stage_log_dir / "checkpoints.jsonl").write_text(
        json.dumps(
            {
                "name": "000002",
                "batch": 2,
                "state_path": "state://000002",
                "sampler_path": "sampler://000002",
            }
        )
        + "\n"
    )

    curriculum._seed_stage_sampler_checkpoint_from_previous_stage(
        stage_runtime=stage_runtime,
        previous_stage_runtime=previous_stage_runtime,
        start_batch=2,
    )

    seeded_entry = json.loads((stage_runtime.stage_log_dir / "checkpoints.jsonl").read_text())
    assert seeded_entry == {
        "name": "000002",
        "batch": 2,
        "sampler_path": "sampler://000002",
    }


def test_replace_checkpoint_name_updates_last_path_segment() -> None:
    assert (
        curriculum._replace_checkpoint_name(
            "tinker://model:train:0/sampler_weights/000000",
            "000004",
        )
        == "tinker://model:train:0/sampler_weights/000004"
    )
    assert curriculum._replace_checkpoint_name("sampler://000002", "000004") == "sampler://000004"


def test_curriculum_skips_duplicate_stage_end_checkpoint(tmp_path, monkeypatch) -> None:
    shared_training_client = _FakeTrainingClient()
    root_logger = _FakeRootLogger()
    save_calls: list[tuple[str, str]] = []

    def fake_build_stage_train_config(
        *,
        stage_runtime: curriculum.StageRuntime,
        config: curriculum.CurriculumConfig,
        actor_runtime,
        rollout_runner,
        rollout_runner_id: str,
    ) -> rl_train.Config:
        del actor_runtime, rollout_runner, rollout_runner_id
        return rl_train.Config(
            learning_rate=stage_runtime.stage.learning_rate,
            dataset_builder=_FakeDatasetBuilder(stage_runtime.num_batches),
            model_name=config.model_name,
            max_tokens=stage_runtime.stage.max_tokens,
            log_path=str(stage_runtime.stage_log_dir),
            eval_every=config.eval_every,
            save_every=config.save_every,
            renderer_name=config.renderer_name,
            wandb_project=None,
            lora_rank=stage_runtime.stage.lora_rank,
            num_groups_to_log=0,
            rollout_json_export=False,
        )

    async def fake_save_checkpoint_async(
        *,
        training_client,
        name: str,
        log_path: str,
        loop_state: dict[str, object],
        kind: str = "state",
        ttl_seconds=None,
    ) -> dict[str, str]:
        del training_client, ttl_seconds
        log_dir = Path(log_path)
        log_dir.mkdir(parents=True, exist_ok=True)
        save_key = (str(log_dir), name)
        if save_key in save_calls:
            raise AssertionError(f"duplicate checkpoint save attempted for {save_key}")
        save_calls.append(save_key)
        paths: dict[str, str] = {}
        if kind in ("state", "both"):
            paths["state_path"] = f"state://{name}"
        if kind in ("sampler", "both"):
            paths["sampler_path"] = f"sampler://{name}"
        with (log_dir / "checkpoints.jsonl").open("a") as handle:
            handle.write(json.dumps({"name": name, **loop_state, **paths}) + "\n")
        return paths

    async def fake_training_func(
        *,
        start_batch: int,
        end_batch: int,
        num_batches: int,
        cfg: rl_train.Config,
        training_client,
        kl_reference_client,
        evaluators,
        dataset,
        ml_logger,
        tokenizer,
    ) -> None:
        del start_batch, num_batches, kl_reference_client, evaluators, dataset, ml_logger, tokenizer
        await fake_save_checkpoint_async(
            training_client=training_client,
            name=f"{end_batch:06d}",
            log_path=cfg.log_path,
            kind="both",
            loop_state={"batch": end_batch},
        )

    async def fake_create_training_client(
        *,
        config: curriculum.CurriculumConfig,
        service_client,
        user_metadata: dict[str, str],
        resume_state_path: str | None,
        lora_rank: int,
    ):
        del config, service_client, user_metadata, resume_state_path, lora_rank
        return shared_training_client

    monkeypatch.setattr(curriculum, "build_actor_runtime", lambda config: object())
    monkeypatch.setattr(curriculum, "create_rollout_runner", lambda config: _FakeRolloutRunner())
    monkeypatch.setattr(curriculum, "build_stage_train_config", fake_build_stage_train_config)
    monkeypatch.setattr(curriculum, "_select_training_function", lambda cfg: fake_training_func)
    monkeypatch.setattr(curriculum, "create_training_client", fake_create_training_client)
    monkeypatch.setattr(curriculum.ml_log, "setup_logging", lambda **kwargs: root_logger)
    monkeypatch.setattr(curriculum.checkpoint_utils, "save_checkpoint_async", fake_save_checkpoint_async)

    class _DummyServiceClient:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

    monkeypatch.setattr(curriculum.tinker, "ServiceClient", _DummyServiceClient)

    config = curriculum.CurriculumConfig(
        model_name="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        renderer_name="qwen3_disable_thinking",
        log_root=str(tmp_path),
        wandb_project="demo",
        openrouter_model="google/gemini-3-flash-preview",
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_api_key_env_var="OPENROUTER_API_KEY",
        http_referer="https://local.codex",
        x_title="demo",
        actor_max_turns=2,
        session_timeout_seconds=10.0,
        default_idle_sleep_seconds=0.1,
        max_message_length=2000,
        save_every=1,
        eval_every=0,
        num_groups_to_log=0,
        rollout_json_export=False,
        stages=(
            curriculum.StageSpec(
                name="stage1_tool_accuracy",
                environment_kind="tool_accuracy",
                num_train_examples=4,
                num_eval_examples=0,
                batch_size=2,
                group_size=2,
                max_tokens=32,
                max_turns=4,
                max_trajectory_tokens=None,
                train_seed=1,
                eval_seed=101,
                learning_rate=3e-5,
                lora_rank=16,
            ),
            curriculum.StageSpec(
                name="stage2_full_press",
                environment_kind="full_press",
                num_train_examples=4,
                num_eval_examples=0,
                batch_size=2,
                group_size=2,
                max_tokens=64,
                max_turns=6,
                max_trajectory_tokens=None,
                train_seed=2,
                eval_seed=102,
                learning_rate=2e-5,
                lora_rank=16,
            ),
        ),
        run_name="curriculum-duplicate-checkpoint-test",
    )

    manifest_path = asyncio.run(curriculum.run_curriculum(config))
    manifest = json.loads(manifest_path.read_text())

    assert manifest["status"] == "completed"
    assert save_calls.count((str(tmp_path / "curriculum-duplicate-checkpoint-test" / "stage1_tool_accuracy"), "000002")) == 1
    assert save_calls.count((str(tmp_path / "curriculum-duplicate-checkpoint-test" / "stage2_full_press"), "000004")) == 1
    assert save_calls[-1][1] == "final"


def test_ensure_stage_end_checkpoint_recovers_existing_sampler_save(tmp_path, monkeypatch) -> None:
    stage_runtime = curriculum.StageRuntime(
        stage=curriculum.StageSpec(
            name="stage1_tool_accuracy",
            environment_kind="tool_accuracy",
            num_train_examples=64,
            num_eval_examples=8,
            batch_size=16,
            group_size=4,
            max_tokens=256,
            max_turns=14,
            max_trajectory_tokens=None,
            train_seed=21,
            eval_seed=10021,
            learning_rate=2e-5,
            lora_rank=32,
        ),
        stage_log_dir=tmp_path / "stage1_tool_accuracy",
        global_start_batch=0,
        global_end_batch=4,
    )
    stage_runtime.stage_log_dir.mkdir(parents=True, exist_ok=True)
    (stage_runtime.stage_log_dir / "checkpoints.jsonl").write_text(
        json.dumps(
            {
                "name": "000000",
                "batch": 0,
                "sampler_path": "tinker://session/sampler_weights/000000",
            }
        )
        + "\n"
    )
    save_calls: list[tuple[str, str, str]] = []

    async def fake_save_checkpoint_async(
        *,
        training_client,
        name: str,
        log_path: str,
        loop_state: dict[str, object],
        kind: str = "state",
        ttl_seconds=None,
    ) -> dict[str, str]:
        del training_client, ttl_seconds, loop_state
        save_calls.append((log_path, name, kind))
        assert kind == "state"
        return {"state_path": f"state://{name}"}

    monkeypatch.setattr(curriculum.checkpoint_utils, "save_checkpoint_async", fake_save_checkpoint_async)

    state_ckpt, sampler_ckpt = asyncio.run(
        curriculum._ensure_stage_end_checkpoint(
            training_client=object(),
            stage_runtime=stage_runtime,
            ttl_seconds=None,
        )
    )

    assert save_calls == [(str(stage_runtime.stage_log_dir), "000004", "state")]
    assert state_ckpt == {
        "name": "000004",
        "batch": 4,
        "stage": "stage1_tool_accuracy",
        "stage_batch": 4,
        "state_path": "state://000004",
        "sampler_path": "tinker://session/sampler_weights/000004",
    }
    assert sampler_ckpt == state_ckpt


def test_stage_metrics_logger_accepts_bound_method_configs(tmp_path) -> None:
    stage_cfg = rl_train.Config(
        learning_rate=3e-5,
        dataset_builder=_FakeDatasetBuilder(1),
        model_name="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        max_tokens=32,
        log_path=str(tmp_path / "stage"),
        eval_every=0,
        save_every=1,
        renderer_name="qwen3_disable_thinking",
        wandb_project=None,
        lora_rank=8,
        num_groups_to_log=0,
        rollout_json_export=False,
        extra_metrics_provider=_FakeRolloutRunner().snapshot_metrics,
    )

    logger = curriculum.StageMetricsLogger(
        root_logger=_FakeRootLogger(),
        stage_log_dir=tmp_path / "stage",
        stage_config=stage_cfg,
    )
    logger.close()

    config_path = tmp_path / "stage" / "config.json"
    assert config_path.exists()
    logged_config = json.loads(config_path.read_text())
    assert logged_config["extra_metrics_provider"].endswith("snapshot_metrics")
