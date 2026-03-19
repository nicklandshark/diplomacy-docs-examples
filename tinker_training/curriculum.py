from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDORED_COOKBOOK_ROOT = REPO_ROOT / "vendor" / "tinker-cookbook"
if str(VENDORED_COOKBOOK_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDORED_COOKBOOK_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tinker_cookbook import checkpoint_utils
from tinker_cookbook.rl import train as rl_train
from tinker_cookbook.rl.metric_util import RLTestSetEvaluator
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset
from tinker_cookbook.utils import ml_log
import tinker

from tinker_training.diplomacy_adapter import (
    ActorRuntimeConfig,
    DiplomacyDatasetBuilder,
    OpenRouterHeaders,
    RuntimePolicyConfig,
    build_actor_configs,
)
from tinker_training.rollout_backends import (
    ModalTrajectorySandboxRunner,
    TrajectorySandboxRunner,
    register_trajectory_runner,
    unregister_trajectory_runner,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
DEFAULT_LOG_ROOT = "~/tinker-runs/diplomacy-grpo"
DEFAULT_WANDB_PROJECT = "diplomacy-grpo"
DEFAULT_OPENROUTER_MODEL = "google/gemini-3-flash-preview"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_API_KEY_ENV_VAR = "OPENROUTER_API_KEY"


@dataclass(frozen=True)
class StageSpec:
    name: str
    environment_kind: Literal["tool_accuracy", "full_press"]
    num_train_examples: int
    num_eval_examples: int
    batch_size: int
    group_size: int
    max_tokens: int
    max_turns: int
    max_trajectory_tokens: int | None
    train_seed: int
    eval_seed: int
    learning_rate: float
    lora_rank: int


@dataclass(frozen=True)
class ModalRolloutConfig:
    app_name: str = "diplomacy-grpo-rollouts"
    timeout_seconds: int = 900
    cpu: float = 2.0
    memory_mb: int = 4096
    single_use_containers: bool = True
    max_inputs: int = 1
    retries: int = 0


@dataclass(frozen=True)
class CurriculumConfig:
    model_name: str
    renderer_name: str
    log_root: str
    wandb_project: str
    openrouter_model: str
    openrouter_base_url: str
    openrouter_api_key_env_var: str
    http_referer: str
    x_title: str
    actor_max_turns: int
    session_timeout_seconds: float
    default_idle_sleep_seconds: float
    max_message_length: int
    save_every: int
    eval_every: int
    num_groups_to_log: int
    rollout_json_export: bool
    stages: tuple[StageSpec, ...]
    modal_rollout: ModalRolloutConfig = field(default_factory=ModalRolloutConfig)
    run_name: str | None = None
    initial_checkpoint_path: str | None = None
    tracked_instruction_block: str | None = None


@dataclass(frozen=True)
class StageRuntime:
    stage: StageSpec
    stage_log_dir: Path
    global_start_batch: int
    global_end_batch: int

    @property
    def num_batches(self) -> int:
        return self.global_end_batch - self.global_start_batch


@dataclass(frozen=True)
class ResumeState:
    completed_stage_results: list[dict[str, Any]]
    next_stage_index: int
    start_batch: int
    resume_state_path: str | None


class OffsetRLDataset(RLDataset):
    def __init__(self, base_dataset: RLDataset, *, batch_offset: int) -> None:
        self.base_dataset = base_dataset
        self.batch_offset = batch_offset

    def get_batch(self, index: int) -> list[EnvGroupBuilder]:
        translated_index = index - self.batch_offset
        if translated_index < 0 or translated_index >= len(self.base_dataset):
            raise IndexError(
                f"Batch index {index} is outside offset dataset "
                f"[{self.batch_offset}, {self.batch_offset + len(self.base_dataset)})"
            )
        return list(self.base_dataset.get_batch(translated_index))

    def __len__(self) -> int:
        return self.batch_offset + len(self.base_dataset)


class StageMetricsLogger(ml_log.Logger):
    def __init__(self, *, root_logger: ml_log.Logger, stage_log_dir: Path, stage_config: Any) -> None:
        self.root_logger = root_logger
        self.stage_logger = ml_log.JsonLogger(stage_log_dir)
        self.stage_logger.log_hparams(_serialize_stage_config(stage_config))

    def log_hparams(self, config: Any) -> None:
        self.stage_logger.log_hparams(config)

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self.root_logger.log_metrics(metrics, step=step)
        self.stage_logger.log_metrics(metrics, step=step)

    def close(self) -> None:
        self.stage_logger.close()

    def sync(self) -> None:
        self.root_logger.sync()
        self.stage_logger.sync()

    def get_logger_url(self) -> str | None:
        return self.root_logger.get_logger_url()


def _serialize_stage_config(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _serialize_stage_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize_stage_config(item) for item in value]
    if callable(value):
        return getattr(value, "__qualname__", repr(value))
    if is_dataclass(value):
        return {
            field_info.name: _serialize_stage_config(getattr(value, field_info.name))
            for field_info in fields(value)
        }
    if hasattr(value, "__dict__"):
        return {
            key: _serialize_stage_config(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return repr(value)


def _sanitize_run_name_component(text: str) -> str:
    sanitized = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return sanitized or "diplomacy"


def resolve_run_name(config: CurriculumConfig) -> str:
    if config.run_name:
        return config.run_name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_suffix = config.model_name.split("/", 1)[-1]
    return f"{_sanitize_run_name_component(model_suffix)}-diplomacy-{timestamp}"


def build_actor_runtime(config: CurriculumConfig) -> ActorRuntimeConfig:
    return ActorRuntimeConfig(
        actor_configs=build_actor_configs(
            base_url=config.openrouter_base_url,
            api_key_env_var=config.openrouter_api_key_env_var,
            model=config.openrouter_model,
        ),
        actor_max_turns=config.actor_max_turns,
        session_timeout_seconds=config.session_timeout_seconds,
        default_idle_sleep_seconds=config.default_idle_sleep_seconds,
        openrouter_headers=OpenRouterHeaders(
            http_referer=config.http_referer,
            x_title=config.x_title,
        ),
    )


def create_rollout_runner(config: CurriculumConfig) -> TrajectorySandboxRunner:
    return ModalTrajectorySandboxRunner(
        app_name=config.modal_rollout.app_name,
        timeout_seconds=config.modal_rollout.timeout_seconds,
        cpu=config.modal_rollout.cpu,
        memory_mb=config.modal_rollout.memory_mb,
        single_use_containers=config.modal_rollout.single_use_containers,
        max_inputs=config.modal_rollout.max_inputs,
        retries=config.modal_rollout.retries,
    )


def validate_curriculum(config: CurriculumConfig) -> None:
    if not config.stages:
        raise ValueError("Curriculum must contain at least one stage.")
    lora_ranks = {stage.lora_rank for stage in config.stages}
    if len(lora_ranks) != 1:
        raise ValueError(
            "All stages must use the same LoRA rank because the curriculum now stays "
            "inside a single Tinker training run."
        )
    for stage in config.stages:
        if stage.batch_size <= 0:
            raise ValueError(f"{stage.name}: batch_size must be positive.")
        if stage.num_train_examples < stage.batch_size:
            raise ValueError(
                f"{stage.name}: num_train_examples ({stage.num_train_examples}) must be at least "
                f"batch_size ({stage.batch_size}) so the stage produces at least one batch."
            )


def build_stage_runtimes(config: CurriculumConfig, root_log_dir: Path) -> tuple[StageRuntime, ...]:
    global_batch_cursor = 0
    runtimes: list[StageRuntime] = []
    for stage in config.stages:
        num_batches = stage.num_train_examples // stage.batch_size
        stage_log_dir = root_log_dir / stage.name
        stage_log_dir.mkdir(parents=True, exist_ok=True)
        runtimes.append(
            StageRuntime(
                stage=stage,
                stage_log_dir=stage_log_dir,
                global_start_batch=global_batch_cursor,
                global_end_batch=global_batch_cursor + num_batches,
            )
        )
        global_batch_cursor += num_batches
    return tuple(runtimes)


def build_stage_train_config(
    *,
    stage_runtime: StageRuntime,
    config: CurriculumConfig,
    actor_runtime: ActorRuntimeConfig,
    rollout_runner: TrajectorySandboxRunner,
    rollout_runner_id: str,
) -> rl_train.Config:
    stage = stage_runtime.stage
    policy_config = RuntimePolicyConfig(
        max_turns=stage.max_turns,
        max_message_length=config.max_message_length,
        max_trajectory_tokens=stage.max_trajectory_tokens,
    )
    dataset_builder = DiplomacyDatasetBuilder(
        environment_kind=stage.environment_kind,
        model_name_for_tokenizer=config.model_name,
        renderer_name=config.renderer_name,
        actor_runtime=actor_runtime,
        policy_config=policy_config,
        tracked_instruction_block=config.tracked_instruction_block,
        batch_size=stage.batch_size,
        group_size=stage.group_size,
        num_train_examples=stage.num_train_examples,
        num_eval_examples=stage.num_eval_examples,
        train_seed=stage.train_seed,
        eval_seed=stage.eval_seed,
        rollout_runner_id=rollout_runner_id,
    )
    return rl_train.Config(
        learning_rate=stage.learning_rate,
        dataset_builder=dataset_builder,
        model_name=config.model_name,
        max_tokens=stage.max_tokens,
        log_path=str(stage_runtime.stage_log_dir),
        eval_every=config.eval_every,
        save_every=config.save_every,
        renderer_name=config.renderer_name,
        wandb_project=config.wandb_project,
        lora_rank=stage.lora_rank,
        num_groups_to_log=config.num_groups_to_log,
        rollout_json_export=config.rollout_json_export,
        extra_metrics_provider=rollout_runner.snapshot_metrics,
    )


def load_stage_checkpoints(stage_log_dir: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    return (
        checkpoint_utils.get_last_checkpoint(str(stage_log_dir), required_key="state_path"),
        checkpoint_utils.get_last_checkpoint(str(stage_log_dir), required_key="sampler_path"),
    )


def _checkpoint_covers_stage_end(
    checkpoint: dict[str, Any] | None,
    *,
    expected_name: str,
    expected_batch: int,
) -> bool:
    if checkpoint is None:
        return False
    return checkpoint.get("name") == expected_name or checkpoint.get("batch") == expected_batch


def _stage_end_checkpoint_already_saved(
    *,
    state_ckpt: dict[str, Any] | None,
    sampler_ckpt: dict[str, Any] | None,
    expected_name: str,
    expected_batch: int,
) -> bool:
    return _checkpoint_covers_stage_end(
        state_ckpt,
        expected_name=expected_name,
        expected_batch=expected_batch,
    ) and _checkpoint_covers_stage_end(
        sampler_ckpt,
        expected_name=expected_name,
        expected_batch=expected_batch,
    )


def _seed_stage_sampler_checkpoint_from_previous_stage(
    *,
    stage_runtime: StageRuntime,
    previous_stage_runtime: StageRuntime | None,
    start_batch: int,
) -> None:
    if previous_stage_runtime is None:
        return
    if start_batch != stage_runtime.global_start_batch or start_batch <= 0:
        return

    _, existing_sampler_ckpt = load_stage_checkpoints(stage_runtime.stage_log_dir)
    if existing_sampler_ckpt is not None:
        return

    _, previous_sampler_ckpt = load_stage_checkpoints(previous_stage_runtime.stage_log_dir)
    if previous_sampler_ckpt is None or previous_sampler_ckpt.get("batch") != start_batch:
        return

    stage_runtime.stage_log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_entry = {
        "name": previous_sampler_ckpt.get("name", f"{start_batch:06d}"),
        "batch": start_batch,
        "sampler_path": previous_sampler_ckpt["sampler_path"],
    }
    with (stage_runtime.stage_log_dir / "checkpoints.jsonl").open("a") as handle:
        handle.write(json.dumps(checkpoint_entry) + "\n")


def _build_stage_manifest_entry(
    *,
    stage_runtime: StageRuntime,
    state_ckpt: dict[str, Any] | None,
    sampler_ckpt: dict[str, Any] | None,
    rollout_backend: str,
    rollout_backend_stats: dict[str, Any],
    status: str,
) -> dict[str, Any]:
    stage = stage_runtime.stage
    return {
        "stage": stage.name,
        "status": status,
        "environment_kind": stage.environment_kind,
        "log_dir": str(stage_runtime.stage_log_dir),
        "state_path": state_ckpt["state_path"] if state_ckpt is not None else None,
        "sampler_path": sampler_ckpt["sampler_path"] if sampler_ckpt is not None else None,
        "global_start_batch": stage_runtime.global_start_batch,
        "global_end_batch": stage_runtime.global_end_batch,
        "train_batches": stage_runtime.num_batches,
        "dropped_train_examples": stage.num_train_examples % stage.batch_size,
        "train_examples": stage.num_train_examples,
        "eval_examples": stage.num_eval_examples,
        "batch_size": stage.batch_size,
        "group_size": stage.group_size,
        "max_tokens": stage.max_tokens,
        "max_turns": stage.max_turns,
        "learning_rate": stage.learning_rate,
        "lora_rank": stage.lora_rank,
        "rollout_backend": rollout_backend,
        "rollout_backend_stats": rollout_backend_stats,
        "updated_at": datetime.now().isoformat(),
    }


def _upsert_manifest_stage(manifest: dict[str, Any], stage_entry: dict[str, Any]) -> None:
    existing_entries = manifest.setdefault("stages", [])
    for index, current_entry in enumerate(existing_entries):
        if current_entry.get("stage") == stage_entry.get("stage"):
            existing_entries[index] = stage_entry
            return
    existing_entries.append(stage_entry)


def discover_resume_state(
    *,
    stage_runtimes: tuple[StageRuntime, ...],
    config: CurriculumConfig,
    rollout_runner: TrajectorySandboxRunner,
) -> ResumeState:
    completed_stage_results: list[dict[str, Any]] = []
    resume_state_path: str | None = None
    next_stage_index = 0
    start_batch = 0
    for stage_runtime in stage_runtimes:
        state_ckpt, sampler_ckpt = load_stage_checkpoints(stage_runtime.stage_log_dir)
        if state_ckpt is None:
            break

        recorded_batch = int(state_ckpt.get("batch", stage_runtime.global_start_batch))
        if recorded_batch >= stage_runtime.global_end_batch:
            completed_stage_results.append(
                _build_stage_manifest_entry(
                    stage_runtime=stage_runtime,
                    state_ckpt=state_ckpt,
                    sampler_ckpt=sampler_ckpt,
                    rollout_backend="modal",
                    rollout_backend_stats=dict(rollout_runner.summary()),
                    status="completed",
                )
            )
            resume_state_path = state_ckpt["state_path"]
            next_stage_index += 1
            start_batch = stage_runtime.global_end_batch
            continue

        resume_state_path = state_ckpt["state_path"]
        start_batch = recorded_batch
        break
    else:
        start_batch = stage_runtimes[-1].global_end_batch if stage_runtimes else 0

    return ResumeState(
        completed_stage_results=completed_stage_results,
        next_stage_index=next_stage_index,
        start_batch=start_batch,
        resume_state_path=resume_state_path,
    )


def _select_training_function(cfg: rl_train.Config):
    if cfg.async_config is not None:
        return rl_train.do_async_training
    if cfg.stream_minibatch_config is not None:
        return rl_train.do_sync_training_with_stream_minibatch
    return rl_train.do_sync_training


async def create_training_client(
    *,
    config: CurriculumConfig,
    service_client: tinker.ServiceClient,
    user_metadata: dict[str, str],
    resume_state_path: str | None,
    lora_rank: int,
) -> tinker.TrainingClient:
    if resume_state_path:
        await checkpoint_utils.check_renderer_name_for_checkpoint_async(
            service_client, resume_state_path, config.renderer_name
        )
        training_client = await service_client.create_training_client_from_state_with_optimizer_async(
            resume_state_path,
            user_metadata=user_metadata,
        )
        logger.info("Resumed curriculum training from %s", resume_state_path)
        return training_client

    if config.initial_checkpoint_path:
        await checkpoint_utils.check_renderer_name_for_checkpoint_async(
            service_client, config.initial_checkpoint_path, config.renderer_name
        )
        training_client = await service_client.create_training_client_from_state_async(
            config.initial_checkpoint_path,
            user_metadata=user_metadata,
        )
        logger.info("Loaded initial curriculum weights from %s", config.initial_checkpoint_path)
        return training_client

    training_client = await service_client.create_lora_training_client_async(
        config.model_name,
        rank=lora_rank,
        user_metadata=user_metadata,
    )
    logger.info("Created fresh curriculum training client for %s", config.model_name)
    return training_client


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2))


async def run_curriculum(config: CurriculumConfig) -> Path:
    validate_curriculum(config)
    run_name = resolve_run_name(config)
    actor_runtime = build_actor_runtime(config)
    root_log_dir = Path(config.log_root).expanduser() / run_name
    root_log_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = root_log_dir / "curriculum_manifest.json"
    stage_runtimes = build_stage_runtimes(config, root_log_dir)
    total_batches = stage_runtimes[-1].global_end_batch

    rollout_runner = create_rollout_runner(config)
    await rollout_runner.start()
    rollout_runner_id = register_trajectory_runner(rollout_runner)
    current_stage_name: str | None = None
    manifest: dict[str, Any] = {
        "run_name": run_name,
        "created_at": datetime.now().isoformat(),
        "status": "running",
        "single_tinker_training_run": True,
        "model_name": config.model_name,
        "renderer_name": config.renderer_name,
        "wandb_project": config.wandb_project,
        "openrouter_model": config.openrouter_model,
        "openrouter_base_url": config.openrouter_base_url,
        "rollout_backend": "modal",
        "modal_rollout": asdict(config.modal_rollout),
        "total_batches": total_batches,
        "stages": [],
    }
    root_logger: ml_log.Logger | None = None

    try:
        resume_state = discover_resume_state(
            stage_runtimes=stage_runtimes,
            config=config,
            rollout_runner=rollout_runner,
        )
        manifest["stages"] = list(resume_state.completed_stage_results)
        manifest["resumed_from_state_path"] = resume_state.resume_state_path
        manifest["current_stage"] = (
            stage_runtimes[resume_state.next_stage_index].stage.name
            if resume_state.next_stage_index < len(stage_runtimes)
            else None
        )
        _write_manifest(manifest_path, manifest)

        if resume_state.next_stage_index >= len(stage_runtimes):
            manifest["status"] = "completed"
            manifest["completed_at"] = datetime.now().isoformat()
            manifest["rollout_backend_stats"] = dict(rollout_runner.summary())
            _write_manifest(manifest_path, manifest)
            logger.info("Curriculum already complete. Manifest: %s", manifest_path)
            return manifest_path

        root_logger = ml_log.setup_logging(
            log_dir=str(root_log_dir),
            wandb_project=config.wandb_project,
            config=config,
            wandb_name=run_name,
        )

        service_client = tinker.ServiceClient()
        user_metadata: dict[str, str] = {}
        if wandb_link := root_logger.get_logger_url():
            user_metadata["wandb_link"] = wandb_link
        checkpoint_utils.add_renderer_name_to_user_metadata(user_metadata, config.renderer_name)

        training_client = await create_training_client(
            config=config,
            service_client=service_client,
            user_metadata=user_metadata,
            resume_state_path=resume_state.resume_state_path,
            lora_rank=config.stages[0].lora_rank,
        )
        tokenizer = training_client.get_tokenizer()

        start_batch = resume_state.start_batch
        for stage_index, stage_runtime in enumerate(
            stage_runtimes[resume_state.next_stage_index :],
            start=resume_state.next_stage_index,
        ):
            current_stage_name = stage_runtime.stage.name
            logger.info("Starting %s in %s", current_stage_name, stage_runtime.stage_log_dir)

            stage_cfg = build_stage_train_config(
                stage_runtime=stage_runtime,
                config=config,
                actor_runtime=actor_runtime,
                rollout_runner=rollout_runner,
                rollout_runner_id=rollout_runner_id,
            )
            dataset, maybe_test_dataset = await stage_cfg.dataset_builder()
            train_dataset = OffsetRLDataset(
                dataset,
                batch_offset=stage_runtime.global_start_batch,
            )
            evaluators = [builder() for builder in stage_cfg.evaluator_builders]
            if maybe_test_dataset is not None:
                evaluators.append(
                    RLTestSetEvaluator(
                        maybe_test_dataset,
                        max_tokens=stage_cfg.max_tokens,
                    )
                )

            stage_logger = StageMetricsLogger(
                root_logger=root_logger,
                stage_log_dir=stage_runtime.stage_log_dir,
                stage_config=stage_cfg,
            )
            previous_stage_runtime = stage_runtimes[stage_index - 1] if stage_index > 0 else None
            _seed_stage_sampler_checkpoint_from_previous_stage(
                stage_runtime=stage_runtime,
                previous_stage_runtime=previous_stage_runtime,
                start_batch=start_batch,
            )
            state_ckpt, sampler_ckpt = load_stage_checkpoints(stage_runtime.stage_log_dir)
            running_stage_entry = _build_stage_manifest_entry(
                stage_runtime=stage_runtime,
                state_ckpt=state_ckpt,
                sampler_ckpt=sampler_ckpt,
                rollout_backend="modal",
                rollout_backend_stats=dict(rollout_runner.summary()),
                status="running",
            )
            running_stage_entry["resume_batch"] = start_batch
            _upsert_manifest_stage(manifest, running_stage_entry)
            manifest["current_stage"] = current_stage_name
            manifest["updated_at"] = datetime.now().isoformat()
            _write_manifest(manifest_path, manifest)

            try:
                training_func = _select_training_function(stage_cfg)
                await training_func(
                    start_batch=max(start_batch, stage_runtime.global_start_batch),
                    end_batch=stage_runtime.global_end_batch,
                    num_batches=total_batches,
                    cfg=stage_cfg,
                    training_client=training_client,
                    kl_reference_client=None,
                    evaluators=evaluators,
                    dataset=train_dataset,
                    ml_logger=stage_logger,
                    tokenizer=tokenizer,
                )
            finally:
                stage_logger.close()

            state_ckpt, sampler_ckpt = load_stage_checkpoints(stage_runtime.stage_log_dir)
            stage_end_checkpoint_name = f"{stage_runtime.global_end_batch:06d}"
            if not _stage_end_checkpoint_already_saved(
                state_ckpt=state_ckpt,
                sampler_ckpt=sampler_ckpt,
                expected_name=stage_end_checkpoint_name,
                expected_batch=stage_runtime.global_end_batch,
            ):
                await checkpoint_utils.save_checkpoint_async(
                    training_client=training_client,
                    name=stage_end_checkpoint_name,
                    log_path=str(stage_runtime.stage_log_dir),
                    kind="both",
                    loop_state={
                        "batch": stage_runtime.global_end_batch,
                        "stage": current_stage_name,
                        "stage_batch": stage_runtime.num_batches,
                    },
                    ttl_seconds=stage_cfg.ttl_seconds,
                )
                state_ckpt, sampler_ckpt = load_stage_checkpoints(stage_runtime.stage_log_dir)
            stage_result = _build_stage_manifest_entry(
                stage_runtime=stage_runtime,
                state_ckpt=state_ckpt,
                sampler_ckpt=sampler_ckpt,
                rollout_backend="modal",
                rollout_backend_stats=dict(rollout_runner.summary()),
                status="completed",
            )
            if stage_result["state_path"] is None:
                raise RuntimeError(f"No state checkpoint found after {current_stage_name}")
            logger.info("Finished %s", current_stage_name)
            _upsert_manifest_stage(manifest, stage_result)
            manifest["rollout_backend_stats"] = dict(rollout_runner.summary())
            manifest["updated_at"] = datetime.now().isoformat()
            _write_manifest(manifest_path, manifest)
            start_batch = stage_runtime.global_end_batch

        last_stage_runtime = stage_runtimes[-1]
        await checkpoint_utils.save_checkpoint_async(
            training_client=training_client,
            name="final",
            log_path=str(last_stage_runtime.stage_log_dir),
            kind="both",
            loop_state={
                "batch": total_batches,
                "stage": last_stage_runtime.stage.name,
                "stage_batch": last_stage_runtime.num_batches,
            },
            ttl_seconds=None,
        )
        final_stage_state, final_stage_sampler = load_stage_checkpoints(
            last_stage_runtime.stage_log_dir
        )
        _upsert_manifest_stage(
            manifest,
            _build_stage_manifest_entry(
                stage_runtime=last_stage_runtime,
                state_ckpt=final_stage_state,
                sampler_ckpt=final_stage_sampler,
                rollout_backend="modal",
                rollout_backend_stats=dict(rollout_runner.summary()),
                status="completed",
            ),
        )
        manifest["status"] = "completed"
        manifest["current_stage"] = None
        manifest["completed_at"] = datetime.now().isoformat()
        manifest["rollout_backend_stats"] = dict(rollout_runner.summary())
        _write_manifest(manifest_path, manifest)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["current_stage"] = current_stage_name
        manifest["error"] = str(exc)
        manifest["updated_at"] = datetime.now().isoformat()
        manifest["rollout_backend_stats"] = dict(rollout_runner.summary())
        _write_manifest(manifest_path, manifest)
        raise
    finally:
        unregister_trajectory_runner(rollout_runner_id)
        await rollout_runner.aclose()
        if root_logger is not None:
            root_logger.close()

    logger.info("Curriculum finished. Manifest: %s", manifest_path)
    return manifest_path
