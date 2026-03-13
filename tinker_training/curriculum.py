from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass, field
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

from tinker_training.diplomacy_adapter import (
    ActorRuntimeConfig,
    DiplomacyDatasetBuilder,
    OpenRouterHeaders,
    RuntimePolicyConfig,
    build_actor_configs,
)
from tinker_training.rollout_backends import (
    LocalTrajectorySandboxRunner,
    ModalTrajectorySandboxRunner,
    TrajectorySandboxRunner,
    register_trajectory_runner,
    unregister_trajectory_runner,
)

logger = logging.getLogger(__name__)


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
    local_fallback_on_infra_failure: bool = True


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
    rollout_backend: Literal["local", "modal"] = "modal"
    modal_rollout: ModalRolloutConfig = field(default_factory=ModalRolloutConfig)
    run_name: str | None = None
    initial_checkpoint_path: str | None = None


def resolve_run_name(config: CurriculumConfig) -> str:
    if config.run_name:
        return config.run_name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"qwen35-27b-diplomacy-{timestamp}"


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
    if config.rollout_backend == "modal":
        return ModalTrajectorySandboxRunner(
            app_name=config.modal_rollout.app_name,
            timeout_seconds=config.modal_rollout.timeout_seconds,
            cpu=config.modal_rollout.cpu,
            memory_mb=config.modal_rollout.memory_mb,
            single_use_containers=config.modal_rollout.single_use_containers,
            max_inputs=config.modal_rollout.max_inputs,
            retries=config.modal_rollout.retries,
            local_fallback_on_infra_failure=config.modal_rollout.local_fallback_on_infra_failure,
        )
    return LocalTrajectorySandboxRunner()


async def run_stage(
    *,
    stage: StageSpec,
    config: CurriculumConfig,
    actor_runtime: ActorRuntimeConfig,
    rollout_runner: TrajectorySandboxRunner,
    rollout_runner_id: str,
    run_name: str,
    root_log_dir: Path,
    load_checkpoint_path: str | None,
) -> dict[str, Any]:
    stage_log_dir = root_log_dir / stage.name
    stage_log_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Starting %s in %s", stage.name, stage_log_dir)

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
        batch_size=stage.batch_size,
        group_size=stage.group_size,
        num_train_examples=stage.num_train_examples,
        num_eval_examples=stage.num_eval_examples,
        train_seed=stage.train_seed,
        eval_seed=stage.eval_seed,
        rollout_runner_id=rollout_runner_id,
    )
    rl_config = rl_train.Config(
        learning_rate=stage.learning_rate,
        dataset_builder=dataset_builder,
        model_name=config.model_name,
        max_tokens=stage.max_tokens,
        log_path=str(stage_log_dir),
        eval_every=config.eval_every,
        save_every=config.save_every,
        load_checkpoint_path=load_checkpoint_path,
        renderer_name=config.renderer_name,
        wandb_project=config.wandb_project,
        wandb_name=f"{run_name}-{stage.name}",
        lora_rank=stage.lora_rank,
        num_groups_to_log=config.num_groups_to_log,
        rollout_json_export=config.rollout_json_export,
        extra_metrics_provider=rollout_runner.snapshot_metrics,
    )
    await rl_train.main(rl_config)

    state_ckpt = checkpoint_utils.get_last_checkpoint(str(stage_log_dir), required_key="state_path")
    sampler_ckpt = checkpoint_utils.get_last_checkpoint(str(stage_log_dir), required_key="sampler_path")
    if state_ckpt is None:
        raise RuntimeError(f"No state checkpoint found after {stage.name}")

    result = {
        "stage": stage.name,
        "environment_kind": stage.environment_kind,
        "log_dir": str(stage_log_dir),
        "state_path": state_ckpt["state_path"],
        "sampler_path": sampler_ckpt["sampler_path"] if sampler_ckpt is not None else None,
        "train_examples": stage.num_train_examples,
        "eval_examples": stage.num_eval_examples,
        "batch_size": stage.batch_size,
        "group_size": stage.group_size,
        "max_tokens": stage.max_tokens,
        "max_turns": stage.max_turns,
    }
    logger.info("Finished %s", stage.name)
    return result


async def run_curriculum(config: CurriculumConfig) -> Path:
    run_name = resolve_run_name(config)
    actor_runtime = build_actor_runtime(config)
    root_log_dir = Path(config.log_root).expanduser() / run_name
    root_log_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = root_log_dir / "curriculum_manifest.json"

    rollout_runner = create_rollout_runner(config)
    await rollout_runner.start()
    rollout_runner_id = register_trajectory_runner(rollout_runner)
    manifest: dict[str, Any] = {
        "run_name": run_name,
        "created_at": datetime.now().isoformat(),
        "model_name": config.model_name,
        "renderer_name": config.renderer_name,
        "wandb_project": config.wandb_project,
        "openrouter_model": config.openrouter_model,
        "openrouter_base_url": config.openrouter_base_url,
        "rollout_backend": config.rollout_backend,
        "modal_rollout": asdict(config.modal_rollout),
        "stages": [],
    }

    try:
        checkpoint_path = config.initial_checkpoint_path
        for stage in config.stages:
            stage_result = await run_stage(
                stage=stage,
                config=config,
                actor_runtime=actor_runtime,
                rollout_runner=rollout_runner,
                rollout_runner_id=rollout_runner_id,
                run_name=run_name,
                root_log_dir=root_log_dir,
                load_checkpoint_path=checkpoint_path,
            )
            stage_result["rollout_backend"] = config.rollout_backend
            stage_result["rollout_backend_stats"] = dict(rollout_runner.summary())
            manifest["stages"].append(stage_result)
            checkpoint_path = stage_result["state_path"]
            manifest["rollout_backend_stats"] = dict(rollout_runner.summary())
            manifest_path.write_text(json.dumps(manifest, indent=2))
    finally:
        unregister_trajectory_runner(rollout_runner_id)
        await rollout_runner.aclose()

    logger.info("Curriculum finished. Manifest: %s", manifest_path)
    return manifest_path
