from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import tinker

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDORED_COOKBOOK_ROOT = REPO_ROOT / "vendor" / "tinker-cookbook"
if str(VENDORED_COOKBOOK_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDORED_COOKBOOK_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import modal
except ImportError:  # pragma: no cover - optional dependency at import time
    modal = None

from tinker_cookbook.completers import TinkerTokenCompleter, TokensWithLogprobs
from tinker_cookbook.rl.rollouts import do_single_rollout
from tinker_cookbook.rl.types import SamplingRef, Trajectory, Transition

logger = logging.getLogger(__name__)

EnvironmentKind = Literal["tool_accuracy", "full_press"]
_RUNNER_REGISTRY: dict[str, "TrajectorySandboxRunner"] = {}


@dataclass(frozen=True)
class SerializedTransition:
    ob: dict[str, Any]
    ac_tokens: list[int]
    ac_logprobs: list[float] | None
    reward: float
    episode_done: bool
    metrics: dict[str, float | int] = field(default_factory=dict)
    logs: dict[str, str | int | float] = field(default_factory=dict)


@dataclass(frozen=True)
class SerializedTrajectory:
    transitions: list[SerializedTransition]
    final_ob: dict[str, Any]


@dataclass(frozen=True)
class TrajectoryRolloutRequest:
    datum: dict[str, Any]
    environment_kind: EnvironmentKind
    model_name: str
    renderer_name: str
    actor_runtime: Any
    policy_config: Any
    sampling_ref: SamplingRef
    max_tokens: int
    temperature: float
    trajectory_index: int
    group_id: str
    enable_logging: bool = False


@dataclass
class TrajectoryRolloutResult:
    trajectory: SerializedTrajectory | None
    final_reward: float = 0.0
    metrics: dict[str, float | int] = field(default_factory=dict)
    remote_metadata: dict[str, Any] = field(default_factory=dict)
    failure_kind: str | None = None
    failure_message: str | None = None


@runtime_checkable
class TrajectorySandboxRunner(Protocol):
    backend_name: str

    async def start(self) -> None:
        ...

    async def run_trajectory(self, request: TrajectoryRolloutRequest) -> TrajectoryRolloutResult:
        ...

    async def aclose(self) -> None:
        ...

    def summary(self) -> dict[str, Any]:
        ...

    def snapshot_metrics(self, reset: bool = True) -> dict[str, float]:
        ...


def register_trajectory_runner(runner: TrajectorySandboxRunner) -> str:
    runner_id = f"{runner.backend_name}:{uuid.uuid4().hex}"
    _RUNNER_REGISTRY[runner_id] = runner
    return runner_id


def get_trajectory_runner(runner_id: str) -> TrajectorySandboxRunner:
    try:
        return _RUNNER_REGISTRY[runner_id]
    except KeyError as exc:  # pragma: no cover - defensive path
        raise KeyError(f"Unknown rollout runner id: {runner_id}") from exc


def unregister_trajectory_runner(runner_id: str) -> None:
    _RUNNER_REGISTRY.pop(runner_id, None)


def serialize_trajectory(trajectory: Trajectory) -> SerializedTrajectory:
    return SerializedTrajectory(
        transitions=[
            SerializedTransition(
                ob=transition.ob.model_dump(mode="python"),
                ac_tokens=list(transition.ac.tokens),
                ac_logprobs=(
                    list(transition.ac.maybe_logprobs)
                    if transition.ac.maybe_logprobs is not None
                    else None
                ),
                reward=float(transition.reward),
                episode_done=bool(transition.episode_done),
                metrics=dict(transition.metrics),
                logs=dict(transition.logs),
            )
            for transition in trajectory.transitions
        ],
        final_ob=trajectory.final_ob.model_dump(mode="python"),
    )


def deserialize_trajectory(serialized: SerializedTrajectory) -> Trajectory:
    transitions = [
        Transition(
            ob=tinker.ModelInput.model_validate(transition.ob),
            ac=TokensWithLogprobs(
                tokens=list(transition.ac_tokens),
                maybe_logprobs=(
                    list(transition.ac_logprobs) if transition.ac_logprobs is not None else None
                ),
            ),
            reward=float(transition.reward),
            episode_done=bool(transition.episode_done),
            metrics=dict(transition.metrics),
            logs=dict(transition.logs),
        )
        for transition in serialized.transitions
    ]
    return Trajectory(
        transitions=transitions,
        final_ob=tinker.ModelInput.model_validate(serialized.final_ob),
    )


async def _execute_rollout_request(request: TrajectoryRolloutRequest) -> TrajectoryRolloutResult:
    from tinker_training import diplomacy_adapter as adapter

    started_at = time.time()
    try:
        built_env = adapter.build_single_diplomacy_env(
            datum=request.datum,
            environment_kind=request.environment_kind,
            model_name=request.model_name,
            renderer_name=request.renderer_name,
            actor_runtime=request.actor_runtime,
            policy_config=request.policy_config,
        )
        service_client = tinker.ServiceClient(base_url=request.sampling_ref.base_url)
        sampling_client = service_client.create_sampling_client(
            model_path=request.sampling_ref.sampler_path
        )
        policy = TinkerTokenCompleter(
            sampling_client,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        trajectory = await do_single_rollout(policy, built_env.env)
        metrics: dict[str, float | int] = {
            "rollout/backend_seconds": time.time() - started_at,
            "rollout/actor_timeout_count": int(
                built_env.state.get("diplomacy_actor_timeout_count", 0)
            ),
            "rollout/actor_error_count": int(built_env.state.get("diplomacy_actor_error_count", 0)),
            "rollout/rejected_tool_calls": int(built_env.state.get("rejected_tool_calls", 0)),
        }
        return TrajectoryRolloutResult(
            trajectory=serialize_trajectory(trajectory),
            final_reward=0.0,
            metrics=metrics,
            remote_metadata={
                "group_id": request.group_id,
                "trajectory_index": request.trajectory_index,
                "environment_kind": request.environment_kind,
                "sampler_path": request.sampling_ref.sampler_path,
            },
        )
    except Exception as exc:
        logger.exception("Trajectory rollout failed for %s[%s]", request.group_id, request.trajectory_index)
        return TrajectoryRolloutResult(
            trajectory=None,
            metrics={"rollout/backend_seconds": time.time() - started_at},
            remote_metadata={
                "group_id": request.group_id,
                "trajectory_index": request.trajectory_index,
                "environment_kind": request.environment_kind,
            },
            failure_kind=type(exc).__name__,
            failure_message=str(exc),
        )


def _default_modal_image() -> "modal.Image":
    if modal is None:  # pragma: no cover - guarded by caller
        raise RuntimeError("modal is not installed")

    return (
        modal.Image.debian_slim(python_version="3.12")
        .add_local_dir(
            str(REPO_ROOT),
            "/root/project",
            copy=True,
            ignore=[".git", ".venv", "__pycache__", ".tmp"],
        )
        .pip_install(
            "datasets>=2.0.0",
            "diplomacy==1.1.2",
            "openai>=1.0.0",
            "tinker>=0.9.0",
            "verifiers>=0.1.9,<0.1.10",
        )
        .run_commands(
            "cd /root/project && pip install -e 'vendor/tinker-cookbook[wandb,verifiers]'",
            env={"PYTHONPATH": "/root/project:/root/project/vendor/tinker-cookbook"},
        )
    )


def _modal_secret_from_env() -> "modal.Secret | None":
    if modal is None:  # pragma: no cover - guarded by caller
        return None
    env_dict = {
        key: value
        for key in (
            "OPENROUTER_API_KEY",
            "TINKER_API_KEY",
            "TINKER_BASE_URL",
        )
        if (value := os.environ.get(key))
    }
    return modal.Secret.from_dict(env_dict) if env_dict else None


def _is_infrastructure_failure(exc: BaseException) -> bool:
    message = str(exc).lower()
    type_name = type(exc).__name__.lower()
    infra_markers = (
        "timeout",
        "temporarily unavailable",
        "connection",
        "transport",
        "grpc",
        "network",
        "app is not running",
    )
    return any(marker in message for marker in infra_markers) or "modal" in type_name


@dataclass
class LocalTrajectorySandboxRunner:
    backend_name: str = "local"
    call_count: int = 0
    failure_count: int = 0
    _pending_call_count: int = field(default=0, init=False, repr=False)
    _pending_failure_count: int = field(default=0, init=False, repr=False)

    async def start(self) -> None:
        return None

    async def run_trajectory(self, request: TrajectoryRolloutRequest) -> TrajectoryRolloutResult:
        self.call_count += 1
        self._pending_call_count += 1
        result = await _execute_rollout_request(request)
        if result.failure_kind is not None:
            self.failure_count += 1
            self._pending_failure_count += 1
        return result

    async def aclose(self) -> None:
        return None

    def summary(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "call_count": self.call_count,
            "failure_count": self.failure_count,
            "retry_count": 0,
        }

    def snapshot_metrics(self, reset: bool = True) -> dict[str, float]:
        metrics = {
            f"rollout/backend_{self.backend_name}": 1.0,
            "rollout/runner_call_count": float(self._pending_call_count),
            "rollout/runner_failure_count": float(self._pending_failure_count),
            "rollout/runner_retry_count": 0.0,
            "rollout/runner_remote_wall_time_seconds": 0.0,
        }
        if reset:
            self._pending_call_count = 0
            self._pending_failure_count = 0
        return metrics


@dataclass
class ModalTrajectorySandboxRunner:
    app_name: str = "diplomacy-grpo-rollouts"
    timeout_seconds: int = 900
    cpu: float = 2.0
    memory_mb: int = 4096
    single_use_containers: bool = True
    max_inputs: int = 1
    retries: int = 0
    local_fallback_on_infra_failure: bool = True
    backend_name: str = "modal"
    call_count: int = 0
    failure_count: int = 0
    retry_count: int = 0
    remote_wall_time_seconds: float = 0.0
    _app: Any = field(default=None, init=False, repr=False)
    _remote_rollout_fn: Any = field(default=None, init=False, repr=False)
    _app_run_ctx: Any = field(default=None, init=False, repr=False)
    _local_fallback: LocalTrajectorySandboxRunner = field(
        default_factory=LocalTrajectorySandboxRunner,
        init=False,
        repr=False,
    )
    _startup_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _pending_call_count: int = field(default=0, init=False, repr=False)
    _pending_failure_count: int = field(default=0, init=False, repr=False)
    _pending_retry_count: int = field(default=0, init=False, repr=False)
    _pending_remote_wall_time_seconds: float = field(default=0.0, init=False, repr=False)

    async def start(self) -> None:
        if self._remote_rollout_fn is not None:
            return
        async with self._startup_lock:
            if self._remote_rollout_fn is not None:
                return
            if modal is None:  # pragma: no cover - exercised only without modal installed
                raise RuntimeError("modal is required for the modal rollout backend")

            image = _default_modal_image()
            modal_secret = _modal_secret_from_env()
            secrets = [modal_secret] if modal_secret is not None else None
            self._app = modal.App(self.app_name)
            self._remote_rollout_fn = self._app.function(
                image=image,
                timeout=self.timeout_seconds,
                cpu=self.cpu,
                memory=self.memory_mb,
                retries=self.retries,
                single_use_containers=self.single_use_containers,
                max_inputs=self.max_inputs,
                include_source=True,
                env={"PYTHONPATH": "/root/project:/root/project/vendor/tinker-cookbook"},
                secrets=secrets,
            )(_execute_rollout_request)
            self._app_run_ctx = self._app.run.aio()
            try:
                await self._app_run_ctx.__aenter__()
            except Exception:
                self._remote_rollout_fn = None
                self._app_run_ctx = None
                self._app = None
                raise

    async def run_trajectory(self, request: TrajectoryRolloutRequest) -> TrajectoryRolloutResult:
        await self.start()
        assert self._remote_rollout_fn is not None

        self.call_count += 1
        self._pending_call_count += 1
        started_at = time.time()
        try:
            function_call = await self._remote_rollout_fn.spawn.aio(request)
            result = await function_call.get.aio(timeout=self.timeout_seconds)
            elapsed = time.time() - started_at
            self.remote_wall_time_seconds += elapsed
            self._pending_remote_wall_time_seconds += elapsed
            if result.failure_kind is not None:
                self.failure_count += 1
                self._pending_failure_count += 1
            return result
        except Exception as exc:
            self.failure_count += 1
            self._pending_failure_count += 1
            elapsed = time.time() - started_at
            self.remote_wall_time_seconds += elapsed
            self._pending_remote_wall_time_seconds += elapsed
            if self.local_fallback_on_infra_failure and _is_infrastructure_failure(exc):
                self.retry_count += 1
                self._pending_retry_count += 1
                fallback = await self._local_fallback.run_trajectory(request)
                fallback.metrics["rollout/local_fallback_retry"] = 1
                fallback.remote_metadata["fallback_reason"] = str(exc)
                return fallback
            raise

    async def aclose(self) -> None:
        async with self._startup_lock:
            if self._app_run_ctx is not None:
                await self._app_run_ctx.__aexit__(None, None, None)
                self._app_run_ctx = None
            self._remote_rollout_fn = None
            self._app = None
        await self._local_fallback.aclose()

    def summary(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "call_count": self.call_count,
            "failure_count": self.failure_count,
            "retry_count": self.retry_count,
            "remote_wall_time_seconds": self.remote_wall_time_seconds,
            "timeout_seconds": self.timeout_seconds,
            "cpu": self.cpu,
            "memory_mb": self.memory_mb,
            "single_use_containers": self.single_use_containers,
            "max_inputs": self.max_inputs,
            "retries": self.retries,
        }

    def snapshot_metrics(self, reset: bool = True) -> dict[str, float]:
        metrics = {
            f"rollout/backend_{self.backend_name}": 1.0,
            "rollout/runner_call_count": float(self._pending_call_count),
            "rollout/runner_failure_count": float(self._pending_failure_count),
            "rollout/runner_retry_count": float(self._pending_retry_count),
            "rollout/runner_remote_wall_time_seconds": self._pending_remote_wall_time_seconds,
        }
        if reset:
            self._pending_call_count = 0
            self._pending_failure_count = 0
            self._pending_retry_count = 0
            self._pending_remote_wall_time_seconds = 0.0
        return metrics
