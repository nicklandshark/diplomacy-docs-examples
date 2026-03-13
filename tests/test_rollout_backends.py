from __future__ import annotations

import asyncio
import types

import pytest
import tinker

from tinker_training import rollout_backends
from tinker_training.diplomacy_adapter import (
    ActorRuntimeConfig,
    DiplomacyEnvGroupBuilder,
    OpenRouterHeaders,
    RuntimePolicyConfig,
    build_actor_configs,
)
from tinker_training.rollout_backends import (
    ModalTrajectorySandboxRunner,
    TrajectoryRolloutRequest,
    TrajectoryRolloutResult,
    register_trajectory_runner,
    serialize_trajectory,
    unregister_trajectory_runner,
)
from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl.types import SamplingRef, Trajectory, Transition


def _dummy_trajectory() -> Trajectory:
    model_input = tinker.ModelInput.from_ints([1, 2, 3])
    return Trajectory(
        transitions=[
            Transition(
                ob=model_input,
                ac=TokensWithLogprobs(tokens=[4, 5], maybe_logprobs=[-0.1, -0.2]),
                reward=1.25,
                episode_done=True,
                metrics={"reward/final": 1.25},
                logs={"note": "dummy"},
            )
        ],
        final_ob=tinker.ModelInput.from_ints([1, 2, 3, 4, 5]),
    )


class _FakeRunner:
    backend_name = "fake"

    def __init__(self) -> None:
        self.requests: list[TrajectoryRolloutRequest] = []

    async def start(self) -> None:
        return None

    async def run_trajectory(self, request: TrajectoryRolloutRequest) -> TrajectoryRolloutResult:
        self.requests.append(request)
        return TrajectoryRolloutResult(
            trajectory=serialize_trajectory(_dummy_trajectory()),
            metrics={"rollout/backend_seconds": 0.01},
        )

    async def aclose(self) -> None:
        return None

    def summary(self) -> dict[str, int]:
        return {"call_count": len(self.requests)}

    def snapshot_metrics(self, reset: bool = True) -> dict[str, float]:
        return {"rollout/runner_call_count": float(len(self.requests))}


def test_group_builder_delegates_to_registered_runner() -> None:
    actor_runtime = ActorRuntimeConfig(
        actor_configs=build_actor_configs(model="google/gemini-3-flash-preview"),
        actor_max_turns=2,
        session_timeout_seconds=10.0,
        default_idle_sleep_seconds=0.1,
        openrouter_headers=OpenRouterHeaders(),
    )
    builder = DiplomacyEnvGroupBuilder(
        datum={"example_id": "ex-1", "info": {"session_id": "unused"}},
        environment_kind="tool_accuracy",
        model_name="Qwen/Qwen3.5-27B",
        renderer_name="qwen3_5_disable_thinking",
        group_size=3,
        actor_runtime=actor_runtime,
        policy_config=RuntimePolicyConfig(max_turns=1),
    )

    fake_runner = _FakeRunner()
    runner_id = register_trajectory_runner(fake_runner)
    builder.rollout_runner_id = runner_id
    try:
        trajectory_group = asyncio.run(
            builder.run_group_rollout(
                SamplingRef(sampler_path="sampler://test", base_url="https://tinker.example"),
                max_tokens=16,
                temperature=1.0,
                do_remove_constant_reward_groups=False,
                enable_logging=False,
            )
        )
    finally:
        unregister_trajectory_runner(runner_id)

    assert trajectory_group is not None
    assert len(trajectory_group.trajectories_G) == 3
    assert all(req.sampling_ref.sampler_path == "sampler://test" for req in fake_runner.requests)
    assert all(metrics["rollout/backend_fake"] == 1.0 for metrics in trajectory_group.metrics_G)


@pytest.mark.asyncio
async def test_modal_runner_startup_is_shared_across_concurrent_first_use(monkeypatch) -> None:
    enter_count = 0

    class _FakeFunctionCall:
        def __init__(self) -> None:
            self.get = types.SimpleNamespace(aio=self._get)

        async def _get(self, timeout: int):
            del timeout
            return TrajectoryRolloutResult(
                trajectory=serialize_trajectory(_dummy_trajectory()),
                metrics={"rollout/backend_seconds": 0.01},
            )

    class _FakeRemoteFunction:
        def __init__(self) -> None:
            self.spawn = types.SimpleNamespace(aio=self._spawn)

        async def _spawn(self, request: TrajectoryRolloutRequest):
            del request
            return _FakeFunctionCall()

    class _FakeAppRunContext:
        async def __aenter__(self):
            nonlocal enter_count
            enter_count += 1
            return self

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

    class _FakeApp:
        def __init__(self, name: str) -> None:
            self.name = name
            self.run = types.SimpleNamespace(aio=lambda: _FakeAppRunContext())

        def function(self, **kwargs):
            del kwargs

            def decorator(fn):
                del fn
                return _FakeRemoteFunction()

            return decorator

    fake_modal = types.SimpleNamespace(App=_FakeApp)
    monkeypatch.setattr(rollout_backends, "modal", fake_modal)
    monkeypatch.setattr(rollout_backends, "_default_modal_image", lambda: object())
    monkeypatch.setattr(rollout_backends, "_modal_secret_from_env", lambda: None)

    runner = ModalTrajectorySandboxRunner()
    request = TrajectoryRolloutRequest(
        datum={"example_id": "ex-1"},
        environment_kind="tool_accuracy",
        model_name="Qwen/Qwen3.5-27B",
        renderer_name="qwen3_5_disable_thinking",
        actor_runtime=None,
        policy_config=None,
        sampling_ref=SamplingRef(sampler_path="sampler://test", base_url="https://tinker.example"),
        max_tokens=16,
        temperature=1.0,
        trajectory_index=0,
        group_id="group-1",
    )

    try:
        await asyncio.gather(*(runner.run_trajectory(request) for _ in range(4)))
        assert enter_count == 1
    finally:
        await runner.aclose()
