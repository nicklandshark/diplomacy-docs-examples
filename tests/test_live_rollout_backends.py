from __future__ import annotations

import asyncio
import os

import pytest

from data_generator import build_tool_accuracy_dataset
from tinker_training.diplomacy_adapter import (
    ActorRuntimeConfig,
    OpenRouterHeaders,
    RuntimePolicyConfig,
    build_actor_configs,
)
from tinker_training.rollout_backends import (
    LocalTrajectorySandboxRunner,
    ModalTrajectorySandboxRunner,
    TrajectoryRolloutRequest,
)
from tinker_cookbook.rl.types import SamplingRef


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_DIPLOMACY_LIVE_ROLLOUT_TEST") != "1"
    or not os.environ.get("TEST_TINKER_SAMPLER_PATH")
    or not os.environ.get("TINKER_API_KEY")
    or not os.environ.get("OPENROUTER_API_KEY"),
    reason="Requires live Tinker/OpenRouter credentials plus TEST_TINKER_SAMPLER_PATH",
)


def _build_request():
    datum = build_tool_accuracy_dataset(num_sessions=1, seed=123).to_list()[0]
    actor_runtime = ActorRuntimeConfig(
        actor_configs=build_actor_configs(model="google/gemini-3-flash-preview"),
        actor_max_turns=2,
        session_timeout_seconds=90.0,
        default_idle_sleep_seconds=0.5,
        openrouter_headers=OpenRouterHeaders(),
    )
    return TrajectoryRolloutRequest(
        datum=datum,
        environment_kind="tool_accuracy",
        model_name="Qwen/Qwen3.5-27B",
        renderer_name="qwen3_5_disable_thinking",
        actor_runtime=actor_runtime,
        policy_config=RuntimePolicyConfig(max_turns=2),
        sampling_ref=SamplingRef(
            sampler_path=os.environ["TEST_TINKER_SAMPLER_PATH"],
            base_url=os.environ.get("TINKER_BASE_URL"),
        ),
        max_tokens=64,
        temperature=1.0,
        trajectory_index=0,
        group_id="live-test",
    )


def test_local_runner_live_smoke() -> None:
    request = _build_request()
    runner = LocalTrajectorySandboxRunner()
    result = asyncio.run(runner.run_trajectory(request))
    assert result.failure_kind is None
    assert result.trajectory is not None


def test_modal_runner_live_smoke() -> None:
    pytest.importorskip("modal")
    request = _build_request()
    runner = ModalTrajectorySandboxRunner(timeout_seconds=300)
    try:
        result = asyncio.run(runner.run_trajectory(request))
        assert result.failure_kind is None
        assert result.trajectory is not None
    finally:
        asyncio.run(runner.aclose())
