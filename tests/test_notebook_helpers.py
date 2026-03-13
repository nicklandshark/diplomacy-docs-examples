from __future__ import annotations

import sys
from pathlib import Path

from tinker_training.notebook_helpers import (
    build_train_command,
    format_missing_required_env_message,
    get_process_state,
    launch_once,
    preflight_env,
    resolve_manifest_path,
)


class _FakeProcess:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.returncode = None

    def poll(self):
        return self.returncode


class _FakePopenFactory:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str]] = []
        self.next_pid = 1000
        self.processes: list[_FakeProcess] = []

    def __call__(self, command: list[str], cwd: str):
        self.calls.append((list(command), cwd))
        process = _FakeProcess(self.next_pid)
        self.next_pid += 1
        self.processes.append(process)
        return process


def test_build_train_command_and_manifest_path() -> None:
    command = build_train_command(
        script_path=Path("scripts/train_tinker_grpo_curriculum.py"),
        model_name="Qwen/Qwen3.5-27B",
        rollout_backend="modal",
        log_root="~/runs",
        wandb_project="demo",
        openrouter_model="google/gemini-3-flash-preview",
        modal_app_name="rollout-app",
        modal_timeout_seconds=123,
        modal_cpu=2.0,
        modal_memory_mb=4096,
        stage1_train_examples=10,
        stage1_group_size=2,
        stage1_train_seed=1,
        stage2_train_examples=20,
        stage2_group_size=4,
        stage2_train_seed=2,
        renderer_name="qwen3_5_disable_thinking",
        run_name="demo-run",
        initial_checkpoint_path="checkpoint://start",
    )

    assert command[:4] == [
        sys.executable,
        "scripts/train_tinker_grpo_curriculum.py",
        "--model-name",
        "Qwen/Qwen3.5-27B",
    ]
    assert "--renderer-name" in command
    assert "--run-name" in command
    assert "--initial-checkpoint-path" in command
    assert resolve_manifest_path(log_root="~/runs", run_name="demo-run", manifest_override=None).as_posix().endswith(
        "demo-run/curriculum_manifest.json"
    )


def test_launch_once_is_idempotent_per_button_event(tmp_path) -> None:
    popen_factory = _FakePopenFactory()
    state: dict[str, object] = {}
    command = ["python", "scripts/train_tinker_grpo_curriculum.py"]
    cwd = tmp_path / "project"
    cwd.mkdir()

    snapshot, message = launch_once(
        state,
        launch_token=1,
        command=command,
        cwd=cwd,
        popen_factory=popen_factory,
    )
    assert snapshot["pid"] == 1000
    assert "Spawned PID 1000" in message
    assert len(popen_factory.calls) == 1

    snapshot, message = launch_once(
        state,
        launch_token=1,
        command=command,
        cwd=cwd,
        popen_factory=popen_factory,
    )
    assert snapshot["pid"] == 1000
    assert "already running" in message
    assert len(popen_factory.calls) == 1

    popen_factory.processes[0].returncode = 0
    snapshot, message = launch_once(
        state,
        launch_token=1,
        command=command,
        cwd=cwd,
        popen_factory=popen_factory,
    )
    assert message == "Launch already handled for the current button event."
    assert len(popen_factory.calls) == 1

    snapshot, message = launch_once(
        state,
        launch_token=2,
        command=command,
        cwd=cwd,
        popen_factory=popen_factory,
    )
    assert snapshot["pid"] == 1001
    assert len(popen_factory.calls) == 2
    assert get_process_state(state)["last_launch_token"] == 2


def test_preflight_env_reports_required_and_optional_vars(monkeypatch) -> None:
    for name in (
        "TINKER_API_KEY",
        "OPENROUTER_API_KEY",
        "WANDB_API_KEY",
        "TINKER_BASE_URL",
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setenv("TINKER_API_KEY", "tinker-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setenv("WANDB_API_KEY", "wandb-key")
    monkeypatch.setenv("TINKER_BASE_URL", "https://tinker.example")
    monkeypatch.setenv("MODAL_TOKEN_ID", "modal-id")

    preflight = preflight_env()

    assert preflight["required"] == {
        "TINKER_API_KEY": True,
        "OPENROUTER_API_KEY": True,
        "WANDB_API_KEY": True,
    }
    assert preflight["optional"] == {
        "TINKER_BASE_URL": True,
    }
    assert preflight["modal_env_auth"] == {
        "MODAL_TOKEN_ID": True,
        "MODAL_TOKEN_SECRET": False,
    }
    assert preflight["required_ok"] is True
    assert preflight["modal_env_auth_ok"] is False


def test_format_missing_required_env_message(monkeypatch) -> None:
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")

    assert (
        format_missing_required_env_message()
        == "missing required env vars: TINKER_API_KEY, WANDB_API_KEY"
    )


def test_launch_once_blocks_without_required_env(tmp_path) -> None:
    popen_factory = _FakePopenFactory()
    state: dict[str, object] = {}
    command = [sys.executable, "scripts/train_tinker_grpo_curriculum.py"]
    cwd = tmp_path / "project"
    cwd.mkdir()

    snapshot, message = launch_once(
        state,
        launch_token=1,
        command=command,
        cwd=cwd,
        launch_blocker="missing required env vars: TINKER_API_KEY",
        popen_factory=popen_factory,
    )

    assert snapshot["pid"] is None
    assert message == "Launch blocked: missing required env vars: TINKER_API_KEY"
    assert len(popen_factory.calls) == 0
    assert get_process_state(state)["last_launch_token"] == 1
