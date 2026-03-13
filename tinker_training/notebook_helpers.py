from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def build_train_command(
    *,
    script_path: Path,
    model_name: str,
    rollout_backend: str,
    log_root: str,
    wandb_project: str,
    openrouter_model: str,
    modal_app_name: str,
    modal_timeout_seconds: int,
    modal_cpu: float,
    modal_memory_mb: int,
    stage1_train_examples: int,
    stage1_group_size: int,
    stage1_train_seed: int,
    stage2_train_examples: int,
    stage2_group_size: int,
    stage2_train_seed: int,
    renderer_name: str | None = None,
    run_name: str | None = None,
    initial_checkpoint_path: str | None = None,
    python_executable: str | None = None,
) -> list[str]:
    command = [
        python_executable or sys.executable,
        str(script_path),
        "--model-name",
        model_name,
        "--rollout-backend",
        rollout_backend,
        "--log-root",
        log_root,
        "--wandb-project",
        wandb_project,
        "--openrouter-model",
        openrouter_model,
        "--modal-app-name",
        modal_app_name,
        "--modal-timeout-seconds",
        str(int(modal_timeout_seconds)),
        "--modal-cpu",
        str(float(modal_cpu)),
        "--modal-memory-mb",
        str(int(modal_memory_mb)),
        "--stage1-train-examples",
        str(int(stage1_train_examples)),
        "--stage1-group-size",
        str(int(stage1_group_size)),
        "--stage1-train-seed",
        str(int(stage1_train_seed)),
        "--stage2-train-examples",
        str(int(stage2_train_examples)),
        "--stage2-group-size",
        str(int(stage2_group_size)),
        "--stage2-train-seed",
        str(int(stage2_train_seed)),
    ]
    if renderer_name and renderer_name.strip():
        command.extend(["--renderer-name", renderer_name.strip()])
    if run_name and run_name.strip():
        command.extend(["--run-name", run_name.strip()])
    if initial_checkpoint_path and initial_checkpoint_path.strip():
        command.extend(["--initial-checkpoint-path", initial_checkpoint_path.strip()])
    return command


def resolve_manifest_path(
    *,
    log_root: str,
    run_name: str | None,
    manifest_override: str | None,
) -> Path:
    if manifest_override and manifest_override.strip():
        return Path(manifest_override).expanduser()
    resolved_run_name = run_name.strip() if run_name and run_name.strip() else "<generated at runtime>"
    return Path(log_root).expanduser() / resolved_run_name / "curriculum_manifest.json"


def get_process_state(state: dict[str, Any]) -> dict[str, Any]:
    process = state.get("process")
    pid = process.pid if process is not None else state.get("pid")
    return {
        "pid": pid,
        "command": list(state.get("command", [])),
        "started_at": state.get("started_at"),
        "returncode": process.poll() if process is not None else state.get("returncode"),
        "last_launch_token": state.get("last_launch_token"),
    }


def launch_once(
    state: dict[str, Any],
    *,
    launch_token: Any,
    command: Sequence[str],
    cwd: Path,
    launch_blocker: str | None = None,
    popen_factory: Any = subprocess.Popen,
) -> tuple[dict[str, Any], str]:
    snapshot = get_process_state(state)
    if not launch_token:
        return snapshot, "Launch idle."

    if snapshot["last_launch_token"] == launch_token:
        if snapshot["pid"] is not None and snapshot["returncode"] is None:
            return snapshot, f"Training already running with PID {snapshot['pid']}."
        return snapshot, "Launch already handled for the current button event."

    if snapshot["pid"] is not None and snapshot["returncode"] is None:
        state["last_launch_token"] = launch_token
        return get_process_state(state), f"Training already running with PID {snapshot['pid']}."

    if launch_blocker:
        state["last_launch_token"] = launch_token
        return get_process_state(state), f"Launch blocked: {launch_blocker}"

    process = popen_factory(list(command), cwd=str(cwd))
    state["process"] = process
    state["pid"] = process.pid
    state["command"] = list(command)
    state["started_at"] = time.time()
    state["returncode"] = None
    state["last_launch_token"] = launch_token
    return get_process_state(state), f"Spawned PID {process.pid}."


def preflight_env() -> dict[str, Any]:
    required = {
        "TINKER_API_KEY": bool(os.environ.get("TINKER_API_KEY")),
        "OPENROUTER_API_KEY": bool(os.environ.get("OPENROUTER_API_KEY")),
        "WANDB_API_KEY": bool(os.environ.get("WANDB_API_KEY")),
    }
    optional = {
        "TINKER_BASE_URL": bool(os.environ.get("TINKER_BASE_URL")),
    }
    modal_env_auth = {
        "MODAL_TOKEN_ID": bool(os.environ.get("MODAL_TOKEN_ID")),
        "MODAL_TOKEN_SECRET": bool(os.environ.get("MODAL_TOKEN_SECRET")),
    }
    return {
        "required": required,
        "optional": optional,
        "modal_env_auth": modal_env_auth,
        "required_ok": all(required.values()),
        "modal_env_auth_ok": all(modal_env_auth.values()),
    }


def format_missing_required_env_message(preflight: dict[str, Any] | None = None) -> str | None:
    snapshot = preflight if preflight is not None else preflight_env()
    missing = [name for name, is_present in snapshot["required"].items() if not is_present]
    if not missing:
        return None
    return "missing required env vars: " + ", ".join(missing)
