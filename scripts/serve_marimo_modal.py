#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tinker_training.modal_image import marimo_ui_image

PROJECT_ROOT = "/root/project"
NOTEBOOK_PATH = f"{PROJECT_ROOT}/notebooks/tinker_grpo_modal.py"
MARIMO_PORT = 2718
MARIMO_STARTUP_TIMEOUT_SECONDS = 60
MARIMO_CONTAINER_TIMEOUT_SECONDS = 24 * 60 * 60
RUNS_VOLUME_NAME = "diplomacy-grpo-runs"
RUNS_MOUNT_PATH = "/root/tinker-runs"


def _marimo_modal_secret_from_local_env() -> modal.Secret | None:
    env_dict = {
        key: value
        for key in (
            "TINKER_API_KEY",
            "OPENROUTER_API_KEY",
            "WANDB_API_KEY",
            "TINKER_BASE_URL",
            "MODAL_TOKEN_ID",
            "MODAL_TOKEN_SECRET",
            "MARIMO_TOKEN_PASSWORD",
        )
        if (value := os.environ.get(key))
    }
    return modal.Secret.from_dict(env_dict) if env_dict else None


image = marimo_ui_image()

app = modal.App("diplomacy-grpo-marimo-ui")
runs_volume = modal.Volume.from_name(RUNS_VOLUME_NAME, create_if_missing=True)
marimo_secret = _marimo_modal_secret_from_local_env()


@app.function(
    image=image,
    secrets=[marimo_secret] if marimo_secret is not None else None,
    volumes={RUNS_MOUNT_PATH: runs_volume},
    cpu=2.0,
    memory=4096,
    timeout=MARIMO_CONTAINER_TIMEOUT_SECONDS,
    startup_timeout=MARIMO_STARTUP_TIMEOUT_SECONDS,
    min_containers=1,
    max_containers=1,
    scaledown_window=3600,
    name="serve-marimo",
)
@modal.concurrent(max_inputs=32)
@modal.web_server(MARIMO_PORT, startup_timeout=MARIMO_STARTUP_TIMEOUT_SECONDS)
def serve_marimo() -> None:
    token_password = os.environ.get("MARIMO_TOKEN_PASSWORD", "").strip()
    if not token_password:
        raise RuntimeError(
            "MARIMO_TOKEN_PASSWORD must be set before running `modal serve`."
        )

    command = [
        sys.executable,
        "-m",
        "marimo",
        "run",
        NOTEBOOK_PATH,
        "--headless",
        "--host",
        "0.0.0.0",
        "--port",
        str(MARIMO_PORT),
        "--token-password",
        token_password,
    ]
    process = subprocess.Popen(command, cwd=PROJECT_ROOT)
    time.sleep(2)
    if process.poll() is not None:
        raise RuntimeError(f"marimo exited early with return code {process.returncode}")
