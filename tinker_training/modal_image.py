from __future__ import annotations

from functools import cache
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = "/root/project"
VENDORED_COOKBOOK_ROOT = REPO_ROOT / "vendor" / "tinker-cookbook"
VENDORED_PACKAGE_ROOT = VENDORED_COOKBOOK_ROOT / "tinker_cookbook"
PYTORCH_CPU_INDEX_URL = "https://download.pytorch.org/whl/cpu"

IMAGE_IGNORE = [
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".tmp",
]

RUNTIME_ENV = {
    "HOME": "/root",
    "PYTHONPATH": "/root/project:/root/project/vendor/tinker-cookbook",
}

RUNTIME_SOURCE_DIRS = (
    "environments",
    "tinker_training",
)

RUNTIME_TOP_LEVEL_FILES = (
    "data_generator.py",
    "rlm_backend.py",
    "rlm_chatroom_backend.py",
)

TRAINING_SCRIPT_FILES = ("train_tinker_grpo_curriculum.py",)

SHARED_RUNTIME_PACKAGES = (
    "chz",
    "diplomacy==1.1.2",
    "numpy",
    "openai>=1.0.0",
    "pillow",
    "tiktoken>=0.12.0",
    "tinker>=0.9.0",
    "transformers>=4.57.6,<5.0.0",
    "verifiers>=0.1.9,<0.1.10",
)

# The remote rollout workers execute already-defined policies against the
# Diplomacy environments and scoring rubrics. They do not run training loops,
# W&B logging, or Modal app creation inside the container.
ROLLOUT_RUNTIME_PACKAGES = SHARED_RUNTIME_PACKAGES

# The Modal-hosted marimo container also launches the local training process, so
# it needs the training/logging stack in addition to the shared environment code.
TRAINING_RUNTIME_PACKAGES = SHARED_RUNTIME_PACKAGES + (
    # Keep Modal's notebook runtime aligned with the repo lockfile. The
    # published UI currently renders correctly on 0.20.4 but not on 0.21.x.
    "marimo==0.20.4",
    "modal>=1.3.1",
    "rich",
    "scipy",
    "termcolor",
    "tqdm",
    "wandb",
)


@cache
def _torch_cpu_image() -> modal.Image:
    image = modal.Image.debian_slim(python_version="3.12")
    # PyTorch's official CPU wheel index avoids pulling the full CUDA stack
    # for the Modal-hosted notebook and rollout workers.
    return image.uv_pip_install("torch", index_url=PYTORCH_CPU_INDEX_URL)


@cache
def _dependency_base_image(packages: tuple[str, ...]) -> modal.Image:
    image = _torch_cpu_image()
    image = image.uv_pip_install(*packages)
    return image.env(RUNTIME_ENV)


def _with_runtime_sources(
    image: modal.Image,
    *,
    include_notebooks: bool,
    include_training_scripts: bool,
) -> modal.Image:
    image = image.add_local_dir(
        str(VENDORED_PACKAGE_ROOT),
        f"{PROJECT_ROOT}/vendor/tinker-cookbook/tinker_cookbook",
        copy=False,
        ignore=IMAGE_IGNORE,
    )
    for directory in RUNTIME_SOURCE_DIRS:
        image = image.add_local_dir(
            str(REPO_ROOT / directory),
            f"{PROJECT_ROOT}/{directory}",
            copy=False,
            ignore=IMAGE_IGNORE,
        )
    for filename in RUNTIME_TOP_LEVEL_FILES:
        image = image.add_local_file(
            REPO_ROOT / filename,
            f"{PROJECT_ROOT}/{filename}",
            copy=False,
        )
    if include_training_scripts:
        for filename in TRAINING_SCRIPT_FILES:
            image = image.add_local_file(
                REPO_ROOT / "scripts" / filename,
                f"{PROJECT_ROOT}/scripts/{filename}",
                copy=False,
            )
    if include_notebooks:
        image = image.add_local_dir(
            str(REPO_ROOT / "notebooks"),
            f"{PROJECT_ROOT}/notebooks",
            copy=False,
            ignore=IMAGE_IGNORE,
        )
    return image


@cache
def rollout_runtime_image() -> modal.Image:
    return _with_runtime_sources(
        _dependency_base_image(ROLLOUT_RUNTIME_PACKAGES),
        include_notebooks=False,
        include_training_scripts=False,
    )


@cache
def marimo_ui_image() -> modal.Image:
    return _with_runtime_sources(
        _dependency_base_image(TRAINING_RUNTIME_PACKAGES),
        include_notebooks=True,
        include_training_scripts=True,
    )
