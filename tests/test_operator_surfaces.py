from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_marimo_notebook_contains_cli_and_manifest_controls() -> None:
    notebook_path = REPO_ROOT / "notebooks" / "tinker_grpo_modal.py"
    text = notebook_path.read_text()
    assert "train_tinker_grpo_curriculum.py" in text
    assert "resolve_manifest_path" in text
    assert "launch_once" in text
    assert "mo.ui.button" in text
    assert "on_click=lambda value: value + 1" in text
    assert "mo.ui.checkbox" not in text
    assert "## Controls" in text
    assert "mo.vstack(" in text
    assert 'importlib.import_module("tinker_training.notebook_helpers")' in text
    assert "from tinker_training.notebook_helpers import" not in text
    assert "import marimo as mo" in text
    assert "return (mo,)" in text
    assert "TINKER_API_KEY" in text
    assert "OPENROUTER_API_KEY" in text
    assert "WANDB_API_KEY" in text
    assert "MODAL_TOKEN_ID" in text
    assert "format_missing_required_env_message" in text
    assert "launch_blocker=" in text


def test_docs_reference_modal_and_marimo() -> None:
    docs_path = REPO_ROOT / "docs" / "tinker-grpo-curriculum.md"
    text = docs_path.read_text()
    assert "uv run marimo edit notebooks/tinker_grpo_modal.py" in text
    assert "uv run python scripts/train_tinker_grpo_curriculum.py" in text
    assert "source .venv/bin/activate" in text
    assert "Launch training" in text
    assert "--rollout-backend modal" in text
    assert "Modal single-use container" in text or "single-use containers" in text
    assert "still requires Tinker, OpenRouter, and W&B credentials" in text


def test_readme_references_marimo_notebook() -> None:
    readme_path = REPO_ROOT / "README.md"
    text = readme_path.read_text()
    assert "uv run marimo edit notebooks/tinker_grpo_modal.py" in text
    assert "uv run python scripts/train_tinker_grpo_curriculum.py" in text
    assert "source .venv/bin/activate" in text
    assert "curriculum_manifest.json" in text
    assert "Launch training" in text
    assert "Launch training` stays blocked" in text
