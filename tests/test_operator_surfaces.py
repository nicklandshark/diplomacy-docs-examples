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
    assert "mo.ui.checkbox" not in text


def test_docs_reference_modal_and_marimo() -> None:
    docs_path = REPO_ROOT / "docs" / "tinker-grpo-curriculum.md"
    text = docs_path.read_text()
    assert "uv run marimo edit notebooks/tinker_grpo_modal.py" in text
    assert "source .venv/bin/activate" in text
    assert "Launch training" in text
    assert "--rollout-backend modal" in text
    assert "Modal single-use container" in text or "single-use containers" in text


def test_readme_references_marimo_notebook() -> None:
    readme_path = REPO_ROOT / "README.md"
    text = readme_path.read_text()
    assert "uv run marimo edit notebooks/tinker_grpo_modal.py" in text
    assert "source .venv/bin/activate" in text
    assert "curriculum_manifest.json" in text
    assert "Launch training" in text
