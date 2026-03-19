from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_marimo_notebook_contains_cli_and_manifest_controls() -> None:
    notebook_path = REPO_ROOT / "notebooks" / "tinker_grpo_modal.py"
    text = notebook_path.read_text()
    assert text.count("@app.cell") >= 3
    assert "train_tinker_grpo_curriculum.py" in text
    assert "resolve_manifest_path" in text
    assert "launch_once" in text
    assert "mo.ui.run_button" in text
    assert "mo.ui.refresh" in text
    assert "mo.ui.checkbox" in text
    assert "mo.ui.dropdown" not in text
    assert "# Tinker GRPO Operator Notebook" in text
    assert "## Controls" in text
    assert "Use this page to launch and monitor one hybrid GRPO training run for Diplomacy." in text
    assert "GRPO (Group Relative Policy Optimization)" in text
    assert "Diplomacy is a multi-agent negotiation game" in text
    assert "`tool_accuracy`" in text
    assert "`target_execution`" in text
    assert "`supported_target`" in text
    assert "`cooperative_press`" in text
    assert "`full_press`" in text
    assert "shared hybrid curriculum" in text
    assert "Prompt family" in text
    assert "**Control guide**" in text
    assert "**Core controls**" in text
    assert "**Modal controls**" in text
    assert "### Hybrid Phase Schedule" in text
    assert "### Per-Environment Turn Limits" in text
    assert "## Hybrid Progress" in text
    assert "mo.vstack(" in text
    assert 'importlib.import_module("tinker_training.notebook_helpers")' in text
    assert "from tinker_training.notebook_helpers import" not in text
    assert "import marimo as mo" in text
    assert "**How to run it**" not in text
    assert "## Cloud Run" not in text
    assert "marimo run notebooks/tinker_grpo_modal.py" not in text
    assert "modal serve scripts/serve_marimo_modal.py" not in text
    assert "uv tool install modal" not in text
    assert "modal token new" not in text
    assert "TINKER_API_KEY" in text
    assert "OPENROUTER_API_KEY" in text
    assert "WANDB_API_KEY" in text
    assert "Modal auth" not in text
    assert "format_missing_required_env_message" in text
    assert "launch_blocker=" in text
    assert "Auto-refresh process and manifest" in text
    assert "refresh_monitor.value" in text
    assert "default `full_v1`" not in text
    assert "legacy two-stage" not in text


def test_readme_contains_curriculum_guide() -> None:
    readme_path = REPO_ROOT / "README.md"
    text = readme_path.read_text()
    assert "modal serve scripts/serve_marimo_modal.py" in text
    assert "uv run modal serve scripts/serve_marimo_modal.py" not in text
    assert "### Step 1" in text
    assert "### Step 5" in text
    assert "MARIMO_TOKEN_PASSWORD" in text
    assert 'openssl rand -base64 24' in text
    assert 'echo "$MARIMO_TOKEN_PASSWORD"' in text
    assert "uv tool install modal" in text
    assert "modal token new" in text
    assert "Start the Modal-hosted notebook with the recommended command" in text
    assert "#### Speed Expectations" in text
    assert "45-90" in text
    assert "2-3" in text
    assert "@modal.web_server" in text
    assert "Launch training" in text
    assert "single-use containers" in text
    assert "run artifacts persist in the Modal Volume" in text
    assert "MODAL_TOKEN_ID" not in text
    assert "MODAL_TOKEN_SECRET" not in text
    assert "now Modal-only" not in text
    assert "default cloud path is now" not in text
    assert "older remote-host pattern" not in text
    assert "old Verifiers Prime" not in text


def test_readme_references_marimo_notebook() -> None:
    readme_path = REPO_ROOT / "README.md"
    text = readme_path.read_text()
    assert "# Tinker GRPO Curriculum" in text
    assert "notebooks/tinker_grpo_modal.py" in text
    assert "curriculum_manifest.json" in text
    assert "## Outputs" in text
    assert "## CLI Parameters" in text
    assert "## Weights & Biases" in text
    assert "wandb_project" in text
    assert "run_name" in text
    assert "## Next Steps" in text
    assert "mix into curriculum training" in text
    assert "Modal workers return trajectories and metrics; they do not open their own W&B runs." not in text
    assert "stage 1 defaults" not in text
    assert "stage 2 defaults" not in text


def test_curriculum_doc_removed() -> None:
    assert not (REPO_ROOT / "docs" / "tinker-grpo-curriculum.md").exists()


def test_modal_marimo_launcher_exists() -> None:
    script_path = REPO_ROOT / "scripts" / "serve_marimo_modal.py"
    text = script_path.read_text()
    assert "modal.web_server" in text
    assert "marimo" in text
    assert "notebooks/tinker_grpo_modal.py" in text
    assert "Volume.from_name" in text
    assert "MARIMO_TOKEN_PASSWORD" in text
    assert "modal deploy" not in text
    assert "allow_concurrent_inputs" not in text
    assert "@modal.concurrent(max_inputs=32)" in text
    assert "container_idle_timeout" not in text


def test_modal_image_helper_uses_locked_dependency_layer() -> None:
    helper_path = REPO_ROOT / "tinker_training" / "modal_image.py"
    text = helper_path.read_text()
    assert "uv_pip_install(" in text
    assert "https://download.pytorch.org/whl/cpu" in text
    assert "ROLLOUT_RUNTIME_PACKAGES" in text
    assert "TRAINING_RUNTIME_PACKAGES" in text
    assert "add_local_dir(" in text
    assert "vendor/tinker-cookbook/tinker_cookbook" in text
    assert "copy=False" in text
    assert '"datasets"' not in text
    assert '"cloudpickle"' not in text
    assert "pip install -e 'vendor/tinker-cookbook[wandb,verifiers]'" not in text
