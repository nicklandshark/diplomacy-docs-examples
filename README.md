# Diplomacy Docs Examples

This repo contains the Diplomacy environment docs plus the Tinker GRPO curriculum trainer and marimo operator notebook.

## Main Docs

- [docs/guide.md](docs/guide.md): broader builder guide for the Diplomacy training setup
- [docs/tinker-grpo-curriculum.md](docs/tinker-grpo-curriculum.md): concrete trainer architecture, CLI usage, Modal rollout isolation, and notebook workflow

## Marimo Notebook

The operator notebook lives at `notebooks/tinker_grpo_modal.py`.

Install dependencies:

```bash
uv sync
```

Set the required credentials:

```bash
export TINKER_API_KEY=...
export OPENROUTER_API_KEY=...
export WANDB_API_KEY=...
```

If you are using Modal rollouts, authenticate locally first:

```bash
modal token new
```

Open the notebook:

```bash
uv run marimo edit notebooks/tinker_grpo_modal.py
```

If you prefer to activate the virtualenv first:

```bash
source .venv/bin/activate
marimo edit notebooks/tinker_grpo_modal.py
```

The notebook launches the trainer with the same Python interpreter that launched marimo. If required credentials are missing, `Launch training` stays blocked and lists the missing env vars instead of spawning a failing subprocess.

What the notebook does:

- lets you configure the training run with widgets
- shows the exact CLI command that will be launched
- runs preflight checks for local credentials
- launches the trainer with the `Launch training` button instead of a reactive checkbox
- shows process status for the current notebook session
- reads `curriculum_manifest.json` for run monitoring

The launch button is idempotent per click event, so rerendering the notebook does not spawn duplicate training jobs.

## CLI Entry Point

The notebook is only a control surface. The actual trainer is still the CLI:

```bash
uv run python scripts/train_tinker_grpo_curriculum.py \
  --model-name Qwen/Qwen3.5-27B \
  --wandb-project diplomacy-grpo \
  --openrouter-model google/gemini-3-flash-preview \
  --rollout-backend modal
```

Use `--rollout-backend local` for an in-process smoke run. This only skips Modal isolation; it still requires Tinker, OpenRouter, and W&B credentials.
