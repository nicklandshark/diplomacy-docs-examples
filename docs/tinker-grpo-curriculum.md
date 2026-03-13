# Tinker GRPO Curriculum

This repo now includes a curriculum trainer that can run Diplomacy GRPO locally or with Modal-isolated rollouts.

## Architecture

- `scripts/train_tinker_grpo_curriculum.py` is the CLI entrypoint.
- `tinker_training/curriculum.py` holds the reusable config objects and `run_curriculum()`.
- `tinker_training/diplomacy_adapter.py` keeps the Diplomacy runtime, direct tools, and Verifiers rubric scoring.
- `tinker_training/rollout_backends.py` provides:
  - `LocalTrajectorySandboxRunner`
  - `ModalTrajectorySandboxRunner`
- `notebooks/tinker_grpo_modal.py` is the marimo operator surface.

## What it does

- trains a Tinker policy with grouped GRPO rollouts
- runs a two-stage curriculum:
  - `tool_accuracy`
  - `full_press`
- keeps the existing Diplomacy environment and Verifiers reward logic
- keeps the trainable policy on-policy by sampling the current Tinker checkpoint every batch
- keeps counterpart powers on OpenRouter, defaulting to `google/gemini-3-flash-preview`
- can isolate each trajectory in its own Modal single-use container
- logs checkpoints, rollout summaries, HTML traces, W&B metrics, and rollout backend stats

## Environment Variables

```bash
export TINKER_API_KEY=...
export OPENROUTER_API_KEY=...
export WANDB_API_KEY=...
```

For Modal rollouts, authenticate locally with:

```bash
modal token new
```

## Install

```bash
uv sync
```

The cookbook is vendored under `vendor/tinker-cookbook/`, and `pyproject.toml` points `tinker_cookbook` at that local copy.

## Default Run

This uses Modal for rollout isolation.

```bash
python scripts/train_tinker_grpo_curriculum.py \
  --model-name Qwen/Qwen3.5-27B \
  --wandb-project diplomacy-grpo \
  --openrouter-model google/gemini-3-flash-preview \
  --rollout-backend modal
```

## Local Smoke Run

This keeps rollouts in-process and is useful for debugging the trainer path without Modal.

```bash
python scripts/train_tinker_grpo_curriculum.py \
  --model-name Qwen/Qwen3.5-27B \
  --rollout-backend local \
  --stage1-train-examples 8 \
  --stage1-eval-examples 2 \
  --stage2-train-examples 4 \
  --stage2-eval-examples 2
```

## Modal Options

Important flags:

- `--modal-app-name`
- `--modal-timeout-seconds`
- `--modal-cpu`
- `--modal-memory-mb`
- `--disable-local-fallback-on-infra-failure`

The production Modal backend uses:

- `single_use_containers=True`
- `max_inputs=1`
- `retries=0`

Each trajectory reconstructs the current Tinker sampler from the latest saved `sampler_path`, runs the Diplomacy env and OpenRouter actors inside the container, and returns a standard trajectory payload back to the local trainer.

## Marimo Notebook

Open the operator notebook with:

```bash
uv run marimo edit notebooks/tinker_grpo_modal.py
```

The notebook is a control surface over the CLI trainer, not a separate training implementation.

If you want bare `marimo` on your shell path instead, activate the repo virtualenv first:

```bash
source .venv/bin/activate
marimo edit notebooks/tinker_grpo_modal.py
```

Before opening it, make sure you have:

- run `uv sync`
- exported `TINKER_API_KEY`, `OPENROUTER_API_KEY`, and `WANDB_API_KEY`
- run `modal token new` locally if you plan to use Modal rollouts

The notebook provides:

- reactive config widgets
- environment and key preflight checks
- the exact CLI command that will be run
- one-shot subprocess launch of the CLI trainer
- process status for the current notebook session
- manifest inspection for `curriculum_manifest.json`

Recommended workflow:

1. Open the notebook with `uv run marimo edit notebooks/tinker_grpo_modal.py`.
2. Fill in the model, backend, stage sizes, seeds, and Modal resource knobs.
3. Confirm the preflight block shows the required credentials.
4. Inspect the rendered CLI command before launching.
5. Click `Launch training` once to start the trainer subprocess.
6. Watch the process panel and manifest panel as the run advances.

Important behavior:

- the notebook launch uses a button, not a reactive checkbox
- the launch handler is idempotent for a single button event, so notebook rerenders do not spawn duplicate jobs
- the notebook only tracks one active subprocess per notebook session
- if you want to resume a run, use the `Initial checkpoint` field and launch a new subprocess deliberately

What to monitor in the notebook:

- the exact command being run
- whether the local API keys are present
- the current `curriculum_manifest.json`
- stage log directories and checkpoint paths written into the manifest

The manifest viewer is most useful after you provide a concrete run name. If `Run name` is left blank, the trainer generates it at runtime, so the notebook cannot know the final manifest path until the process starts writing it.

## Outputs

Each run writes to:

`~/tinker-runs/diplomacy-grpo/<run-name>/`

Important files:

- `curriculum_manifest.json`: stage order, checkpoints, rollout backend stats
- `stage1_tool_accuracy/`: first-stage logs and checkpoints
- `stage2_full_press/`: second-stage logs and checkpoints

## Notes

- The trained policy is always the current Tinker sampler for the tracked power.
- Counterpart powers are separate LLM actors on OpenRouter.
- W&B logging stays in the local trainer process; remote workers return metrics but do not open their own W&B runs.
- The old Verifiers Prime sandbox path is not used by this trainer.
