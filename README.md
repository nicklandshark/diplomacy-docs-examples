# Tinker GRPO Curriculum

This repo includes a Diplomacy curriculum trainer with a marimo operator UI. The rollout path is Modal-only.

## Mental Model

Think about the stack in five layers:

1. Browser
   You interact with the marimo notebook UI from a browser.
2. Marimo notebook
   `notebooks/tinker_grpo_modal.py` is only a control surface. It renders the command, checks credentials, launches the trainer once, and watches `curriculum_manifest.json`.
3. Trainer process
   `scripts/train_tinker_grpo_curriculum.py` and `tinker_training/curriculum.py` own the real training loop, the Tinker training client, checkpointing, manifests, and W&B logging.
4. Modal rollout workers
   Every rollout trajectory is executed in Modal. The operator path uses a Modal rollout backend with no fallback to local execution. By default the runner uses single-use containers with one input per worker so each trajectory is isolated.
5. Environment + model services
   Each Modal worker reconstructs one Diplomacy environment episode through `tinker_training/diplomacy_adapter.py`, queries the Tinker sampler, and uses OpenRouter for the background counterpart powers.

The notebook does not talk to the Diplomacy environment directly. It does not hold the training client. It does not run rollouts itself.

## Architecture

- `scripts/train_tinker_grpo_curriculum.py`: CLI entrypoint.
- `tinker_training/curriculum.py`: hybrid curriculum config, checkpoint manifests, one shared Tinker training run.
- `tinker_training/diplomacy_adapter.py`: Diplomacy environment adapter and grouped rollout builder.
- `tinker_training/rollout_backends.py`: Modal rollout execution layer.
- `notebooks/tinker_grpo_modal.py`: marimo UI.

The hybrid environment mix includes:

- `tool_accuracy`
- `target_execution`
- `supported_target`
- `cooperative_press`
- `full_press`

## Credentials

Required:

```bash
export TINKER_API_KEY=...
export OPENROUTER_API_KEY=...
export WANDB_API_KEY=...
```

Purpose of each key:

- `TINKER_API_KEY`: authenticates the training client and sampler checkpoint access.
- `OPENROUTER_API_KEY`: authenticates the background Diplomacy counterpart actors.
- `WANDB_API_KEY`: authenticates Weights & Biases logging.

Optional:

- `TINKER_BASE_URL`: override the default Tinker API endpoint.

## Install

```bash
uv sync
```

Use `uv sync` if you want to run the CLI or notebook locally.

The cloud Modal launcher below does not need the full project environment on your machine. It only needs the `modal` CLI plus this repo checkout.

The cookbook is vendored under `vendor/tinker-cookbook/`, and `pyproject.toml` points `tinker_cookbook` at that local copy.

## Marimo

Here's how to run the notebook in the cloud with Modal.

### Step 1

Generate a marimo login password and export it with the required credentials:

```bash
export MARIMO_TOKEN_PASSWORD="$(openssl rand -base64 24)"
export TINKER_API_KEY=...
export OPENROUTER_API_KEY=...
export WANDB_API_KEY=...
```

### Step 2

Install the Modal CLI and sign in:

```bash
uv tool install modal
modal token new
```

### Step 3

Start the Modal-hosted notebook with the recommended command:

```bash
modal serve scripts/serve_marimo_modal.py
```

#### Speed Expectations

- `Cold start`: expect about `45-90` seconds with the current image layout.
- `Warm restart`: usually about `2-3` seconds.
- `Full rebuild`: if Modal has to rebuild everything from scratch, it can still take a few minutes.

### Step 4

Open the URL printed by Modal in your browser.

### Step 5

Print the password you generated and enter it in the browser:

```bash
echo "$MARIMO_TOKEN_PASSWORD"
```

### Step 6

Click `Launch training`.

What this does:

- the marimo notebook runs in a Modal web container
- the trainer subprocess also runs in that same Modal web container
- run artifacts persist in the Modal Volume mounted at `/root/tinker-runs`
- rollout trajectories execute in separate Modal workers
- Tinker stays remote as the training API
- your laptop is only the browser UI

The launcher uses `@modal.web_server`, mounts persistent run storage, and pins the UI to one container so the notebook's local subprocess and manifest polling work cleanly. If you redeploy or the web container restarts, the active trainer subprocess stops, but checkpoints and manifests remain in the mounted volume.

## Notebook Controls

The notebook exposes the parameters most people should touch.

Core controls:

- `Model to train`: base checkpoint for the trainable policy.
- `Use thinking-enabled renderer`: opt into the model's thinking-capable default renderer when supported.
- `Renderer override`: only use this if you need to force a specific renderer.
- `Run name`: output folder and W&B run name.
- `Log root`: parent directory for manifests, checkpoints, and logs.
- `Initial checkpoint`: warm-start checkpoint if you are not resuming from the existing run folder.
- `Save every` / `Eval every`: cadence in global curriculum batches.
- `Rich-log groups per batch`: how many groups get HTML/logtree output.
- `Prompt family`: environment-specific tracked-policy prompts under `prompts/gepa-full-press/`.
- `Hybrid total batches`: how long to run the mixed easy-to-hard schedule.
- `Learning rate`: one global learning rate for the whole hybrid run.
- `LoRA rank`: shared adapter rank for the full hybrid curriculum.

Hybrid controls:

- the environment schedule is fixed in code and gradually shifts from easy-heavy to full-press-heavy
- the notebook shows the phase schedule and per-environment turn limits as read-only summaries
- evaluation is split by environment so W&B can show which task families are improving

Modal controls:

- `OpenRouter actor model`: model used by the non-trained counterpart powers.
- `Modal app name`: app name for rollout workers.
- `Modal timeout`, `Modal CPU`, `Modal memory`: per-worker resource sizing.

## Outputs

Each run writes to:

`~/tinker-runs/diplomacy-grpo/<run-name>/`

Important files:

- `curriculum_manifest.json`: run metadata, hybrid phase status, checkpoint paths, backend stats, and overall status.
- `hybrid_v1/`: hybrid config snapshot, metrics, traces, and checkpoints.

## CLI Parameters

See the live CLI help for the full list:

```bash
uv run python scripts/train_tinker_grpo_curriculum.py --help
```

Most important flags:

- `--model-name`: default is `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`.
- `--renderer-name`: optional explicit renderer override.
- `--enable-thinking`: use the thinking-capable renderer when available.
- `--run-name`: stable output folder name. If omitted, one is generated from the model name and timestamp.
- `--initial-checkpoint-path`: warm-start from an existing checkpoint when no local curriculum checkpoints exist yet.
- `--modal-app-name`, `--modal-timeout-seconds`, `--modal-cpu`, `--modal-memory-mb`: rollout-worker sizing.
- `--openrouter-model`: default background actor model is `google/gemini-3-flash-preview`.
- `--save-every`, `--eval-every`, `--num-groups-to-log`: checkpointing and observability cadence.

## Notes

- The trained policy is the Tinker sampler for the tracked power.
- Counterpart powers are separate OpenRouter actors.
- W&B logging stays in the trainer process.

## Weights & Biases

Each training run logs to the W&B project you set in the notebook or CLI.

To view the run:

- open the W&B project named by your `wandb_project` setting
- find the run with the same name as your curriculum `run_name`
- use `curriculum_manifest.json` and the stage log directories if you need to match local artifacts back to the W&B run

What to look for first:

- training progress across batches and whether the run survives all hybrid phases
- reward and rubric metrics by environment, especially whether easier environments improve first and `full_press` gets non-zero signal later
- evaluation cadence and checkpoint cadence, to confirm saves and evals are happening when expected
- spikes in rollout failures, invalid tool use, or other regressions that suggest environment or reward issues rather than model improvement

## Next Steps

To make the model better:

- improve reward design so the policy gets credit for the behaviors you actually want
- tune the hybrid schedule, batch size, group size, and learning rate once the basic run is stable
- review `curriculum_manifest.json`, hybrid checkpoints, and traces to find where the policy is failing

To improve environment design:

- make task rows more varied so the policy does not overfit narrow prompts or board situations
- add clearer process rewards where sparse outcome rewards are too weak to train against directly
- tighten failure gates when invalid tool use or illegal submissions are getting partial credit

To add more environments to mix into curriculum training:

- add a new environment that teaches one distinct skill, such as persuasion, longer-horizon coordination, or recovery from bad board states
- keep the environment adapter thin so new environments plug into the same Tinker training loop
- add the environment to the hybrid mixture once its reward and termination logic are stable
