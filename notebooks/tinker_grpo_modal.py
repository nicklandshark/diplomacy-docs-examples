import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Tinker GRPO Operator Notebook

    This notebook is a control surface for running the Diplomacy curriculum trainer.

    **RL in this repo**

    - A **trajectory** is one full episode: the model sees an observation, produces text and tool calls, the environment updates, and the episode ends with a reward.
    - A **rollout** is the process of generating that trajectory from the current policy.
    - The trainer runs a curriculum:
      - `tool_accuracy` teaches the model to use the Diplomacy tool interface correctly.
      - `full_press` trains the broader negotiation-and-orders task after the tool behavior is stable.

    **GRPO in this repo**

    - **GRPO** trains on a **group** of rollouts for the same task instead of a single sample.
    - Rewards are compared across that group, which gives the model a stronger learning signal than treating each trajectory in isolation.
    - In this setup, the current Tinker checkpoint is sampled on-policy every batch, so each batch uses the latest saved sampler.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Tinker concepts used here**

    - **Training client**: updates the model weights.
    - **Sampling client / sampler path**: a saved, queryable view of the current policy used for rollout generation.
    - **Renderer**: formats observations, assistant output, and tool calls in the model-specific way expected by Qwen.
    - **Env group builder**: constructs the grouped tasks used by GRPO.
    - **Modal rollout backend**: runs one trajectory per isolated container while the local process keeps orchestration, checkpointing, and W&B logging.
    - **OpenRouter actor model**: powers the non-trained counterpart powers. The tracked power is the policy being trained.

    **What the controls below change**

    - model / renderer: the policy architecture and formatting layer
    - stage sizes / seeds: the curriculum data volume and reproducibility knobs
    - rollout backend / Modal resources: where rollouts execute and how isolated workers are sized
    - W&B project / run name / checkpoint: experiment tracking and resume control
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Environment variables used by this notebook**

    Required for the trainer path configured here:

    - `TINKER_API_KEY`: used by Tinker training and sampling clients
    - `OPENROUTER_API_KEY`: used by the non-trained counterpart powers on OpenRouter
    - `WANDB_API_KEY`: used for Weights & Biases experiment logging

    Optional:

    - `TINKER_BASE_URL`: override the default Tinker API endpoint
    - `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`: env-based Modal auth if you do not want to rely on `modal token new`

    The preflight block below checks these variables directly.
    """)
    return


@app.cell
def _():
    import importlib
    import json
    import shlex
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    script_path = repo_root / "scripts" / "train_tinker_grpo_curriculum.py"
    process_state: dict[str, object] = {}
    notebook_helpers = importlib.import_module("tinker_training.notebook_helpers")
    build_train_command = notebook_helpers.build_train_command
    format_missing_required_env_message = notebook_helpers.format_missing_required_env_message
    get_process_state = notebook_helpers.get_process_state
    launch_once = notebook_helpers.launch_once
    preflight_env = notebook_helpers.preflight_env
    resolve_manifest_path = notebook_helpers.resolve_manifest_path
    return (
        Path,
        build_train_command,
        format_missing_required_env_message,
        get_process_state,
        json,
        launch_once,
        preflight_env,
        process_state,
        repo_root,
        resolve_manifest_path,
        script_path,
        shlex,
    )


@app.cell
def _(mo):
    model_name = mo.ui.text(value="Qwen/Qwen3.5-27B", label="Model")
    renderer_name = mo.ui.text(value="", label="Renderer override")
    rollout_backend = mo.ui.dropdown(
        options=["modal", "local"],
        value="modal",
        label="Rollout backend",
    )
    log_root = mo.ui.text(value="~/tinker-runs/diplomacy-grpo", label="Log root")
    run_name = mo.ui.text(value="", label="Run name")
    wandb_project = mo.ui.text(value="diplomacy-grpo", label="W&B project")
    resume_checkpoint = mo.ui.text(value="", label="Initial checkpoint")

    stage1_train = mo.ui.number(value=128, label="Stage 1 train")
    stage1_group = mo.ui.number(value=4, label="Stage 1 group")
    stage1_seed = mo.ui.number(value=21, label="Stage 1 seed")
    stage2_train = mo.ui.number(value=96, label="Stage 2 train")
    stage2_group = mo.ui.number(value=4, label="Stage 2 group")
    stage2_seed = mo.ui.number(value=37, label="Stage 2 seed")

    openrouter_model = mo.ui.text(
        value="google/gemini-3-flash-preview",
        label="OpenRouter actor model",
    )
    modal_app_name = mo.ui.text(value="diplomacy-grpo-rollouts", label="Modal app")
    modal_timeout = mo.ui.number(value=900, label="Modal timeout (s)")
    modal_cpu = mo.ui.number(value=2.0, label="Modal CPU")
    modal_memory = mo.ui.number(value=4096, label="Modal memory (MB)")
    launch_button = mo.ui.button(
        value=0,
        on_click=lambda value: value + 1,
        label="Launch training",
    )
    manifest_path = mo.ui.text(value="", label="Manifest path override")
    return (
        launch_button,
        log_root,
        manifest_path,
        modal_app_name,
        modal_cpu,
        modal_memory,
        modal_timeout,
        model_name,
        openrouter_model,
        renderer_name,
        resume_checkpoint,
        rollout_backend,
        run_name,
        stage1_group,
        stage1_seed,
        stage1_train,
        stage2_group,
        stage2_seed,
        stage2_train,
        wandb_project,
    )


@app.cell
def _(
    launch_button,
    log_root,
    manifest_path,
    modal_app_name,
    modal_cpu,
    modal_memory,
    modal_timeout,
    mo,
    model_name,
    openrouter_model,
    renderer_name,
    resume_checkpoint,
    rollout_backend,
    run_name,
    stage1_group,
    stage1_seed,
    stage1_train,
    stage2_group,
    stage2_seed,
    stage2_train,
    wandb_project,
):
    mo.vstack(
        [
            mo.md("## Controls"),
            mo.hstack([model_name, renderer_name, rollout_backend], wrap=True, justify="start"),
            mo.hstack([log_root, run_name, wandb_project], wrap=True, justify="start"),
            mo.hstack([resume_checkpoint, manifest_path], wrap=True, justify="start"),
            mo.md("### Stage 1"),
            mo.hstack([stage1_train, stage1_group, stage1_seed], wrap=True, justify="start"),
            mo.md("### Stage 2"),
            mo.hstack([stage2_train, stage2_group, stage2_seed], wrap=True, justify="start"),
            mo.md("### Backend"),
            mo.hstack(
                [openrouter_model, modal_app_name, modal_timeout, modal_cpu, modal_memory],
                wrap=True,
                justify="start",
            ),
            launch_button,
        ],
        gap=1.0,
    )
    return


@app.cell
def _(
    build_train_command,
    log_root,
    manifest_path,
    modal_app_name,
    modal_cpu,
    modal_memory,
    modal_timeout,
    model_name,
    openrouter_model,
    renderer_name,
    resolve_manifest_path,
    resume_checkpoint,
    rollout_backend,
    run_name,
    script_path,
    shlex,
    stage1_group,
    stage1_seed,
    stage1_train,
    stage2_group,
    stage2_seed,
    stage2_train,
    wandb_project,
):
    command_args = build_train_command(
        script_path=script_path,
        model_name=model_name.value,
        rollout_backend=rollout_backend.value,
        log_root=log_root.value,
        wandb_project=wandb_project.value,
        openrouter_model=openrouter_model.value,
        modal_app_name=modal_app_name.value,
        modal_timeout_seconds=int(modal_timeout.value),
        modal_cpu=float(modal_cpu.value),
        modal_memory_mb=int(modal_memory.value),
        stage1_train_examples=int(stage1_train.value),
        stage1_group_size=int(stage1_group.value),
        stage1_train_seed=int(stage1_seed.value),
        stage2_train_examples=int(stage2_train.value),
        stage2_group_size=int(stage2_group.value),
        stage2_train_seed=int(stage2_seed.value),
        renderer_name=renderer_name.value,
        run_name=run_name.value,
        initial_checkpoint_path=resume_checkpoint.value,
    )
    command = shlex.join(command_args)
    chosen_manifest = str(
        resolve_manifest_path(
            log_root=log_root.value,
            run_name=run_name.value,
            manifest_override=manifest_path.value,
        )
    )
    return chosen_manifest, command, command_args


@app.cell
def _(command, json, mo, preflight_env, rollout_backend):
    preflight = preflight_env()
    required_json = json.dumps(preflight["required"], indent=2)
    optional_json = json.dumps(preflight["optional"], indent=2)
    modal_json = json.dumps(preflight["modal_env_auth"], indent=2)
    missing_required = [
        name for name, is_present in preflight["required"].items() if not is_present
    ]
    missing_text = (
        "All required trainer env vars are present."
        if not missing_required
        else "Missing required env vars: " + ", ".join(missing_required)
    )
    modal_note = ""
    if rollout_backend.value == "modal":
        modal_note = (
            "\n\nModal note:\n"
            "- Env-based auth is optional.\n"
            "- If `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` are unset, make sure `modal token new` has already authenticated this machine."
        )
    mo.md(
        f"""
        ## Launch

        ```bash
        {command}
        ```

        Required env vars:

        ```json
        {required_json}
        ```

        Optional env vars:

        ```json
        {optional_json}
        ```

        Modal auth env vars:

        ```json
        {modal_json}
        ```

        Status: `{missing_text}`{modal_note}
        """
    )
    return


@app.cell
def _(
    command_args,
    format_missing_required_env_message,
    get_process_state,
    launch_button,
    launch_once,
    mo,
    process_state: dict[str, object],
    repo_root,
):
    launch_blocker = format_missing_required_env_message()
    snapshot, launch_result = launch_once(
        process_state,
        launch_token=launch_button.value,
        command=command_args,
        cwd=repo_root,
        launch_blocker=launch_blocker,
    )
    process_view = get_process_state(process_state)
    mo.md(f"## Process\n\n{launch_result}")
    return


@app.cell
def _(Path, chosen_manifest, json, mo):
    manifest_data = None
    manifest_file = Path(chosen_manifest).expanduser()
    if manifest_file.exists():
        manifest_data = json.loads(manifest_file.read_text())

    mo.md(
        "## Manifest\n\n"
        + (
            f"```json\n{json.dumps(manifest_data, indent=2)}\n```"
            if manifest_data is not None
            else f"`{manifest_file}` does not exist yet."
        )
    )
    return


if __name__ == "__main__":
    app.run()
