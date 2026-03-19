import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import importlib
    import json
    import shlex
    import sys
    import time
    from pathlib import Path

    import marimo as mo

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    script_path = repo_root / "scripts" / "train_tinker_grpo_curriculum.py"
    notebook_helpers = importlib.import_module("tinker_training.notebook_helpers")
    build_train_command = notebook_helpers.build_train_command
    env_var_purposes = notebook_helpers.ENV_VAR_PURPOSES
    format_missing_required_env_message = (
        notebook_helpers.format_missing_required_env_message
    )
    hybrid_phase_summary = notebook_helpers.hybrid_phase_summary
    hybrid_turn_summary = notebook_helpers.hybrid_turn_summary
    launch_once = notebook_helpers.launch_once
    make_default_run_name = notebook_helpers.make_default_run_name
    notebook_defaults = notebook_helpers.notebook_defaults
    preflight_env = notebook_helpers.preflight_env
    resolve_manifest_path = notebook_helpers.resolve_manifest_path

    process_state = globals().setdefault("_NOTEBOOK_PROCESS_STATE", {})
    defaults = notebook_defaults()
    return (
        Path,
        build_train_command,
        defaults,
        env_var_purposes,
        format_missing_required_env_message,
        hybrid_phase_summary,
        hybrid_turn_summary,
        json,
        launch_once,
        make_default_run_name,
        mo,
        preflight_env,
        process_state,
        repo_root,
        resolve_manifest_path,
        script_path,
        shlex,
        time,
    )


@app.cell
def _(defaults, make_default_run_name, mo):
    model_name = mo.ui.text(
        value=defaults["model_name"],
        label="Model to train",
    )
    enable_thinking = mo.ui.checkbox(
        value=False,
        label="Use thinking-enabled renderer when available",
    )
    renderer_name = mo.ui.text(
        value="",
        label="Renderer override (optional)",
    )
    run_name = mo.ui.text(
        value=make_default_run_name(defaults["model_name"]),
        label="Run name",
    )
    log_root = mo.ui.text(
        value=defaults["log_root"],
        label="Log root",
    )
    wandb_project = mo.ui.text(
        value=defaults["wandb_project"],
        label="W&B project",
    )
    initial_checkpoint = mo.ui.text(
        value="",
        label="Initial checkpoint (optional)",
    )
    prompt_family_dir = mo.ui.text(
        value=defaults["prompt_family_dir"],
        label="Prompt family",
    )
    manifest_path = mo.ui.text(
        value="",
        label="Manifest path override (optional)",
    )
    hybrid_total_batches = mo.ui.number(
        value=defaults["hybrid_total_batches"],
        label="Hybrid total batches",
    )
    save_every = mo.ui.number(
        value=defaults["save_every"],
        label="Save every N global batches",
    )
    eval_every = mo.ui.number(
        value=defaults["eval_every"],
        label="Eval every N global batches",
    )
    num_groups_to_log = mo.ui.number(
        value=defaults["num_groups_to_log"],
        label="Rich-log groups per batch",
    )
    learning_rate = mo.ui.number(
        value=defaults["learning_rate"],
        label="Learning rate",
    )
    lora_rank = mo.ui.number(
        value=defaults["lora_rank"],
        label="LoRA rank",
    )
    openrouter_model = mo.ui.text(
        value=defaults["openrouter_model"],
        label="OpenRouter actor model",
    )
    modal_app_name = mo.ui.text(
        value=defaults["modal_app_name"],
        label="Modal app name",
    )
    modal_timeout = mo.ui.number(
        value=defaults["modal_timeout_seconds"],
        label="Modal timeout (seconds)",
    )
    modal_cpu = mo.ui.number(
        value=defaults["modal_cpu"],
        label="Modal CPU",
    )
    modal_memory = mo.ui.number(
        value=defaults["modal_memory_mb"],
        label="Modal memory (MB)",
    )
    refresh_monitor = mo.ui.refresh(
        options=["5s", "15s", "30s", "60s"],
        default_interval="15s",
        label="Auto-refresh process and manifest",
    )
    launch_button = mo.ui.button(
        value=0,
        on_click=lambda value: value + 1,
        label="Launch training",
    )

    return (
        enable_thinking,
        eval_every,
        hybrid_total_batches,
        initial_checkpoint,
        launch_button,
        learning_rate,
        log_root,
        lora_rank,
        manifest_path,
        modal_app_name,
        modal_cpu,
        modal_memory,
        modal_timeout,
        model_name,
        num_groups_to_log,
        openrouter_model,
        prompt_family_dir,
        refresh_monitor,
        renderer_name,
        run_name,
        save_every,
        wandb_project,
    )


@app.cell
def _(
    build_train_command,
    enable_thinking,
    env_var_purposes,
    eval_every,
    format_missing_required_env_message,
    hybrid_total_batches,
    initial_checkpoint,
    launch_button,
    learning_rate,
    log_root,
    lora_rank,
    manifest_path,
    modal_app_name,
    modal_cpu,
    modal_memory,
    modal_timeout,
    model_name,
    num_groups_to_log,
    openrouter_model,
    prompt_family_dir,
    preflight_env,
    refresh_monitor,
    renderer_name,
    resolve_manifest_path,
    run_name,
    save_every,
    script_path,
    shlex,
    wandb_project,
):
    command_args = build_train_command(
        script_path=script_path,
        model_name=model_name.value,
        log_root=log_root.value,
        wandb_project=wandb_project.value,
        hybrid_total_batches=int(hybrid_total_batches.value),
        prompt_family_dir=prompt_family_dir.value,
        openrouter_model=openrouter_model.value,
        modal_app_name=modal_app_name.value,
        modal_timeout_seconds=int(modal_timeout.value),
        modal_cpu=float(modal_cpu.value),
        modal_memory_mb=int(modal_memory.value),
        save_every=int(save_every.value),
        eval_every=int(eval_every.value),
        num_groups_to_log=int(num_groups_to_log.value),
        learning_rate=float(learning_rate.value),
        lora_rank=int(lora_rank.value),
        enable_thinking=bool(enable_thinking.value),
        renderer_name=renderer_name.value,
        run_name=run_name.value,
        initial_checkpoint_path=initial_checkpoint.value,
    )
    command = shlex.join(command_args)
    manifest_target = str(
        resolve_manifest_path(
            log_root=log_root.value,
            run_name=run_name.value,
            manifest_override=manifest_path.value,
        )
    )

    preflight = preflight_env()
    launch_blocker = format_missing_required_env_message(preflight)

    required_env_lines = "\n".join(
        f"- `{name}`: {env_var_purposes[name]} Present: `{preflight['required'][name]}`"
        for name in ("TINKER_API_KEY", "OPENROUTER_API_KEY", "WANDB_API_KEY")
    )
    optional_env_lines = "\n".join(
        f"- `{name}`: {env_var_purposes[name]} Present: `{preflight['optional'][name]}`"
        for name in ("TINKER_BASE_URL",)
    )

    return (
        command,
        command_args,
        launch_blocker,
        manifest_target,
        optional_env_lines,
        required_env_lines,
    )


@app.cell
def _(
    command,
    defaults,
    enable_thinking,
    hybrid_phase_summary,
    hybrid_total_batches,
    hybrid_turn_summary,
    launch_blocker,
    launch_button,
    learning_rate,
    log_root,
    lora_rank,
    modal_app_name,
    modal_cpu,
    modal_memory,
    modal_timeout,
    model_name,
    mo,
    num_groups_to_log,
    openrouter_model,
    optional_env_lines,
    prompt_family_dir,
    refresh_monitor,
    renderer_name,
    required_env_lines,
    run_name,
    save_every,
    eval_every,
    initial_checkpoint,
    manifest_path,
    wandb_project,
):
    phase_lines = []
    for phase in hybrid_phase_summary():
        end_text = (
            f"{phase['end_batch_exclusive'] - 1}"
            if phase["end_batch_exclusive"] is not None
            else "tail"
        )
        weights = ", ".join(
            f"{environment}={weight:.2f}"
            for environment, weight in phase["environment_weights"].items()
        )
        phase_lines.append(
            f"- `{phase['name']}`: batches `{phase['start_batch']}` to `{end_text}` with {weights}"
        )
    turn_lines = "\n".join(
        f"- `{environment}`: `max_turns={max_turns}`"
        for environment, max_turns in hybrid_turn_summary().items()
    )

    mo.vstack(
        [
            mo.md(
                """
                # Tinker GRPO Operator Notebook

                Use this page to launch and monitor one hybrid GRPO training run for Diplomacy.

                Diplomacy is a multi-agent negotiation game: the policy must read game state and messages,
                communicate with other powers, and submit legal orders under turn limits.

                GRPO (Group Relative Policy Optimization) samples a small group of candidate trajectories for
                each prompt group and updates the trainable policy using their relative rewards instead of a
                separate value model.

                When you click `Launch training`, marimo starts `scripts/train_tinker_grpo_curriculum.py`
                as a subprocess on the machine hosting this notebook. That trainer runs one shared hybrid curriculum
                with a fixed easy-to-hard mixture over:

                1. `tool_accuracy`
                2. `target_execution`
                3. `supported_target`
                4. `cooperative_press`
                5. `full_press`

                The notebook is only the control plane. It launches the trainer, shows the resolved command,
                and monitors `curriculum_manifest.json`. Rollout episodes still execute on Modal workers.
                """
            ),
            mo.md("## Controls"),
            mo.hstack([model_name, enable_thinking, renderer_name], wrap=True, justify="start"),
            mo.hstack([run_name, log_root, wandb_project], wrap=True, justify="start"),
            mo.hstack([initial_checkpoint, prompt_family_dir, manifest_path], wrap=True, justify="start"),
            mo.hstack(
                [hybrid_total_batches, save_every, eval_every, num_groups_to_log, learning_rate, lora_rank],
                wrap=True,
                justify="start",
            ),
            mo.md(
                """
                **Control guide**

                **Core controls**

                - `Model to train`: base checkpoint for the trainable policy.
                - `Renderer override`: only use this if you need to force a specific renderer.
                - `Run name`: output folder and W&B run name.
                - `Log root`: parent directory for manifests, checkpoints, and logs.
                - `Initial checkpoint`: warm-start checkpoint if you are not resuming from the existing run folder.
                - `Prompt family`: the tracked-policy prompts used for the five environment variants.
                - `Hybrid total batches`: total number of train batches in the mixed curriculum.
                - `Save every` / `Eval every`: cadence in global hybrid batches.
                - `Rich-log groups per batch`: how many groups get HTML and logtree output.
                - `Learning rate` / `LoRA rank`: one shared optimizer configuration for the whole hybrid run.

                **Modal controls**

                - `OpenRouter actor model`: model used by the non-trained counterpart powers.
                - `Modal app name`: app name for rollout workers.
                - `Modal timeout` / `Modal CPU` / `Modal memory`: per-worker resource sizing.
                """
            ),
            mo.md("### Hybrid Phase Schedule\n" + "\n".join(phase_lines)),
            mo.md("### Per-Environment Turn Limits\n" + turn_lines),
            mo.md(
                f"""
                ### Fixed Hybrid Defaults

                - `batch_size={defaults['hybrid_batch_size']}`
                - `group_size={defaults['hybrid_group_size']}`
                - `eval_examples_per_environment={defaults['hybrid_eval_examples_per_environment']}`
                - `max_tokens={defaults['hybrid_max_tokens']}`
                """
            ),
            mo.md("### Backend"),
            mo.hstack(
                [openrouter_model, modal_app_name, modal_timeout, modal_cpu, modal_memory],
                wrap=True,
                justify="start",
            ),
            mo.hstack([refresh_monitor, launch_button], justify="start"),
            mo.md(
                f"""
                ## Launch

                ```bash
                {command}
                ```

                **Required environment variables**

                {required_env_lines}

                **Optional**

                {optional_env_lines}
                """
            ),
            (
                mo.md(f"**Launch blocked:** {launch_blocker}")
                if launch_blocker
                else mo.md("")
            ),
        ],
        gap=1.0,
    )
    return command, launch_button


@app.cell
def _(
    Path,
    command_args,
    json,
    launch_blocker,
    launch_button,
    launch_once,
    manifest_target,
    mo,
    process_state,
    refresh_monitor,
    repo_root,
    shlex,
    time,
):
    _refresh_tick = refresh_monitor.value
    process_snapshot, launch_result = launch_once(
        process_state,
        launch_token=launch_button.value,
        command=command_args,
        cwd=repo_root,
        launch_blocker=launch_blocker,
    )

    manifest_file = Path(manifest_target).expanduser()
    manifest_data = json.loads(manifest_file.read_text()) if manifest_file.exists() else None

    process_details = [
        f"- Launch result: {launch_result}",
        f"- PID: `{process_snapshot['pid']}`",
        f"- Return code: `{process_snapshot['returncode']}`",
    ]
    if process_snapshot["started_at"]:
        started_at = time.strftime(
            "%Y-%m-%d %H:%M:%S",
            time.localtime(float(process_snapshot["started_at"])),
        )
        process_details.append(f"- Started at: `{started_at}`")
    if process_snapshot["command"]:
        process_details.append(
            f"- Last command: `{shlex.join(process_snapshot['command'])}`"
        )

    summary_lines = []
    if isinstance(manifest_data, dict):
        summary_lines.extend(
            [
                f"- Status: `{manifest_data.get('status')}`",
                f"- Current stage: `{manifest_data.get('current_stage')}`",
                f"- Current batch: `{manifest_data.get('current_batch')}` / `{manifest_data.get('total_batches')}`",
                f"- Current phase: `{manifest_data.get('current_phase')}`",
                f"- Last checkpoint: `{manifest_data.get('last_checkpoint_name')}`",
            ]
        )
        current_weights = manifest_data.get("current_environment_weights")
        if isinstance(current_weights, dict):
            weights_text = ", ".join(
                f"{environment}={weight:.2f}"
                for environment, weight in current_weights.items()
                if isinstance(weight, (int, float))
            )
            summary_lines.append(f"- Current environment weights: `{weights_text}`")

    manifest_view = (
        f"```json\n{json.dumps(manifest_data, indent=2)}\n```"
        if manifest_data is not None
        else f"`{manifest_file}` does not exist yet."
    )
    mo.vstack(
        [
            mo.md("## Process\n\n" + "\n".join(process_details)),
            mo.md(
                "## Hybrid Progress\n\n"
                + ("\n".join(summary_lines) if summary_lines else "Manifest not written yet.")
            ),
            mo.md("## Manifest\n\n" + manifest_view),
        ],
        gap=1.0,
    )
    return


if __name__ == "__main__":
    app.run()
