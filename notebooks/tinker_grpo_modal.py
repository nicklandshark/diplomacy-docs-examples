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
    launch_once = notebook_helpers.launch_once
    make_default_run_name = notebook_helpers.make_default_run_name
    notebook_defaults = notebook_helpers.notebook_defaults
    preflight_env = notebook_helpers.preflight_env
    resolve_manifest_path = notebook_helpers.resolve_manifest_path

    # Keep the subprocess bookkeeping in kernel-global state so a click on the
    # launch button survives cell reruns without introducing a state self-loop.
    process_state = globals().setdefault("_NOTEBOOK_PROCESS_STATE", {})

    defaults = notebook_defaults()
    return (
        Path,
        build_train_command,
        defaults,
        env_var_purposes,
        format_missing_required_env_message,
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
    tracked_instruction_block_path = defaults.get("tracked_instruction_block_path", "")
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
    lora_rank = mo.ui.number(
        value=defaults["lora_rank"],
        label="LoRA rank (shared across both stages)",
    )

    stage1_train_examples = mo.ui.number(
        value=defaults["stage1_train_examples"],
        label="Stage 1 train examples",
    )
    stage1_eval_examples = mo.ui.number(
        value=defaults["stage1_eval_examples"],
        label="Stage 1 eval examples",
    )
    stage1_batch_size = mo.ui.number(
        value=defaults["stage1_batch_size"],
        label="Stage 1 batch size",
    )
    stage1_group_size = mo.ui.number(
        value=defaults["stage1_group_size"],
        label="Stage 1 group size",
    )
    stage1_max_tokens = mo.ui.number(
        value=defaults["stage1_max_tokens"],
        label="Stage 1 max output tokens",
    )
    stage1_max_turns = mo.ui.number(
        value=defaults["stage1_max_turns"],
        label="Stage 1 max turns",
    )
    stage1_train_seed = mo.ui.number(
        value=defaults["stage1_train_seed"],
        label="Stage 1 train seed",
    )
    stage1_learning_rate = mo.ui.number(
        value=defaults["stage1_learning_rate"],
        label="Stage 1 learning rate",
    )

    stage2_train_examples = mo.ui.number(
        value=defaults["stage2_train_examples"],
        label="Stage 2 train examples",
    )
    stage2_eval_examples = mo.ui.number(
        value=defaults["stage2_eval_examples"],
        label="Stage 2 eval examples",
    )
    stage2_batch_size = mo.ui.number(
        value=defaults["stage2_batch_size"],
        label="Stage 2 batch size",
    )
    stage2_group_size = mo.ui.number(
        value=defaults["stage2_group_size"],
        label="Stage 2 group size",
    )
    stage2_max_tokens = mo.ui.number(
        value=defaults["stage2_max_tokens"],
        label="Stage 2 max output tokens",
    )
    stage2_max_turns = mo.ui.number(
        value=defaults["stage2_max_turns"],
        label="Stage 2 max turns",
    )
    stage2_train_seed = mo.ui.number(
        value=defaults["stage2_train_seed"],
        label="Stage 2 train seed",
    )
    stage2_learning_rate = mo.ui.number(
        value=defaults["stage2_learning_rate"],
        label="Stage 2 learning rate",
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
        initial_checkpoint,
        launch_button,
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
        stage1_batch_size,
        stage1_eval_examples,
        stage1_group_size,
        stage1_learning_rate,
        stage1_max_tokens,
        stage1_max_turns,
        stage1_train_examples,
        stage1_train_seed,
        stage2_batch_size,
        stage2_eval_examples,
        stage2_group_size,
        stage2_learning_rate,
        stage2_max_tokens,
        stage2_max_turns,
        stage2_train_examples,
        stage2_train_seed,
        tracked_instruction_block_path,
        wandb_project,
    )


@app.cell
def _(
    build_train_command,
    enable_thinking,
    env_var_purposes,
    eval_every,
    format_missing_required_env_message,
    initial_checkpoint,
    launch_button,
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
    stage1_batch_size,
    stage1_eval_examples,
    stage1_group_size,
    stage1_learning_rate,
    stage1_max_tokens,
    stage1_max_turns,
    stage1_train_examples,
    stage1_train_seed,
    stage2_batch_size,
    stage2_eval_examples,
    stage2_group_size,
    stage2_learning_rate,
    stage2_max_tokens,
    stage2_max_turns,
    stage2_train_examples,
    stage2_train_seed,
    tracked_instruction_block_path,
    wandb_project,
):
    command_args = build_train_command(
        script_path=script_path,
        model_name=model_name.value,
        log_root=log_root.value,
        wandb_project=wandb_project.value,
        prompt_family_dir=prompt_family_dir.value,
        openrouter_model=openrouter_model.value,
        modal_app_name=modal_app_name.value,
        modal_timeout_seconds=int(modal_timeout.value),
        modal_cpu=float(modal_cpu.value),
        modal_memory_mb=int(modal_memory.value),
        save_every=int(save_every.value),
        eval_every=int(eval_every.value),
        num_groups_to_log=int(num_groups_to_log.value),
        stage1_train_examples=int(stage1_train_examples.value),
        stage1_eval_examples=int(stage1_eval_examples.value),
        stage1_batch_size=int(stage1_batch_size.value),
        stage1_group_size=int(stage1_group_size.value),
        stage1_max_tokens=int(stage1_max_tokens.value),
        stage1_max_turns=int(stage1_max_turns.value),
        stage1_train_seed=int(stage1_train_seed.value),
        stage1_learning_rate=float(stage1_learning_rate.value),
        stage2_train_examples=int(stage2_train_examples.value),
        stage2_eval_examples=int(stage2_eval_examples.value),
        stage2_batch_size=int(stage2_batch_size.value),
        stage2_group_size=int(stage2_group_size.value),
        stage2_max_tokens=int(stage2_max_tokens.value),
        stage2_max_turns=int(stage2_max_turns.value),
        stage2_train_seed=int(stage2_train_seed.value),
        stage2_learning_rate=float(stage2_learning_rate.value),
        lora_rank=int(lora_rank.value),
        enable_thinking=bool(enable_thinking.value),
        renderer_name=renderer_name.value,
        run_name=run_name.value,
        initial_checkpoint_path=initial_checkpoint.value,
        tracked_instruction_block_path=tracked_instruction_block_path,
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
    command_args,
    enable_thinking,
    eval_every,
    initial_checkpoint,
    launch_blocker,
    log_root,
    lora_rank,
    mo,
    num_groups_to_log,
    openrouter_model,
    optional_env_lines,
    prompt_family_dir,
    renderer_name,
    required_env_lines,
    refresh_monitor,
    run_name,
    save_every,
    stage1_batch_size,
    stage1_eval_examples,
    stage1_group_size,
    stage1_learning_rate,
    stage1_max_tokens,
    stage1_max_turns,
    stage1_train_examples,
    stage1_train_seed,
    stage2_batch_size,
    stage2_eval_examples,
    stage2_group_size,
    stage2_learning_rate,
    stage2_max_tokens,
    stage2_max_turns,
    stage2_train_examples,
    stage2_train_seed,
    launch_button,
    modal_app_name,
    modal_cpu,
    modal_memory,
    modal_timeout,
    model_name,
    wandb_project,
    manifest_path,
):
    mo.vstack(
        [
            mo.md(
                """
                # Tinker GRPO Operator Notebook

                Use this page to launch and monitor one multi-stage GRPO training run for Diplomacy.

                Diplomacy is a multi-agent negotiation game: the policy must read game state and messages,
                communicate with other powers, and submit legal orders under turn limits.

                GRPO (Group Relative Policy Optimization) samples a small group of candidate trajectories for
                each prompt group and updates the trainable policy using their relative rewards instead of a
                separate value model.

                When you click `Launch training`, marimo starts `scripts/train_tinker_grpo_curriculum.py`
                as a subprocess on the machine hosting this notebook. That trainer then runs one shared
                curriculum with the default `full_v1` five-stage easy-to-hard progression:

                1. `tool_accuracy`: short-horizon drills for reading, messaging, and legal action tools.
                2. `target_execution`: no-press target-conversion drills focused on adjudicated success.
                3. `supported_target`: one-counterpart coordination where the target requires a specific support pattern.
                4. `cooperative_press`: easier full-press tasks with one cooperative counterpart.
                5. `full_press`: the hardest unrestricted stage, continuing from the same shared checkpoint.

                This notebook is only the control plane. It launches the trainer, shows the resolved command,
                and monitors `curriculum_manifest.json`. Rollout episodes still execute on Modal workers.
                """
            ),
            mo.md("## Controls"),
            mo.hstack([model_name, enable_thinking, renderer_name], wrap=True, justify="start"),
            mo.hstack([run_name, log_root, wandb_project], wrap=True, justify="start"),
            mo.hstack([initial_checkpoint, prompt_family_dir, manifest_path], wrap=True, justify="start"),
            mo.hstack([save_every, eval_every, num_groups_to_log, lora_rank], wrap=True, justify="start"),
            mo.md(
                """
                **Control guide**

                **Core controls**

                - `Model to train`: base checkpoint for the trainable policy.
                - `Use thinking-enabled renderer`: opt into the model's thinking-capable default renderer when supported.
                - `Renderer override`: only use this if you need to force a specific renderer.
                - `Run name`: output folder and W&B run name.
                - `Log root`: parent directory for manifests, checkpoints, and logs.
                - `Initial checkpoint`: warm-start checkpoint if you are not resuming from the existing run folder.
                - `Prompt family`: the stage-specific tracked-policy prompts used by the default five-stage curriculum.
                - `Manifest path override`: override the default `curriculum_manifest.json` location if you need to monitor another path.
                - `Save every` / `Eval every`: cadence in global curriculum batches.
                - `Rich-log groups per batch`: how many groups get HTML and logtree output.
                - `LoRA rank`: shared adapter rank for the full curriculum.

                **Stage controls**

                The numeric stage controls below are legacy two-stage overrides. The default `full_v1`
                notebook path uses the built-in five-stage preset and ignores them unless you deliberately
                launch the legacy curriculum from the command line outside this notebook.

                - `train examples`: number of sampled sessions used for training in that stage.
                - `eval examples`: number of sampled sessions used for evaluation in that stage.
                - `batch size`: number of prompt groups per optimizer batch.
                - `group size`: number of trajectories sampled per prompt group.
                - `max output tokens`: per-trajectory generation cap.
                - `max turns`: environment turn cap.
                - `train seed`: dataset seed.
                - `learning rate`: stage-specific learning rate.

                The default five-stage preset is intentionally conservative to control spend:
                `64/8`, `64/8`, `64/8`, `48/8`, `48/8` train/eval examples across the five stages.

                **Modal controls**

                - `OpenRouter actor model`: model used by the non-trained counterpart powers.
                - `Modal app name`: app name for rollout workers.
                - `Modal timeout` / `Modal CPU` / `Modal memory`: per-worker resource sizing.
                """
            ),
            mo.md("### Stage 1 (`tool_accuracy`)"),
            mo.hstack(
                [
                    stage1_train_examples,
                    stage1_eval_examples,
                    stage1_batch_size,
                    stage1_group_size,
                ],
                wrap=True,
                justify="start",
            ),
            mo.hstack(
                [
                    stage1_max_tokens,
                    stage1_max_turns,
                    stage1_train_seed,
                    stage1_learning_rate,
                ],
                wrap=True,
                justify="start",
            ),
            mo.md("### Stage 2 (`full_press`)"),
            mo.hstack(
                [
                    stage2_train_examples,
                    stage2_eval_examples,
                    stage2_batch_size,
                    stage2_group_size,
                ],
                wrap=True,
                justify="start",
            ),
            mo.hstack(
                [
                    stage2_max_tokens,
                    stage2_max_turns,
                    stage2_train_seed,
                    stage2_learning_rate,
                ],
                wrap=True,
                justify="start",
            ),
            mo.md("### Backend"),
            mo.hstack(
                [openrouter_model, modal_app_name, modal_timeout, modal_cpu, modal_memory],
                wrap=True,
                justify="start",
            ),
            mo.hstack([refresh_monitor], justify="start"),
            launch_button,
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
        ],
        gap=1.0,
    )
    return command_args, launch_blocker


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

    manifest_view = (
        f"```json\n{json.dumps(manifest_data, indent=2)}\n```"
        if manifest_data is not None
        else f"`{manifest_file}` does not exist yet."
    )
    mo.vstack(
        [
            mo.md("## Process\n\n" + "\n".join(process_details)),
            mo.md("## Manifest\n\n" + manifest_view),
        ],
        gap=1.0,
    )
    return


if __name__ == "__main__":
    app.run()
