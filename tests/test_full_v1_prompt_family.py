from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from data_generator import (
    build_cooperative_press_dataset,
    build_supported_target_dataset,
    build_target_execution_dataset,
)
from scripts.train_tinker_grpo_curriculum import build_config
from tinker_training.diplomacy_adapter import DiplomacyDatasetBuilder, HybridDiplomacyDatasetBuilder
from tinker_training.eval_utils import build_actor_runtime, build_runtime_policy
from tinker_training.prompt_family import (
    DEFAULT_PROMPT_FAMILY_DIR,
    PROMPT_FAMILY_ENVIRONMENTS,
    load_prompt_family_blocks,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_load_prompt_family_blocks_reads_all_stage_prompts() -> None:
    blocks = load_prompt_family_blocks()

    assert set(blocks) == set(PROMPT_FAMILY_ENVIRONMENTS)
    assert "Use tools directly." in blocks["full_press"]
    assert "No press is required in this environment." in blocks["target_execution"]


def test_build_config_supports_hybrid_prompt_family(tmp_path: Path) -> None:
    prompt_family_dir = tmp_path / "prompt-family"
    prompt_family_dir.mkdir()
    for environment_kind in PROMPT_FAMILY_ENVIRONMENTS:
        (prompt_family_dir / f"{environment_kind}.txt").write_text(
            f"{environment_kind} prompt\n"
        )

    args = argparse.Namespace(
        prompt_family_dir=str(prompt_family_dir),
        model_name="Qwen/Qwen3-30B-A3B-Instruct-2507",
        renderer_name="qwen3_instruct",
        enable_thinking=False,
        log_root="~/tinker-runs/diplomacy-grpo",
        run_name="hybrid-v1-test",
        wandb_project="diplomacy-grpo",
        initial_checkpoint_path=None,
        hybrid_total_batches=20,
        learning_rate=2e-5,
        lora_rank=32,
        modal_app_name="rollout-app",
        modal_timeout_seconds=900,
        modal_cpu=2.0,
        modal_memory_mb=4096,
        openrouter_model="openai/gpt-5.4-mini",
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_api_key_env_var="OPENROUTER_API_KEY",
        http_referer="https://local.codex",
        x_title="diplomacy-grpo",
        actor_max_turns=6,
        session_timeout_seconds=90.0,
        default_idle_sleep_seconds=0.5,
        max_message_length=2000,
        save_every=10,
        eval_every=5,
        num_groups_to_log=2,
        disable_rollout_json_export=False,
    )

    config = build_config(args)

    assert [stage.name for stage in config.stages] == ["hybrid_v1"]
    assert config.stages[0].learning_rate == 2e-5
    assert config.prompt_blocks == {
        environment_kind: f"{environment_kind} prompt"
        for environment_kind in PROMPT_FAMILY_ENVIRONMENTS
    }


def test_target_execution_dataset_has_no_required_interactions() -> None:
    row = build_target_execution_dataset(num_sessions=1, seed=21).to_list()[0]
    info = row["info"]

    assert info["environment_kind"] == "target_execution"
    assert info["relevant_powers"] == []
    assert info["required_interactions"] == []
    assert info["transition_target"] is not None


def test_supported_target_dataset_has_one_required_counterpart() -> None:
    row = build_supported_target_dataset(num_sessions=1, seed=21).to_list()[0]
    info = row["info"]

    assert info["environment_kind"] == "supported_target"
    assert len(info["relevant_powers"]) == 1
    assert len(info["required_interactions"]) == 2
    assert len(info["relevant_actor_objectives"]) == 1


def test_cooperative_press_dataset_has_one_required_counterpart() -> None:
    row = build_cooperative_press_dataset(num_sessions=1, seed=21).to_list()[0]
    info = row["info"]

    assert info["environment_kind"] == "cooperative_press"
    assert len(info["relevant_powers"]) == 1
    assert len(info["required_interactions"]) == 2
    assert len(info["relevant_actor_objectives"]) == 1


def test_eval_dataset_uses_single_partial_batch_when_eval_set_is_small() -> None:
    dataset_builder = DiplomacyDatasetBuilder(
        environment_kind="tool_accuracy",
        model_name_for_tokenizer="Qwen/Qwen3-30B-A3B-Instruct-2507",
        renderer_name="qwen3_instruct",
        actor_runtime=build_actor_runtime(actor_max_turns=6, session_timeout_seconds=90.0),
        policy_config=build_runtime_policy(max_turns=14),
        batch_size=16,
        group_size=4,
        num_train_examples=64,
        num_eval_examples=8,
        train_seed=21,
        eval_seed=10021,
    )

    _, eval_dataset = asyncio.run(dataset_builder())

    assert eval_dataset is not None
    assert len(eval_dataset) == 1
    assert len(eval_dataset.get_batch(0)) == 8


def test_hybrid_dataset_builder_creates_mixed_train_batches_and_named_evaluators() -> None:
    prompt_blocks = {
        environment_kind: f"{environment_kind} prompt"
        for environment_kind in PROMPT_FAMILY_ENVIRONMENTS
    }
    dataset_builder = HybridDiplomacyDatasetBuilder(
        model_name_for_tokenizer="Qwen/Qwen3-30B-A3B-Instruct-2507",
        renderer_name="qwen3_instruct",
        actor_runtime=build_actor_runtime(actor_max_turns=6, session_timeout_seconds=90.0),
        prompt_blocks=prompt_blocks,
        batch_size=4,
        group_size=2,
        total_batches=6,
        train_seed=21,
        eval_seed_base=10021,
        num_eval_examples_per_environment=2,
    )

    train_dataset, maybe_eval_dataset = asyncio.run(dataset_builder())

    assert maybe_eval_dataset is None
    assert len(train_dataset) == 6
    first_batch = train_dataset.get_batch(0)
    assert len(first_batch) == 4
    assert all(builder.environment_kind == "tool_accuracy" for builder in first_batch)
    named_evaluators = dataset_builder.build_named_evaluators(max_tokens=64)
    assert [evaluator.name for evaluator in named_evaluators] == [
        "test/tool_accuracy",
        "test/target_execution",
        "test/supported_target",
        "test/cooperative_press",
        "test/full_press",
    ]
    metrics = dataset_builder.metrics_for_step(0)
    assert metrics["mix/train_batch_fraction/tool_accuracy"] == 1.0
    assert metrics["mix/current_phase_index"] == 1.0


def test_repo_prompt_tree_only_contains_gepa_full_press_family() -> None:
    prompt_files = sorted(
        path.relative_to(REPO_ROOT / "prompts").as_posix()
        for path in (REPO_ROOT / "prompts").rglob("*.txt")
    )

    assert DEFAULT_PROMPT_FAMILY_DIR == REPO_ROOT / "prompts" / "gepa-full-press"
    assert prompt_files == [
        "gepa-full-press/cooperative_press.txt",
        "gepa-full-press/full_press.txt",
        "gepa-full-press/supported_target.txt",
        "gepa-full-press/target_execution.txt",
        "gepa-full-press/tool_accuracy.txt",
    ]
