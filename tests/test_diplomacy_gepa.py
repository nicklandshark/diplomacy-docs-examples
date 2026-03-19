from __future__ import annotations

from dataclasses import replace

from tinker_training.diplomacy_gepa import (
    SeedResult,
    TurnRecord,
    build_manual_review,
    build_taxonomy_summary,
    classify_seed_result,
    get_gepa_train_pool,
    resolve_seed_pool,
    summarize_rows,
)


def _row(
    *,
    seed: int,
    reward: float = 0.0,
    gate: float = 0.0,
    transition_target: float = 0.0,
    relevant_submission: float = 0.0,
    rejected_tool_calls: float = 0.0,
    wait_count: int = 0,
    read_conversation_count: int = 0,
    compact_order_error: bool = False,
    has_think_close: bool = False,
    stop_condition: str | None = None,
    status: str = "ok",
    failure_kind: str | None = None,
    failure_message: str | None = None,
    observation_excerpt: str = "",
    action_text: str = "ok",
) -> SeedResult:
    row = SeedResult(
        seed=seed,
        status=status,  # type: ignore[arg-type]
        execution_backend="tinker_modal",
        score=reward,
        reward=reward,
        gate=gate,
        transition_target=transition_target,
        relevant_submission=relevant_submission,
        rejected_tool_calls=rejected_tool_calls,
        wait_count=wait_count,
        read_conversation_count=read_conversation_count,
        compact_order_error=compact_order_error,
        has_think_close=has_think_close,
        turns=wait_count + read_conversation_count + 1,
        max_turns=10,
        wall_time_seconds=1.0,
        stop_condition=stop_condition,
        failure_kind=failure_kind,
        failure_message=failure_message,
        turn_records=[
            TurnRecord(
                turn_index=0,
                tools=["wait"] if wait_count else ["send_message"],
                action_text="A MUN-BOH" if compact_order_error else action_text,
                observation_excerpt=observation_excerpt,
                reward=reward,
                episode_done=False,
                metrics={},
            )
        ],
    )
    return row


def test_seed_pools_resolve_and_alternate() -> None:
    name, seeds = resolve_seed_pool("tool_screen")
    assert name == "tool_screen"
    assert seeds[0] == 21
    assert seeds[-1] == 84
    assert len(seeds) == 64

    custom_name, custom_seeds = resolve_seed_pool("9,10-12")
    assert custom_name == "custom"
    assert custom_seeds == [9, 10, 11, 12]

    first_round = get_gepa_train_pool(1)
    second_round = get_gepa_train_pool(2)
    assert first_round[0] == "gepa_train_a"
    assert second_round[0] == "gepa_train_b"


def test_classify_failure_buckets() -> None:
    compact = _row(seed=21, compact_order_error=True)
    assert classify_seed_result(compact) == "compact_order_format_error"

    wait_loop = _row(seed=22, wait_count=3, read_conversation_count=2)
    assert classify_seed_result(wait_loop) == "wait_loop"

    no_reply = _row(
        seed=23,
        wait_count=2,
        read_conversation_count=2,
        action_text="I see only my message so far and Germany hasn't replied.",
    )
    assert classify_seed_result(no_reply) == "counterpart_no_reply"

    rejected = _row(seed=24, rejected_tool_calls=2.0)
    assert classify_seed_result(rejected) == "rejected_tool_call"

    parse = _row(
        seed=25,
        status="error",
        failure_kind="ValueError",
        failure_message="Malformed JSON in tool arguments",
    )
    assert classify_seed_result(parse) == "parse_or_markup_failure"

    timeout = _row(
        seed=251,
        status="error",
        failure_kind="TimeoutError",
        failure_message="rollout exceeded timeout",
    )
    assert classify_seed_result(timeout) == "late_finish"

    late = _row(seed=26, stop_condition="max_turns_reached")
    assert classify_seed_result(late) == "late_finish"

    gate_fail = _row(seed=27, relevant_submission=1.0)
    assert classify_seed_result(gate_fail) == "gate_fail_after_legal_submission"

    parse_metric = _row(seed=271)
    parse_metric = replace(
        parse_metric,
        last_metrics={"parse_error": 1.0},
    )
    assert classify_seed_result(parse_metric) == "parse_or_markup_failure"


def test_summary_taxonomy_and_review_outputs() -> None:
    success = _row(seed=31, reward=1.0, gate=1.0, transition_target=1.0, relevant_submission=1.0)
    failed = _row(seed=32, wait_count=3, read_conversation_count=2)
    failed = replace(failed, dominant_failure=classify_seed_result(failed))
    success = replace(success, dominant_failure=classify_seed_result(success))

    summary = summarize_rows([success, failed])
    assert summary["seed_count"] == 2
    assert summary["success_rate"] == 0.5
    assert summary["wait_loop_rate"] == 0.5

    taxonomy = build_taxonomy_summary([success, failed])
    assert taxonomy["counts"]["wait_loop"] == 1
    assert taxonomy["dominant_failure"] in {"wait_loop", "compact_order_format_error", "rejected_tool_call", "parse_or_markup_failure", "late_finish", "gate_fail_after_legal_submission", "counterpart_no_reply", "other"}

    review = build_manual_review([success, failed], failed_limit=1, success_limit=1)
    assert len(review["failed"]) == 1
    assert len(review["successful"]) == 1


def test_seed_result_json_keeps_failure_bucket_alias() -> None:
    row = _row(seed=41, wait_count=3, read_conversation_count=2)
    row = replace(row, dominant_failure=classify_seed_result(row))

    payload = row.to_json()
    assert payload["failure_bucket"] == "wait_loop"

    restored = SeedResult.from_json(payload)
    assert restored.dominant_failure == "wait_loop"
