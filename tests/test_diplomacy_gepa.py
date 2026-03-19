from __future__ import annotations

from dataclasses import replace

from tinker_training.diplomacy_gepa import (
    SeedResult,
    TurnRecord,
    build_manual_review,
    build_pattern_summary,
    build_taxonomy_summary,
    classify_seed_result,
    get_gepa_train_pool,
    review_background_from_taxonomy,
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
        wait_count=1,
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
    assert summary["row_count"] == 2
    assert summary["success_rate"] == 0.5
    assert summary["wait_loop_rate"] == 0.5
    assert summary["dominant_failure_type"] == "wait_loop"
    assert summary["failure_counts"]["wait_loop"] == 1

    taxonomy = build_taxonomy_summary([success, failed])
    assert taxonomy["counts"]["wait_loop"] == 1
    assert taxonomy["dominant_failure"] in {"wait_loop", "compact_order_format_error", "rejected_tool_call", "parse_or_markup_failure", "late_finish", "gate_fail_after_legal_submission", "counterpart_no_reply", "other"}

    review = build_manual_review([success, failed], failed_limit=1, success_limit=1)
    assert len(review["failed"]) == 1
    assert len(review["successful"]) == 1
    assert "recent_trace" in review["failed"][0]
    assert "turn_trace" in review["failed"][0]
    assert "trajectory_signals" in review["failed"][0]
    assert review["failed"][0]["tool_counts"]


def test_manual_review_includes_contact_and_post_legal_signals() -> None:
    row = _row(seed=51, reward=0.0, relevant_submission=1.0)
    row = replace(
        row,
        turn_records=[
            TurnRecord(
                turn_index=0,
                tools=["send_message"],
                action_text="<tool_call><function=send_message></function></tool_call>",
                observation_excerpt="",
                reward=0.0,
                episode_done=False,
                metrics={},
            ),
            TurnRecord(
                turn_index=1,
                tools=["read_legal_orders"],
                action_text="<tool_call><function=read_legal_orders></function></tool_call>",
                observation_excerpt="",
                reward=0.0,
                episode_done=False,
                metrics={},
            ),
            TurnRecord(
                turn_index=2,
                tools=["submit_orders"],
                action_text="<tool_call><function=submit_orders></function></tool_call>",
                observation_excerpt="",
                reward=0.0,
                episode_done=False,
                metrics={},
            ),
            TurnRecord(
                turn_index=3,
                tools=["finish"],
                action_text="<tool_call><function=finish></function></tool_call>",
                observation_excerpt="",
                reward=0.0,
                episode_done=True,
                metrics={},
            ),
        ],
        dominant_failure="gate_fail_after_legal_submission",
    )
    review = build_manual_review([row], failed_limit=1, success_limit=1)
    signals = review["failed"][0]["trajectory_signals"]
    assert signals["contact_before_read"] is True
    assert signals["post_legal_next_turn_has_submit"] is True
    assert signals["post_legal_next_turn_narrates"] is False
    assert signals["finish_after_submit"] is True


def test_seed_result_json_keeps_failure_bucket_alias() -> None:
    row = _row(seed=41, wait_count=3, read_conversation_count=2)
    row = replace(
        row,
        turn_records=[
            TurnRecord(
                turn_index=0,
                tools=["wait"],
                action_text="wait",
                observation_excerpt="no response yet",
                reward=0.0,
                episode_done=False,
                metrics={},
            ),
            TurnRecord(
                turn_index=1,
                tools=["read_conversation"],
                action_text="read",
                observation_excerpt="still only my message",
                reward=0.0,
                episode_done=False,
                metrics={},
            ),
            TurnRecord(
                turn_index=2,
                tools=["wait"],
                action_text="wait",
                observation_excerpt="still only my message",
                reward=0.0,
                episode_done=False,
                metrics={},
            ),
            TurnRecord(
                turn_index=3,
                tools=["read_conversation"],
                action_text="read",
                observation_excerpt="still only my message",
                reward=0.0,
                episode_done=False,
                metrics={},
            ),
            TurnRecord(
                turn_index=4,
                tools=["wait"],
                action_text="wait",
                observation_excerpt="still only my message",
                reward=0.0,
                episode_done=False,
                metrics={},
            ),
        ],
    )
    row = replace(row, dominant_failure=classify_seed_result(row))

    payload = row.to_json()
    assert payload["failure_bucket"] == "wait_loop"

    restored = SeedResult.from_json(payload)
    assert restored.dominant_failure == "wait_loop"


def test_seed_result_from_json_backfills_json_tool_names() -> None:
    row = _row(seed=61, reward=0.0, relevant_submission=1.0)
    row = replace(
        row,
        turn_records=[
            TurnRecord(
                turn_index=0,
                tools=[],
                action_text='<tool_call>{"name":"send_message","arguments":{"participants":["FRANCE"],"message":"hi"}}</tool_call>',
                observation_excerpt="",
                reward=0.0,
                episode_done=False,
                metrics={},
            ),
            TurnRecord(
                turn_index=1,
                tools=[],
                action_text='<tool_call>{"name":"wait","arguments":{"seconds":5}}</tool_call>',
                observation_excerpt="still only my message",
                reward=0.0,
                episode_done=False,
                metrics={},
            ),
            TurnRecord(
                turn_index=2,
                tools=[],
                action_text='<tool_call>{"name":"read_conversation","arguments":{"participants":["FRANCE"]}}</tool_call>',
                observation_excerpt="still only my message",
                reward=0.0,
                episode_done=False,
                metrics={},
            ),
        ],
        wait_count=0,
        read_conversation_count=0,
        dominant_failure="other",
    )
    restored = SeedResult.from_json(row.to_json())
    assert restored.turn_records[0].tools == ["send_message"]
    assert restored.turn_records[1].tools == ["wait"]
    assert restored.wait_count == 1
    assert restored.read_conversation_count == 1


def test_manual_review_backfills_near_miss_successes_when_no_gate_wins() -> None:
    rows = [
        replace(_row(seed=71, reward=0.85, relevant_submission=1.0), dominant_failure="gate_fail_after_legal_submission"),
        replace(_row(seed=72, reward=0.25, relevant_submission=1.0), dominant_failure="rejected_tool_call"),
        replace(_row(seed=73, reward=-0.25), dominant_failure="wait_loop"),
    ]
    review = build_manual_review(rows, failed_limit=2, success_limit=2)
    assert [entry["seed"] for entry in review["successful"]] == [71, 72]


def test_pattern_summary_and_background_include_trace_signals() -> None:
    legal_but_fail = replace(
        _row(seed=81, reward=0.85, relevant_submission=1.0, gate=0.0),
        turn_records=[
            TurnRecord(
                turn_index=0,
                tools=["send_message"],
                action_text="I will message first <tool_call><function=send_message></function></tool_call>",
                observation_excerpt="",
                reward=0.0,
                episode_done=False,
                metrics={},
            ),
            TurnRecord(
                turn_index=1,
                tools=["read_legal_orders"],
                action_text="<tool_call><function=read_legal_orders></function></tool_call>",
                observation_excerpt="",
                reward=0.0,
                episode_done=False,
                metrics={},
            ),
            TurnRecord(
                turn_index=2,
                tools=["submit_orders"],
                action_text="<tool_call><function=submit_orders></function></tool_call>",
                observation_excerpt="",
                reward=0.0,
                episode_done=False,
                metrics={},
            ),
            TurnRecord(
                turn_index=3,
                tools=["submit_orders"],
                action_text="<tool_call><function=submit_orders></function></tool_call>",
                observation_excerpt="",
                reward=0.0,
                episode_done=True,
                metrics={},
            ),
        ],
        dominant_failure="gate_fail_after_legal_submission",
    )
    patterns = build_pattern_summary([legal_but_fail])
    assert patterns["narration_before_tool_rate"] == 1.0
    assert patterns["multi_submit_rate"] == 1.0
    assert patterns["legal_but_fail_gate_rate"] == 1.0

    background = review_background_from_taxonomy(
        baseline_taxonomy=build_taxonomy_summary([legal_but_fail]),
        pattern_summary=patterns,
    )
    assert "narration before tool calls is common" in background
    assert "many trajectories submit legal orders but still miss the objective" in background
    assert "keep non-essential units on simple legal holds" in background
    assert "request the exact support or hold order" in background
    assert "actual occupation of the target province after adjudication" in background
    assert "repeated submit_orders attempts are common" in background
    assert "do not resubmit variant order sets unless the previous submit_orders call was explicitly rejected" in background
    assert "do not call read_legal_orders again or submit a second variant" in background
    assert "once a legal submit_orders call succeeds, finish immediately" in background
