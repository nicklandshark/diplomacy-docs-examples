from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from statistics import mean
from typing import Any, Literal

DEFAULT_REFLECTION_MODEL = "openai/gpt-5.4-mini"
DEFAULT_HELPER_MODEL = "openai/gpt-5.4-mini"
DEFAULT_HELPER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_HELPER_API_KEY_ENV_VAR = "OPENROUTER_API_KEY"
DEFAULT_HELPER_HTTP_REFERER = "https://local.codex"
DEFAULT_HELPER_X_TITLE = "diplomacy-gepa-bench"
DEFAULT_REPORT_WORKERS = 4
DEFAULT_PER_SEED_TIMEOUT_SECONDS = 180.0

FailureBucket = Literal[
    "wait_loop",
    "compact_order_format_error",
    "rejected_tool_call",
    "parse_or_markup_failure",
    "late_finish",
    "gate_fail_after_legal_submission",
    "counterpart_no_reply",
    "other",
]

SEED_POOLS: dict[str, list[int]] = {
    "tool_screen": list(range(21, 85)),
    "fullpress_screen": list(range(85, 149)),
    "gepa_train_a": list(range(149, 213)),
    "gepa_train_b": list(range(213, 277)),
    "gepa_val": list(range(277, 309)),
    "confirm": list(range(309, 341)),
    "final_compare": list(range(21, 341)),
}

_COMPACT_ORDER_PATTERNS = (
    re.compile(r"\b[AF]\s+[A-Z]{3}-[A-Z]{3}\b"),
    re.compile(r"\b[AF]\s+[A-Z]{3}[A-Z]{3}\b"),
)
_NO_REPLY_PATTERNS = (
    "no messages",
    "no new messages",
    "no unread messages",
    "no response",
    "awaiting reply",
    "only my message",
    "hasn't replied",
    "has not replied",
    "still only my message",
)
_PARSE_MARKUP_PATTERNS = (
    "parse",
    "json",
    "tool arguments",
    "unterminated",
    "unexpected",
    "malformed",
    "</think>",
)
_XML_TOOL_PATTERN = re.compile(r"<function=([^>]+)>")
_JSON_TOOL_PATTERN = re.compile(r'"name"\s*:\s*"([^"]+)"')


@dataclass(frozen=True)
class ExperimentPreset:
    model_name: str
    environment: Literal["tool_accuracy", "full_press"]
    renderer_name: str
    disable_thinking: bool
    temperature: float
    max_turns: int
    actor_max_turns: int
    session_timeout_seconds: float
    max_tokens: int
    helper_model: str = DEFAULT_HELPER_MODEL
    helper_base_url: str = DEFAULT_HELPER_BASE_URL
    helper_api_key_env_var: str = DEFAULT_HELPER_API_KEY_ENV_VAR
    helper_http_referer: str = DEFAULT_HELPER_HTTP_REFERER
    helper_x_title: str = DEFAULT_HELPER_X_TITLE
    tracked_instruction_block: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "ExperimentPreset":
        return cls(**payload)


@dataclass(frozen=True)
class TurnRecord:
    turn_index: int
    tools: list[str]
    action_text: str
    observation_excerpt: str
    reward: float
    episode_done: bool
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SeedResult:
    seed: int
    status: Literal["ok", "error", "skipped"]
    execution_backend: str
    score: float
    reward: float
    gate: float
    transition_target: float
    relevant_submission: float
    rejected_tool_calls: float
    wait_count: int
    read_conversation_count: int
    compact_order_error: bool
    has_think_close: bool
    turns: int
    max_turns: int
    wall_time_seconds: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    stop_condition: str | None = None
    failure_kind: str | None = None
    failure_message: str | None = None
    dominant_failure: FailureBucket | None = None
    last_metrics: dict[str, float] = field(default_factory=dict)
    turn_records: list[TurnRecord] = field(default_factory=list)
    artifact_path: str | None = None

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["failure_bucket"] = payload["dominant_failure"]
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "SeedResult":
        payload = dict(payload)
        if "dominant_failure" not in payload and "failure_bucket" in payload:
            payload["dominant_failure"] = payload["failure_bucket"]
        payload.pop("failure_bucket", None)
        turn_records = [TurnRecord(**record) for record in payload.get("turn_records", [])]
        repaired_turn_records = [
            replace(record, tools=_extract_tool_names(record.action_text))
            if not record.tools and _extract_tool_names(record.action_text)
            else record
            for record in turn_records
        ]
        payload["turn_records"] = repaired_turn_records
        row = cls(**payload)
        if repaired_turn_records:
            observation_snippets = [
                (record.observation_excerpt or "").lower() for record in repaired_turn_records
            ]
            row = replace(
                row,
                wait_count=sum(record.tools.count("wait") for record in repaired_turn_records),
                read_conversation_count=sum(
                    record.tools.count("read_conversation") for record in repaired_turn_records
                ),
                compact_order_error=row.compact_order_error
                or any(
                    "does not belong to" in snippet or "invalid order" in snippet
                    for snippet in observation_snippets
                )
                or _has_compact_order_text(repaired_turn_records),
                has_think_close=row.has_think_close
                or any("</think>" in record.action_text for record in repaired_turn_records),
                turns=len(repaired_turn_records),
            )
            row = replace(row, dominant_failure=classify_seed_result(row))
        return row


def resolve_seed_pool(seed_pool: str) -> tuple[str, list[int]]:
    cleaned = seed_pool.strip()
    if cleaned in SEED_POOLS:
        return cleaned, list(SEED_POOLS[cleaned])
    return "custom", _parse_seed_spec(cleaned)


def get_gepa_train_pool(round_index: int) -> tuple[str, list[int]]:
    pool_name = "gepa_train_a" if round_index % 2 == 1 else "gepa_train_b"
    return pool_name, list(SEED_POOLS[pool_name])


def summarize_rows(rows: list[SeedResult]) -> dict[str, Any]:
    if not rows:
        return {}
    ok_rows = [row for row in rows if row.status == "ok"]
    failed_rows = [row for row in rows if row.status != "ok"]
    failure_counts = {bucket: 0 for bucket in _all_failure_buckets()}
    taxonomy_rates: dict[str, float] = {}
    for bucket in _all_failure_buckets():
        failure_counts[bucket] = sum(1 for row in rows if row.dominant_failure == bucket)
        taxonomy_rates[f"{bucket}_rate"] = mean(
            1.0 if row.dominant_failure == bucket else 0.0 for row in rows
        )
    dominant_failure_type = None
    if any(failure_counts.values()):
        dominant_failure_type = max(
            failure_counts.items(),
            key=lambda item: (item[1], item[0]),
        )[0]
    return {
        "seed_count": len(rows),
        "row_count": len(rows),
        "ok_count": len(ok_rows),
        "error_count": len(failed_rows),
        "mean_score": mean(row.score for row in rows),
        "mean_reward": mean(row.reward for row in rows),
        "gate_pass_rate": mean(row.gate for row in rows),
        "transition_target_rate": mean(row.transition_target for row in rows),
        "relevant_submission_rate": mean(row.relevant_submission for row in rows),
        "mean_rejected_tool_calls": mean(row.rejected_tool_calls for row in rows),
        "rejected_tool_call_rate": mean(1.0 if row.rejected_tool_calls > 0.0 else 0.0 for row in rows),
        "mean_wait_count": mean(row.wait_count for row in rows),
        "mean_read_conversation_count": mean(row.read_conversation_count for row in rows),
        "compact_order_error_rate": mean(1.0 if row.compact_order_error else 0.0 for row in rows),
        "think_close_rate": mean(1.0 if row.has_think_close else 0.0 for row in rows),
        "mean_wall_time_seconds": mean(row.wall_time_seconds for row in rows),
        "mean_prompt_tokens": mean(row.prompt_tokens for row in rows),
        "mean_completion_tokens": mean(row.completion_tokens for row in rows),
        "success_rate": mean(1.0 if row.reward >= 1.0 else 0.0 for row in rows),
        "dominant_failure_type": dominant_failure_type,
        "failure_counts": failure_counts,
        **taxonomy_rates,
    }


def build_taxonomy_summary(rows: list[SeedResult]) -> dict[str, Any]:
    counts = {bucket: 0 for bucket in _all_failure_buckets()}
    for row in rows:
        if row.dominant_failure is not None:
            counts[row.dominant_failure] += 1
    dominant_failure = None
    if any(counts.values()):
        dominant_failure = max(counts.items(), key=lambda item: item[1])[0]
    return {
        "counts": counts,
        "dominant_failure": dominant_failure,
        "examples": {
            bucket: [
                {
                    "seed": row.seed,
                    "score": row.score,
                    "reward": row.reward,
                    "gate": row.gate,
                    "transition_target": row.transition_target,
                    "relevant_submission": row.relevant_submission,
                    "artifact_path": row.artifact_path,
                }
                for row in rows
                if row.dominant_failure == bucket
            ][:5]
            for bucket in _all_failure_buckets()
        },
    }


def build_pattern_summary(rows: list[SeedResult]) -> dict[str, Any]:
    if not rows:
        return {}

    def _rate(predicate: Any) -> float:
        return mean(1.0 if predicate(row) else 0.0 for row in rows)

    def _submit_count(row: SeedResult) -> int:
        return sum(record.tools.count("submit_orders") for record in row.turn_records)

    def _narrates(row: SeedResult) -> bool:
        return any(_first_nonempty_text_before_tool(record.action_text) for record in row.turn_records)

    pattern_rates = {
        "narration_before_tool_rate": _rate(_narrates),
        "contact_before_read_rate": _rate(lambda row: _trajectory_signals(row)["contact_before_read"]),
        "legal_before_submit_rate": _rate(
            lambda row: (
                _trajectory_signals(row)["first_read_legal_orders_turn"] is not None
                and _trajectory_signals(row)["first_submit_orders_turn"] is not None
                and _trajectory_signals(row)["first_read_legal_orders_turn"]
                < _trajectory_signals(row)["first_submit_orders_turn"]
            )
        ),
        "finish_after_submit_rate": _rate(lambda row: _trajectory_signals(row)["finish_after_submit"]),
        "multi_submit_rate": _rate(lambda row: _submit_count(row) > 1),
        "legal_but_fail_gate_rate": _rate(lambda row: row.relevant_submission > 0.0 and row.gate <= 0.0),
        "read_conversation_after_legal_before_submit_rate": _rate(
            lambda row: any(
                "read_conversation" in record.tools
                and _trajectory_signals(row)["first_read_legal_orders_turn"] is not None
                and record.turn_index > _trajectory_signals(row)["first_read_legal_orders_turn"]
                and (
                    _trajectory_signals(row)["first_submit_orders_turn"] is None
                    or record.turn_index < _trajectory_signals(row)["first_submit_orders_turn"]
                )
                for record in row.turn_records
            )
        ),
    }
    pattern_rates["mean_submit_count"] = mean(_submit_count(row) for row in rows)
    return pattern_rates


def build_manual_review(rows: list[SeedResult], *, failed_limit: int = 10, success_limit: int = 5) -> dict[str, Any]:
    failed_rows = sorted(
        [row for row in rows if row.dominant_failure is not None or row.status != "ok"],
        key=lambda row: (row.score, row.seed),
    )[:failed_limit]
    success_candidates = [row for row in rows if row.status == "ok" and row.gate > 0.0]
    if len(success_candidates) < success_limit:
        seen = {row.seed for row in success_candidates}
        for row in sorted(
            [row for row in rows if row.status == "ok" and row.seed not in seen],
            key=lambda row: (-row.score, row.seed),
        ):
            success_candidates.append(row)
            seen.add(row.seed)
            if len(success_candidates) >= success_limit:
                break
    success_rows = sorted(success_candidates, key=lambda row: (-row.score, row.seed))[:success_limit]
    return {
        "failed": [_review_entry(row) for row in failed_rows],
        "successful": [_review_entry(row) for row in success_rows],
    }


def classify_seed_result(result: SeedResult) -> FailureBucket | None:
    if result.status != "ok":
        lowered = f"{result.failure_kind or ''} {result.failure_message or ''}".lower()
        if "timeout" in lowered:
            return "late_finish"
        if any(pattern in lowered for pattern in _PARSE_MARKUP_PATTERNS):
            return "parse_or_markup_failure"
        return "other"

    if result.compact_order_error or _has_compact_order_text(result.turn_records):
        return "compact_order_format_error"

    if result.wait_count >= 3 or (
        result.wait_count >= 2 and result.read_conversation_count >= 2 and result.relevant_submission < 1.0
    ):
        return "wait_loop"

    if _looks_like_counterpart_no_reply(result):
        return "counterpart_no_reply"

    if result.rejected_tool_calls > 0.0 and result.gate <= 0.0:
        return "rejected_tool_call"

    if float(result.last_metrics.get("parse_error", 0.0)) > 0.0:
        return "parse_or_markup_failure"

    if result.has_think_close and result.gate <= 0.0:
        return "parse_or_markup_failure"

    if result.relevant_submission > 0.0 and result.gate <= 0.0:
        return "gate_fail_after_legal_submission"

    if result.stop_condition == "max_turns_reached" or result.turns >= result.max_turns:
        if result.gate <= 0.0:
            return "late_finish"

    return None


def load_preset(path: Path) -> ExperimentPreset:
    return ExperimentPreset.from_json(json.loads(path.read_text()))


def save_preset(path: Path, preset: ExperimentPreset) -> None:
    path.write_text(json.dumps(preset.to_json(), indent=2) + "\n")


def save_seed_result(path: Path, row: SeedResult) -> None:
    path.write_text(json.dumps(row.to_json(), indent=2) + "\n")


def load_seed_result(path: Path) -> SeedResult:
    return SeedResult.from_json(json.loads(path.read_text()))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def review_background_from_taxonomy(
    *,
    baseline_taxonomy: dict[str, Any] | None = None,
    pattern_summary: dict[str, Any] | None = None,
    extra_lines: list[str] | None = None,
) -> str:
    lines = [
        "The candidate is only the tracked-policy instruction block. Role and power lines are added elsewhere.",
        "Optimize for Diplomacy full_press success with legal, timely, tool-native actions.",
        "Preserve direct tool use. Do not add prose about intended tool use.",
        "Prefer concise instructions that bias toward:",
        "- reading legal orders before submitting orders",
        "- sending one targeted coordination message, then acting",
        "- using at most 2 waits before best-effort legal action",
        "- canonical spaced order syntax like 'A MUN - BOH'",
        '- calling finish(summary="DONE") only after submission or clear completion',
    ]
    if baseline_taxonomy:
        dominant = baseline_taxonomy.get("dominant_failure")
        lines.append(f"Current dominant failure bucket: {dominant}.")
        counts = baseline_taxonomy.get("counts", {})
        for bucket in _all_failure_buckets():
            count = counts.get(bucket, 0)
            if count:
                lines.append(f"- {bucket}: {count}")
    if pattern_summary:
        lines.append("Observed trajectory patterns:")
        if pattern_summary.get("narration_before_tool_rate", 0.0) >= 0.25:
            lines.append(
                f"- narration before tool calls is common ({pattern_summary['narration_before_tool_rate']:.0%}); strip it out."
            )
        if pattern_summary.get("legal_but_fail_gate_rate", 0.0) >= 0.25:
            lines.append(
                f"- many trajectories submit legal orders but still miss the objective ({pattern_summary['legal_but_fail_gate_rate']:.0%}); bias toward the anchor/target order, not merely any legal set."
            )
            lines.append(
                "- when only one move is central to the objective, keep non-essential units on simple legal holds unless another order clearly supports that target."
            )
            lines.append(
                "- when coordination is required, the single message should request the exact support or hold order that makes the target adjudicate successfully."
            )
            lines.append(
                "- optimize for actual occupation of the target province after adjudication, not just for submitting the named target move unsupported."
            )
        if pattern_summary.get("multi_submit_rate", 0.0) >= 0.1:
            lines.append(
                f"- repeated submit_orders attempts are common ({pattern_summary['multi_submit_rate']:.0%}); after a rejection, rebuild from scratch with exactly one order per orderable unit."
            )
            lines.append(
                "- do not resubmit variant order sets unless the previous submit_orders call was explicitly rejected by the environment."
            )
            lines.append(
                "- after a submit_orders call succeeds, do not call read_legal_orders again or submit a second variant unless the environment explicitly rejected the prior submit."
            )
        if pattern_summary.get("read_conversation_after_legal_before_submit_rate", 0.0) >= 0.1:
            lines.append(
                "- do not re-open conversations after reading legal orders unless the task explicitly requires it; move directly to submission."
            )
        if pattern_summary.get("finish_after_submit_rate", 1.0) < 0.85:
            lines.append(
                f"- too many trajectories keep acting after a valid submission ({1.0 - pattern_summary['finish_after_submit_rate']:.0%}); once a legal submit_orders call succeeds, finish immediately unless the environment rejected it or new task-critical information arrived."
            )
    if extra_lines:
        lines.extend(extra_lines)
    return "\n".join(lines)


def _parse_seed_spec(spec: str) -> list[int]:
    if not spec:
        return []
    values: list[int] = []
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            step = 1 if end >= start else -1
            values.extend(list(range(start, end + step, step)))
        else:
            values.append(int(part))
    return values


def _all_failure_buckets() -> tuple[FailureBucket, ...]:
    return (
        "wait_loop",
        "compact_order_format_error",
        "rejected_tool_call",
        "parse_or_markup_failure",
        "late_finish",
        "gate_fail_after_legal_submission",
        "counterpart_no_reply",
        "other",
    )


def _has_compact_order_text(turn_records: list[TurnRecord]) -> bool:
    for record in turn_records:
        text = record.action_text
        if any(pattern.search(text) for pattern in _COMPACT_ORDER_PATTERNS):
            return True
    return False


def _looks_like_counterpart_no_reply(result: SeedResult) -> bool:
    if result.read_conversation_count < 2 or result.wait_count < 1:
        return False
    searchable = " ".join(
        f"{record.observation_excerpt.lower()} {record.action_text.lower()}" for record in result.turn_records
    )
    return any(pattern in searchable for pattern in _NO_REPLY_PATTERNS)


def _first_turn_with_tool(row: SeedResult, tool_name: str) -> int | None:
    for record in row.turn_records:
        if tool_name in record.tools:
            return record.turn_index
    return None


def _first_nonempty_text_before_tool(action_text: str) -> str:
    prefix, _, _ = action_text.partition("<tool_call>")
    return prefix.strip()


def _trajectory_signals(row: SeedResult) -> dict[str, Any]:
    send_turn = _first_turn_with_tool(row, "send_message")
    read_turn = _first_turn_with_tool(row, "read_conversation")
    legal_turn = _first_turn_with_tool(row, "read_legal_orders")
    submit_turn = _first_turn_with_tool(row, "submit_orders")
    finish_turn = _first_turn_with_tool(row, "finish")
    post_legal_turn = None
    post_legal_has_submit = False
    post_legal_narrates = False
    if legal_turn is not None:
        for record in row.turn_records:
            if record.turn_index <= legal_turn:
                continue
            post_legal_turn = record.turn_index
            post_legal_has_submit = "submit_orders" in record.tools
            post_legal_narrates = bool(_first_nonempty_text_before_tool(record.action_text))
            break
    return {
        "first_send_message_turn": send_turn,
        "first_read_conversation_turn": read_turn,
        "first_read_legal_orders_turn": legal_turn,
        "first_submit_orders_turn": submit_turn,
        "first_finish_turn": finish_turn,
        "contact_before_read": (
            send_turn is not None and (read_turn is None or send_turn < read_turn)
        ),
        "respected_wait_hard_cap": row.wait_count <= 2,
        "post_legal_next_turn": post_legal_turn,
        "post_legal_next_turn_has_submit": post_legal_has_submit,
        "post_legal_next_turn_narrates": post_legal_narrates,
        "finish_after_submit": (
            submit_turn is not None and finish_turn is not None and finish_turn > submit_turn
        ),
    }


def _review_entry(row: SeedResult) -> dict[str, Any]:
    tool_counts: dict[str, int] = {}
    narration_before_tool_turns: list[int] = []
    for record in row.turn_records:
        for tool in record.tools:
            tool_counts[tool] = tool_counts.get(tool, 0) + 1
        if _first_nonempty_text_before_tool(record.action_text):
            narration_before_tool_turns.append(record.turn_index)
    return {
        "seed": row.seed,
        "status": row.status,
        "score": row.score,
        "reward": row.reward,
        "gate": row.gate,
        "transition_target": row.transition_target,
        "relevant_submission": row.relevant_submission,
        "dominant_failure": row.dominant_failure,
        "artifact_path": row.artifact_path,
        "trajectory_signals": _trajectory_signals(row),
        "tool_counts": tool_counts,
        "narration_before_tool_turns": narration_before_tool_turns,
        "recent_trace": [
            {
                "turn_index": record.turn_index,
                "tools": record.tools,
                "action_excerpt": record.action_text[:240],
                "observation_excerpt": record.observation_excerpt[:180],
                "metrics": record.metrics,
            }
            for record in row.turn_records[-3:]
        ],
        "turn_trace": [
            {
                "turn_index": record.turn_index,
                "tools": record.tools,
                "reward": record.reward,
                "episode_done": record.episode_done,
                "action_text": record.action_text[:800],
                "observation_excerpt": record.observation_excerpt[:300],
                "metrics": record.metrics,
            }
            for record in row.turn_records
        ],
    }


def _extract_tool_names(action_text: str) -> list[str]:
    if not action_text:
        return []
    xml_names = _XML_TOOL_PATTERN.findall(action_text)
    if xml_names:
        return xml_names
    names: list[str] = []
    for tool_call in re.findall(r"<tool_call>(.*?)</tool_call>", action_text, flags=re.DOTALL):
        for name in _JSON_TOOL_PATTERN.findall(tool_call):
            names.append(name)
    return names
