from __future__ import annotations

import json

from scripts.optimize_qwen35_gepa import load_completed_rows, record_metric_call, resolve_reflection_lm
from tinker_training.diplomacy_gepa import SeedResult


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout: float,
        default_headers: dict[str, str],
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.default_headers = default_headers
        self.calls: list[dict[str, object]] = []
        self.chat = self
        self.completions = self

    def create(self, *, model: str, messages: list[dict[str, str]]) -> _FakeCompletion:
        self.calls.append({"model": model, "messages": messages})
        return _FakeCompletion("ok")


def test_resolve_reflection_lm_wraps_openai_compatible_client(monkeypatch) -> None:
    created: dict[str, _FakeClient] = {}

    def _fake_openai(
        *,
        api_key: str,
        base_url: str,
        timeout: float,
        default_headers: dict[str, str],
    ) -> _FakeClient:
        client = _FakeClient(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            default_headers=default_headers,
        )
        created["client"] = client
        return client

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("scripts.optimize_qwen35_gepa.OpenAI", _fake_openai)

    lm = resolve_reflection_lm(
        reflection_lm="openai/gpt-5.4-mini",
        helper_base_url="https://openrouter.ai/api/v1",
        helper_api_key_env_var="OPENROUTER_API_KEY",
        helper_http_referer="https://local.codex",
        helper_x_title="diplomacy-gepa-bench",
        timeout_seconds=45.0,
    )

    assert callable(lm)
    assert lm("hello") == "ok"
    assert created["client"].api_key == "test-key"
    assert created["client"].base_url == "https://openrouter.ai/api/v1"
    assert created["client"].timeout == 45.0
    assert created["client"].default_headers == {
        "HTTP-Referer": "https://local.codex",
        "X-Title": "diplomacy-gepa-bench",
    }
    assert created["client"].calls == [
        {
            "model": "openai/gpt-5.4-mini",
            "messages": [{"role": "user", "content": "hello"}],
        }
    ]


def test_resolve_reflection_lm_leaves_callables_unchanged() -> None:
    sentinel = lambda prompt: "ok"
    resolved = resolve_reflection_lm(
        reflection_lm=sentinel,
        helper_base_url="https://openrouter.ai/api/v1",
        helper_api_key_env_var="OPENROUTER_API_KEY",
        helper_http_referer="https://local.codex",
        helper_x_title="diplomacy-gepa-bench",
        timeout_seconds=45.0,
    )
    assert resolved is sentinel


def test_record_metric_call_writes_candidate_and_row_artifacts(tmp_path) -> None:
    row = SeedResult(
        seed=87,
        status="ok",
        execution_backend="tinker_modal",
        score=0.25,
        reward=0.0,
        gate=0.0,
        transition_target=0.0,
        relevant_submission=1.0,
        rejected_tool_calls=0.0,
        wait_count=0,
        read_conversation_count=1,
        compact_order_error=False,
        has_think_close=False,
        turns=5,
        max_turns=10,
        wall_time_seconds=12.0,
        dominant_failure="gate_fail_after_legal_submission",
    )

    record_metric_call(
        metric_dir=tmp_path,
        call_index=3,
        candidate="Use tools directly.",
        row=row,
    )

    candidate_files = list((tmp_path / "candidates").glob("*.txt"))
    row_files = list((tmp_path / "rows").glob("*.json"))
    assert len(candidate_files) == 1
    assert len(row_files) == 1
    row_payload = json.loads(row_files[0].read_text())
    assert row_payload["call_index"] == 3
    assert row_payload["seed_result"]["seed"] == 87
    lines = (tmp_path / "metric_calls.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["dominant_failure"] == "gate_fail_after_legal_submission"


def test_load_completed_rows_requires_all_seed_artifacts(tmp_path) -> None:
    rows_dir = tmp_path / "rows"
    rows_dir.mkdir()
    row = SeedResult(
        seed=87,
        status="ok",
        execution_backend="tinker_modal",
        score=0.25,
        reward=0.0,
        gate=0.0,
        transition_target=0.0,
        relevant_submission=1.0,
        rejected_tool_calls=0.0,
        wait_count=0,
        read_conversation_count=1,
        compact_order_error=False,
        has_think_close=False,
        turns=5,
        max_turns=10,
        wall_time_seconds=12.0,
        dominant_failure="gate_fail_after_legal_submission",
    )
    (rows_dir / "seed_0087.json").write_text(json.dumps(row.to_json()))

    assert load_completed_rows(output_dir=tmp_path, seeds=[87, 88]) is None

    row_b = SeedResult(
        seed=88,
        status="ok",
        execution_backend="tinker_modal",
        score=1.0,
        reward=1.0,
        gate=1.0,
        transition_target=1.0,
        relevant_submission=1.0,
        rejected_tool_calls=0.0,
        wait_count=0,
        read_conversation_count=1,
        compact_order_error=False,
        has_think_close=False,
        turns=4,
        max_turns=10,
        wall_time_seconds=10.0,
        dominant_failure=None,
    )
    (rows_dir / "seed_0088.json").write_text(json.dumps(row_b.to_json()))

    loaded = load_completed_rows(output_dir=tmp_path, seeds=[87, 88])
    assert loaded is not None
    assert [row.seed for row in loaded] == [87, 88]
    assert loaded[1].reward == 1.0
