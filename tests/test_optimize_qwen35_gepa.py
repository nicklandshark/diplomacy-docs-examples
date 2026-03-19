from __future__ import annotations

import argparse
import json

from scripts.optimize_qwen35_gepa import (
    GEPAPromptEvaluator,
    extract_tool_names,
    is_transient_seed_error,
    load_completed_rows,
    load_metric_cache,
    make_screen_presets,
    repair_saved_screen,
    record_metric_call,
    resolve_reflection_lm,
)
from tinker_training.diplomacy_gepa import ExperimentPreset, SeedResult


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


def test_extract_tool_names_supports_xml_and_json_formats() -> None:
    assert extract_tool_names('<tool_call><function=read_phase_status></function></tool_call>') == [
        "read_phase_status"
    ]
    assert extract_tool_names(
        '<tool_call>{"name":"submit_orders","arguments":{"orders":["A PAR - PIC"]}}</tool_call>'
    ) == ["submit_orders"]


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


def test_load_metric_cache_restores_candidate_seed_entries(tmp_path) -> None:
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
    call_index, cache = load_metric_cache(tmp_path)
    assert call_index == 1
    assert len(cache) == 1
    cached_row = next(iter(cache.values()))
    assert cached_row.seed == 87
    assert cached_row.score == 0.25


def test_load_metric_cache_skips_transient_conflict_errors(tmp_path) -> None:
    transient = SeedResult(
        seed=87,
        status="error",
        execution_backend="tinker_modal",
        score=-1.0,
        reward=-1.0,
        gate=0.0,
        transition_target=0.0,
        relevant_submission=0.0,
        rejected_tool_calls=0.0,
        wait_count=0,
        read_conversation_count=0,
        compact_order_error=False,
        has_think_close=False,
        turns=0,
        max_turns=10,
        wall_time_seconds=1.0,
        failure_kind="ConflictError",
        failure_message="The app is stopped or disabled",
        dominant_failure="other",
    )
    record_metric_call(
        metric_dir=tmp_path,
        call_index=1,
        candidate="Use tools directly.",
        row=transient,
    )

    call_index, cache = load_metric_cache(tmp_path)
    assert call_index == 0
    assert cache == {}
    assert is_transient_seed_error(transient) is True


def test_gepa_prompt_evaluator_uses_cached_metric_without_starting_runner(monkeypatch, tmp_path) -> None:
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
    candidate = "Use tools directly."
    record_metric_call(
        metric_dir=tmp_path,
        call_index=1,
        candidate=candidate,
        row=row,
    )

    started = {"value": False}

    class _FakeRunner:
        async def start(self) -> None:
            started["value"] = True

        async def aclose(self) -> None:
            return None

    class _FakeModalPoolEvaluator:
        def __init__(self, *, preset, app_name, per_seed_timeout_seconds) -> None:
            self.runner = _FakeRunner()

    monkeypatch.setattr("scripts.optimize_qwen35_gepa.ModalPoolEvaluator", _FakeModalPoolEvaluator)
    preset = ExperimentPreset(
        model_name="Qwen/Qwen3.5-27B",
        environment="full_press",
        renderer_name="qwen3_5_disable_thinking",
        disable_thinking=True,
        temperature=1.0,
        max_turns=10,
        actor_max_turns=6,
        session_timeout_seconds=120.0,
        max_tokens=512,
        tracked_instruction_block=candidate,
    )
    evaluator = GEPAPromptEvaluator(
        preset=preset,
        app_name="test-gepa",
        per_seed_timeout_seconds=180.0,
        metric_dir=tmp_path,
    )
    try:
        score, payload = evaluator(candidate, {"seed": 87})
    finally:
        evaluator.close()

    assert score == 0.25
    assert payload["seed"] == 87
    assert started["value"] is False


def test_repair_saved_screen_rewrites_ranked_results_from_cached_rows(tmp_path) -> None:
    screen_dir = tmp_path / "screen"
    config_a = screen_dir / "001-config-a"
    config_b = screen_dir / "002-config-b"
    for config in (config_a, config_b):
        (config / "rows").mkdir(parents=True)

    preset_a = ExperimentPreset(
        model_name="Qwen/Qwen3.5-27B",
        environment="full_press",
        renderer_name="qwen3_5_disable_thinking",
        disable_thinking=True,
        temperature=1.0,
        max_turns=10,
        actor_max_turns=6,
        session_timeout_seconds=120.0,
        max_tokens=512,
        tracked_instruction_block="prompt-a",
    )
    preset_b = ExperimentPreset(
        model_name="Qwen/Qwen3-30B-A3B-Instruct-2507",
        environment="full_press",
        renderer_name="qwen3_instruct",
        disable_thinking=False,
        temperature=1.0,
        max_turns=14,
        actor_max_turns=6,
        session_timeout_seconds=120.0,
        max_tokens=512,
        tracked_instruction_block="prompt-b",
    )
    (config_a / "preset.json").write_text(json.dumps(preset_a.to_json()))
    (config_b / "preset.json").write_text(json.dumps(preset_b.to_json()))

    row_a1 = SeedResult(
        seed=85,
        status="ok",
        execution_backend="tinker_modal",
        score=0.0,
        reward=0.0,
        gate=0.0,
        transition_target=0.0,
        relevant_submission=1.0,
        rejected_tool_calls=0.0,
        wait_count=1,
        read_conversation_count=1,
        compact_order_error=False,
        has_think_close=False,
        turns=6,
        max_turns=10,
        wall_time_seconds=10.0,
        dominant_failure="gate_fail_after_legal_submission",
    )
    row_a2 = SeedResult(
        seed=86,
        status="ok",
        execution_backend="tinker_modal",
        score=0.0,
        reward=0.0,
        gate=0.0,
        transition_target=0.0,
        relevant_submission=1.0,
        rejected_tool_calls=1.0,
        wait_count=0,
        read_conversation_count=1,
        compact_order_error=False,
        has_think_close=False,
        turns=6,
        max_turns=10,
        wall_time_seconds=9.0,
        dominant_failure="rejected_tool_call",
    )
    row_b1 = SeedResult(
        seed=85,
        status="ok",
        execution_backend="tinker_modal",
        score=0.85,
        reward=0.85,
        gate=0.0,
        transition_target=1.0,
        relevant_submission=1.0,
        rejected_tool_calls=0.0,
        wait_count=0,
        read_conversation_count=1,
        compact_order_error=False,
        has_think_close=False,
        turns=6,
        max_turns=14,
        wall_time_seconds=8.0,
        dominant_failure="gate_fail_after_legal_submission",
    )
    row_b2 = SeedResult(
        seed=86,
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
        turns=5,
        max_turns=14,
        wall_time_seconds=7.0,
        dominant_failure=None,
    )
    (config_a / "rows" / "seed_0085.json").write_text(json.dumps(row_a1.to_json()))
    (config_a / "rows" / "seed_0086.json").write_text(json.dumps(row_a2.to_json()))
    (config_b / "rows" / "seed_0085.json").write_text(json.dumps(row_b1.to_json()))
    (config_b / "rows" / "seed_0086.json").write_text(json.dumps(row_b2.to_json()))

    ranked = repair_saved_screen(screen_dir=screen_dir, seeds=[85, 86], top_k=1)

    assert len(ranked) == 2
    assert ranked[0]["preset_path"].endswith("002-config-b/preset.json")
    assert (screen_dir / "top_1_preset.json").exists()
    ranked_results = json.loads((screen_dir / "ranked_results.json").read_text())
    assert ranked_results[0]["summary"]["mean_reward"] == 0.925
    assert (config_a / "review.json").exists()
    assert (config_b / "patterns.json").exists()


def test_make_screen_presets_overrides_environment_for_loaded_presets(tmp_path) -> None:
    preset = ExperimentPreset(
        model_name="Qwen/Qwen3-30B-A3B-Instruct-2507",
        environment="tool_accuracy",
        renderer_name="qwen3_instruct",
        disable_thinking=False,
        temperature=1.0,
        max_turns=10,
        actor_max_turns=6,
        session_timeout_seconds=120.0,
        max_tokens=512,
        tracked_instruction_block="prompt-a",
    )
    preset_path = tmp_path / "preset.json"
    preset_path.write_text(json.dumps(preset.to_json()))

    args = argparse.Namespace(
        preset_paths=[str(preset_path)],
        model_name="Qwen/Qwen3-30B-A3B-Instruct-2507",
        environment="full_press",
        renderer_name=None,
        disable_thinking=False,
        temperature=1.0,
        temperatures=None,
        max_turns=10,
        max_turns_options=None,
        actor_max_turns=6,
        actor_max_turns_options=None,
        prompt_candidate_paths=None,
        session_timeout_seconds=120.0,
        max_tokens=512,
        helper_model="openai/gpt-5.4-mini",
        helper_base_url="https://openrouter.ai/api/v1",
        helper_api_key_env_var="OPENROUTER_API_KEY",
        helper_http_referer="https://local.codex",
        helper_x_title="diplomacy-gepa-bench",
    )

    presets = make_screen_presets(args)

    assert len(presets) == 1
    assert presets[0].environment == "full_press"
