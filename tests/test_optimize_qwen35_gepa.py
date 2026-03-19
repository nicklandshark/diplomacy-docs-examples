from __future__ import annotations

from scripts.optimize_qwen35_gepa import resolve_reflection_lm


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
    def __init__(self, *, api_key: str, base_url: str, default_headers: dict[str, str]) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.default_headers = default_headers
        self.calls: list[dict[str, object]] = []
        self.chat = self
        self.completions = self

    def create(self, *, model: str, messages: list[dict[str, str]]) -> _FakeCompletion:
        self.calls.append({"model": model, "messages": messages})
        return _FakeCompletion("ok")


def test_resolve_reflection_lm_wraps_openai_compatible_client(monkeypatch) -> None:
    created: dict[str, _FakeClient] = {}

    def _fake_openai(*, api_key: str, base_url: str, default_headers: dict[str, str]) -> _FakeClient:
        client = _FakeClient(api_key=api_key, base_url=base_url, default_headers=default_headers)
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
    )

    assert callable(lm)
    assert lm("hello") == "ok"
    assert created["client"].api_key == "test-key"
    assert created["client"].base_url == "https://openrouter.ai/api/v1"
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
    )
    assert resolved is sentinel
