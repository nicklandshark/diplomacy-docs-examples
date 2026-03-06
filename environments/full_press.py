from __future__ import annotations

from typing import Any

import verifiers as vf

import rlm_backend as backend
from data_generator import build_full_press_dataset


def build_rubric() -> vf.Rubric:
    rubric = vf.Rubric(
        funcs=[backend.transition_target_satisfied_metric],
        weights=[1.0],
    )
    rubric.add_metric(backend.full_press_gate_metric)
    rubric.add_metric(backend.relevant_actor_submission_metric)
    rubric.add_metric(backend.rejected_tool_call_count_metric)
    rubric.add_metric(backend.finalize_diplomacy_state_metric)
    return rubric


def load_environment(
    *,
    actor_configs: dict[str, backend.ActorConfig | dict[str, Any]],
    num_train_examples: int = 100,
    max_turns: int = 50,
    sub_tool_max_turns: int = 5,
    max_sub_llm_parallelism: int = 5,
    repl_language: str = "bash",
    conversation_render_line_cap: int = backend.DEFAULT_CONVERSATION_RENDER_LINE_CAP,
    global_session_timeout_seconds: float = backend.DEFAULT_SESSION_TIMEOUT_SECONDS,
    default_idle_sleep_seconds: float = backend.DEFAULT_IDLE_SLEEP_SECONDS,
    actor_max_turns: int = 50,
    seed: int | None = None,
    max_message_length: int = backend.DEFAULT_MAX_MESSAGE_LENGTH,
    code_execution_timeout: int = 120,
    **kwargs: Any,
) -> backend.AsyncDiplomacyRLMEnv:
    dataset = build_full_press_dataset(num_sessions=num_train_examples, seed=seed)
    rubric = build_rubric()
    return backend.load_environment(
        dataset=dataset,
        rubric=rubric,
        actor_configs=actor_configs,
        max_turns=max_turns,
        sub_tool_max_turns=sub_tool_max_turns,
        max_sub_llm_parallelism=max_sub_llm_parallelism,
        repl_language=repl_language,
        conversation_render_line_cap=conversation_render_line_cap,
        global_session_timeout_seconds=global_session_timeout_seconds,
        default_idle_sleep_seconds=default_idle_sleep_seconds,
        actor_max_turns=actor_max_turns,
        max_message_length=max_message_length,
        code_execution_timeout=code_execution_timeout,
        **kwargs,
    )
