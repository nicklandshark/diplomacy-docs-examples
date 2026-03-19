from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from tinker_cookbook.rl.types import SamplingRef
from tinker_training.diplomacy_adapter import (
    ActorRuntimeConfig,
    OpenRouterHeaders,
    RuntimePolicyConfig,
    build_actor_configs,
)
from tinker_training.diplomacy_gepa import SeedResult, TurnRecord, classify_seed_result
from tinker_training.rollout_backends import deserialize_trajectory
from data_generator import (
    EnvironmentKind,
    build_cooperative_press_dataset,
    build_full_press_dataset,
    build_supported_target_dataset,
    build_target_execution_dataset,
    build_tool_accuracy_dataset,
)

TOOL_PATTERN = re.compile(r"<function=([^>]+)>")
JSON_TOOL_PATTERN = re.compile(r'"name"\s*:\s*"([^"]+)"')
DEFAULT_HELPER_MODEL = "openai/gpt-5.4-mini"
DEFAULT_HELPER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_HELPER_API_KEY_ENV_VAR = "OPENROUTER_API_KEY"
DEFAULT_HELPER_HTTP_REFERER = "https://local.codex"
DEFAULT_HELPER_X_TITLE = "diplomacy-curriculum-bench"


def load_datum(environment: EnvironmentKind, seed: int) -> dict[str, Any]:
    if environment == "tool_accuracy":
        return build_tool_accuracy_dataset(num_sessions=1, seed=seed).to_list()[0]
    if environment == "target_execution":
        return build_target_execution_dataset(num_sessions=1, seed=seed).to_list()[0]
    if environment == "supported_target":
        return build_supported_target_dataset(num_sessions=1, seed=seed).to_list()[0]
    if environment == "cooperative_press":
        return build_cooperative_press_dataset(num_sessions=1, seed=seed).to_list()[0]
    return build_full_press_dataset(num_sessions=1, seed=seed).to_list()[0]


def build_actor_runtime(
    *,
    actor_max_turns: int,
    session_timeout_seconds: float,
    background_actor_model: str = DEFAULT_HELPER_MODEL,
    background_actor_base_url: str = DEFAULT_HELPER_BASE_URL,
    background_actor_api_key_env_var: str = DEFAULT_HELPER_API_KEY_ENV_VAR,
    http_referer: str = DEFAULT_HELPER_HTTP_REFERER,
    x_title: str = DEFAULT_HELPER_X_TITLE,
) -> ActorRuntimeConfig:
    return ActorRuntimeConfig(
        actor_configs=build_actor_configs(
            base_url=background_actor_base_url,
            api_key_env_var=background_actor_api_key_env_var,
            model=background_actor_model,
        ),
        actor_max_turns=actor_max_turns,
        session_timeout_seconds=session_timeout_seconds,
        default_idle_sleep_seconds=0.5,
        openrouter_headers=OpenRouterHeaders(
            http_referer=http_referer,
            x_title=x_title,
        ),
    )


def build_runtime_policy(max_turns: int) -> RuntimePolicyConfig:
    return RuntimePolicyConfig(max_turns=max_turns)


def build_sampling_ref(
    *,
    model_name: str,
    checkpoint_path: str | None = None,
    base_url: str | None = None,
) -> SamplingRef:
    if checkpoint_path:
        return SamplingRef(
            sampler_path=checkpoint_path,
            base_model=model_name,
            base_url=base_url,
        )
    return SamplingRef(base_model=model_name, base_url=base_url)


def extract_tool_names(action_text: str) -> list[str]:
    if not action_text:
        return []
    xml_names = TOOL_PATTERN.findall(action_text)
    if xml_names:
        return xml_names
    names: list[str] = []
    for tool_call in re.findall(r"<tool_call>(.*?)</tool_call>", action_text, flags=re.DOTALL):
        names.extend(JSON_TOOL_PATTERN.findall(tool_call))
    return names


def render_observation_excerpt(model_input: Any, tokenizer: Any) -> str:
    if not hasattr(model_input, "model_dump"):
        return str(model_input)
    payload = model_input.model_dump(mode="python")
    chunks = payload.get("chunks", [])
    rendered_chunks: list[str] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            rendered_chunks.append(str(chunk))
            continue
        tokens = chunk.get("tokens")
        if isinstance(tokens, list) and tokens:
            try:
                rendered_chunks.append(tokenizer.decode(tokens, skip_special_tokens=False))
                continue
            except Exception:
                pass
        if "text" in chunk:
            rendered_chunks.append(str(chunk["text"]))
            continue
        if "content" in chunk:
            rendered_chunks.append(str(chunk["content"]))
            continue
        rendered_chunks.append(json.dumps(chunk, ensure_ascii=False))
    text = "".join(rendered_chunks).strip()
    if not text:
        return json.dumps(payload, ensure_ascii=False)
    return text[-2000:]


def infer_stop_condition(
    *,
    turn_records: Sequence[TurnRecord],
    merged_metrics: dict[str, float],
    max_turns: int,
) -> str | None:
    if merged_metrics.get("tool/finish_called", 0.0) > 0.0:
        return "finish_called"
    if merged_metrics.get("episode/max_turns_reached", 0.0) > 0.0:
        return "max_turns_reached"
    if turn_records and turn_records[-1].episode_done:
        return "episode_done"
    if len(turn_records) >= max_turns:
        return "max_turns_reached"
    return None


def extract_objective_metrics(
    *,
    merged_metrics: dict[str, float],
    environment: EnvironmentKind,
) -> tuple[float, float, float]:
    if environment in {"supported_target", "cooperative_press", "full_press"}:
        return (
            float(merged_metrics.get("rubric/full_press_gate_metric", 0.0)),
            float(merged_metrics.get("rubric/transition_target_satisfied_metric", 0.0)),
            float(merged_metrics.get("rubric/relevant_actor_submission_metric", 0.0)),
        )
    if environment == "target_execution":
        transition = float(merged_metrics.get("rubric/transition_target_without_press_metric", 0.0))
        complete_legal = float(merged_metrics.get("rubric/complete_legal_submission_metric", 0.0))
        return transition, transition, complete_legal
    constraints = float(merged_metrics.get("rubric/constraints_satisfied_metric", 0.0))
    complete_legal = float(merged_metrics.get("rubric/complete_legal_submission_metric", 0.0))
    send_recall = float(merged_metrics.get("rubric/required_send_recall_metric", 0.0))
    read_recall = float(merged_metrics.get("rubric/required_read_recall_metric", 0.0))
    invalid_tool_budget = float(merged_metrics.get("rubric/invalid_tool_budget_pass_metric", 0.0))
    gate = min(constraints, complete_legal, send_recall, read_recall, invalid_tool_budget)
    transition_target = complete_legal
    relevant_submission = send_recall
    return gate, transition_target, relevant_submission


def seed_result_from_tinker_result(
    *,
    environment: EnvironmentKind,
    seed: int,
    result: Any,
    tokenizer: Any,
    max_turns: int,
    wall_time_seconds: float,
    artifact_path: Path,
) -> SeedResult:
    if result.failure_kind is not None or result.trajectory is None:
        row = SeedResult(
            seed=seed,
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
            max_turns=max_turns,
            wall_time_seconds=wall_time_seconds,
            failure_kind=result.failure_kind,
            failure_message=result.failure_message,
            artifact_path=str(artifact_path),
        )
        return replace(row, dominant_failure=classify_seed_result(row))

    trajectory = deserialize_trajectory(result.trajectory)
    merged_metrics: dict[str, float] = {}
    turn_records: list[TurnRecord] = []
    decoded_actions: list[str] = []
    observation_snippets: list[str] = []
    prompt_tokens = 0
    completion_tokens = 0
    for index, transition in enumerate(trajectory.transitions):
        prompt_tokens += int(getattr(transition.ob, "length", 0) or 0)
        completion_tokens += len(transition.ac.tokens)
        action_text = tokenizer.decode(transition.ac.tokens, skip_special_tokens=False)
        decoded_actions.append(action_text)
        tool_names = extract_tool_names(action_text)
        observation_excerpt = render_observation_excerpt(transition.ob, tokenizer)[:500]
        observation_snippets.append(observation_excerpt.lower())
        metrics = {
            key: float(value)
            for key, value in transition.metrics.items()
            if isinstance(value, (int, float))
        }
        merged_metrics.update(metrics)
        turn_records.append(
            TurnRecord(
                turn_index=index,
                tools=tool_names,
                action_text=action_text,
                observation_excerpt=observation_excerpt,
                reward=float(transition.reward),
                episode_done=bool(transition.episode_done),
                metrics=metrics,
            )
        )

    merged_metrics.update(
        {
            key: float(value)
            for key, value in result.metrics.items()
            if isinstance(value, (int, float))
        }
    )
    reward = sum(float(transition.reward) for transition in trajectory.transitions) + float(
        result.final_reward
    )
    wait_count = sum(record.tools.count("wait") for record in turn_records)
    read_conversation_count = sum(
        record.tools.count("read_conversation") for record in turn_records
    )
    compact_order_error = any(
        "does not belong to" in snippet or "invalid order" in snippet for snippet in observation_snippets
    )
    has_think_close = any("</think>" in action_text for action_text in decoded_actions)
    stop_condition = infer_stop_condition(
        turn_records=turn_records,
        merged_metrics=merged_metrics,
        max_turns=max_turns,
    )
    gate, transition_target, relevant_submission = extract_objective_metrics(
        merged_metrics=merged_metrics,
        environment=environment,
    )
    rejected_tool_calls = float(
        merged_metrics.get(
            "rubric/rejected_tool_call_count_metric",
            merged_metrics.get("rollout/rejected_tool_calls", 0.0),
        )
    )
    score = (
        reward
        + 0.10 * transition_target
        + 0.05 * relevant_submission
        - 0.05 * rejected_tool_calls
        - 0.02 * max(0, wait_count - 1)
        - (0.05 if compact_order_error else 0.0)
        - (0.05 if has_think_close and environment in {"supported_target", "cooperative_press", "full_press"} else 0.0)
    )
    row = SeedResult(
        seed=seed,
        status="ok",
        execution_backend="tinker_modal",
        score=score,
        reward=reward,
        gate=gate,
        transition_target=transition_target,
        relevant_submission=relevant_submission,
        rejected_tool_calls=rejected_tool_calls,
        wait_count=wait_count,
        read_conversation_count=read_conversation_count,
        compact_order_error=compact_order_error,
        has_think_close=has_think_close,
        turns=len(turn_records),
        max_turns=max_turns,
        wall_time_seconds=wall_time_seconds,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        stop_condition=stop_condition,
        last_metrics=merged_metrics,
        turn_records=turn_records,
        artifact_path=str(artifact_path),
    )
    return replace(row, dominant_failure=classify_seed_result(row))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
