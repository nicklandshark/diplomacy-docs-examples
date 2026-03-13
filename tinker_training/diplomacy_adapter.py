from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence, cast

import chz
import verifiers as vf
from openai import AsyncOpenAI

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDORED_COOKBOOK_ROOT = REPO_ROOT / "vendor" / "tinker-cookbook"
if str(VENDORED_COOKBOOK_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDORED_COOKBOOK_ROOT))

from tinker_cookbook import model_info, renderers, tokenizer_utils
from tinker_cookbook.renderers.base import Message, Renderer
from tinker_cookbook.rl.message_env import EnvFromMessageEnv, MessageEnv, MessageStepResult
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder, SamplingRef, TrajectoryGroup
from tinker_cookbook.tool_use import simple_tool_result, tool
from tinker_cookbook.tool_use.tools import handle_tool_call
from tinker_cookbook.tool_use.types import Tool, ToolInput, ToolResult

import environments.full_press as full_press_env
import environments.tool_accuracy as tool_accuracy_env
import rlm_backend as backend
from data_generator import (
    STANDARD_POWERS,
    build_full_press_dataset,
    build_tool_accuracy_dataset,
)
from rlm_chatroom_backend import (
    DEFAULT_CONVERSATION_RENDER_LINE_CAP,
    DEFAULT_MAX_MESSAGE_LENGTH,
    _append_chatroom_tool_trace,
    _channel_type_for_participants,
    _ensure_chatroom_tool_trace,
    _ensure_list_of_strings,
    _format_time_ago,
    _render_message_text,
)
from tinker_training.rollout_backends import (
    LocalTrajectorySandboxRunner,
    TrajectoryRolloutRequest,
    deserialize_trajectory,
    get_trajectory_runner,
)

logger = logging.getLogger(__name__)

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_HTTP_REFERER = "https://local.codex"
DEFAULT_OPENROUTER_X_TITLE = "diplomacy-grpo"

EnvironmentKind = Literal["tool_accuracy", "full_press"]


@chz.chz
class OpenRouterHeaders:
    http_referer: str = DEFAULT_OPENROUTER_HTTP_REFERER
    x_title: str = DEFAULT_OPENROUTER_X_TITLE


@chz.chz
class ActorRuntimeConfig:
    actor_configs: dict[str, backend.ActorConfig]
    actor_max_turns: int
    session_timeout_seconds: float
    default_idle_sleep_seconds: float
    openrouter_headers: OpenRouterHeaders


@chz.chz
class RuntimePolicyConfig:
    max_turns: int
    max_message_length: int = DEFAULT_MAX_MESSAGE_LENGTH
    conversation_render_line_cap: int = DEFAULT_CONVERSATION_RENDER_LINE_CAP
    failed_parse_reward: float = -0.25
    max_trajectory_tokens: int | None = None


@dataclass(frozen=True)
class EpisodeContext:
    datum: dict[str, Any]
    state: dict[str, Any]
    runtime: backend.DiplomacySessionRuntime
    tool_executor: "DiplomacyToolExecutor"


@dataclass(frozen=True)
class BuiltDiplomacyEnv:
    env: Env
    state: dict[str, Any]


def _null_async_context():
    @contextlib.asynccontextmanager
    async def _manager():
        yield

    return _manager()


def get_default_renderer_name(model_name: str, *, disable_thinking: bool) -> str:
    recommended = model_info.get_recommended_renderer_name(model_name)
    if not disable_thinking:
        return recommended

    disable_map = {
        "qwen3": "qwen3_disable_thinking",
        "qwen3_5": "qwen3_5_disable_thinking",
        "kimi_k25": "kimi_k25_disable_thinking",
    }
    return disable_map.get(recommended, recommended)


def build_actor_configs(
    *,
    base_url: str = DEFAULT_OPENROUTER_BASE_URL,
    api_key_env_var: str = "OPENROUTER_API_KEY",
    model: str = "google/gemini-3-flash-preview",
    timeout: float = 180.0,
    sampling_args: dict[str, Any] | None = None,
) -> dict[str, backend.ActorConfig]:
    sampling_args = {} if sampling_args is None else dict(sampling_args)
    return {
        power: backend.ActorConfig(
            api_base_url=base_url,
            api_key_var=api_key_env_var,
            model=model,
            timeout=timeout,
            sampling_args=sampling_args,
        )
        for power in STANDARD_POWERS
    }


def _make_actor_client(
    actor_config: backend.ActorConfig,
    headers: OpenRouterHeaders,
) -> AsyncOpenAI:
    api_key = os.environ.get(actor_config.api_key_var, "").strip()
    if not api_key:
        raise ValueError(
            f"Environment variable {actor_config.api_key_var} is not set for actor model access."
        )
    return AsyncOpenAI(
        api_key=api_key,
        base_url=actor_config.api_base_url,
        timeout=actor_config.timeout,
        default_headers={
            "HTTP-Referer": headers.http_referer,
            "X-Title": headers.x_title,
        },
    )


def _get_rubric_for_environment(environment_kind: EnvironmentKind) -> vf.Rubric:
    if environment_kind == "tool_accuracy":
        return tool_accuracy_env.build_rubric()
    return full_press_env.build_rubric()


def _build_system_prompt(
    *,
    agent_name: str,
    is_background_actor: bool,
    default_idle_sleep_seconds: float,
) -> str:
    role_line = (
        "You are a hidden counterpart policy in the Diplomacy environment."
        if is_background_actor
        else "You are the trainable tracked policy in the Diplomacy environment."
    )
    return "\n".join(
        [
            role_line,
            f"Your power: {agent_name}",
            "Use tools directly. Do not describe intended tool use in prose.",
            "Recommended loop:",
            "1. Read phase status and inbox notifications.",
            "2. Read the relevant conversation or legal orders before acting.",
            "3. Send messages or submit orders when you have enough information.",
            f"4. Use wait(seconds) when you expect a reply; the default poll interval is {default_idle_sleep_seconds:.2f}s.",
            '5. Call finish(summary="DONE") only when your work for this phase is complete.',
        ]
    )


def _copy_prompt_messages(prompt: list[dict[str, Any]]) -> list[Message]:
    copied: list[Message] = []
    for message in prompt:
        copied.append(
            cast(
                Message,
                {
                    key: value
                    for key, value in message.items()
                    if key in {"role", "content", "tool_calls", "name"}
                },
            )
        )
    return copied


class DiplomacyToolExecutor:
    def __init__(
        self,
        *,
        state: dict[str, Any],
        runtime: backend.DiplomacySessionRuntime,
        agent_name: str,
        controls_episode: bool,
        record_tool_trace: bool,
        max_message_length: int,
        default_idle_sleep_seconds: float,
    ) -> None:
        self.state = state
        self.runtime = runtime
        self.agent_name = agent_name
        self.controls_episode = controls_episode
        self.record_tool_trace = record_tool_trace
        self.max_message_length = max_message_length
        self.default_idle_sleep_seconds = default_idle_sleep_seconds

    def all_tools(self) -> list[Tool]:
        return [
            self.read_phase_status,
            self.read_inbox_notifications,
            self.read_conversation,
            self.send_message,
            self.read_legal_orders,
            self.submit_orders,
            self.wait,
            self.finish,
        ]

    def _contacts_for_actor(self) -> list[str]:
        info = self.state.get("info", {})
        if not isinstance(info, dict):
            return []
        contacts_map = info.get("contacts_map", {})
        if isinstance(contacts_map, dict):
            contacts = contacts_map.get(self.agent_name, [])
            if isinstance(contacts, list) and all(isinstance(item, str) for item in contacts):
                return list(contacts)
        contacts = info.get("contacts", [])
        if isinstance(contacts, list) and all(isinstance(item, str) for item in contacts):
            return list(contacts)
        return []

    def _append_trace(
        self,
        *,
        tool_name: str,
        participants: list[str] | None = None,
        channel_type: Literal["dm", "group", "system"] = "system",
        success: bool,
        error: str | None = None,
        has_unread_notifications: bool | None = None,
        notification_count: int | None = None,
        total_messages: int | None = None,
        offset: int | None = None,
        limit: int | None = None,
        result_preview: str | None = None,
    ) -> None:
        if not self.record_tool_trace:
            return
        _append_chatroom_tool_trace(
            self.state,
            tool_name=tool_name,
            agent_name=self.agent_name,
            participants=participants,
            channel_type=channel_type,
            success=success,
            error=error,
            has_unread_notifications=has_unread_notifications,
            notification_count=notification_count,
            total_messages=total_messages,
            offset=offset,
            limit=limit,
            result_preview=result_preview,
        )

    def _reject_tool_call(self) -> None:
        if not self.record_tool_trace:
            return
        backend._increment_rejected_tool_call(self.state, self.runtime, self.agent_name)

    def _validate_participants(
        self,
        participants: Any,
        *,
        require_non_empty: bool,
    ) -> tuple[list[str] | None, str | None]:
        participants_list = _ensure_list_of_strings(participants)
        if participants_list is None:
            return None, "Error: participants must be a list of strings."

        contacts = self._contacts_for_actor()
        cleaned: list[str] = []
        seen: set[str] = set()
        for participant in participants_list:
            if participant == self.agent_name:
                return None, "Error: do not include yourself in participants."
            if participant in seen:
                continue
            seen.add(participant)
            cleaned.append(participant)

        if require_non_empty and not cleaned:
            return None, "Error: participants list cannot be empty."

        for participant in cleaned:
            if participant not in contacts:
                return None, f"Error: participant '{participant}' is not in your contacts."
        return sorted(cleaned), None

    @tool
    def read_inbox_notifications(self) -> ToolResult:
        """Return summaries of unread messages and mark them as read."""
        rows = self.runtime.db.consume_unread_notifications(self.agent_name)
        if not rows:
            result = "no new notifications"
            self._append_trace(
                tool_name="read_inbox_notifications",
                success=True,
                channel_type="system",
                has_unread_notifications=False,
                notification_count=0,
                result_preview=result,
            )
            return simple_tool_result(result)

        total_count = len(rows)
        grouped: dict[str, dict[str, Any]] = {}
        now = time.time()
        for row in rows:
            channel_id = str(row["channel_id"])
            participants = json.loads(str(row["participants"]))
            entry = grouped.setdefault(
                channel_id,
                {
                    "count": 0,
                    "participants": participants,
                    "last_sender": "",
                    "last_sent_at": 0.0,
                },
            )
            entry["count"] += 1
            sent_at = float(row["sent_at"])
            if sent_at >= entry["last_sent_at"]:
                entry["last_sent_at"] = sent_at
                entry["last_sender"] = str(row["sender"])

        lines = [f"you have received {total_count} new message(s)"]
        ordered = sorted(grouped.values(), key=lambda item: float(item["last_sent_at"]), reverse=True)
        for item in ordered:
            participants = sorted(cast(list[str], item["participants"]))
            others = [name for name in participants if name != self.agent_name]
            count = int(item["count"])
            last_sender = str(item["last_sender"])
            last_sent = float(item["last_sent_at"])
            time_ago = _format_time_ago(last_sent, now=now)
            if len(participants) == 2 and len(others) == 1:
                lines.append(f"from: {others[0]} ({count}) last sent {time_ago}")
            else:
                display = ["you" if name == self.agent_name else name for name in participants]
                lines.append(
                    f"from: [{', '.join(display)}] ({count}) last message sent by {last_sender} {time_ago}"
                )
        result = "\n".join(lines)
        self._append_trace(
            tool_name="read_inbox_notifications",
            success=True,
            channel_type="system",
            has_unread_notifications=True,
            notification_count=total_count,
            result_preview=result[:200],
        )
        return simple_tool_result(result)

    @tool
    def read_phase_status(self) -> ToolResult:
        """Return current phase status, public board summary, orderable locations, and current submission."""
        if self.runtime.game is None:
            raise RuntimeError("game runtime is unavailable")

        orderable_locations = list(self.runtime.game.get_orderable_locations(self.agent_name))
        submission = self.runtime.canonical_submissions.get(self.agent_name, {})
        lines = [
            backend._stable_board_summary(self.runtime.game),
            "",
            f"orderable locations for {self.agent_name}: {json.dumps(orderable_locations)}",
            "current submission:",
        ]
        if submission:
            lines.extend(f"- {order}" for order in backend._submission_for_display(submission))
        else:
            lines.append("- (none)")
        result = "\n".join(lines)
        self._append_trace(
            tool_name="read_phase_status",
            success=True,
            channel_type="system",
            result_preview=result[:200],
        )
        return simple_tool_result(result)

    @tool
    def read_conversation(
        self,
        participants: list[str],
        offset: int | None = None,
        limit: int | None = None,
    ) -> ToolResult:
        """Read the conversation with the specified participants."""
        validated, err = self._validate_participants(participants, require_non_empty=True)
        if err:
            self._reject_tool_call()
            self._append_trace(
                tool_name="read_conversation",
                participants=participants if isinstance(participants, list) else [],
                channel_type=(
                    _channel_type_for_participants(participants)
                    if isinstance(participants, list) and participants
                    else "system"
                ),
                success=False,
                error=err,
                offset=offset if isinstance(offset, int) else None,
                limit=limit if isinstance(limit, int) else None,
                result_preview=err,
            )
            return simple_tool_result(err)
        assert validated is not None

        if (offset is None) != (limit is None):
            error = "Error: offset and limit must be provided together."
            self._reject_tool_call()
            self._append_trace(
                tool_name="read_conversation",
                participants=validated,
                channel_type=_channel_type_for_participants(validated),
                success=False,
                error=error,
                offset=offset if isinstance(offset, int) else None,
                limit=limit if isinstance(limit, int) else None,
                result_preview=error,
            )
            return simple_tool_result(error)

        if offset is not None and (not isinstance(offset, int) or offset < 0):
            error = "Error: offset must be a non-negative integer."
            self._reject_tool_call()
            self._append_trace(
                tool_name="read_conversation",
                participants=validated,
                channel_type=_channel_type_for_participants(validated),
                success=False,
                error=error,
                limit=limit if isinstance(limit, int) else None,
                result_preview=error,
            )
            return simple_tool_result(error)

        if limit is not None and (not isinstance(limit, int) or limit <= 0):
            error = "Error: limit must be a positive integer."
            self._reject_tool_call()
            self._append_trace(
                tool_name="read_conversation",
                participants=validated,
                channel_type=_channel_type_for_participants(validated),
                success=False,
                error=error,
                offset=offset if isinstance(offset, int) else None,
                result_preview=error,
            )
            return simple_tool_result(error)

        all_participants = sorted([self.agent_name, *validated])
        (
            _channel_id,
            canonical,
            resolved_offset,
            resolved_limit,
            total_messages,
            rendered_line_count,
            rows,
            error_type,
        ) = self.runtime.db.list_messages(participants=all_participants, offset=offset, limit=limit)

        if error_type == "not_found":
            error = f"No conversation found with [{', '.join(validated)}]."
            self._append_trace(
                tool_name="read_conversation",
                participants=validated,
                channel_type=_channel_type_for_participants(validated),
                success=False,
                error=error,
                offset=offset,
                limit=limit,
                result_preview=error,
            )
            return simple_tool_result(error)

        if error_type == "render_line_cap_exceeded":
            error = (
                f"Error: Conversation renders to {rendered_line_count} lines. "
                "Use offset and limit to read it."
            )
            self._append_trace(
                tool_name="read_conversation",
                participants=validated,
                channel_type=_channel_type_for_participants(validated),
                success=False,
                error=error,
                total_messages=total_messages,
                offset=offset,
                limit=limit,
                result_preview=error[:200],
            )
            return simple_tool_result(error)

        assert canonical is not None
        if total_messages == 0 or len(rows) == 0:
            range_summary = f"messages none of {total_messages}"
        else:
            range_summary = f"messages {resolved_offset + 1}-{resolved_offset + len(rows)} of {total_messages}"
        lines = [
            f"[channel: {' <-> '.join(canonical)} -- offset {resolved_offset}, limit {resolved_limit}, {range_summary}]",
            "",
        ]
        for row in rows:
            lines.append(
                _render_message_text(
                    sender=str(row["sender"]),
                    content=str(row["content"]),
                    sent_at=float(row["sent_at"]),
                    actor=self.agent_name,
                )
            )
        if len(lines) == 2:
            lines.append("(no messages)")
        result = "\n".join(lines)
        self._append_trace(
            tool_name="read_conversation",
            participants=validated,
            channel_type=_channel_type_for_participants(validated),
            success=True,
            total_messages=total_messages,
            offset=resolved_offset,
            limit=resolved_limit,
            result_preview=result[:200],
        )
        return simple_tool_result(result)

    @tool
    def send_message(self, participants: list[str], message: str) -> ToolResult:
        """Send a message to one or more other powers."""
        validated, err = self._validate_participants(participants, require_non_empty=True)
        if err:
            self._reject_tool_call()
            self._append_trace(
                tool_name="send_message",
                participants=participants if isinstance(participants, list) else [],
                channel_type=(
                    _channel_type_for_participants(participants)
                    if isinstance(participants, list) and participants
                    else "system"
                ),
                success=False,
                error=err,
                result_preview=err,
            )
            return simple_tool_result(err)
        assert validated is not None

        if not isinstance(message, str) or not message.strip():
            error = "Error: message cannot be empty."
            self._reject_tool_call()
            self._append_trace(
                tool_name="send_message",
                participants=validated,
                channel_type=_channel_type_for_participants(validated),
                success=False,
                error=error,
                result_preview=error,
            )
            return simple_tool_result(error)

        if len(message) > self.max_message_length:
            error = f"Error: message exceeds {self.max_message_length} character limit."
            self._reject_tool_call()
            self._append_trace(
                tool_name="send_message",
                participants=validated,
                channel_type=_channel_type_for_participants(validated),
                success=False,
                error=error,
                result_preview=error,
            )
            return simple_tool_result(error)

        self.runtime.db.insert_message(
            sender=self.agent_name,
            participants=sorted([self.agent_name, *validated]),
            content=message,
            sent_at=time.time(),
        )
        result = f"Message sent to [{', '.join(validated)}] channel."
        self._append_trace(
            tool_name="send_message",
            participants=validated,
            channel_type=_channel_type_for_participants(validated),
            success=True,
            result_preview=result,
        )
        return simple_tool_result(result)

    @tool
    def read_legal_orders(self) -> ToolResult:
        """List all legal orders for your current power."""
        if self.runtime.game is None:
            return simple_tool_result("Error: game runtime is unavailable.")
        orderable_locations = list(self.runtime.game.get_orderable_locations(self.agent_name))
        legal_orders = self.runtime.game.get_all_possible_orders()
        lines = [f"[legal orders for {self.agent_name} -- phase {self.runtime.game.get_current_phase()}]"]
        if not orderable_locations:
            lines.append("(no orderable locations)")
        for location in orderable_locations:
            lines.append(f"{location}:")
            for order in cast(list[str], legal_orders.get(location, [])):
                lines.append(f"- {order}")
        result = "\n".join(lines)
        self._append_trace(
            tool_name="read_legal_orders",
            success=True,
            channel_type="system",
            result_preview=result[:200],
        )
        return simple_tool_result(result)

    @tool
    def submit_orders(self, orders: list[str]) -> ToolResult:
        """Submit a complete legal order set for your current power."""
        canonical_map, error = backend._canonicalize_submission(
            self.runtime,
            power=self.agent_name,
            orders=orders,
        )
        if error:
            self._reject_tool_call()
            self._append_trace(
                tool_name="submit_orders",
                success=False,
                channel_type="system",
                error=error,
                result_preview=error,
            )
            return simple_tool_result(error)
        assert canonical_map is not None
        with self.runtime.lock:
            self.runtime.canonical_submissions[self.agent_name] = canonical_map
            self.runtime.current_submissions[self.agent_name] = backend._submission_for_display(
                canonical_map
            )
        result = "Submission accepted:\n" + "\n".join(
            f"- {order}" for order in self.runtime.current_submissions[self.agent_name]
        )
        self._append_trace(
            tool_name="submit_orders",
            success=True,
            channel_type="system",
            result_preview=result[:200],
        )
        return simple_tool_result(result)

    @tool
    async def wait(self, seconds: float | None = None) -> ToolResult:
        """Sleep briefly while waiting for counterpart actors to respond."""
        requested = self.default_idle_sleep_seconds if seconds is None else float(seconds)
        sleep_for = min(max(requested, 0.0), 10.0)
        await asyncio.sleep(sleep_for)
        result = f"Waited {sleep_for:.2f} seconds."
        self._append_trace(
            tool_name="wait",
            success=True,
            channel_type="system",
            result_preview=result,
        )
        return simple_tool_result(result)

    @tool
    def finish(self, summary: str = "DONE") -> ToolResult:
        """Mark your turn as complete and stop the episode."""
        if self.controls_episode:
            self.state["final_answer"] = summary
            self.state["stop_condition"] = "finish_called"
        self._append_trace(
            tool_name="finish",
            success=True,
            channel_type="system",
            result_preview=summary[:200],
        )
        return simple_tool_result(
            f"Episode marked complete with summary: {summary}",
            should_stop=True,
            metrics={"finish_called": 1.0},
        )


def _tool_to_openai_schema(tool_obj: Tool) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool_obj.name,
            "description": tool_obj.description,
            "parameters": tool_obj.parameters_schema,
        },
    }


async def _run_openrouter_actor(
    *,
    state: dict[str, Any],
    runtime: backend.DiplomacySessionRuntime,
    actor_name: str,
    actor_prompt: str,
    actor_config: backend.ActorConfig,
    actor_runtime: ActorRuntimeConfig,
    max_message_length: int,
) -> str:
    tool_executor = DiplomacyToolExecutor(
        state=state,
        runtime=runtime,
        agent_name=actor_name,
        controls_episode=False,
        record_tool_trace=False,
        max_message_length=max_message_length,
        default_idle_sleep_seconds=actor_runtime.default_idle_sleep_seconds,
    )
    tools = tool_executor.all_tools()
    tool_map = {tool_obj.name: tool_obj for tool_obj in tools}
    tool_schemas = [_tool_to_openai_schema(tool_obj) for tool_obj in tools]
    history: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": _build_system_prompt(
                agent_name=actor_name,
                is_background_actor=True,
                default_idle_sleep_seconds=actor_runtime.default_idle_sleep_seconds,
            ),
        },
        {"role": "user", "content": actor_prompt},
    ]
    client = _make_actor_client(actor_config, actor_runtime.openrouter_headers)
    try:
        for _ in range(actor_runtime.actor_max_turns):
            response = await client.chat.completions.create(
                model=actor_config.model,
                messages=history,
                tools=tool_schemas,
                **actor_config.sampling_args,
            )
            choice = response.choices[0]
            assistant_message = choice.message.model_dump(exclude_none=True)
            history.append(assistant_message)

            tool_calls = assistant_message.get("tool_calls", [])
            if not tool_calls:
                return "completed"

            should_stop = False
            for tool_call in tool_calls:
                tool_name = tool_call.get("function", {}).get("name", "")
                tool_obj = tool_map.get(tool_name)
                if tool_obj is None:
                    history.append(
                        {
                            "role": "tool",
                            "content": f"Unknown tool: {tool_name}",
                            "tool_call_id": tool_call.get("id", ""),
                            "name": tool_name,
                        }
                    )
                    continue
                try:
                    arguments = json.loads(tool_call.get("function", {}).get("arguments", "") or "{}")
                    if not isinstance(arguments, dict):
                        raise ValueError("Tool arguments must decode to an object.")
                except Exception as exc:
                    history.append(
                        {
                            "role": "tool",
                            "content": f"Error: {exc}",
                            "tool_call_id": tool_call.get("id", ""),
                            "name": tool_name,
                        }
                    )
                    continue

                tool_result = await tool_obj.run(
                    ToolInput(arguments=arguments, call_id=tool_call.get("id", ""))
                )
                history.extend(cast(list[dict[str, Any]], tool_result.messages))
                should_stop = should_stop or tool_result.should_stop
            if should_stop:
                return "completed"
        return "max_turns"
    finally:
        with contextlib.suppress(Exception):
            await client.close()


async def _run_actor_task(
    *,
    state: dict[str, Any],
    runtime: backend.DiplomacySessionRuntime,
    actor_name: str,
    actor_runtime: ActorRuntimeConfig,
    max_message_length: int,
) -> None:
    actor_statuses = state.get("actor_statuses", {})
    if not isinstance(actor_statuses, dict):
        actor_statuses = {}
        state["actor_statuses"] = actor_statuses

    actor_config = actor_runtime.actor_configs.get(actor_name)
    if actor_config is None:
        actor_statuses[actor_name] = "error:missing_actor_config"
        state["diplomacy_actor_error_count"] = int(state.get("diplomacy_actor_error_count", 0)) + 1
        return

    info = state.get("info", {})
    if not isinstance(info, dict):
        actor_statuses[actor_name] = "error:invalid_info"
        state["diplomacy_actor_error_count"] = int(state.get("diplomacy_actor_error_count", 0)) + 1
        return
    actor_prompt = str(cast(dict[str, Any], info.get("actor_prompts", {})).get(actor_name, "")).strip()
    if not actor_prompt:
        actor_statuses[actor_name] = "completed"
        return

    actor_statuses[actor_name] = "running"
    try:
        stop_condition = await asyncio.wait_for(
            _run_openrouter_actor(
                state=state,
                runtime=runtime,
                actor_name=actor_name,
                actor_prompt=actor_prompt,
                actor_config=actor_config,
                actor_runtime=actor_runtime,
                max_message_length=max_message_length,
            ),
            timeout=max(actor_runtime.session_timeout_seconds + 5.0, 5.0),
        )
    except asyncio.TimeoutError:
        actor_statuses[actor_name] = "timeout"
        state["diplomacy_actor_timeout_count"] = int(state.get("diplomacy_actor_timeout_count", 0)) + 1
    except asyncio.CancelledError:
        actor_statuses[actor_name] = "cancelled"
        raise
    except Exception as exc:
        actor_statuses[actor_name] = f"error:{type(exc).__name__}"
        state["diplomacy_actor_error_count"] = int(state.get("diplomacy_actor_error_count", 0)) + 1
        logger.exception("Background actor %s failed", actor_name)
    else:
        actor_statuses[actor_name] = stop_condition


def _initialize_episode_context(
    *,
    datum: dict[str, Any],
    actor_runtime: ActorRuntimeConfig,
    policy_config: RuntimePolicyConfig,
) -> EpisodeContext:
    info = dict(cast(dict[str, Any], datum["info"]))
    session_id = str(info.get("session_id", ""))
    if not session_id:
        raise ValueError("datum.info.session_id is required")

    runtime = backend._SESSION_REGISTRY.acquire(
        session_id,
        conversation_render_line_cap=policy_config.conversation_render_line_cap,
    )
    scoring_hold = backend._SESSION_REGISTRY.acquire(
        session_id,
        conversation_render_line_cap=policy_config.conversation_render_line_cap,
    )
    deadline = time.time() + actor_runtime.session_timeout_seconds
    info["session_deadline_epoch"] = deadline
    info["default_idle_sleep_seconds"] = actor_runtime.default_idle_sleep_seconds

    state: dict[str, Any] = {
        "prompt": datum["prompt"],
        "answer": datum["answer"],
        "task": datum["task"],
        "example_id": datum["example_id"],
        "info": info,
        "reward": None,
        "metrics": None,
        "error": None,
        "final_answer": "",
        "stop_condition": None,
        "timing": {
            "generation_ms": 0.0,
            "scoring_ms": 0.0,
            "total_ms": 0.0,
            "start_time": time.time(),
        },
        "diplomacy_session_id": session_id,
        "diplomacy_scoring_hold_session_id": scoring_hold.session_id,
        "_diplomacy_scoring_hold_active": True,
        "_diplomacy_primary_ref_released": False,
        "session_deadline_epoch": deadline,
        "actor_statuses": {
            name: "scheduled"
            for name in [
                str(item)
                for item in info.get("relevant_powers", [])
                if isinstance(item, str)
            ]
        },
        "diplomacy_actor_timeout_count": 0,
        "diplomacy_actor_error_count": 0,
        "rejected_tool_calls": 0,
    }
    _ensure_chatroom_tool_trace(state)
    backend._seed_runtime(runtime, cast(backend.DiplomacyRowInfo, info))

    actor_tasks = {
        actor_name: asyncio.create_task(
            _run_actor_task(
                state=state,
                runtime=runtime,
                actor_name=actor_name,
                actor_runtime=actor_runtime,
                max_message_length=policy_config.max_message_length,
            ),
            name=f"diplomacy-actor-{actor_name}",
        )
        for actor_name in state["actor_statuses"].keys()
    }
    state["diplomacy_actor_tasks"] = actor_tasks

    tracked_power = str(info.get("tracked_power", info.get("agent_name", "")))
    tool_executor = DiplomacyToolExecutor(
        state=state,
        runtime=runtime,
        agent_name=tracked_power,
        controls_episode=True,
        record_tool_trace=True,
        max_message_length=policy_config.max_message_length,
        default_idle_sleep_seconds=actor_runtime.default_idle_sleep_seconds,
    )
    return EpisodeContext(datum=datum, state=state, runtime=runtime, tool_executor=tool_executor)


async def _cleanup_episode_state(state: dict[str, Any]) -> None:
    actor_tasks = state.get("diplomacy_actor_tasks", {})
    if isinstance(actor_tasks, dict):
        pending = [task for task in actor_tasks.values() if isinstance(task, asyncio.Task) and not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            with contextlib.suppress(Exception):
                await asyncio.gather(*pending, return_exceptions=True)

    session_id = state.get("diplomacy_session_id")
    if isinstance(session_id, str) and session_id and not state.get("_diplomacy_primary_ref_released"):
        backend._SESSION_REGISTRY.release(session_id)
        state["_diplomacy_primary_ref_released"] = True


class DiplomacyMessageEnv(MessageEnv):
    def __init__(
        self,
        *,
        episode_context: EpisodeContext,
        renderer: Renderer,
        environment_kind: EnvironmentKind,
        actor_runtime: ActorRuntimeConfig,
        policy_config: RuntimePolicyConfig,
    ) -> None:
        self.episode_context = episode_context
        self.renderer = renderer
        self.environment_kind = environment_kind
        self.actor_runtime = actor_runtime
        self.policy_config = policy_config
        self.history: list[Message] = []
        self.turn_count = 0
        self.cleaned_up = False
        self.tool_dict = {
            tool_obj.name: tool_obj for tool_obj in self.episode_context.tool_executor.all_tools()
        }
        tool_specs = [tool_obj.to_spec() for tool_obj in self.tool_dict.values()]
        info = self.episode_context.state.get("info", {})
        tracked_power = str(cast(dict[str, Any], info).get("tracked_power", cast(dict[str, Any], info).get("agent_name", "")))
        system_prompt = _build_system_prompt(
            agent_name=tracked_power,
            is_background_actor=False,
            default_idle_sleep_seconds=actor_runtime.default_idle_sleep_seconds,
        )
        self.initial_messages = (
            renderer.create_conversation_prefix_with_tools(tools=tool_specs, system_prompt=system_prompt)
            + _copy_prompt_messages(cast(list[dict[str, Any]], self.episode_context.state["prompt"]))
        )

    async def initial_observation(self) -> list[Message]:
        if not self.history:
            self.history = list(self.initial_messages)
        return self.history

    async def _score_episode(self) -> tuple[float, dict[str, float]]:
        state = self.episode_context.state
        rubric = _get_rubric_for_environment(self.environment_kind)
        if not state.get("completion"):
            state["completion"] = self.history[len(self.initial_messages) :]
        await rubric.score_rollout(state, score_sem=_null_async_context())
        reward = float(state.get("reward") or 0.0)
        metrics = {
            f"rubric/{key}": float(value)
            for key, value in cast(dict[str, float], state.get("metrics") or {}).items()
        }
        metrics["reward/final"] = reward
        return reward, metrics

    async def _cleanup_if_needed(self) -> None:
        if self.cleaned_up:
            return
        self.cleaned_up = True
        await _cleanup_episode_state(self.episode_context.state)

    async def step(self, message: Message) -> MessageStepResult:
        self.turn_count += 1
        step_metrics: dict[str, float] = {}
        self.history.append(message)

        tool_calls = list(message.get("tool_calls") or [])
        should_stop = False
        for tool_call in tool_calls:
            tool_result = await handle_tool_call(self.tool_dict, tool_call)
            self.history.extend(tool_result.messages)
            should_stop = should_stop or tool_result.should_stop
            for key, value in tool_result.metrics.items():
                step_metrics[f"tool/{key}"] = step_metrics.get(f"tool/{key}", 0.0) + float(value)

        done = should_stop or not tool_calls or self.turn_count >= self.policy_config.max_turns
        if self.turn_count >= self.policy_config.max_turns:
            step_metrics["episode/max_turns_reached"] = 1.0
            if not self.episode_context.state.get("stop_condition"):
                self.episode_context.state["stop_condition"] = "max_turns_reached"

        reward = 0.0
        if done:
            if not self.episode_context.state.get("final_answer"):
                text_content = renderers.get_text_content(message)
                if text_content.strip():
                    self.episode_context.state["final_answer"] = text_content.strip()
            try:
                reward, final_metrics = await self._score_episode()
                step_metrics.update(final_metrics)
            finally:
                await self._cleanup_if_needed()

        return MessageStepResult(
            reward=reward,
            episode_done=done,
            next_messages=self.history,
            metrics=step_metrics,
        )


def build_single_diplomacy_env(
    *,
    datum: dict[str, Any],
    environment_kind: EnvironmentKind,
    model_name: str,
    renderer_name: str,
    actor_runtime: ActorRuntimeConfig,
    policy_config: RuntimePolicyConfig,
) -> BuiltDiplomacyEnv:
    tokenizer = tokenizer_utils.get_tokenizer(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    episode_context = _initialize_episode_context(
        datum=json.loads(json.dumps(datum)),
        actor_runtime=actor_runtime,
        policy_config=policy_config,
    )
    message_env = DiplomacyMessageEnv(
        episode_context=episode_context,
        renderer=renderer,
        environment_kind=environment_kind,
        actor_runtime=actor_runtime,
        policy_config=policy_config,
    )
    env = EnvFromMessageEnv(
        renderer=renderer,
        message_env=message_env,
        failed_parse_reward=policy_config.failed_parse_reward,
        max_trajectory_tokens=policy_config.max_trajectory_tokens,
    )
    return BuiltDiplomacyEnv(env=env, state=episode_context.state)


class DiplomacyEnvGroupBuilder(EnvGroupBuilder):
    def __init__(
        self,
        *,
        datum: dict[str, Any],
        environment_kind: EnvironmentKind,
        model_name: str,
        renderer_name: str,
        group_size: int,
        actor_runtime: ActorRuntimeConfig,
        policy_config: RuntimePolicyConfig,
        rollout_runner_id: str | None = None,
    ) -> None:
        self.datum = datum
        self.environment_kind = environment_kind
        self.model_name = model_name
        self.renderer_name = renderer_name
        self.group_size = group_size
        self.actor_runtime = actor_runtime
        self.policy_config = policy_config
        self._rollout_runner_id: str | None = None
        self.requires_in_process_rollout = False
        self.rollout_runner_id = rollout_runner_id

    @property
    def rollout_runner_id(self) -> str | None:
        return self._rollout_runner_id

    @rollout_runner_id.setter
    def rollout_runner_id(self, value: str | None) -> None:
        self._rollout_runner_id = value
        self.requires_in_process_rollout = value is not None

    async def make_envs(self) -> Sequence[Env]:
        envs: list[Env] = []
        for _ in range(self.group_size):
            built_env = build_single_diplomacy_env(
                datum=self.datum,
                environment_kind=self.environment_kind,
                model_name=self.model_name,
                renderer_name=self.renderer_name,
                actor_runtime=self.actor_runtime,
                policy_config=self.policy_config,
            )
            envs.append(built_env.env)
        return envs

    async def run_group_rollout(
        self,
        sampling_ref: SamplingRef,
        *,
        max_tokens: int,
        temperature: float,
        do_remove_constant_reward_groups: bool,
        enable_logging: bool,
    ) -> TrajectoryGroup | None:
        del do_remove_constant_reward_groups
        runner = (
            get_trajectory_runner(self.rollout_runner_id)
            if self.rollout_runner_id is not None
            else LocalTrajectorySandboxRunner()
        )
        group_id = f"{self.datum.get('example_id', 'example')}:{uuid.uuid4().hex}"
        requests = [
            TrajectoryRolloutRequest(
                datum=json.loads(json.dumps(self.datum)),
                environment_kind=self.environment_kind,
                model_name=self.model_name,
                renderer_name=self.renderer_name,
                actor_runtime=self.actor_runtime,
                policy_config=self.policy_config,
                sampling_ref=sampling_ref,
                max_tokens=max_tokens,
                temperature=temperature,
                trajectory_index=index,
                group_id=group_id,
                enable_logging=enable_logging and index == 0,
            )
            for index in range(self.group_size)
        ]
        results = await asyncio.gather(*(runner.run_trajectory(request) for request in requests))
        trajectories = []
        final_rewards = []
        group_metrics = []
        for result in results:
            if result.failure_kind is not None or result.trajectory is None:
                raise RuntimeError(
                    f"Trajectory rollout failed for group {group_id}: "
                    f"{result.failure_kind or 'unknown'}: {result.failure_message or ''}"
                )
            trajectories.append(deserialize_trajectory(result.trajectory))
            final_rewards.append(float(result.final_reward))
            numeric_metrics = {
                key: float(value)
                for key, value in result.metrics.items()
                if isinstance(value, (int, float))
            }
            numeric_metrics[f"rollout/backend_{runner.backend_name}"] = 1.0
            group_metrics.append(numeric_metrics)
        return TrajectoryGroup(
            trajectories_G=trajectories,
            final_rewards_G=final_rewards,
            metrics_G=group_metrics,
        )

    def logging_tags(self) -> list[str]:
        return ["diplomacy", self.environment_kind]


class DiplomacyRLDataset(RLDataset):
    def __init__(self, env_group_builders: list[DiplomacyEnvGroupBuilder], batch_size: int):
        self.env_group_builders = env_group_builders
        self.batch_size = batch_size

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.batch_size
        end = start + self.batch_size
        return self.env_group_builders[start:end]

    def __len__(self) -> int:
        return len(self.env_group_builders) // self.batch_size


@chz.chz
class DiplomacyDatasetBuilder(RLDatasetBuilder):
    environment_kind: EnvironmentKind
    model_name_for_tokenizer: str
    renderer_name: str
    actor_runtime: ActorRuntimeConfig
    policy_config: RuntimePolicyConfig
    batch_size: int
    group_size: int
    num_train_examples: int
    num_eval_examples: int = 0
    train_seed: int = 0
    eval_seed: int = 10_000
    rollout_runner_id: str | None = None

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        if self.environment_kind == "tool_accuracy":
            train_rows = build_tool_accuracy_dataset(
                num_sessions=self.num_train_examples,
                seed=self.train_seed,
            ).to_list()
            eval_rows = (
                build_tool_accuracy_dataset(
                    num_sessions=self.num_eval_examples,
                    seed=self.eval_seed,
                ).to_list()
                if self.num_eval_examples > 0
                else []
            )
        else:
            train_rows = build_full_press_dataset(
                num_sessions=self.num_train_examples,
                seed=self.train_seed,
            ).to_list()
            eval_rows = (
                build_full_press_dataset(
                    num_sessions=self.num_eval_examples,
                    seed=self.eval_seed,
                ).to_list()
                if self.num_eval_examples > 0
                else []
            )

        train_dataset = DiplomacyRLDataset(
            [
                DiplomacyEnvGroupBuilder(
                    datum=row,
                    environment_kind=self.environment_kind,
                    model_name=self.model_name_for_tokenizer,
                    renderer_name=self.renderer_name,
                    group_size=self.group_size,
                    actor_runtime=self.actor_runtime,
                    policy_config=self.policy_config,
                    rollout_runner_id=self.rollout_runner_id,
                )
                for row in train_rows
            ],
            batch_size=self.batch_size,
        )
        eval_dataset = (
            DiplomacyRLDataset(
                [
                    DiplomacyEnvGroupBuilder(
                        datum=row,
                        environment_kind=self.environment_kind,
                        model_name=self.model_name_for_tokenizer,
                        renderer_name=self.renderer_name,
                        group_size=self.group_size,
                        actor_runtime=self.actor_runtime,
                        policy_config=self.policy_config,
                        rollout_runner_id=self.rollout_runner_id,
                    )
                    for row in eval_rows
                ],
                batch_size=self.batch_size,
            )
            if eval_rows
            else None
        )
        return train_dataset, eval_dataset
