from __future__ import annotations

import asyncio
import contextlib
import contextvars
import copy
import json
import os
import socket
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import verifiers as vf
from datasets import Dataset
from diplomacy import Game
from verifiers.envs.experimental.rlm_env import RLMEnv
from verifiers.types import ClientConfig, RolloutInput, State

from data_generator import (
    EXPECTED_FINAL_ANSWER,
    STANDARD_POWERS,
    DiplomacyRowInfo,
    TransitionTargetSpec,
)
from rlm_chatroom_backend import (
    ChatroomDB,
    DEFAULT_CONVERSATION_RENDER_LINE_CAP,
    DEFAULT_MAX_MESSAGE_LENGTH,
    _append_chatroom_tool_trace,
    _channel_type_for_participants,
    _count_rendered_message_lines,
    _ensure_chatroom_tool_trace,
    _ensure_list_of_strings,
    _format_time_ago,
    _render_message_text,
)


DEFAULT_IDLE_SLEEP_SECONDS = 1.0
DEFAULT_SESSION_TIMEOUT_SECONDS = 30.0
DEFAULT_INVALID_TOOL_BUDGET = 3


@dataclass
class ActorConfig:
    api_base_url: str
    api_key_var: str
    model: str
    sampling_args: dict[str, Any] = field(default_factory=dict)
    client_type: Literal["openai_chat_completions"] = "openai_chat_completions"
    timeout: float = 3600.0

    def to_client_config(self) -> ClientConfig:
        return ClientConfig(
            client_type=self.client_type,
            api_key_var=self.api_key_var,
            api_base_url=self.api_base_url,
            timeout=self.timeout,
        )


@dataclass
class DiplomacySessionRuntime:
    session_id: str
    db: ChatroomDB
    root_dir: str
    db_path: str
    ref_count: int = 0
    seeded: bool = False
    lock: threading.RLock = field(default_factory=threading.RLock)
    initial_phase: str = ""
    initial_state: dict[str, Any] | None = None
    game: Game | None = None
    tracked_power: str = ""
    relevant_powers: list[str] = field(default_factory=list)
    environment_kind: str = ""
    current_submissions: dict[str, list[str]] = field(default_factory=dict)
    canonical_submissions: dict[str, dict[str, str]] = field(default_factory=dict)
    deterministic_orders: dict[str, list[str]] = field(default_factory=dict)
    rejected_tool_calls: int = 0


class SessionRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, DiplomacySessionRuntime] = {}

    def acquire(
        self,
        session_id: str,
        *,
        conversation_render_line_cap: int,
    ) -> DiplomacySessionRuntime:
        with self._lock:
            runtime = self._sessions.get(session_id)
            if runtime is None:
                root_dir = tempfile.mkdtemp(prefix=f"diplomacy_rlm_{session_id}_")
                db_path = os.path.join(root_dir, f"diplomacy-{session_id}.db")
                runtime = DiplomacySessionRuntime(
                    session_id=session_id,
                    db=ChatroomDB(
                        db_path,
                        conversation_render_line_cap=conversation_render_line_cap,
                    ),
                    root_dir=root_dir,
                    db_path=db_path,
                )
                self._sessions[session_id] = runtime
            runtime.ref_count += 1
            return runtime

    def get(self, session_id: str) -> DiplomacySessionRuntime | None:
        with self._lock:
            return self._sessions.get(session_id)

    def release(self, session_id: str) -> None:
        with self._lock:
            runtime = self._sessions.get(session_id)
            if runtime is None:
                return
            runtime.ref_count -= 1
            if runtime.ref_count > 0:
                return
            self._sessions.pop(session_id, None)
        runtime.db.close()
        try:
            Path(runtime.db_path).unlink(missing_ok=True)
        except OSError:
            pass
        try:
            Path(runtime.root_dir).rmdir()
        except OSError:
            pass


_SESSION_REGISTRY = SessionRegistry()


def _normalize_actor_configs(
    actor_configs: dict[str, ActorConfig | dict[str, Any]],
) -> dict[str, ActorConfig]:
    normalized: dict[str, ActorConfig] = {}
    for actor_name, config in actor_configs.items():
        if isinstance(config, ActorConfig):
            normalized[actor_name] = config
        elif isinstance(config, dict):
            normalized[actor_name] = ActorConfig(**config)
        else:
            raise TypeError(
                f"Unsupported actor config type for {actor_name}: {type(config).__name__}"
            )
    return normalized


def _new_game_from_state(initial_state: dict[str, Any]) -> Game:
    game = Game()
    game.set_state(copy.deepcopy(initial_state))
    return game


def _allocate_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(sock.getsockname()[1])


def _submission_for_display(submission: dict[str, str]) -> list[str]:
    return [submission[location] for location in sorted(submission)]


def _required_sends(state: State) -> list[dict[str, Any]]:
    info = state.get("info", {})
    if not isinstance(info, dict):
        return []
    raw = info.get("required_interactions", [])
    if not isinstance(raw, list):
        return []
    return [
        item
        for item in raw
        if isinstance(item, dict) and "DM" in str(item.get("description", ""))
    ]


def _required_reads(state: State) -> list[dict[str, Any]]:
    info = state.get("info", {})
    if not isinstance(info, dict):
        return []
    raw = info.get("required_interactions", [])
    if not isinstance(raw, list):
        return []
    return [
        item
        for item in raw
        if isinstance(item, dict) and "read" in str(item.get("description", "")).lower()
    ]


def _interaction_key(item: dict[str, Any]) -> tuple[tuple[str, ...], str]:
    participants = item.get("participants", [])
    if not isinstance(participants, list):
        participants = []
    return tuple(sorted(str(name) for name in participants)), str(item.get("channel_type", ""))


def _parse_order_location(order: str) -> str | None:
    parts = order.split()
    if not parts:
        return None
    if parts[0] in {"A", "F"}:
        if len(parts) < 2:
            return None
        return parts[1]
    return parts[0]


def _parse_destination(order: str) -> str | None:
    parts = order.split()
    if "-" not in parts:
        return None
    dash_index = parts.index("-")
    if dash_index + 1 >= len(parts):
        return None
    return parts[dash_index + 1]


def _orderable_location_key(location: str, orderable_locations: list[str]) -> str | None:
    if location in orderable_locations:
        return location
    base = location.split("/")[0]
    if base in orderable_locations:
        return base
    return None


def _unit_for_location(state: dict[str, Any], power: str, location: str) -> str | None:
    units_by_power = state.get("units", {})
    if not isinstance(units_by_power, dict):
        return None
    for unit in cast(list[str], units_by_power.get(power, [])):
        clean = unit[1:] if unit.startswith("*") else unit
        parts = clean.split()
        if len(parts) == 2 and parts[1] == location:
            return clean
    return None


def _power_occupies_location(state: dict[str, Any], power: str, location: str) -> bool:
    units_by_power = state.get("units", {})
    if not isinstance(units_by_power, dict):
        return False
    return any(unit == f"A {location}" or unit == f"F {location}" for unit in cast(list[str], units_by_power.get(power, [])))


def _evaluate_transition_target(
    *,
    target: TransitionTargetSpec,
    pre_state: dict[str, Any],
    post_state: dict[str, Any],
    results: dict[str, Any],
) -> bool:
    kind = str(target.get("kind", ""))
    power = str(target.get("power", ""))
    if kind == "unit_at":
        unit_type = str(target.get("unit_type", ""))
        location = str(target.get("location", ""))
        return f"{unit_type} {location}" in cast(list[str], post_state.get("units", {}).get(power, []))
    if kind == "location_occupied_by":
        return _power_occupies_location(post_state, power, str(target.get("location", "")))
    if kind in {"move_succeeded", "move_failed"}:
        from_location = str(target.get("from_location", ""))
        to_location = str(target.get("to_location", ""))
        unit = _unit_for_location(pre_state, power, from_location)
        if unit is None:
            return False
        unit_type = unit.split()[0]
        moved = f"{unit_type} {to_location}" in cast(list[str], post_state.get("units", {}).get(power, []))
        if kind == "move_succeeded":
            return moved and not results.get(unit)
        return (not moved) and bool(results.get(unit))
    if kind == "unit_dislodged":
        location = str(target.get("location", ""))
        unit = _unit_for_location(pre_state, power, location)
        if unit is None:
            return False
        dislodged_units = cast(list[str], post_state.get("units", {}).get(power, []))
        return f"*{unit}" in dislodged_units
    return False


def _stable_board_summary(game: Game) -> str:
    units = game.get_units()
    centers = game.get_centers()
    lines = [f"phase: {game.get_current_phase()}", "board summary:"]
    for power in STANDARD_POWERS:
        power_units = ", ".join(cast(list[str], units.get(power, []))) or "(none)"
        power_centers = ", ".join(cast(list[str], centers.get(power, []))) or "(none)"
        lines.append(f"- {power}: units [{power_units}] centers [{power_centers}]")
    return "\n".join(lines)


def _increment_rejected_tool_call(state: State, runtime: DiplomacySessionRuntime, agent_name: str) -> None:
    if agent_name != runtime.tracked_power:
        return
    runtime.rejected_tool_calls += 1
    state["rejected_tool_calls"] = runtime.rejected_tool_calls


def _canonicalize_submission(
    runtime: DiplomacySessionRuntime,
    *,
    power: str,
    orders: list[str],
) -> tuple[dict[str, str] | None, str | None]:
    if runtime.game is None:
        return None, "Error: game runtime is unavailable."
    orderable_locations = list(runtime.game.get_orderable_locations(power))
    if not orderable_locations:
        if orders:
            return None, f"Error: {power} has no orderable locations in this phase."
        return {}, None
    if not isinstance(orders, list) or not orders:
        return None, "Error: orders must be a non-empty list of strings."
    if not all(isinstance(order, str) and order.strip() for order in orders):
        return None, "Error: orders must be non-empty strings."
    seen_locations: set[str] = set()
    for order in orders:
        location = _parse_order_location(order)
        if location is None:
            return None, f"Error: unable to parse order location from {order!r}."
        orderable_key = _orderable_location_key(location, orderable_locations)
        if orderable_key is None:
            return None, f"Error: order {order!r} does not belong to {power}."
        if orderable_key in seen_locations:
            return None, f"Error: duplicate order submitted for {location}."
        seen_locations.add(orderable_key)

    scratch = _new_game_from_state(runtime.initial_state or runtime.game.get_state())
    scratch.set_orders(power, orders, expand=True, replace=True)
    canonical_orders = cast(list[str], scratch.get_orders(power))
    if len(canonical_orders) != len(orderable_locations):
        return None, "Error: submission must cover all orderable locations with legal orders."
    legal_orders = scratch.get_all_possible_orders()
    canonical_map: dict[str, str] = {}
    for order in canonical_orders:
        location = _parse_order_location(order)
        if location is None:
            return None, f"Error: unable to canonicalize order {order!r}."
        orderable_key = _orderable_location_key(location, orderable_locations)
        if orderable_key is None:
            return None, f"Error: order {order!r} does not belong to {power}."
        if order not in cast(list[str], legal_orders.get(orderable_key, [])):
            return None, f"Error: order {order!r} is not legal for {power}."
        canonical_map[orderable_key] = order
    if set(canonical_map) != set(orderable_locations):
        return None, "Error: submission must cover all orderable locations with legal orders."
    return canonical_map, None


def _seed_runtime(runtime: DiplomacySessionRuntime, info: DiplomacyRowInfo) -> None:
    with runtime.lock:
        if runtime.seeded:
            return
        initial_state = info.get("initial_state")
        initial_phase = str(info.get("initial_phase", ""))
        if not isinstance(initial_state, dict):
            raise ValueError("initial_state missing from info")
        game = _new_game_from_state(initial_state)
        if initial_phase and game.get_current_phase() != initial_phase:
            raise ValueError(
                f"initial_phase mismatch: expected {initial_phase}, got {game.get_current_phase()}"
            )
        runtime.initial_phase = game.get_current_phase()
        runtime.initial_state = copy.deepcopy(initial_state)
        runtime.game = game
        runtime.tracked_power = str(info.get("tracked_power", info.get("agent_name", "")))
        runtime.relevant_powers = [
            str(item) for item in info.get("relevant_powers", []) if isinstance(item, str)
        ]
        runtime.environment_kind = str(info.get("environment_kind", ""))

        initial_messages = info.get("initial_messages", [])
        if isinstance(initial_messages, list):
            for item in initial_messages:
                if not isinstance(item, dict):
                    continue
                sender = item.get("sender")
                participants = _ensure_list_of_strings(item.get("participants"))
                message = item.get("message")
                sent_at = item.get("sent_at")
                if not isinstance(sender, str) or participants is None or not isinstance(message, str):
                    continue
                runtime.db.insert_message(
                    sender=sender,
                    participants=sorted(set([sender, *participants])),
                    content=message,
                    sent_at=float(sent_at) if isinstance(sent_at, (int, float)) else time.time(),
                )

        deterministic_orders = info.get("deterministic_orders", {})
        if isinstance(deterministic_orders, dict):
            for power, orders in deterministic_orders.items():
                if not isinstance(power, str) or not isinstance(orders, list):
                    continue
                canonical_map, error = _canonicalize_submission(runtime, power=power, orders=cast(list[str], orders))
                if error:
                    raise ValueError(f"Invalid deterministic orders for {power}: {error}")
                runtime.current_submissions[power] = _submission_for_display(canonical_map or {})
                runtime.canonical_submissions[power] = canonical_map or {}
                runtime.deterministic_orders[power] = list(runtime.current_submissions[power])
        runtime.rejected_tool_calls = 0
        runtime.seeded = True


class DiplomacyRLMEnv(RLMEnv):
    def __init__(
        self,
        *,
        max_message_length: int = DEFAULT_MAX_MESSAGE_LENGTH,
        conversation_render_line_cap: int = DEFAULT_CONVERSATION_RENDER_LINE_CAP,
        **kwargs: Any,
    ) -> None:
        self.max_message_length = max_message_length
        self.conversation_render_line_cap = conversation_render_line_cap

        self._root_tool_context_var: contextvars.ContextVar[dict[str, Any] | None] = (
            contextvars.ContextVar("diplomacy_rlm_root_tool_context", default=None)
        )
        self._rollout_state_var: contextvars.ContextVar[State | None] = contextvars.ContextVar(
            "diplomacy_rlm_rollout_state",
            default=None,
        )
        self._sub_tool_state_var: contextvars.ContextVar[State | None] = contextvars.ContextVar(
            "diplomacy_rlm_sub_tool_state",
            default=None,
        )

        root_tools = self._build_root_tools()
        sub_tools = self._build_sub_tools()
        self.root_tool_map = {
            getattr(tool, "__name__", tool.__class__.__name__): tool for tool in root_tools
        }
        self.sub_tool_map = {
            getattr(tool, "__name__", tool.__class__.__name__): tool for tool in sub_tools
        }
        super().__init__(root_tools=root_tools, sub_tools=sub_tools, **kwargs)

    def _build_worker_env_vars(self, state: State) -> dict[str, str]:
        parent_builder = getattr(super(), "_build_worker_env_vars", None)
        env_vars = parent_builder(state) if callable(parent_builder) else {}
        info = state.get("info", {})
        if isinstance(info, dict):
            contacts = info.get("contacts", [])
            env_vars["AGENT_NAME"] = str(info.get("agent_name", ""))
            env_vars["CONTACTS_BOOK"] = json.dumps(contacts if isinstance(contacts, list) else [])
            deadline = info.get("session_deadline_epoch")
            if isinstance(deadline, (int, float)):
                env_vars["SESSION_DEADLINE_EPOCH"] = str(float(deadline))
            idle_sleep = info.get("default_idle_sleep_seconds")
            if isinstance(idle_sleep, (int, float)):
                env_vars["DEFAULT_IDLE_SLEEP_SECONDS"] = str(float(idle_sleep))
        return env_vars

    async def _run_sub_llm(self, client, model, messages):
        state = self._rollout_state_var.get()
        if state is None:
            raise RuntimeError("sub-LLM call is missing rollout state context")
        token = self._sub_tool_state_var.set(state)
        try:
            return await super()._run_sub_llm(client, model, messages)
        finally:
            self._sub_tool_state_var.reset(token)

    async def _handle_sub_llm_request(self, request: Any) -> Any:
        rollout_id = request.match_info["rollout_id"]
        context = self.active_rollouts.get(rollout_id)
        state = context.get("state") if isinstance(context, dict) else None
        token = self._rollout_state_var.set(state if isinstance(state, dict) else None)
        try:
            return await super()._handle_sub_llm_request(request)
        finally:
            self._rollout_state_var.reset(token)

    async def setup_state(self, state: State, **kwargs: Any) -> State:
        info = state.get("info", {})
        if not isinstance(info, dict):
            raise ValueError("state.info must be a dictionary")
        session_id = str(info.get("session_id", ""))
        if not session_id:
            raise ValueError("session_id missing from info")
        runtime = _SESSION_REGISTRY.acquire(
            session_id,
            conversation_render_line_cap=self.conversation_render_line_cap,
        )
        state["diplomacy_session_id"] = session_id
        _ensure_chatroom_tool_trace(state)
        try:
            state = await super().setup_state(state, **kwargs)
        except Exception:
            _SESSION_REGISTRY.release(session_id)
            raise

        _seed_runtime(runtime, cast(DiplomacyRowInfo, info))
        state["rejected_tool_calls"] = runtime.rejected_tool_calls

        agent_name = str(info.get("agent_name", ""))
        contacts = info.get("contacts", [])
        task_prompt = str(info.get("task_prompt", ""))
        agent_prompt = str(info.get("agent_prompt", task_prompt))
        deadline = info.get("session_deadline_epoch")
        idle_sleep = info.get("default_idle_sleep_seconds")
        wait_guidance = ""
        if isinstance(deadline, (int, float)):
            wait_guidance += (
                f"\nSession deadline (unix epoch): {float(deadline):.3f}\n"
                "If you are waiting on a reply, sleep in bash and poll again."
            )
        if isinstance(idle_sleep, (int, float)):
            wait_guidance += (
                f"\nDefault idle sleep: {float(idle_sleep):.3f} seconds.\n"
                "When waiting on messages, sleep approximately this long before polling again."
            )
        scaffold = f"""
You are {agent_name}, a Diplomacy agent in a messaging and order-submission environment.

Your identity: {agent_name}
Your contacts: {json.dumps(contacts if isinstance(contacts, list) else [])}

Tool split:
- Root-only tools you can call directly: read_phase_status, read_inbox_notifications, call_python_repl.
- Sub-LLM-only tools: read_conversation, send_message, read_legal_orders, submit_orders.
- A sub-LLM CANNOT call read_phase_status or read_inbox_notifications.

Recommended workflow:
1) Call read_phase_status directly as a root tool.
2) Call read_inbox_notifications directly as a root tool.
3) Use call_python_repl to invoke llm_batch with prompts that instruct sub-LLMs to use read_conversation, send_message, read_legal_orders, and submit_orders.
4) After you have completed the task, use call_python_repl to finalize with:
   answer["content"] = "DONE"
   answer["ready"] = True

Important:
- read_inbox_notifications marks unread notifications as read.
- participants must list only other powers, never yourself.
- submit_orders replaces the current submission for that power.
- If a counterpart has not replied yet, sleep in Python or bash, then poll again.
- Continue working until you have completed the task or the session deadline has passed.{wait_guidance}

Task:
{agent_prompt}
""".strip()
        state["rlm_system_prompt"] = f"{scaffold}\n\n{state['rlm_system_prompt']}"
        return state

    @vf.cleanup
    async def cleanup_diplomacy_session(self, state: State) -> None:
        session_id = state.get("diplomacy_session_id")
        if isinstance(session_id, str) and session_id:
            _SESSION_REGISTRY.release(session_id)

    def _get_runtime_for_state(self, state: State) -> DiplomacySessionRuntime:
        session_id = state.get("diplomacy_session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("diplomacy session is not initialized")
        runtime = _SESSION_REGISTRY.get(session_id)
        if runtime is None:
            raise ValueError("diplomacy session runtime not found")
        return runtime

    def _state_identity(self, state: State) -> tuple[str, list[str]]:
        info = state.get("info", {})
        if not isinstance(info, dict):
            raise ValueError("state.info must be a dictionary")
        agent_name = info.get("agent_name")
        contacts = info.get("contacts", [])
        if not isinstance(agent_name, str) or not agent_name:
            raise ValueError("agent_name missing from state info")
        contacts_list = _ensure_list_of_strings(contacts)
        if contacts_list is None:
            raise ValueError("contacts must be a list[str]")
        return agent_name, contacts_list

    def _validate_participants(
        self,
        *,
        actor: str,
        contacts: list[str],
        participants: Any,
        require_non_empty: bool,
    ) -> tuple[list[str] | None, str | None]:
        participants_list = _ensure_list_of_strings(participants)
        if participants_list is None:
            return None, "Error: participants must be a list of strings."
        cleaned: list[str] = []
        seen: set[str] = set()
        for participant in participants_list:
            if participant == actor:
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

    def _build_root_tools(self) -> list[Any]:
        env = self

        async def read_inbox_notifications() -> str:
            """Return summaries of unread messages and mark them as read."""
            context = env._root_tool_context_var.get()
            state = context.get("state") if context else None
            if state is None:
                raise RuntimeError("read_inbox_notifications called outside of root tool context")
            agent, _ = env._state_identity(state)
            runtime = env._get_runtime_for_state(state)
            rows = runtime.db.consume_unread_notifications(agent)
            if not rows:
                _append_chatroom_tool_trace(
                    state,
                    tool_name="read_inbox_notifications",
                    agent_name=agent,
                    success=True,
                    channel_type="system",
                    has_unread_notifications=False,
                    notification_count=0,
                    result_preview="no new notifications",
                )
                return "no new notifications"
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
                others = [name for name in participants if name != agent]
                count = int(item["count"])
                last_sender = str(item["last_sender"])
                last_sent = float(item["last_sent_at"])
                time_ago = _format_time_ago(last_sent, now=now)
                if len(participants) == 2 and len(others) == 1:
                    lines.append(f"from: {others[0]} ({count}) last sent {time_ago}")
                else:
                    display = ["you" if name == agent else name for name in participants]
                    lines.append(
                        f"from: [{', '.join(display)}] ({count}) last message sent by {last_sender} {time_ago}"
                    )
            result = "\n".join(lines)
            _append_chatroom_tool_trace(
                state,
                tool_name="read_inbox_notifications",
                agent_name=agent,
                success=True,
                channel_type="system",
                has_unread_notifications=True,
                notification_count=total_count,
                result_preview=result[:200],
            )
            return result

        async def read_phase_status() -> str:
            """Return current phase status, public board summary, orderable locations, and current submission."""
            context = env._root_tool_context_var.get()
            state = context.get("state") if context else None
            if state is None:
                raise RuntimeError("read_phase_status called outside of root tool context")
            agent, _ = env._state_identity(state)
            runtime = env._get_runtime_for_state(state)
            if runtime.game is None:
                raise RuntimeError("game runtime is unavailable")
            orderable_locations = list(runtime.game.get_orderable_locations(agent))
            submission = runtime.canonical_submissions.get(agent, {})
            lines = [
                _stable_board_summary(runtime.game),
                "",
                f"orderable locations for {agent}: {json.dumps(orderable_locations)}",
                "current submission:",
            ]
            if submission:
                lines.extend(f"- {order}" for order in _submission_for_display(submission))
            else:
                lines.append("- (none)")
            result = "\n".join(lines)
            _append_chatroom_tool_trace(
                state,
                tool_name="read_phase_status",
                agent_name=agent,
                success=True,
                channel_type="system",
                result_preview=result[:200],
            )
            return result

        return [read_inbox_notifications, read_phase_status]

    def _build_sub_tools(self) -> list[Any]:
        env = self

        def read_conversation(
            participants: list[str],
            offset: int | None = None,
            limit: int | None = None,
        ) -> str:
            state = env._sub_tool_state_var.get()
            if state is None:
                return "Error: sub-tool context is unavailable."
            try:
                actor, contacts = env._state_identity(state)
                runtime = env._get_runtime_for_state(state)
            except Exception as exc:
                return f"Error: {exc}"

            validated, err = env._validate_participants(
                actor=actor,
                contacts=contacts,
                participants=participants,
                require_non_empty=True,
            )
            if err:
                if runtime := _SESSION_REGISTRY.get(cast(str, state.get("diplomacy_session_id", ""))):
                    _increment_rejected_tool_call(state, runtime, actor)
                _append_chatroom_tool_trace(
                    state,
                    tool_name="read_conversation",
                    agent_name=actor,
                    participants=participants if isinstance(participants, list) else [],
                    channel_type=_channel_type_for_participants(participants) if isinstance(participants, list) and participants else "system",
                    success=False,
                    error=err,
                    offset=offset if isinstance(offset, int) else None,
                    limit=limit if isinstance(limit, int) else None,
                    result_preview=err,
                )
                return err
            assert validated is not None
            if (offset is None) != (limit is None):
                error = "Error: offset and limit must be provided together."
                _increment_rejected_tool_call(state, runtime, actor)
                _append_chatroom_tool_trace(
                    state,
                    tool_name="read_conversation",
                    agent_name=actor,
                    participants=validated,
                    channel_type=_channel_type_for_participants(validated),
                    success=False,
                    error=error,
                    offset=offset if isinstance(offset, int) else None,
                    limit=limit if isinstance(limit, int) else None,
                    result_preview=error,
                )
                return error
            if offset is not None and (not isinstance(offset, int) or offset < 0):
                error = "Error: offset must be a non-negative integer."
                _increment_rejected_tool_call(state, runtime, actor)
                _append_chatroom_tool_trace(
                    state,
                    tool_name="read_conversation",
                    agent_name=actor,
                    participants=validated,
                    channel_type=_channel_type_for_participants(validated),
                    success=False,
                    error=error,
                    limit=limit if isinstance(limit, int) else None,
                    result_preview=error,
                )
                return error
            if limit is not None and (not isinstance(limit, int) or limit <= 0):
                error = "Error: limit must be a positive integer."
                _increment_rejected_tool_call(state, runtime, actor)
                _append_chatroom_tool_trace(
                    state,
                    tool_name="read_conversation",
                    agent_name=actor,
                    participants=validated,
                    channel_type=_channel_type_for_participants(validated),
                    success=False,
                    error=error,
                    offset=offset if isinstance(offset, int) else None,
                    result_preview=error,
                )
                return error
            all_participants = sorted([actor, *validated])
            (
                _channel_id,
                canonical,
                resolved_offset,
                resolved_limit,
                total_messages,
                rendered_line_count,
                rows,
                error_type,
            ) = runtime.db.list_messages(participants=all_participants, offset=offset, limit=limit)
            if error_type == "not_found":
                error = f"No conversation found with [{', '.join(validated)}]."
                _append_chatroom_tool_trace(
                    state,
                    tool_name="read_conversation",
                    agent_name=actor,
                    participants=validated,
                    channel_type=_channel_type_for_participants(validated),
                    success=False,
                    error=error,
                    offset=offset,
                    limit=limit,
                    result_preview=error,
                )
                return error
            if error_type == "render_line_cap_exceeded":
                error = (
                    f"Error: Conversation renders to {rendered_line_count} lines. "
                    f"Use offset and limit to read it.\n"
                    f'Example: read_conversation(participants=["{validated[0]}"], offset=0, limit=50)'
                )
                _append_chatroom_tool_trace(
                    state,
                    tool_name="read_conversation",
                    agent_name=actor,
                    participants=validated,
                    channel_type=_channel_type_for_participants(validated),
                    success=False,
                    error=error,
                    total_messages=total_messages,
                    offset=offset,
                    limit=limit,
                    result_preview=error[:200],
                )
                return error
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
                        actor=actor,
                    )
                )
            if len(lines) == 2:
                lines.append("(no messages)")
            result = "\n".join(lines)
            _append_chatroom_tool_trace(
                state,
                tool_name="read_conversation",
                agent_name=actor,
                participants=validated,
                channel_type=_channel_type_for_participants(validated),
                success=True,
                total_messages=total_messages,
                offset=resolved_offset,
                limit=resolved_limit,
                result_preview=result[:200],
            )
            return result

        def send_message(participants: list[str], message: str) -> str:
            state = env._sub_tool_state_var.get()
            if state is None:
                return "Error: sub-tool context is unavailable."
            try:
                actor, contacts = env._state_identity(state)
                runtime = env._get_runtime_for_state(state)
            except Exception as exc:
                return f"Error: {exc}"
            validated, err = env._validate_participants(
                actor=actor,
                contacts=contacts,
                participants=participants,
                require_non_empty=True,
            )
            if err:
                _increment_rejected_tool_call(state, runtime, actor)
                _append_chatroom_tool_trace(
                    state,
                    tool_name="send_message",
                    agent_name=actor,
                    participants=participants if isinstance(participants, list) else [],
                    channel_type=_channel_type_for_participants(participants) if isinstance(participants, list) and participants else "system",
                    success=False,
                    error=err,
                    result_preview=err,
                )
                return err
            assert validated is not None
            if not isinstance(message, str) or not message.strip():
                error = "Error: message cannot be empty."
                _increment_rejected_tool_call(state, runtime, actor)
                _append_chatroom_tool_trace(
                    state,
                    tool_name="send_message",
                    agent_name=actor,
                    participants=validated,
                    channel_type=_channel_type_for_participants(validated),
                    success=False,
                    error=error,
                    result_preview=error,
                )
                return error
            if len(message) > env.max_message_length:
                error = f"Error: message exceeds {env.max_message_length} character limit."
                _increment_rejected_tool_call(state, runtime, actor)
                _append_chatroom_tool_trace(
                    state,
                    tool_name="send_message",
                    agent_name=actor,
                    participants=validated,
                    channel_type=_channel_type_for_participants(validated),
                    success=False,
                    error=error,
                    result_preview=error,
                )
                return error
            runtime.db.insert_message(
                sender=actor,
                participants=sorted([actor, *validated]),
                content=message,
                sent_at=time.time(),
            )
            result = f"Message sent to [{', '.join(validated)}] channel."
            _append_chatroom_tool_trace(
                state,
                tool_name="send_message",
                agent_name=actor,
                participants=validated,
                channel_type=_channel_type_for_participants(validated),
                success=True,
                result_preview=result,
            )
            return result

        def read_legal_orders() -> str:
            state = env._sub_tool_state_var.get()
            if state is None:
                return "Error: sub-tool context is unavailable."
            try:
                actor, _ = env._state_identity(state)
                runtime = env._get_runtime_for_state(state)
            except Exception as exc:
                return f"Error: {exc}"
            if runtime.game is None:
                return "Error: game runtime is unavailable."
            orderable_locations = list(runtime.game.get_orderable_locations(actor))
            legal_orders = runtime.game.get_all_possible_orders()
            lines = [f"[legal orders for {actor} -- phase {runtime.game.get_current_phase()}]"]
            if not orderable_locations:
                lines.append("(no orderable locations)")
            for location in orderable_locations:
                lines.append(f"{location}:")
                for order in cast(list[str], legal_orders.get(location, [])):
                    lines.append(f"- {order}")
            result = "\n".join(lines)
            _append_chatroom_tool_trace(
                state,
                tool_name="read_legal_orders",
                agent_name=actor,
                success=True,
                channel_type="system",
                result_preview=result[:200],
            )
            return result

        def submit_orders(orders: list[str]) -> str:
            state = env._sub_tool_state_var.get()
            if state is None:
                return "Error: sub-tool context is unavailable."
            try:
                actor, _ = env._state_identity(state)
                runtime = env._get_runtime_for_state(state)
            except Exception as exc:
                return f"Error: {exc}"
            canonical_map, error = _canonicalize_submission(runtime, power=actor, orders=orders)
            if error:
                _increment_rejected_tool_call(state, runtime, actor)
                _append_chatroom_tool_trace(
                    state,
                    tool_name="submit_orders",
                    agent_name=actor,
                    success=False,
                    channel_type="system",
                    error=error,
                    result_preview=error,
                )
                return error
            assert canonical_map is not None
            with runtime.lock:
                runtime.canonical_submissions[actor] = canonical_map
                runtime.current_submissions[actor] = _submission_for_display(canonical_map)
            result = "Submission accepted:\n" + "\n".join(f"- {order}" for order in runtime.current_submissions[actor])
            _append_chatroom_tool_trace(
                state,
                tool_name="submit_orders",
                agent_name=actor,
                success=True,
                channel_type="system",
                result_preview=result[:200],
            )
            return result

        return [read_conversation, send_message, read_legal_orders, submit_orders]


class AsyncDiplomacyRLMEnv(DiplomacyRLMEnv):
    def __init__(
        self,
        *,
        actor_configs: dict[str, ActorConfig | dict[str, Any]],
        actor_max_turns: int = 50,
        global_session_timeout_seconds: float = DEFAULT_SESSION_TIMEOUT_SECONDS,
        default_idle_sleep_seconds: float = DEFAULT_IDLE_SLEEP_SECONDS,
        **kwargs: Any,
    ) -> None:
        self.global_session_timeout_seconds = float(global_session_timeout_seconds)
        self.default_idle_sleep_seconds = float(default_idle_sleep_seconds)
        self.actor_max_turns = int(actor_max_turns)
        self.actor_configs = _normalize_actor_configs(actor_configs)
        super().__init__(**kwargs)

    def _required_actor_names(self, state: State | None = None) -> list[str]:
        if state is not None:
            info = state.get("info", {})
            if isinstance(info, dict):
                actor_names = info.get("relevant_powers")
                if isinstance(actor_names, list) and all(isinstance(item, str) for item in actor_names):
                    return list(actor_names)
        return []

    def _validate_actor_configs(self, state: State | None = None) -> None:
        missing = [name for name in self._required_actor_names(state) if name not in self.actor_configs]
        if missing:
            raise ValueError(f"actor_configs missing required actors: {missing}")

    async def setup_state(self, state: State, **kwargs: Any) -> State:
        info = state.get("info", {})
        if not isinstance(info, dict):
            raise ValueError("state.info must be a dictionary")
        self._validate_actor_configs(state)
        deadline = time.time() + self.global_session_timeout_seconds
        info = dict(info)
        info["session_deadline_epoch"] = deadline
        info["default_idle_sleep_seconds"] = self.default_idle_sleep_seconds
        info.setdefault("agent_prompt", str(info.get("task_prompt", "")))
        info.setdefault("tracked_power", str(info.get("agent_name", "")))
        state["info"] = info
        state = await super().setup_state(state, **kwargs)

        session_id = str(info.get("session_id", ""))
        scoring_hold = _SESSION_REGISTRY.acquire(
            session_id,
            conversation_render_line_cap=self.conversation_render_line_cap,
        )
        state["diplomacy_scoring_hold_session_id"] = scoring_hold.session_id
        state["_diplomacy_scoring_hold_active"] = True
        state["actor_statuses"] = {name: "scheduled" for name in self._required_actor_names(state)}
        state["diplomacy_actor_timeout_count"] = 0
        state["diplomacy_actor_error_count"] = 0
        state["session_deadline_epoch"] = deadline
        state["diplomacy_actor_tasks"] = await self._spawn_actor_rollouts(state)
        return state

    async def cleanup_diplomacy_session(self, state: State) -> None:
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
            _SESSION_REGISTRY.release(session_id)
            state["_diplomacy_primary_ref_released"] = True

    def _actor_env_kwargs(self) -> dict[str, Any]:
        sandbox_labels = sorted(set([*(self.sandbox_labels or []), "diplomacy-actor"]))
        kwargs: dict[str, Any] = {
            "max_message_length": self.max_message_length,
            "conversation_render_line_cap": self.conversation_render_line_cap,
            "sub_tool_max_turns": self.sub_tool_max_turns,
            "max_iterations": self.actor_max_turns,
            "max_output_length": self.max_output_length,
            "max_sub_llm_parallelism": self.max_sub_llm_parallelism,
            "repl_language": self.repl_language,
            "code_execution_timeout": self.code_execution_timeout,
            "abort_on_code_timeout": self.abort_on_code_timeout,
            "max_startup_wait_seconds": self.max_startup_wait_seconds,
            "pip_install_packages": self.pip_install_packages,
            "interception_port": _allocate_local_port(),
            "sandbox_labels": sandbox_labels,
            "retain_filesystem_after_rollout": False,
        }
        optional_attrs = [
            "sub_prompt_verbosity",
            "root_prompt_verbosity",
            "expose_message_history",
            "sandbox_docker_image",
            "sandbox_start_command",
            "sandbox_cpu_cores",
            "sandbox_memory_gb",
            "sandbox_disk_size_gb",
            "sandbox_gpu_count",
            "sandbox_timeout_minutes",
            "sandbox_environment_vars",
            "sandbox_team_id",
            "sandbox_advanced_configs",
            "sandbox_client_max_workers",
            "sandbox_client_max_connections",
            "sandbox_client_max_keepalive_connections",
        ]
        for name in optional_attrs:
            if hasattr(self, name):
                kwargs[name] = getattr(self, name)
        return kwargs

    def _build_actor_input(self, state: State, actor_name: str) -> RolloutInput:
        info = state.get("info", {})
        if not isinstance(info, dict):
            raise ValueError("state.info must be a dictionary")
        contacts_map = info.get("contacts_map", {})
        if not isinstance(contacts_map, dict):
            raise ValueError("contacts_map missing from state")
        actor_prompts = info.get("actor_prompts", {})
        if not isinstance(actor_prompts, dict):
            raise ValueError("actor_prompts missing from state")
        prompt = str(actor_prompts.get(actor_name, "")).strip()
        if not prompt:
            raise ValueError(f"No actor prompt configured for {actor_name}")
        actor_info: DiplomacyRowInfo = {
            "session_id": str(info.get("session_id", "")),
            "agent_name": actor_name,
            "contacts": [str(name) for name in contacts_map.get(actor_name, [])] if isinstance(contacts_map.get(actor_name, []), list) else [],
            "contacts_map": {str(key): [str(name) for name in value] for key, value in contacts_map.items() if isinstance(value, list)},
            "task_prompt": prompt,
            "agent_prompt": prompt,
            "task_config": dict(info.get("task_config", {})) if isinstance(info.get("task_config", {}), dict) else {},
            "expected_answer": EXPECTED_FINAL_ANSWER,
            "initial_messages": [],
            "initial_phase": str(info.get("initial_phase", "")),
            "initial_state": copy.deepcopy(cast(dict[str, Any], info.get("initial_state", {}))),
            "relevant_powers": [],
            "actor_prompts": {},
            "required_interactions": [],
            "order_constraints": [],
            "transition_target": None,
            "relevant_actor_objectives": [],
            "deterministic_orders": {str(key): list(value) for key, value in cast(dict[str, list[str]], info.get("deterministic_orders", {})).items()},
            "environment_kind": cast(Literal["tool_accuracy", "full_press"], str(info.get("environment_kind", "tool_accuracy"))),
            "session_deadline_epoch": float(state.get("session_deadline_epoch", time.time())),
            "default_idle_sleep_seconds": float(info.get("default_idle_sleep_seconds", self.default_idle_sleep_seconds)),
            "tracked_power": str(info.get("tracked_power", info.get("agent_name", ""))),
        }
        return cast(
            RolloutInput,
            {
                "prompt": [{"role": "user", "content": prompt}],
                "answer": EXPECTED_FINAL_ANSWER,
                "task": f"diplomacy-{info.get('environment_kind', 'tool_accuracy')}",
                "example_id": -1,
                "info": actor_info,
            },
        )

    def _build_actor_env(self) -> DiplomacyRLMEnv:
        placeholder = Dataset.from_list(
            [
                {
                    "example_id": 0,
                    "prompt": [{"role": "user", "content": "placeholder"}],
                    "answer": EXPECTED_FINAL_ANSWER,
                    "task": "diplomacy-actor",
                }
            ]
        )
        return DiplomacyRLMEnv(
            dataset=placeholder,
            rubric=vf.Rubric(funcs=[correct_answer], weights=[1.0]),
            score_rollouts=False,
            **self._actor_env_kwargs(),
        )

    async def _run_actor_rollout(self, actor_name: str, state: State) -> None:
        actor_config = self.actor_configs[actor_name]
        actor_client = vf.AsyncOpenAI(
            api_key=os.environ.get(actor_config.api_key_var, "EMPTY"),
            base_url=actor_config.api_base_url,
            timeout=actor_config.timeout,
        )
        actor_env = self._build_actor_env()
        actor_input = self._build_actor_input(state, actor_name)
        actor_statuses = state.get("actor_statuses", {})
        if isinstance(actor_statuses, dict):
            actor_statuses[actor_name] = "running"
        try:
            actor_state = await asyncio.wait_for(
                actor_env.rollout(
                    actor_input,
                    actor_client,
                    actor_config.model,
                    actor_config.sampling_args,
                ),
                timeout=max(self.global_session_timeout_seconds + 5.0, 5.0),
            )
        except asyncio.TimeoutError:
            state["diplomacy_actor_timeout_count"] = int(state.get("diplomacy_actor_timeout_count", 0)) + 1
            if isinstance(actor_statuses, dict):
                actor_statuses[actor_name] = "timeout"
        except asyncio.CancelledError:
            if isinstance(actor_statuses, dict):
                actor_statuses[actor_name] = "cancelled"
            raise
        except Exception as exc:
            state["diplomacy_actor_error_count"] = int(state.get("diplomacy_actor_error_count", 0)) + 1
            if isinstance(actor_statuses, dict):
                actor_statuses[actor_name] = f"error:{type(exc).__name__}"
        else:
            stop_condition = actor_state.get("stop_condition")
            if isinstance(actor_statuses, dict):
                actor_statuses[actor_name] = str(stop_condition or "completed")
        finally:
            with contextlib.suppress(Exception):
                await actor_client.close()
            with contextlib.suppress(Exception):
                await actor_env._teardown()

    async def _spawn_actor_rollouts(self, state: State) -> dict[str, asyncio.Task[Any]]:
        tasks: dict[str, asyncio.Task[Any]] = {}
        for actor_name in self._required_actor_names(state):
            tasks[actor_name] = asyncio.create_task(
                self._run_actor_rollout(actor_name, state),
                name=f"diplomacy-actor-{actor_name}",
            )
        return tasks


async def correct_answer(state: State, answer: str) -> float:
    final_answer = str(state.get("final_answer", "")).strip()
    if final_answer == str(answer).strip() and str(answer).strip() != "":
        return 1.0
    return 0.0


def _compute_interaction_analysis(state: State) -> dict[str, Any]:
    cached = state.get("diplomacy_analysis")
    if isinstance(cached, dict):
        return cached
    info = state.get("info", {})
    if not isinstance(info, dict):
        info = {}
    session_id = str(state.get("diplomacy_session_id", ""))
    runtime = _SESSION_REGISTRY.get(session_id) if session_id else None
    tool_trace = [item for item in state.get("chatroom_tool_trace", []) if isinstance(item, dict)]
    successful_sends = [
        item for item in tool_trace if item.get("tool_name") == "send_message" and item.get("success") is True
    ]
    successful_reads = [
        item for item in tool_trace if item.get("tool_name") == "read_conversation" and item.get("success") is True
    ]
    required_sends = _required_sends(state)
    required_reads = _required_reads(state)
    successful_send_keys = {_interaction_key(cast(dict[str, Any], item)) for item in successful_sends}
    matched_required_send_count = sum(
        1 for item in required_sends if _interaction_key(cast(dict[str, Any], item)) in successful_send_keys
    )
    matched_required_read_count = 0
    tracked_power = str(info.get("tracked_power", info.get("agent_name", "")))
    if runtime is not None:
        for item in required_reads:
            interaction_key = _interaction_key(cast(dict[str, Any], item))
            send_times = [
                float(send_item.get("timestamp", 0.0))
                for send_item in successful_sends
                if _interaction_key(cast(dict[str, Any], send_item)) == interaction_key
            ]
            if not send_times:
                continue
            sent_after = min(send_times)
            participants = cast(list[str], item["participants"])
            rows = runtime.db.get_messages(participants=[tracked_power, *participants])
            reply_times = [
                float(row["sent_at"])
                for row in rows
                if str(row["sender"]) != tracked_power and float(row["sent_at"]) >= sent_after
            ]
            if not reply_times:
                continue
            reply_after = min(reply_times)
            has_read = any(
                _interaction_key(cast(dict[str, Any], read_item)) == interaction_key
                and float(read_item.get("timestamp", 0.0)) >= reply_after
                for read_item in successful_reads
            )
            if has_read:
                matched_required_read_count += 1
    analysis = {
        "matched_required_send_count": matched_required_send_count,
        "matched_required_read_count": matched_required_read_count,
        "required_send_total": len(required_sends),
        "required_read_total": len(required_reads),
        "rejected_tool_calls": int(state.get("rejected_tool_calls", 0)),
    }
    state["diplomacy_analysis"] = analysis
    return analysis


async def constraints_satisfied_metric(state: State) -> float:
    info = state.get("info", {})
    if not isinstance(info, dict):
        return 0.0
    session_id = str(state.get("diplomacy_session_id", ""))
    runtime = _SESSION_REGISTRY.get(session_id) if session_id else None
    if runtime is None:
        return 0.0
    tracked_power = str(info.get("tracked_power", info.get("agent_name", "")))
    submission = runtime.canonical_submissions.get(tracked_power, {})
    constraints = info.get("order_constraints", [])
    if not isinstance(constraints, list) or not constraints:
        return 1.0
    matched = 0
    total = 0
    for item in constraints:
        if not isinstance(item, dict):
            continue
        location = str(item.get("location", ""))
        canonical_order = str(item.get("canonical_order", ""))
        total += 1
        if submission.get(location) == canonical_order:
            matched += 1
    if total <= 0:
        return 1.0
    return float(matched) / float(total)


async def complete_legal_submission_metric(state: State) -> float:
    info = state.get("info", {})
    if not isinstance(info, dict):
        return 0.0
    session_id = str(state.get("diplomacy_session_id", ""))
    runtime = _SESSION_REGISTRY.get(session_id) if session_id else None
    if runtime is None or runtime.game is None:
        return 0.0
    tracked_power = str(info.get("tracked_power", info.get("agent_name", "")))
    submission = runtime.canonical_submissions.get(tracked_power)
    if submission is None:
        return 0.0
    return 1.0 if set(submission) == set(runtime.game.get_orderable_locations(tracked_power)) else 0.0


async def required_send_recall_metric(state: State) -> float:
    analysis = _compute_interaction_analysis(state)
    total = int(analysis["required_send_total"])
    if total <= 0:
        return 1.0
    return float(analysis["matched_required_send_count"]) / float(total)


async def required_read_recall_metric(state: State) -> float:
    analysis = _compute_interaction_analysis(state)
    total = int(analysis["required_read_total"])
    if total <= 0:
        return 1.0
    return float(analysis["matched_required_read_count"]) / float(total)


async def invalid_tool_budget_pass_metric(state: State) -> float:
    return 1.0 if int(_compute_interaction_analysis(state)["rejected_tool_calls"]) <= DEFAULT_INVALID_TOOL_BUDGET else 0.0


async def rejected_tool_call_count_metric(state: State) -> float:
    return float(_compute_interaction_analysis(state)["rejected_tool_calls"])


async def _wait_for_relevant_actors(state: State) -> None:
    actor_tasks = state.get("diplomacy_actor_tasks", {})
    if not isinstance(actor_tasks, dict) or not actor_tasks:
        return
    pending = [task for task in actor_tasks.values() if isinstance(task, asyncio.Task) and not task.done()]
    if not pending:
        return
    deadline = float(state.get("session_deadline_epoch", time.time()))
    timeout = max(0.0, deadline - time.time())
    if timeout <= 0.0:
        return
    with contextlib.suppress(Exception):
        await asyncio.wait(pending, timeout=timeout)


async def _tracked_submission_gate(state: State) -> bool:
    return await complete_legal_submission_metric(state) >= 1.0


async def _relevant_actor_gate(state: State) -> bool:
    await _wait_for_relevant_actors(state)
    info = state.get("info", {})
    if not isinstance(info, dict):
        return False
    session_id = str(state.get("diplomacy_session_id", ""))
    runtime = _SESSION_REGISTRY.get(session_id) if session_id else None
    if runtime is None or runtime.game is None:
        return False
    actor_statuses = state.get("actor_statuses", {})
    if not isinstance(actor_statuses, dict):
        actor_statuses = {}
    for power in [str(item) for item in info.get("relevant_powers", []) if isinstance(item, str)]:
        status = actor_statuses.get(power, "")
        if status == "timeout" or (isinstance(status, str) and status.startswith("error:")):
            return False
        submission = runtime.canonical_submissions.get(power)
        if submission is None:
            return False
        if set(submission) != set(runtime.game.get_orderable_locations(power)):
            return False
    return True


async def full_press_gate_metric(state: State) -> float:
    send_ok = await required_send_recall_metric(state) >= 1.0
    read_ok = await required_read_recall_metric(state) >= 1.0
    budget_ok = await invalid_tool_budget_pass_metric(state) >= 1.0
    tracked_ok = await _tracked_submission_gate(state)
    actor_ok = await _relevant_actor_gate(state)
    return 1.0 if (send_ok and read_ok and budget_ok and tracked_ok and actor_ok) else 0.0


async def transition_target_satisfied_metric(state: State) -> float:
    gate = await full_press_gate_metric(state)
    if gate < 1.0:
        return 0.0
    info = state.get("info", {})
    if not isinstance(info, dict):
        return 0.0
    target = info.get("transition_target")
    if not isinstance(target, dict):
        return 0.0
    session_id = str(state.get("diplomacy_session_id", ""))
    runtime = _SESSION_REGISTRY.get(session_id) if session_id else None
    if runtime is None or runtime.game is None or runtime.initial_state is None:
        return 0.0
    adjudication_game = _new_game_from_state(runtime.initial_state)
    for power in STANDARD_POWERS:
        submission = runtime.current_submissions.get(power)
        if submission is not None:
            adjudication_game.set_orders(power, submission)
    phase_data = adjudication_game.process()
    return 1.0 if _evaluate_transition_target(
        target=cast(TransitionTargetSpec, target),
        pre_state=runtime.initial_state,
        post_state=adjudication_game.get_state(),
        results=cast(dict[str, Any], phase_data.results),
    ) else 0.0


async def relevant_actor_submission_metric(state: State) -> float:
    return 1.0 if await _relevant_actor_gate(state) else 0.0


async def finalize_diplomacy_state_metric(state: State) -> float:
    if state.get("_diplomacy_finalized"):
        return 0.0
    state["_diplomacy_finalized"] = True
    actor_tasks = state.get("diplomacy_actor_tasks", {})
    if isinstance(actor_tasks, dict):
        pending = [task for task in actor_tasks.values() if isinstance(task, asyncio.Task) and not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            with contextlib.suppress(Exception):
                await asyncio.gather(*pending, return_exceptions=True)
    session_id = state.get("diplomacy_scoring_hold_session_id")
    if isinstance(session_id, str) and session_id and state.get("_diplomacy_scoring_hold_active"):
        _SESSION_REGISTRY.release(session_id)
        state["_diplomacy_scoring_hold_active"] = False
    return 0.0


def load_environment(
    *,
    dataset: Dataset,
    rubric: vf.Rubric,
    actor_configs: dict[str, ActorConfig | dict[str, Any]],
    max_turns: int = 50,
    sub_tool_max_turns: int = 5,
    max_sub_llm_parallelism: int = 5,
    repl_language: str = "bash",
    conversation_render_line_cap: int = DEFAULT_CONVERSATION_RENDER_LINE_CAP,
    global_session_timeout_seconds: float = DEFAULT_SESSION_TIMEOUT_SECONDS,
    default_idle_sleep_seconds: float = DEFAULT_IDLE_SLEEP_SECONDS,
    actor_max_turns: int = 50,
    max_message_length: int = DEFAULT_MAX_MESSAGE_LENGTH,
    code_execution_timeout: int = 120,
    **kwargs: Any,
) -> AsyncDiplomacyRLMEnv:
    sandbox_labels = kwargs.pop("sandbox_labels", [])
    if not (isinstance(sandbox_labels, list) and all(isinstance(label, str) for label in sandbox_labels)):
        raise ValueError(f"sandbox_labels must be list[str]; got {sandbox_labels}")
    sandbox_labels = sorted(set(["diplomacy-rlm", *sandbox_labels]))
    kwargs.setdefault("interception_port", _allocate_local_port())
    return AsyncDiplomacyRLMEnv(
        dataset=dataset,
        rubric=rubric,
        actor_configs=actor_configs,
        actor_max_turns=actor_max_turns,
        global_session_timeout_seconds=global_session_timeout_seconds,
        default_idle_sleep_seconds=default_idle_sleep_seconds,
        max_iterations=max_turns,
        sub_tool_max_turns=sub_tool_max_turns,
        max_sub_llm_parallelism=max_sub_llm_parallelism,
        repl_language=repl_language,
        conversation_render_line_cap=conversation_render_line_cap,
        max_message_length=max_message_length,
        code_execution_timeout=code_execution_timeout,
        sandbox_labels=sandbox_labels,
        **kwargs,
    )
