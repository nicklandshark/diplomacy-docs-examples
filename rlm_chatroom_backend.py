from __future__ import annotations

import contextvars
import json
import os
import random
import sqlite3
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, TypedDict

import verifiers as vf
from datasets import Dataset
from verifiers.envs.experimental.rlm_env import RLMEnv
from verifiers.types import State


DEFAULT_AGENT_NAMES = ["bob", "ted", "carol", "alice"]
DEFAULT_TASK = "rlm-chatroom"
DEFAULT_MAX_MESSAGE_LENGTH = 2000
DEFAULT_CONVERSATION_RENDER_LINE_CAP = 2000
DEFAULT_EXPECTED_FINAL_ANSWER = "DONE"


class ChatroomRowInfo(TypedDict, total=False):
    session_id: str
    agent_name: str
    contacts: list[str]
    contacts_map: dict[str, list[str]]
    task_prompt: str
    agent_prompt: str
    task_config: dict[str, Any]
    expected_answer: str
    initial_messages: list[dict[str, Any]]
    player_agent: str
    actor_agents: list[str]
    agent_prompts: dict[str, str]
    grading_spec: dict[str, Any]
    session_deadline_epoch: float
    default_idle_sleep_seconds: float


class ChatroomToolTrace(TypedDict, total=False):
    tool_name: str
    timestamp: float
    agent_name: str
    participants: list[str]
    channel_type: Literal["dm", "group", "system"]
    success: bool
    error: str | None
    has_unread_notifications: bool | None
    notification_count: int | None
    total_messages: int | None
    offset: int | None
    limit: int | None
    result_preview: str | None


@dataclass
class ChatroomInteractionSpec:
    participants: list[str]
    channel_type: Literal["dm", "group"]
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "participants": sorted(self.participants),
            "channel_type": self.channel_type,
            "description": self.description,
        }


@dataclass
class ChatroomGradingSpec:
    required_sends: list[ChatroomInteractionSpec] = field(default_factory=list)
    required_reads: list[ChatroomInteractionSpec] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "required_sends": [item.to_dict() for item in self.required_sends],
            "required_reads": [item.to_dict() for item in self.required_reads],
        }


@dataclass
class ChatroomConfig:
    agent_names: list[str]
    contacts_map: dict[str, list[str]]
    max_message_length: int = DEFAULT_MAX_MESSAGE_LENGTH
    conversation_render_line_cap: int = DEFAULT_CONVERSATION_RENDER_LINE_CAP


@dataclass
class ChatroomSessionSpec:
    session_id: str
    agent_names: list[str]
    contacts_map: dict[str, list[str]]
    task_prompt: str
    task_config: dict[str, Any] = field(default_factory=dict)
    expected_answers: dict[str, str] = field(default_factory=dict)
    initial_messages: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class MultiAgentChatroomSessionSpec:
    session_id: str
    player_agent: str
    actor_agents: list[str]
    agent_names: list[str]
    contacts_map: dict[str, list[str]]
    bob_prompt: str
    agent_prompts: dict[str, str]
    grading_spec: ChatroomGradingSpec
    task_config: dict[str, Any] = field(default_factory=dict)
    initial_messages: list[dict[str, Any]] = field(default_factory=list)
    expected_answer: str = DEFAULT_EXPECTED_FINAL_ANSWER


class ChatroomTaskBackend(Protocol):
    def build_sessions(
        self,
        *,
        num_sessions: int,
        agent_names: list[str],
        contacts_map: dict[str, list[str]],
        seed: int | None = None,
    ) -> list[ChatroomSessionSpec]: ...


class DefaultChatroomTaskBackend:
    def build_sessions(
        self,
        *,
        num_sessions: int,
        agent_names: list[str],
        contacts_map: dict[str, list[str]],
        seed: int | None = None,
    ) -> list[ChatroomSessionSpec]:
        rng = random.Random(seed)
        sessions: list[ChatroomSessionSpec] = []
        for idx in range(num_sessions):
            session_id = f"session_{idx:05d}_{rng.randint(1000, 9999)}"
            task_prompt = (
                "Coordinate with your contacts using chat messages if needed. "
                "When finished, set your final answer to your own agent name exactly."
            )
            expected_answers = {name: name for name in agent_names}
            sessions.append(
                ChatroomSessionSpec(
                    session_id=session_id,
                    agent_names=list(agent_names),
                    contacts_map={k: list(v) for k, v in contacts_map.items()},
                    task_prompt=task_prompt,
                    task_config={"backend": "default", "index": idx},
                    expected_answers=expected_answers,
                )
            )
        return sessions


@dataclass
class SessionRuntime:
    session_id: str
    db: "ChatroomDB"
    root_dir: str
    db_path: str
    ref_count: int = 0
    seeded: bool = False


class ChatroomDB:
    def __init__(
        self,
        path: str,
        *,
        conversation_render_line_cap: int = DEFAULT_CONVERSATION_RENDER_LINE_CAP,
    ) -> None:
        self.path = path
        self.conversation_render_line_cap = conversation_render_line_cap
        self._write_lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA busy_timeout=5000;")
        self._create_schema()

    def close(self) -> None:
        self._conn.close()

    def _create_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS channels (
                    id              TEXT PRIMARY KEY,
                    participant_key TEXT UNIQUE,
                    participants    TEXT,
                    created_at      REAL
                );
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    channel_id  TEXT    REFERENCES channels(id),
                    sender      TEXT,
                    content     TEXT,
                    sent_at     REAL
                );
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS notifications (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent       TEXT,
                    channel_id  TEXT    REFERENCES channels(id),
                    message_id  INTEGER REFERENCES messages(id),
                    is_read     INTEGER DEFAULT 0
                );
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_notifications_unread
                ON notifications(agent, is_read)
                WHERE is_read = 0;
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_channel
                ON messages(channel_id, sent_at);
                """
            )

    @staticmethod
    def canonical_participants(participants: list[str]) -> list[str]:
        return sorted(set(participants))

    @classmethod
    def participant_key(cls, participants: list[str]) -> str:
        return ",".join(cls.canonical_participants(participants))

    def get_or_create_channel(self, participants: list[str], created_at: float | None = None) -> tuple[str, list[str]]:
        created_at = created_at if created_at is not None else time.time()
        canonical = self.canonical_participants(participants)
        key = self.participant_key(canonical)
        with self._write_lock:
            row = self._conn.execute(
                "SELECT id, participants FROM channels WHERE participant_key = ?",
                (key,),
            ).fetchone()
            if row is not None:
                return str(row["id"]), json.loads(str(row["participants"]))
            channel_id = str(uuid.uuid4())
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO channels (id, participant_key, participants, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (channel_id, key, json.dumps(canonical), float(created_at)),
                )
            return channel_id, canonical

    def get_channel(self, participants: list[str]) -> sqlite3.Row | None:
        key = self.participant_key(participants)
        return self._conn.execute(
            "SELECT * FROM channels WHERE participant_key = ?",
            (key,),
        ).fetchone()

    def insert_message(
        self,
        *,
        sender: str,
        participants: list[str],
        content: str,
        sent_at: float | None = None,
    ) -> tuple[str, int]:
        sent_at = sent_at if sent_at is not None else time.time()
        with self._write_lock:
            channel_id, canonical = self.get_or_create_channel(participants, created_at=sent_at)
            with self._conn:
                cursor = self._conn.execute(
                    """
                    INSERT INTO messages (channel_id, sender, content, sent_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (channel_id, sender, content, float(sent_at)),
                )
                message_id = int(cursor.lastrowid)
                recipients = [participant for participant in canonical if participant != sender]
                for agent in recipients:
                    self._conn.execute(
                        """
                        INSERT INTO notifications (agent, channel_id, message_id, is_read)
                        VALUES (?, ?, ?, 0)
                        """,
                        (agent, channel_id, message_id),
                    )
            return channel_id, message_id

    def consume_unread_notifications(self, agent: str) -> list[sqlite3.Row]:
        with self._write_lock:
            rows = self._conn.execute(
                """
                SELECT
                    n.id AS notification_id,
                    n.channel_id AS channel_id,
                    n.message_id AS message_id,
                    m.sender AS sender,
                    m.sent_at AS sent_at,
                    c.participants AS participants
                FROM notifications n
                JOIN messages m ON n.message_id = m.id
                JOIN channels c ON n.channel_id = c.id
                WHERE n.agent = ? AND n.is_read = 0
                ORDER BY m.sent_at ASC, n.id ASC
                """,
                (agent,),
            ).fetchall()
            if rows:
                with self._conn:
                    self._conn.execute(
                        "UPDATE notifications SET is_read = 1 WHERE agent = ? AND is_read = 0",
                        (agent,),
                    )
            return rows

    def list_messages(
        self,
        *,
        participants: list[str],
        offset: int | None,
        limit: int | None,
    ) -> tuple[str | None, list[str], int, int, int, int | None, list[sqlite3.Row], str | None]:
        channel = self.get_channel(participants)
        if channel is None:
            return None, participants, 0, 0, 0, None, [], "not_found"

        channel_id = str(channel["id"])
        canonical = json.loads(str(channel["participants"]))

        total_messages = int(
            self._conn.execute(
                "SELECT COUNT(*) FROM messages WHERE channel_id = ?",
                (channel_id,),
            ).fetchone()[0]
        )

        if offset is None and limit is None:
            rows = self._conn.execute(
                """
                SELECT sender, content, sent_at
                FROM messages
                WHERE channel_id = ?
                ORDER BY sent_at ASC, id ASC
                """
                ,
                (channel_id,),
            ).fetchall()
            rendered_line_count = sum(
                _count_rendered_message_lines(
                    sender=str(row["sender"]),
                    content=str(row["content"]),
                    sent_at=float(row["sent_at"]),
                )
                for row in rows
            )
            if rendered_line_count > self.conversation_render_line_cap:
                return (
                    channel_id,
                    canonical,
                    0,
                    total_messages,
                    total_messages,
                    rendered_line_count,
                    [],
                    "render_line_cap_exceeded",
                )
            return (
                channel_id,
                canonical,
                0,
                total_messages,
                total_messages,
                rendered_line_count,
                rows,
                None,
            )

        resolved_offset = 0 if offset is None else offset
        resolved_limit = total_messages if limit is None else limit
        rows = self._conn.execute(
            """
            SELECT sender, content, sent_at
            FROM messages
            WHERE channel_id = ?
            ORDER BY sent_at ASC, id ASC
            LIMIT ? OFFSET ?
            """,
            (channel_id, resolved_limit, resolved_offset),
        ).fetchall()

        return (
            channel_id,
            canonical,
            resolved_offset,
            resolved_limit,
            total_messages,
            None,
            rows,
            None,
        )

    def get_messages(self, *, participants: list[str]) -> list[sqlite3.Row]:
        channel = self.get_channel(participants)
        if channel is None:
            return []
        channel_id = str(channel["id"])
        return self._conn.execute(
            """
            SELECT sender, content, sent_at
            FROM messages
            WHERE channel_id = ?
            ORDER BY sent_at ASC, id ASC
            """,
            (channel_id,),
        ).fetchall()


class SessionRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, SessionRuntime] = {}

    def acquire(
        self,
        session_id: str,
        *,
        conversation_render_line_cap: int,
    ) -> SessionRuntime:
        with self._lock:
            runtime = self._sessions.get(session_id)
            if runtime is None:
                root_dir = tempfile.mkdtemp(prefix=f"rlm_chatroom_{session_id}_")
                db_path = os.path.join(root_dir, f"chatroom-{session_id}.db")
                db = ChatroomDB(
                    db_path,
                    conversation_render_line_cap=conversation_render_line_cap,
                )
                runtime = SessionRuntime(
                    session_id=session_id,
                    db=db,
                    root_dir=root_dir,
                    db_path=db_path,
                    ref_count=0,
                    seeded=False,
                )
                self._sessions[session_id] = runtime
            runtime.ref_count += 1
            return runtime

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

    def get(self, session_id: str) -> SessionRuntime | None:
        with self._lock:
            return self._sessions.get(session_id)


_SESSION_REGISTRY = SessionRegistry()


def _validate_contacts_map(agent_names: list[str], contacts_map: dict[str, list[str]]) -> dict[str, list[str]]:
    normalized: dict[str, list[str]] = {}
    known = set(agent_names)
    for agent in agent_names:
        contacts = contacts_map.get(agent, [])
        filtered: list[str] = []
        for contact in contacts:
            if contact == agent:
                continue
            if contact not in known:
                continue
            if contact not in filtered:
                filtered.append(contact)
        normalized[agent] = sorted(filtered)
    return normalized


def _default_contacts_map(agent_names: list[str]) -> dict[str, list[str]]:
    return {
        agent: sorted([other for other in agent_names if other != agent])
        for agent in agent_names
    }


def _format_time_ago(sent_at: float, now: float | None = None) -> str:
    now = now if now is not None else time.time()
    delta = max(0, int(now - sent_at))
    if delta < 60:
        return "just now"
    if delta < 3600:
        return f"{delta // 60}m ago"
    if delta < 86400:
        return f"{delta // 3600}h ago"
    return f"{delta // 86400}d ago"


def _format_timestamp(sent_at: float) -> str:
    return datetime.fromtimestamp(sent_at, tz=UTC).strftime("%H:%M:%S")


def _render_message_text(sender: str, content: str, sent_at: float, *, actor: str | None = None) -> str:
    timestamp = _format_timestamp(sent_at)
    sender_display = f"{sender} (you)" if actor is not None and sender == actor else sender
    return f"{sender_display}  {timestamp}  {content}"


def _count_rendered_message_lines(sender: str, content: str, sent_at: float) -> int:
    return len(_render_message_text(sender, content, sent_at).split("\n"))


def _ensure_list_of_strings(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        return None
    if not all(isinstance(item, str) for item in value):
        return None
    return value


def _channel_type_for_participants(participants: list[str]) -> Literal["dm", "group"]:
    return "dm" if len(participants) == 1 else "group"


def _ensure_chatroom_tool_trace(state: State) -> list[ChatroomToolTrace]:
    trace = state.get("chatroom_tool_trace")
    if isinstance(trace, list):
        return trace
    state["chatroom_tool_trace"] = []
    return state["chatroom_tool_trace"]


def _append_chatroom_tool_trace(
    state: State,
    *,
    tool_name: str,
    agent_name: str,
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
    trace = _ensure_chatroom_tool_trace(state)
    trace.append(
        ChatroomToolTrace(
            tool_name=tool_name,
            timestamp=time.time(),
            agent_name=agent_name,
            participants=sorted(participants or []),
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
    )


def build_dataset(
    *,
    num_sessions: int,
    agent_names: list[str],
    contacts_map: dict[str, list[str]],
    task_backend: ChatroomTaskBackend,
    seed: int | None,
) -> Dataset:
    sessions = task_backend.build_sessions(
        num_sessions=num_sessions,
        agent_names=agent_names,
        contacts_map=contacts_map,
        seed=seed,
    )

    rows: list[dict[str, Any]] = []
    for session in sessions:
        session_contacts = _validate_contacts_map(session.agent_names, session.contacts_map)
        for agent in session.agent_names:
            info: ChatroomRowInfo = {
                "session_id": session.session_id,
                "agent_name": agent,
                "contacts": list(session_contacts.get(agent, [])),
                "contacts_map": {k: list(v) for k, v in session_contacts.items()},
                "task_prompt": session.task_prompt,
                "task_config": dict(session.task_config),
                "expected_answer": session.expected_answers.get(agent, ""),
                "initial_messages": list(session.initial_messages),
            }
            rows.append(
                {
                    "example_id": f"{session.session_id}:{agent}",
                    "prompt": [{"role": "user", "content": session.task_prompt}],
                    "answer": session.expected_answers.get(agent, ""),
                    "info": info,
                    "task": DEFAULT_TASK,
                }
            )
    return Dataset.from_list(rows)


class RLMChatroomEnv(RLMEnv):
    def __init__(
        self,
        *,
        max_message_length: int = DEFAULT_MAX_MESSAGE_LENGTH,
        conversation_render_line_cap: int = DEFAULT_CONVERSATION_RENDER_LINE_CAP,
        **kwargs: Any,
    ) -> None:
        self.max_message_length = max_message_length
        self.conversation_render_line_cap = conversation_render_line_cap

        self._chatroom_root_tool_context_var: contextvars.ContextVar[dict[str, Any] | None] = (
            contextvars.ContextVar("rlm_chatroom_root_tool_context", default=None)
        )
        self._sub_tool_state_var: contextvars.ContextVar[State | None] = contextvars.ContextVar(
            "rlm_chatroom_sub_tool_state",
            default=None,
        )

        root_tools = self._build_root_tools()
        sub_tools = self._build_sub_tools()
        self._chatroom_root_tool_map = {
            getattr(tool, "__name__", tool.__class__.__name__): tool for tool in root_tools
        }
        self._chatroom_sub_tool_map = {
            getattr(tool, "__name__", tool.__class__.__name__): tool for tool in sub_tools
        }

        super().__init__(
            root_tools=root_tools,
            sub_tools=sub_tools,
            **kwargs,
        )
        if not hasattr(self, "_root_tool_context_var"):
            self._root_tool_context_var = self._chatroom_root_tool_context_var
        if not hasattr(self, "root_tool_map"):
            self.root_tool_map = dict(self._chatroom_root_tool_map)
        if not hasattr(self, "sub_tool_map"):
            self.sub_tool_map = dict(self._chatroom_sub_tool_map)

    def _build_worker_env_vars(self, state: State) -> dict[str, str]:
        parent_builder = getattr(super(), "_build_worker_env_vars", None)
        if callable(parent_builder):
            env_vars = parent_builder(state)
        else:
            env_vars = {}
        info = state.get("info", {})
        if isinstance(info, dict):
            agent_name = str(info.get("agent_name", ""))
            contacts = info.get("contacts", [])
            contacts_list = contacts if isinstance(contacts, list) else []
            env_vars["AGENT_NAME"] = agent_name
            env_vars["CONTACTS_BOOK"] = json.dumps(contacts_list)
            deadline = info.get("session_deadline_epoch")
            if isinstance(deadline, (int, float)):
                env_vars["SESSION_DEADLINE_EPOCH"] = str(float(deadline))
            idle_sleep = info.get("default_idle_sleep_seconds")
            if isinstance(idle_sleep, (int, float)):
                env_vars["DEFAULT_IDLE_SLEEP_SECONDS"] = str(float(idle_sleep))
        return env_vars

    async def _run_sub_llm(self, state, client, model, messages):
        token = self._sub_tool_state_var.set(state)
        try:
            return await super()._run_sub_llm(state, client, model, messages)
        finally:
            self._sub_tool_state_var.reset(token)

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
        state["chatroom_session_id"] = session_id
        _ensure_chatroom_tool_trace(state)

        try:
            state = await super().setup_state(state, **kwargs)
        except Exception:
            _SESSION_REGISTRY.release(session_id)
            raise

        if not runtime.seeded:
            initial_messages = info.get("initial_messages", [])
            if isinstance(initial_messages, list):
                for item in initial_messages:
                    if not isinstance(item, dict):
                        continue
                    sender = item.get("sender")
                    participants = item.get("participants")
                    message = item.get("message")
                    sent_at = item.get("sent_at")
                    if not isinstance(sender, str) or not isinstance(message, str):
                        continue
                    participants_list = _ensure_list_of_strings(participants)
                    if participants_list is None:
                        continue
                    participants_full = sorted(set([sender, *participants_list]))
                    runtime.db.insert_message(
                        sender=sender,
                        participants=participants_full,
                        content=message,
                        sent_at=float(sent_at) if isinstance(sent_at, (int, float)) else time.time(),
                    )
            runtime.seeded = True

        agent_name = str(info.get("agent_name", ""))
        contacts = info.get("contacts", [])
        task_prompt = str(info.get("task_prompt", ""))
        agent_prompt = str(info.get("agent_prompt", task_prompt))
        contacts_render = json.dumps(contacts if isinstance(contacts, list) else [])
        session_deadline_epoch = info.get("session_deadline_epoch")
        default_idle_sleep_seconds = info.get("default_idle_sleep_seconds")

        wait_guidance = ""
        if isinstance(session_deadline_epoch, (int, float)):
            wait_guidance += (
                f"\nSession deadline (unix epoch): {float(session_deadline_epoch):.3f}\n"
                "Do not finish just because your inbox is currently empty. "
                "If you are waiting on others, sleep in the bash REPL and poll again."
            )
        if isinstance(default_idle_sleep_seconds, (int, float)):
            wait_guidance += (
                f"\nDefault idle sleep: {float(default_idle_sleep_seconds):.3f} seconds.\n"
                "When waiting for messages, prefer sleeping approximately this long before polling again."
            )

        chatroom_scaffold = f"""
You are {agent_name}, an agent in a multi-agent messaging environment.
You communicate with other agents by delegating sub-LLMs to read and send messages on your behalf.
You CANNOT read or send messages directly.

Your identity: {agent_name}
Your contacts (agents you can communicate with): {contacts_render}

Workflow:
1) Run read_inbox_notifications to check new message summaries.
2) Run llm_batch with prompts that instruct sub-LLMs to call read_conversation and/or send_message.
3) Use sub-LLM responses as your only source of message content.
4) Repeat as needed and finalize with RLM_CONTENT and RLM_READY.

Important:
- read_inbox_notifications marks unread notifications as read.
- read_conversation and send_message are sub-LLM tools only.
- participants must list only other agents, never yourself.
- Waiting should happen inside the bash REPL using sleep.
- Continue working until you have completed your task or the session deadline has passed.{wait_guidance}

Task:
{agent_prompt}
""".strip()

        state["rlm_system_prompt"] = f"{chatroom_scaffold}\n\n{state['rlm_system_prompt']}"
        return state

    @vf.cleanup
    async def cleanup_chatroom_session(self, state: State) -> None:
        session_id = state.get("chatroom_session_id")
        if isinstance(session_id, str) and session_id:
            _SESSION_REGISTRY.release(session_id)

    def _get_runtime_for_state(self, state: State) -> SessionRuntime:
        session_id = state.get("chatroom_session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("chatroom session is not initialized")
        runtime = _SESSION_REGISTRY.get(session_id)
        if runtime is None:
            raise ValueError("chatroom session runtime not found")
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
            """
            Return summaries of unread messages and mark them as read.
            """
            context_var = getattr(
                env, "_root_tool_context_var", env._chatroom_root_tool_context_var
            )
            context = context_var.get()
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
            ordered = sorted(
                grouped.values(),
                key=lambda item: float(item["last_sent_at"]),
                reverse=True,
            )
            for item in ordered:
                participants = sorted(item["participants"])
                others = [name for name in participants if name != agent]
                count = int(item["count"])
                last_sender = str(item["last_sender"])
                last_sent = float(item["last_sent_at"])
                time_ago = _format_time_ago(last_sent, now=now)

                if len(participants) == 2 and len(others) == 1:
                    lines.append(f"from: {others[0]} ({count}) last sent {time_ago}")
                else:
                    display = ["you" if name == agent else name for name in participants]
                    sender_display = "you" if last_sender == agent else last_sender
                    lines.append(
                        "from: "
                        f"[{', '.join(display)}] ({count}) "
                        f"last message sent by {sender_display} {time_ago}"
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

        return [read_inbox_notifications]

    def _build_sub_tools(self) -> list[Any]:
        env = self

        def read_conversation(
            participants: list[str],
            offset: int | None = None,
            limit: int | None = None,
        ) -> str:
            """
            Read conversation messages from a channel that includes the caller.
            """
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
            ) = runtime.db.list_messages(
                participants=all_participants,
                offset=offset,
                limit=limit,
            )

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
            separator = " <-> "
            header_names = separator.join(canonical)
            if total_messages == 0 or len(rows) == 0:
                range_summary = f"messages none of {total_messages}"
            else:
                range_start = resolved_offset + 1
                range_end = resolved_offset + len(rows)
                range_summary = f"messages {range_start}-{range_end} of {total_messages}"
            lines = [
                f"[channel: {header_names} -- offset {resolved_offset}, limit {resolved_limit}, {range_summary}]",
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
            """
            Send a message to a conversation that includes the caller.
            """
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

            all_participants = sorted([actor, *validated])
            runtime.db.insert_message(
                sender=actor,
                participants=all_participants,
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

        return [read_conversation, send_message]


async def correct_answer(state: State, answer: str) -> float:
    final_answer = str(state.get("final_answer", "")).strip()
    if final_answer == str(answer).strip() and str(answer).strip() != "":
        return 1.0
    return 0.0


def load_environment(
    num_train_examples: int = 100,
    agent_names: list[str] | None = None,
    contacts_map: dict[str, list[str]] | None = None,
    max_turns: int = 50,
    sub_tool_max_turns: int = 5,
    max_sub_llm_parallelism: int = 5,
    max_message_length: int = DEFAULT_MAX_MESSAGE_LENGTH,
    conversation_render_line_cap: int = DEFAULT_CONVERSATION_RENDER_LINE_CAP,
    code_execution_timeout: int = 120,
    repl_language: str = "bash",
    seed: int | None = None,
    task_backend: ChatroomTaskBackend | None = None,
    **kwargs: Any,
) -> RLMChatroomEnv:
    resolved_agent_names = list(agent_names or DEFAULT_AGENT_NAMES)
    if len(resolved_agent_names) < 2:
        raise ValueError("agent_names must contain at least two agents")

    if contacts_map is None:
        resolved_contacts_map = _default_contacts_map(resolved_agent_names)
    else:
        resolved_contacts_map = _validate_contacts_map(resolved_agent_names, contacts_map)

    backend = task_backend or DefaultChatroomTaskBackend()
    dataset = build_dataset(
        num_sessions=num_train_examples,
        agent_names=resolved_agent_names,
        contacts_map=resolved_contacts_map,
        task_backend=backend,
        seed=seed,
    )

    rubric = vf.Rubric(funcs=[correct_answer], weights=[1.0])

    sandbox_labels = kwargs.pop("sandbox_labels", [])
    if not (isinstance(sandbox_labels, list) and all(isinstance(label, str) for label in sandbox_labels)):
        raise ValueError(f"sandbox_labels must be list[str]; got {sandbox_labels}")
    sandbox_labels = sorted(set(["rlm-chatroom", *sandbox_labels]))

    return RLMChatroomEnv(
        dataset=dataset,
        rubric=rubric,
        max_iterations=max_turns,
        sub_tool_max_turns=sub_tool_max_turns,
        max_sub_llm_parallelism=max_sub_llm_parallelism,
        max_message_length=max_message_length,
        conversation_render_line_cap=conversation_render_line_cap,
        code_execution_timeout=code_execution_timeout,
        repl_language=repl_language,
        sandbox_labels=sandbox_labels,
        **kwargs,
    )
