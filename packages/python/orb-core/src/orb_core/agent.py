"""The agent interface: one contract between agents and channels.

A channel (web chat, voice, later Slack) turns its medium into text turns. An agent turns a session and a user
text into a stream of events. The channel renders the events: "token" events become words on the screen or
audio in the room, and the other events feed the sidebar, the logs and the tests.

    store = SessionStore({"discussion": DiscussionAgent(...)})
    session = store.create("Should we cache API responses?")
    async for event in session.turn("We should cache everything for a day.", channel="web"):
        ...

Every agent keeps its own state inside its sessions. A channel never reads that state directly. It reads
`session.state()`, a JSON-safe dict.
"""
from __future__ import annotations

import json
import re
import secrets
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

# read: what the reader found in the user message. decision: what the agent chose to do and why.
# token: a piece of the reply text. reply: the full reply. check: did the reply do its job.
# state: the session state after the turn. error: a failure that ends the turn.
EVENT_TYPES = ("read", "decision", "token", "reply", "check", "state", "error")
CHANNELS = ("web", "voice", "slack", "test")
SESSION_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,47}$")
MAX_TOPIC = 2000
MAX_TEXT = 8000


@dataclass(frozen=True)
class Event:
    type: str
    data: Any = None

    def __post_init__(self):
        if self.type not in EVENT_TYPES:
            raise ValueError(f"unknown event type {self.type!r}; use one of {EVENT_TYPES}")

    def to_json(self) -> str:
        return json.dumps({"type": self.type, "data": self.data}, ensure_ascii=False, default=str)


@runtime_checkable
class Session(Protocol):
    id: str
    topic: str
    agent_name: str

    def turn(self, text: str, channel: str = "web", replace_last: bool = False) -> AsyncIterator[Event]:
        """Process one user message and stream the events of the reply.

        replace_last=True: the user never heard the reply to their last message and went on talking, so `text`
        is that message said further. The agent undoes the last turn first, when `text` really continues it."""

    def state(self) -> dict:
        """A JSON-safe view of the session for channels and the UI."""

    async def settle(self) -> None:
        """Wait until the background work of the last turn is done (for example the read of the AI reply)."""

    def snapshot(self) -> dict:
        """A JSON-safe record of everything, for saving."""


@runtime_checkable
class Agent(Protocol):
    name: str

    def new_session(self, topic: str, session_id: str) -> Session:
        """Start a session. The agent may do slow setup in the background."""

    async def health(self) -> dict:
        """What the agent needs, and whether each part answers."""


class InputError(ValueError):
    """The input breaks a limit or a format rule. The message says which one."""


def new_session_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S").lower() + "-" + secrets.token_hex(3)


def check_session_id(session_id: str) -> str:
    if not SESSION_ID.match(session_id or ""):
        raise InputError("A session ID must be 3 to 48 characters: lower-case letters, digits and hyphens.")
    return session_id


def check_text(text: str, limit: int = MAX_TEXT, what: str = "The message") -> str:
    text = (text or "").strip()
    if not text:
        raise InputError(f"{what} is empty.")
    if len(text) > limit:
        raise InputError(f"{what} has {len(text)} characters. The limit is {limit}.")
    return text


class SessionStore:
    """Keeps the live sessions in memory and saves a JSON snapshot of a session after each turn."""

    def __init__(self, agents: dict[str, Agent], save_dir: str | Path | None = None, max_sessions: int = 200):
        if not agents:
            raise ValueError("SessionStore needs at least one agent")
        self.agents = agents
        self.default_agent = next(iter(agents))
        self.save_dir = Path(save_dir) if save_dir else None
        self.max_sessions = max_sessions
        self._sessions: dict[str, Session] = {}
        self._last_used: dict[str, float] = {}

    def create(self, topic: str, agent: str | None = None, session_id: str | None = None) -> Session:
        name = agent or self.default_agent
        if name not in self.agents:
            raise InputError(f"There is no agent named {name!r}. Agents: {', '.join(self.agents)}.")
        topic = check_text(topic, MAX_TOPIC, "The topic")
        sid = check_session_id(session_id) if session_id else new_session_id()
        if sid in self._sessions:
            raise InputError(f"Session {sid} exists already.")
        self._evict()
        session = self.agents[name].new_session(topic, sid)
        self._sessions[sid] = session
        self._last_used[sid] = time.monotonic()
        return session

    def get(self, session_id: str) -> Session | None:
        session = self._sessions.get(session_id)
        if session is not None:
            self._last_used[session_id] = time.monotonic()
        return session

    def get_or_create(self, session_id: str, topic: str, agent: str | None = None) -> Session:
        return self.get(check_session_id(session_id)) or self.create(topic, agent, session_id)

    def list(self) -> list[dict]:
        return [{"id": s.id, "topic": s.topic, "agent": s.agent_name} for s in self._sessions.values()]

    def save(self, session: Session) -> Path | None:
        if not self.save_dir:
            return None
        self.save_dir.mkdir(parents=True, exist_ok=True)
        dest = self.save_dir / f"{session.id}.json"
        tmp = dest.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(session.snapshot(), indent=1, ensure_ascii=False, default=str))
        tmp.replace(dest)  # a reader never sees a half-written file
        return dest

    def _evict(self):
        while len(self._sessions) >= self.max_sessions:
            oldest = min(self._last_used, key=self._last_used.get)
            self._sessions.pop(oldest, None)
            self._last_used.pop(oldest, None)
