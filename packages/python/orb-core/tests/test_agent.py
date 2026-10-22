import json

import pytest

from orb_core.agent import Agent, Event, InputError, Session, SessionStore, check_session_id, new_session_id


class EchoSession:
    def __init__(self, topic, sid):
        self.id, self.topic, self.agent_name, self.said = sid, topic, "echo", []

    async def turn(self, text, channel="web"):
        self.said.append(text)
        yield Event("token", text)
        yield Event("reply", {"text": text})

    def state(self):
        return {"id": self.id, "said": self.said}

    async def settle(self):
        return None

    def snapshot(self):
        return self.state()


class EchoAgent:
    name = "echo"

    def new_session(self, topic, session_id):
        return EchoSession(topic, session_id)

    async def health(self):
        return {"ok": True}


def test_fakes_satisfy_the_protocols():
    assert isinstance(EchoAgent(), Agent)
    assert isinstance(EchoSession("t", "abc"), Session)


def test_event_rejects_unknown_types_and_serializes():
    with pytest.raises(ValueError):
        Event("nope")
    assert json.loads(Event("token", "hi").to_json()) == {"type": "token", "data": "hi"}


def test_session_ids():
    sid = new_session_id()
    assert check_session_id(sid) == sid
    for bad in ["", "AB", "a", "has space", "../etc", "x" * 49, "-start"]:
        with pytest.raises(InputError):
            check_session_id(bad)


async def _collect(session, text):
    return [e async for e in session.turn(text)]


def test_store_create_get_save(tmp_path):
    import asyncio

    store = SessionStore({"echo": EchoAgent()}, save_dir=tmp_path)
    s = store.create("A topic")
    assert store.get(s.id) is s
    events = asyncio.run(_collect(s, "hello"))
    assert [e.type for e in events] == ["token", "reply"]
    path = store.save(s)
    assert json.loads(path.read_text()) == {"id": s.id, "said": ["hello"]}
    assert not list(tmp_path.glob("*.tmp"))


def test_store_rejects_bad_input():
    store = SessionStore({"echo": EchoAgent()})
    with pytest.raises(InputError):
        store.create("   ")
    with pytest.raises(InputError):
        store.create("x" * 2001)
    with pytest.raises(InputError):
        store.create("ok", agent="missing")
    store.create("ok", session_id="room-1")
    with pytest.raises(InputError):
        store.create("ok", session_id="room-1")


def test_store_get_or_create_and_eviction():
    store = SessionStore({"echo": EchoAgent()}, max_sessions=2)
    a = store.get_or_create("aaa", "first")
    assert store.get_or_create("aaa", "ignored") is a
    store.create("b", session_id="bbb")
    store.get("aaa")  # aaa is now the most recently used
    store.create("c", session_id="ccc")  # evicts bbb
    assert store.get("bbb") is None and store.get("aaa") is a
