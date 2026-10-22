"""The agent service API with the real discussion agent, a scripted reader and a fake LLM. No network."""
import json
import time

import jwt
import pytest
from fastapi.testclient import TestClient

from orb_core.agent import SessionStore
from orb_discussion.agent import DiscussionAgent
from orb_agent.server import create_app
from orb_agent.voice import VoiceConfig


class _A:
    def __init__(self, p):
        self.noul = p


class _R:
    def __init__(self, answers):
        self.answers, self.model = answers, "scripted"
        self.usage = type("U", (), {"input_tokens": 500})()


class ScriptedReader:
    def __init__(self, script):
        self.script = list(script)

    def system_one(self, state, questions):
        item = self.script.pop(0) if self.script else {}
        return _R({q: _A(item.get(q, 0.0)) for q in questions})


class FakeLLM:
    model = "fake"

    async def stream(self, messages, options=None):
        for w in ["Hashing", " is", " safer."]:
            yield w

    async def warm(self, messages):
        pass

    async def health(self):
        return {"ok": True, "model": "fake"}


@pytest.fixture
def app_and_dir(tmp_path):
    reader = ScriptedReader([{"claims": 1, "new_option": 1}, {"new_option": 1, "disagrees": 1}])
    agent = DiscussionAgent(reader, FakeLLM(), load_source=lambda url: {"url": url, "error": "offline test"})
    store = SessionStore({"discussion": agent}, save_dir=tmp_path)
    return create_app(store, VoiceConfig("ws://127.0.0.1:7880", "devkey", "secret", dev=True)), tmp_path


def test_page_and_health(app_and_dir):
    c = TestClient(app_and_dir[0])
    assert "orb" in c.get("/").text
    h = c.get("/health").json()
    assert h["agents"]["discussion"]["llm"]["ok"] is True and h["voice"]["dev"] is True
    assert "secret" not in json.dumps(h)


def test_session_turn_and_save(app_and_dir):
    app, folder = app_and_dir
    c = TestClient(app)
    s = c.post("/v1/sessions", json={"topic": "Should our assistant store passwords?"})
    assert s.status_code == 201
    sid = s.json()["id"]
    r = c.post(f"/v1/sessions/{sid}/turns", json={"text": "We should store passwords in a JSON file.", "channel": "web"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/x-ndjson")
    events = [json.loads(line) for line in r.text.splitlines() if line]
    assert [e["type"] for e in events] == ["read", "decision", "token", "token", "token", "reply", "check", "state"]
    assert events[1]["data"]["persona"] == "Rival" and events[-2]["data"]["done"] is True
    state = c.get(f"/v1/sessions/{sid}").json()
    assert [m["speaker"] for m in state["messages"]] == ["human", "ai"] and state["stage"] == "Test"
    assert c.get(f"/v1/sessions/{sid}?since={state['version']}").json() == {"version": state["version"], "unchanged": True}
    for _ in range(100):  # the save runs in the background after the turn
        if (folder / f"{sid}.json").is_file():
            break
        time.sleep(0.02)
    saved = json.loads((folder / f"{sid}.json").read_text())
    assert saved["trace"][1]["speaker"] == "ai"


def test_errors_are_clear(app_and_dir):
    c = TestClient(app_and_dir[0])
    assert c.get("/v1/sessions/nope-404").status_code == 404
    assert "lower-case letters" in c.get("/v1/sessions/BAD%20ID").json()["error"]
    sid = c.post("/v1/sessions", json={"topic": "T"}).json()["id"]
    assert c.post(f"/v1/sessions/{sid}/turns", json={"text": "hi", "channel": "fax"}).status_code == 400
    assert c.post(f"/v1/sessions/{sid}/turns", json={"text": ""}).status_code == 422
    assert c.post("/v1/sessions", json={"topic": "T", "agent": "nope"}).status_code == 400


def test_put_creates_once(app_and_dir):
    c = TestClient(app_and_dir[0])
    a = c.put("/v1/sessions/room-7", json={"topic": "Voice room"}).json()
    b = c.put("/v1/sessions/room-7", json={"topic": "Another topic"}).json()
    assert a["id"] == b["id"] == "room-7" and b["topic"] == "Voice room"


def test_voice_token_is_for_the_session_room(app_and_dir):
    c = TestClient(app_and_dir[0])
    sid = c.post("/v1/sessions", json={"topic": "T"}).json()["id"]
    t = c.post("/v1/voice/token", json={"session_id": sid}).json()
    claims = jwt.decode(t["token"], "secret", algorithms=["HS256"])
    assert t["room"] == f"orb-{sid}" and claims["video"]["room"] == t["room"] and claims["video"]["roomJoin"] is True
    assert c.post("/v1/voice/token", json={"session_id": "missing-1"}).status_code == 404


def test_voice_off_gives_503(tmp_path):
    store = SessionStore({"discussion": DiscussionAgent(ScriptedReader([]), FakeLLM())})
    c = TestClient(create_app(store, None))
    sid = c.post("/v1/sessions", json={"topic": "T"}).json()["id"]
    assert c.post("/v1/voice/token", json={"session_id": sid}).status_code == 503


def test_voice_config_rules(monkeypatch):
    for k in ("LIVEKIT_PUBLIC_URL", "LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
        monkeypatch.delenv(k, raising=False)
    assert VoiceConfig.from_env().dev is True  # loopback with no keys: the dev server credentials
    monkeypatch.setenv("LIVEKIT_URL", "wss://voice.example.com")
    assert VoiceConfig.from_env() is None  # a remote server with no keys: voice is off
    monkeypatch.setenv("LIVEKIT_API_KEY", "k")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "short")
    with pytest.raises(ValueError, match="32 or more"):
        VoiceConfig.from_env()
