"""The agent service API with the real discussion agent, a scripted reader and a fake LLM. No network."""
import asyncio
import json
from pathlib import Path
import time

import jwt
import pytest
from fastapi.testclient import TestClient

from orb_core.agent import SessionStore
from orb_discussion.agent import DiscussionAgent
from orb_agent.server import create_app
from orb_agent.voice import VoiceConfig, host_port, is_local, listening, livekit_answers

LOCAL = "http://127.0.0.1:8100"
TUNNEL = "https://quiet-river-1234.trycloudflare.com"


def local(app):
    """A client on this computer. A request with another Host counts as remote, and the gate refuses it (access.py)."""
    return TestClient(app, base_url=LOCAL)


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
    c = local(app_and_dir[0])
    assert "orb" in c.get("/").text
    h = c.get("/health").json()
    assert h["agents"]["discussion"]["llm"]["ok"] is True and h["voice"]["dev"] is True
    assert "secret" not in json.dumps(h)


def test_session_turn_and_save(app_and_dir):
    app, folder = app_and_dir
    c = local(app)
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
    c = local(app_and_dir[0])
    assert c.get("/v1/sessions/nope-404").status_code == 404
    assert "lower-case letters" in c.get("/v1/sessions/BAD%20ID").json()["error"]
    sid = c.post("/v1/sessions", json={"topic": "T"}).json()["id"]
    assert c.post(f"/v1/sessions/{sid}/turns", json={"text": "hi", "channel": "fax"}).status_code == 400
    assert c.post(f"/v1/sessions/{sid}/turns", json={"text": ""}).status_code == 422
    assert c.post("/v1/sessions", json={"topic": "T", "agent": "nope"}).status_code == 400


def test_put_creates_once(app_and_dir):
    c = local(app_and_dir[0])
    a = c.put("/v1/sessions/room-7", json={"topic": "Voice room"}).json()
    b = c.put("/v1/sessions/room-7", json={"topic": "Another topic"}).json()
    assert a["id"] == b["id"] == "room-7" and b["topic"] == "Voice room"


def test_voice_token_is_for_the_session_room(app_and_dir):
    c = local(app_and_dir[0])
    sid = c.post("/v1/sessions", json={"topic": "T"}).json()["id"]
    t = c.post("/v1/voice/token", json={"session_id": sid}).json()
    claims = jwt.decode(t["token"], "secret", algorithms=["HS256"])
    assert t["room"] == f"orb-{sid}" and claims["video"]["room"] == t["room"] and claims["video"]["roomJoin"] is True
    assert c.post("/v1/voice/token", json={"session_id": "missing-1"}).status_code == 404


def test_voice_off_gives_503(tmp_path):
    store = SessionStore({"discussion": DiscussionAgent(ScriptedReader([]), FakeLLM())})
    c = local(create_app(store, None))
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


def test_session_config_picks_an_llm_and_the_worker_reads_it(tmp_path):
    made = []

    class NamedLLM(FakeLLM):
        def __init__(self, model):
            self.model = model

        async def stream(self, messages, options=None):
            yield f"from {self.model}"

    def model_agent(model):
        made.append(model)
        reader = ScriptedReader([{"claims": 1}])
        return DiscussionAgent(reader, NamedLLM(model), load_source=lambda url: {"url": url, "error": "offline"})

    async def installed():
        return ["fake", "qwen3:4b"]

    base = DiscussionAgent(ScriptedReader([]), FakeLLM(), load_source=lambda url: {"url": url, "error": "offline"})
    app = create_app(SessionStore({"discussion": base}, save_dir=tmp_path), None, model_agent=model_agent,
                     installed_models=installed, default_model="fake")
    c = local(app)
    assert c.get("/v1/models").json()["llm"] == {"default": "fake", "installed": ["fake", "qwen3:4b"]}

    cfg = {"stt_engine": "parakeet", "llm_model": "qwen3:4b", "tts_engine": "kokoro", "tts_voice": "hf_alpha",
           "tts_lang": "hi"}
    s = c.post("/v1/sessions", json={"topic": "Bench", "config": cfg}).json()
    assert s["config"] == cfg and made == ["qwen3:4b"]
    sid = s["id"]
    lines = [json.loads(x) for x in c.post(f"/v1/sessions/{sid}/turns", json={"text": "We should hash passwords."}).iter_lines() if x]
    assert "".join(e["data"] for e in lines if e["type"] == "token") == "from qwen3:4b"
    assert c.get(f"/v1/sessions/{sid}").json()["config"]["llm_model"] == "qwen3:4b"  # what the voice worker reads

    # a second session with the same model reuses its agent; the default model needs no new agent
    c.post("/v1/sessions", json={"topic": "Bench 2", "config": {"llm_model": "qwen3:4b"}})
    c.post("/v1/sessions", json={"topic": "Default", "config": {"llm_model": "fake"}})
    assert made == ["qwen3:4b"]


def test_session_config_refuses_unknown_models_and_llm_changes(tmp_path):
    async def installed():
        return ["fake"]

    base = DiscussionAgent(ScriptedReader([]), FakeLLM(), load_source=lambda url: {"url": url, "error": "offline"})
    app = create_app(SessionStore({"discussion": base}, save_dir=tmp_path), None, installed_models=installed,
                     model_agent=lambda m: base, default_model="fake")
    c = local(app)
    r = c.post("/v1/sessions", json={"topic": "x", "config": {"llm_model": "llama-huge"}})
    assert r.status_code == 400 and "no model 'llama-huge'" in r.json()["error"] and "fake" in r.json()["error"]
    sid = c.post("/v1/sessions", json={"topic": "x"}).json()["id"]
    assert c.put(f"/v1/sessions/{sid}/config", json={"tts_voice": "am_michael"}).json() == {"tts_voice": "am_michael"}
    r = c.put(f"/v1/sessions/{sid}/config", json={"llm_model": "other"})
    assert r.status_code == 400 and "fixed when it starts" in r.json()["error"]


def test_the_bench_tools_are_tabs_of_the_chat_page(app_and_dir):
    import re

    c = local(app_and_dir[0])
    r = c.get("/bench", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == "/"   # the old address of the bench page still works
    html = c.get("/").text
    for tab in ("chat", "stt", "tts", "llm", "suites"):
        assert f'data-tab="{tab}"' in html and f'id="tab-{tab}"' in html
    for bar in ("m-stt", "m-llm", "m-tts", "m-voice", "m-lang", "m-style"):
        assert f'id="{bar}"' in html
    # every element that the script looks up by id is in the page: a typo would stop the whole script
    script = html[html.index("<script>", html.index("livekit-client")):]
    used = set(re.findall(r'\$\("([a-zA-Z0-9_-]+)"\)', script))
    ids = set(re.findall(r'\bid="([a-zA-Z0-9_-]+)"', html))
    assert used - ids == set(), f"the script uses ids that the page does not have: {sorted(used - ids)}"


def test_the_talk_tab_is_an_orb_and_two_buttons_with_no_session_start_screen(app_and_dir):
    c = local(app_and_dir[0])
    html = c.get("/").text
    for needed in ('id="orb"', 'id="orb-canvas"', 'id="orb-state"', 'id="talk"', 'id="new-session"', "New session"):
        assert needed in html
    # no topic, no Start button, no examples, no text box: a session starts by itself when the person presses Talk
    for gone in ('id="topic"', 'id="go"', 'id="start"', 'id="examples"', 'id="composer"', 'id="text"', 'id="send"',
                 "startSession", "loadSession", "history.replaceState"):
        assert gone not in html, gone
    assert 'topic: "Open discussion"' in html            # the session that Talk makes has a fixed topic
    # the orb is LiveKit's Aura visualizer: the credit and the license files are shipped with it
    assert "agent-audio-visualizer-aura" in html and "Apache-2.0" in html
    assert "lk.agent.state" in html                       # LiveKit's own signal for the state of the agent
    vendor = Path(__file__).parents[1] / "src" / "orb_agent" / "static" / "vendor"
    assert "Apache License" in (vendor / "LICENSE-livekit-components-js.txt").read_text()
    assert "LiveKit, Inc." in (vendor / "NOTICE-livekit-components-js.txt").read_text()
    # the states of LiveKit's table, and the speed of each: idle 10, listening 20, thinking 30, speaking 70
    for state, speed in (("idle", 10), ("listening", 20), ("thinking", 30), ("speaking", 70)):
        assert f"{state}: " in html and f"o.speed = {speed};" in html


def test_the_captions_show_what_the_stt_heard_and_what_the_tts_speaks(app_and_dir):
    html = local(app_and_dir[0]).get("/").text
    for needed in ('id="captions"', 'id="cap-you"', 'id="cap-agent"', 'id="cap-you-who"', 'id="cap-agent-who"'):
        assert needed in html
    # both come from LiveKit's transcription streams, and the page tells the two senders apart by identity
    assert 'registerTextStreamHandler("lk.transcription"' in html and "lk.segment_id" in html
    assert "lk.transcription_final" in html and "sender.identity === myIdentity" in html
    assert "heard by" in html and "spoken by" in html     # the label names the model: useful when models are compared


def test_the_real_service_is_private_to_this_computer(tmp_path):
    base = DiscussionAgent(ScriptedReader([]), FakeLLM(), load_source=lambda url: {"url": url, "error": "offline"})
    voice = VoiceConfig("ws://127.0.0.1:7880", "devkey", "secret", dev=True)

    def app(key=None):
        return create_app(SessionStore({"discussion": base}, save_dir=tmp_path), voice, share_key=key)

    tunnel = TestClient(app(), base_url=TUNNEL)
    for path in ("/", "/health", "/v1/sessions", "/v1/voice/status", "/v1/models"):
        assert tunnel.get(path).status_code == 403, path        # no key is set: nothing goes to a remote client
    assert tunnel.post("/v1/voice/token", json={"session_id": "x-123"}).status_code == 403
    assert tunnel.post("/v1/sessions", json={"topic": "T"}).status_code == 403
    assert local(app()).get("/health").status_code == 200

    shared = TestClient(app("0123456789abcdef"), base_url=TUNNEL)
    assert shared.get("/health").status_code == 403
    assert shared.get("/health", headers={"x-orb-key": "0123456789abcdef"}).status_code == 200
    with pytest.raises(ValueError, match="16 or more"):
        app("short")


def probe_of(up):
    calls = []

    async def probe(host, port):
        calls.append((host, port))
        return (host, port) in up

    probe.calls = calls
    return probe


def status_of(voice, probe=None, worker_port=8081):
    store = SessionStore({"discussion": DiscussionAgent(ScriptedReader([]), FakeLLM())})
    c = local(create_app(store, voice, probe=probe, worker_port=worker_port))
    status = c.get("/health").json()["voice"]
    assert c.get("/v1/voice/status").json() == status           # the light route says the same as /health
    return status


def test_the_voice_status_says_what_runs():
    dev = VoiceConfig("ws://127.0.0.1:7880", "devkey", "secret", dev=True)
    up = probe_of({("127.0.0.1", 7880), ("127.0.0.1", 8081)})
    assert status_of(dev, up) == {"url": "ws://127.0.0.1:7880", "dev": True, "local": True, "proxy": True, "livekit": True, "worker": True}
    assert set(up.calls) == {("127.0.0.1", 7880), ("127.0.0.1", 8081)}
    assert status_of(dev, probe_of({("127.0.0.1", 7880)}))["worker"] is False
    down = status_of(dev, probe_of(set()))
    assert down["livekit"] is False and down["worker"] is False
    assert status_of(dev, up, worker_port=None)["worker"] is None   # no worker port: not checked
    unchecked = status_of(dev)                                       # no probe: nothing is checked
    assert unchecked["livekit"] is None and unchecked["worker"] is None


def test_the_voice_status_of_a_server_on_the_internet():
    cloud = VoiceConfig("wss://orb-1a2b3c.livekit.cloud", "APIkey", "s" * 40, dev=False)
    status = status_of(cloud, probe_of({("orb-1a2b3c.livekit.cloud", 443), ("127.0.0.1", 8081)}))
    assert status == {"url": "wss://orb-1a2b3c.livekit.cloud", "dev": False, "local": False, "proxy": False, "livekit": True, "worker": True}
    assert "s" * 40 not in json.dumps(status) and "APIkey" not in json.dumps(status)
    assert status_of(None) is None                                   # voice off


def test_host_and_port_of_a_voice_url_and_the_listening_check():
    assert host_port("ws://127.0.0.1:7880") == ("127.0.0.1", 7880)
    assert host_port("wss://orb.livekit.cloud") == ("orb.livekit.cloud", 443)
    assert host_port("ws://voice.example.com") == ("voice.example.com", 80)
    assert is_local("ws://localhost:7880") and is_local("ws://[::1]:7880") and not is_local("wss://voice.example.com")
    assert VoiceConfig("ws://localhost:7880", "k", "s", dev=False).local is True

    async def run():
        server = await asyncio.start_server(lambda r, w: w.close(), "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            assert await listening("127.0.0.1", port) is True
        finally:
            server.close()
            await server.wait_closed()
        assert await listening("127.0.0.1", port) is False           # the port is closed now
    asyncio.run(run())


def test_the_page_says_why_voice_cannot_start_on_another_device(app_and_dir):
    html = local(app_and_dir[0]).get("/").text
    # the browser knows two things that the server does not: whether the page is secure, and which host it was opened on
    for needed in ("window.isSecureContext", "/v1/voice/status", "LIVEKIT_PUBLIC_URL", "LOCAL_HOSTS.includes(location.hostname)",
                   "Voice: not available", "block?.hard", "Voice is not available here",
                   "function signalUrl", "v.proxy", "pc connection"):
        assert needed in html, needed


def test_livekit_answers_only_when_its_health_page_says_ok():
    requests = []

    async def serve(body, status="200 OK", silent=False):
        async def handle(reader, writer):
            requests.append(await reader.readuntil(b"\r\n\r\n"))
            if silent:
                await reader.read()                      # the client gives up (its time limit) and closes
            else:
                data = body.encode()
                head = f"HTTP/1.1 {status}\r\nContent-Type: text/plain\r\nContent-Length: {len(data)}\r\nConnection: close\r\n\r\n"
                writer.write(head.encode() + data)
                await writer.drain()
            writer.close()
        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        return server, server.sockets[0].getsockname()[1]

    async def answer(body, status="200 OK", silent=False):
        server, port = await serve(body, status, silent)
        try:
            return await livekit_answers(f"ws://127.0.0.1:{port}", timeout=0.3), await livekit_answers(f"http://127.0.0.1:{port}", timeout=0.3)
        finally:
            server.close()
            await server.wait_closed()

    async def run():
        assert await answer("OK") == (True, True)                                   # ws:// is asked as http://
        assert await answer("OK\n") == (True, True)
        assert await answer("<html>ERR_NGROK_3200: the tunnel is offline</html>", "404 Not Found") == (False, False)
        assert await answer("<html>a page</html>") == (False, False)                # 200, but not LiveKit's page
        assert await answer("OK", "502 Bad Gateway") == (False, False)
        assert await answer("", silent=True) == (False, False)                      # accepts and never answers: the time limit
        assert await livekit_answers("ws://127.0.0.1:1", timeout=0.3) is False      # nothing listens
    asyncio.run(asyncio.wait_for(run(), 20))                                       # a test must not hang the suite
    assert all(b"ngrok-skip-browser-warning: 1" in r for r in requests)            # a free ngrok domain would show a page otherwise


def test_the_voice_status_checks_a_public_livekit_by_its_address():
    cloud = VoiceConfig("wss://backless.example.dev", "APIkey", "s" * 40, dev=False)
    asked = []

    async def livekit_probe(url):
        asked.append(url)
        return url == "wss://backless.example.dev"

    tcp = probe_of({("127.0.0.1", 8081)})
    store = SessionStore({"discussion": DiscussionAgent(ScriptedReader([]), FakeLLM())})
    status = local(create_app(store, cloud, probe=tcp, worker_port=8081, livekit_probe=livekit_probe)).get("/v1/voice/status").json()
    assert status == {"url": "wss://backless.example.dev", "dev": False, "local": False, "proxy": False, "livekit": True, "worker": True}
    assert asked == ["wss://backless.example.dev"] and tcp.calls == [("127.0.0.1", 8081)]   # the worker by TCP, the server by address


# --- Gemini Live (docs/design/gemini-live.md) ------------------------------------------------------------------------
class CountingLLM(FakeLLM):
    def __init__(self):
        self.warmed = 0

    async def warm(self, messages):
        self.warmed += 1


def gemini_app(tmp_path, key: bool = True):
    llm = CountingLLM()
    agent = DiscussionAgent(ScriptedReader([]), llm, load_source=lambda url: {"url": url, "error": "offline test"})
    app = create_app(SessionStore({"discussion": agent}, save_dir=tmp_path), None, gemini_available=lambda: key)
    return local(app), llm


def test_models_list_the_gemini_options_and_whether_the_key_is_set(tmp_path):
    c, _ = gemini_app(tmp_path)
    g = c.get("/v1/models").json()["gemini"]
    assert g["available"] is True and g["default_model"] == "gemini-3.8-live" and g["default_voice"] == "Kore"
    assert "gemini-3.8-live-extended-thinking" in g["models"] and "Kore" in g["voices"]
    assert gemini_app(tmp_path, key=False)[0].get("/v1/models").json()["gemini"]["available"] is False


def test_a_gemini_session_never_warms_the_llm_and_an_orb_session_does(tmp_path):
    c, llm = gemini_app(tmp_path)
    cfg = {"voice_mode": "gemini-live", "gemini_model": "gemini-3.8-live", "gemini_voice": "Kore"}
    s = c.post("/v1/sessions", json={"topic": "Open discussion", "config": cfg}).json()
    assert s["config"] == cfg
    c.get(f"/v1/sessions/{s['id']}")  # the background prepare has run by the next request
    assert llm.warmed == 0
    c.post("/v1/sessions", json={"topic": "Open discussion"})
    c.get("/v1/sessions")
    assert llm.warmed == 1


@pytest.mark.parametrize("cfg, why", [
    ({"voice_mode": "telepathy"}, "voice_mode must be one of"),
    ({"voice_mode": "gemini-live", "gemini_model": "gemini-1.0-pro"}, "gemini_model must be one of"),
    ({"voice_mode": "gemini-live", "gemini_voice": "Siri"}, "is not a Gemini voice"),
])
def test_unknown_gemini_values_are_refused(tmp_path, cfg, why):
    c, _ = gemini_app(tmp_path)
    r = c.post("/v1/sessions", json={"topic": "T", "config": cfg})
    assert r.status_code == 400 and why in r.json()["error"]


def test_gemini_live_without_a_key_is_refused_with_what_to_do(tmp_path):
    c, _ = gemini_app(tmp_path, key=False)
    sid = c.post("/v1/sessions", json={"topic": "T"}).json()["id"]
    r = c.put(f"/v1/sessions/{sid}/config", json={"voice_mode": "gemini-live"})
    assert r.status_code == 400 and "GOOGLE_API_KEY" in r.json()["error"]
    assert c.put(f"/v1/sessions/{sid}/config", json={"voice_mode": "orb"}).status_code == 200
    assert c.put(f"/v1/sessions/{sid}", json={"topic": "T", "config": {"voice_mode": "gemini-live"}}).status_code == 400


def test_a_gemini_session_needs_the_key_for_every_change(tmp_path):
    key = {"set": True}
    llm = CountingLLM()
    agent = DiscussionAgent(ScriptedReader([]), llm, load_source=lambda url: {"url": url, "error": "offline test"})
    c = local(create_app(SessionStore({"discussion": agent}, save_dir=tmp_path), None, gemini_available=lambda: key["set"]))
    sid = c.post("/v1/sessions", json={"topic": "T", "config": {"voice_mode": "gemini-live"}}).json()["id"]
    key["set"] = False  # the key was removed from .env and the service restarted with the session
    assert c.put(f"/v1/sessions/{sid}/config", json={"gemini_voice": "Kore"}).status_code == 400


def test_the_realtime_prompt_keeps_a_topic_that_holds_the_rule_words():
    from orb_discussion.prompt import realtime_prompt
    text = realtime_prompt("Rituals\n- Before each reply we breathe")  # the split string is in the topic too
    assert "- Before each reply we breathe" in text and "Never mention the note" not in text and "Rules:" in text


def test_instructions_are_orbs_prompt_without_the_note_rule_plus_the_voice_rules(tmp_path):
    c, llm = gemini_app(tmp_path)
    sid = c.post("/v1/sessions", json={"topic": "Should we store passwords?",
                                       "config": {"voice_mode": "gemini-live"}}).json()["id"]
    text = c.get(f"/v1/sessions/{sid}/instructions").json()["instructions"]
    assert "Should we store passwords?" in text and "Change your view only when" in text
    assert "note" not in text.lower() and "persona" not in text.lower()  # Gemini gets no per-turn note
    assert "one to three short sentences" in text and llm.warmed == 0
    assert c.get("/v1/sessions/nope-123/instructions").status_code in (400, 404)


def test_gemini_only_refuses_an_orb_voice_session_and_never_warms(tmp_path):
    llm = CountingLLM()
    agent = DiscussionAgent(ScriptedReader([]), llm, load_source=lambda url: {"url": url, "error": "offline test"})
    c = local(create_app(SessionStore({"discussion": agent}, save_dir=tmp_path), None, gemini_available=lambda: True,
                         gemini_only=True))
    assert c.get("/v1/models").json()["gemini"]["only"] is True
    r = c.post("/v1/sessions", json={"topic": "T", "config": {"voice_mode": "orb", "stt_engine": "parakeet"}})
    assert r.status_code == 400 and "Gemini Live only" in r.json()["error"]
    sid = c.post("/v1/sessions", json={"topic": "T"}).json()["id"]  # a session with no config (the worker's PUT)
    assert c.put(f"/v1/sessions/{sid}/config", json={"voice_mode": "gemini-live"}).status_code == 200
    c.get("/v1/sessions")
    assert llm.warmed == 0
