"""Gemini Live (docs/design/gemini-live.md, GL-2): the model options, the plugin's lists, the error messages, and a
call wired with a fake session. No call goes to Google."""
import asyncio
from typing import get_args

import httpx
import pytest
from livekit.agents import llm
from livekit.plugins.google.realtime import api_proto

from orb_core import gemini
from orb_voice import gemini_live, main


def test_orbs_gemini_lists_are_the_plugins():
    assert set(gemini.MODELS) <= set(get_args(api_proto.LiveAPIModels))
    assert set(gemini.VOICES) == set(get_args(api_proto.Voice))


def test_options_carry_the_session_choice_transcripts_and_long_call_settings():
    o = gemini_live.options({"gemini_model": "gemini-3.8-live-extended-thinking", "gemini_voice": "Kore"})
    assert (o["model"], o["voice"]) == ("gemini-3.8-live-extended-thinking", "Kore")
    assert o["input_audio_transcription"] is not None and o["output_audio_transcription"] is not None
    assert o["context_window_compression"].sliding_window is not None and o["session_resumption"] is not None
    assert gemini_live.options({"gemini_model": "gpt-9", "gemini_voice": "Siri"})["model"] == gemini.DEFAULT_MODEL
    assert gemini_live.options({})["voice"] == gemini.DEFAULT_VOICE


def test_the_model_is_built_without_connecting(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key-for-the-test")
    m = gemini_live.make_model({"gemini_voice": "Kore"})
    assert m.model == "gemini-3.8-live" and m.capabilities.turn_detection and m.capabilities.user_transcription


def model_error(e: Exception) -> llm.RealtimeModelError:
    return llm.RealtimeModelError(timestamp=0.0, label="gemini", error=e, recoverable=False)


@pytest.mark.parametrize("error, says", [
    (Exception("1007 API key not valid. Please pass a valid API key."), "refused the API key"),
    (Exception("received 1011 RESOURCE_EXHAUSTED: You exceeded your current quota"), "over its quota"),
    (Exception("1008 policy violation: models/gemini-9 is not found"), "does not accept this model"),
    (ConnectionError("[Errno 8] nodename nor servname provided, or not known"), "cannot be reached"),
    (RuntimeError("something new"), "voice worker log has the details"),
])
def test_errors_become_a_message_that_says_what_to_do(error, says):
    assert says in gemini_live.explain(model_error(error))


def test_instructions_come_from_the_agent_service():
    def handler(req):
        assert req.url.path == "/v1/sessions/web-1/instructions"
        return httpx.Response(200, json={"instructions": "Be brief."})

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await gemini_live.fetch_instructions(http, "http://agent", "web-1")

    assert asyncio.run(go()) == "Be brief."


class FakeSession:
    """An AgentSession that records what the call gives it and lets the test fire its events."""
    made: list["FakeSession"] = []

    def __init__(self, **kwargs):
        self.kwargs, self.handlers, self.started = kwargs, {}, None
        FakeSession.made.append(self)

    def on(self, name):
        def add(fn):
            self.handlers.setdefault(name, []).append(fn)
            return fn
        return add

    def fire(self, name, ev):
        for fn in self.handlers.get(name, []):
            fn(ev)

    async def start(self, room, agent):
        self.started = (room, agent)


class FakeBench:
    def __init__(self):
        self.sent = []

    def send(self, kind, **data):
        self.sent.append((kind, data))


class Ctx:
    room = object()


def gemini_call(monkeypatch, instructions_status=200):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key-for-the-test")
    monkeypatch.setattr(main, "AgentSession", FakeSession)
    FakeSession.made = []

    def handler(req):
        return httpx.Response(instructions_status, json={"instructions": "orb rules"})

    bench = FakeBench()

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            s = main.Settings.from_env()
            await main.gemini_call(Ctx(), http, s, "web-1", {"voice_mode": "gemini-live", "gemini_voice": "Kore"}, bench)

    asyncio.run(go())
    return FakeSession.made[0], bench


def test_a_gemini_call_uses_only_the_realtime_model_and_the_session_instructions(monkeypatch):
    session, bench = gemini_call(monkeypatch)
    assert set(session.kwargs) == {"llm"}  # no STT, TTS, VAD or turn detector: nothing of the speech service
    assert session.kwargs["llm"].model == "gemini-3.8-live"
    assert session.started[1].instructions == "orb rules"
    kind, data = bench.sent[-1]
    assert kind == "session" and data["tts"] == "gemini" and data["voice"] == "Kore"


def test_a_worker_without_the_key_tells_the_page_and_does_not_crash(monkeypatch):
    monkeypatch.setattr(main, "AgentSession", FakeSession)
    FakeSession.made = []
    bench = FakeBench()

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"instructions": "x"}))) as http:
            monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
            await main.gemini_call(Ctx(), http, main.Settings.from_env(), "web-1", {"voice_mode": "gemini-live"}, bench)

    asyncio.run(go())
    assert FakeSession.made == [] and bench.sent == [("error", {"error": bench.sent[0][1]["error"]})]
    assert "no API key in the voice worker" in bench.sent[0][1]["error"]


def test_a_gemini_call_starts_with_the_voice_rules_when_the_agent_service_fails(monkeypatch):
    session, _ = gemini_call(monkeypatch, instructions_status=500)
    assert session.started[1].instructions == gemini.VOICE_RULES.strip()


def test_a_gemini_error_reaches_the_page_once_and_retries_stay_quiet(monkeypatch):
    session, bench = gemini_call(monkeypatch)
    retry = llm.RealtimeModelError(timestamp=0.0, label="gemini", error=Exception("1011 blip"), recoverable=True)
    session.fire("error", type("Ev", (), {"error": retry})())
    fatal = model_error(Exception("API key not valid"))
    for _ in range(2):
        session.fire("error", type("Ev", (), {"error": fatal})())
    errors = [d["error"] for k, d in bench.sent if k == "error"]
    assert len(errors) == 1 and "refused the API key" in errors[0]


def test_both_modes_share_the_transcript_and_state_events(monkeypatch):
    session, bench = gemini_call(monkeypatch)
    session.fire("user_input_transcribed", type("Ev", (), {"is_final": True, "transcript": "Namaste, kaise ho?"})())
    session.fire("agent_state_changed", type("Ev", (), {"new_state": "speaking"})())
    kinds = [k for k, _ in bench.sent]
    assert "transcript" in kinds and "agent_state" in kinds
    assert not any(k == "polished" for k in kinds)  # no polish: the speech service is not used
