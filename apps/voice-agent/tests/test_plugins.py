"""The plugins against fake services: STT, TTS and LLM over HTTP, the undo rule, and interim words."""
import asyncio
import json

import httpx
import numpy as np
import pytest
from livekit import rtc
from livekit.agents import APIConnectOptions, APIError, APIStatusError, llm, stt, utils, vad
from livekit.agents.types import NOT_GIVEN

from orb_voice.plugins import LiveSTT, OrbLLM, OrbSTT, OrbTTS, TurnTracker, turn_text

NO_RETRY = APIConnectOptions(max_retry=0)


def frame(value: int = 1000, n: int = 320, rate: int = 16000) -> rtc.AudioFrame:
    return rtc.AudioFrame(data=np.full(n, value, np.int16).tobytes(), sample_rate=rate, num_channels=1,
                          samples_per_channel=n)


def client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def ndjson(*events: dict) -> bytes:
    return b"".join(json.dumps(e).encode() + b"\n" for e in events)


def chat(*messages: tuple[str, str]) -> llm.ChatContext:
    ctx = llm.ChatContext.empty()
    for role, text in messages:
        ctx.add_message(role=role, content=text)
    return ctx


async def reply_text(stream: llm.LLMStream) -> str:
    parts = []
    try:
        async for chunk in stream:
            if chunk.delta and chunk.delta.content:
                parts.append(chunk.delta.content)
    finally:
        await stream.aclose()
    return "".join(parts)


# --- STT -------------------------------------------------------------------------------------------------------------

def test_stt_sends_pcm16_with_rate_language_and_engine():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["path"], seen["params"], seen["bytes"] = req.url.path, dict(req.url.params), len(req.content)
        return httpx.Response(200, json={"text": "hello there", "language": "en", "ms": 12.0, "audio_ms": 40})

    async def go():
        async with client(handler) as http:
            return await OrbSTT(http, "http://speech", engine="whistle").recognize(buffer=[frame(), frame()])

    ev = asyncio.run(go())
    assert ev.alternatives[0].text == "hello there" and ev.alternatives[0].end_time == 0.04
    assert seen == {"path": "/v1/stt", "params": {"sample_rate": "16000", "language": "en", "engine": "whistle"},
                    "bytes": 2 * 320 * 2}


def test_stt_errors_carry_the_service_message():
    def handler(req):
        return httpx.Response(413, json={"error": "The audio is longer than 60 s."})

    async def go():
        async with client(handler) as http:
            await OrbSTT(http, "http://speech").recognize(buffer=frame(), conn_options=NO_RETRY)

    with pytest.raises(APIStatusError, match="longer than 60 s"):
        asyncio.run(go())


# --- TTS -------------------------------------------------------------------------------------------------------------

def test_tts_sends_the_options_and_streams_pcm_at_24khz():
    pcm = np.arange(4800, dtype=np.int16).tobytes()  # 0.2 s
    bodies = []

    def handler(req):
        bodies.append(json.loads(req.content))
        return httpx.Response(200, content=pcm, headers={"X-Sample-Rate": "24000"})

    async def go():
        async with client(handler) as http:
            stream = OrbTTS(http, "http://speech", voice="am_michael").synthesize("Hello there.")
            frames = []
            try:
                async for audio in stream:
                    frames.append(audio.frame)
            finally:
                await stream.aclose()
            return frames

    frames = asyncio.run(go())
    assert bodies == [{"text": "Hello there.", "voice": "am_michael"}]  # empty options are not sent
    assert frames and frames[0].sample_rate == 24000
    assert 4800 <= sum(f.samples_per_channel for f in frames) < 9600


def test_tts_refuses_another_sample_rate():
    def handler(req):
        return httpx.Response(200, content=bytes(960), headers={"X-Sample-Rate": "16000"})

    async def go():
        async with client(handler) as http:
            stream = OrbTTS(http, "http://speech").synthesize("Hi there.", conn_options=NO_RETRY)
            try:
                async for _ in stream:
                    pass
            finally:
                await stream.aclose()

    with pytest.raises(APIError, match="16000 Hz"):
        asyncio.run(go())


# --- LLM: the agent service, and the undo rule ---------------------------------------------------------------------

def test_turn_text_joins_the_words_since_the_agent_spoke():
    ctx = chat(("user", "old turn"), ("assistant", "old reply"), ("user", "I think we should use Go"),
               ("user", "because the team knows it"))
    assert turn_text(ctx) == "I think we should use Go because the team knows it"
    assert turn_text(chat(("assistant", "hi"))) == ""
    pieces = chat(("user", "I think we should use Go for the back end. because the team knows it. and it scales."))
    assert turn_text(pieces) == "I think we should use Go for the back end because the team knows it and it scales."
    assert turn_text(chat(("user", "It works. Then we ship."))) == "It works. Then we ship."  # real sentences stay


def test_llm_streams_tokens_forwards_events_and_undoes_unheard_turns():
    bodies, events = [], []

    def handler(req):
        bodies.append(json.loads(req.content))
        return httpx.Response(200, content=ndjson(
            {"type": "read", "data": {"seconds": 0.3}}, {"type": "decision", "data": {"persona": "Rival"}},
            {"type": "token", "data": "Hello "}, {"type": "token", "data": "there."},
            {"type": "reply", "data": {"text": "Hello there.", "timing": {}}}, {"type": "state", "data": {"x": 1}}))

    async def go():
        async with client(handler) as http:
            tracker = TurnTracker()
            orb = OrbLLM(http, "http://agent", "s1", tracker=tracker, on_event=events.append)
            first = await reply_text(orb.chat(chat_ctx=chat(("user", "we should store"))))
            # the person goes on before any reply audio played: the turn is said again, with more words
            await reply_text(orb.chat(chat_ctx=chat(("user", "we should store passwords"))))
            tracker.played()  # now the agent's audio reached the room
            await reply_text(orb.chat(chat_ctx=chat(("assistant", "Hello there."), ("user", "a new point"))))
            return first

    assert asyncio.run(go()) == "Hello there."
    assert [b["replace_last"] for b in bodies] == [False, True, False]
    assert {b["channel"] for b in bodies} == {"voice"} and bodies[2]["text"] == "a new point"
    assert [e["type"] for e in events[:3]] == ["read", "decision", "reply"]  # tokens and state are not events


@pytest.mark.parametrize("response", [
    httpx.Response(200, content=ndjson({"type": "error", "data": "The reader failed."})),
    httpx.Response(404, json={"error": "No session s1."}),
])
def test_llm_failures_are_reported_and_never_retried(response):
    calls = []

    def handler(req):
        calls.append(req)
        return response

    async def go():
        async with client(handler) as http:
            orb = OrbLLM(http, "http://agent", "s1", tracker=TurnTracker())
            await reply_text(orb.chat(chat_ctx=chat(("user", "hello there"))))  # default options allow 3 retries

    with pytest.raises(APIError):
        asyncio.run(go())
    assert len(calls) == 1  # a turn changes the session, so a retry would read it twice


# --- LiveSTT: interim words while the person speaks ------------------------------------------------------------------

class FakeVAD(vad.VAD):
    """Voiced frames are speech; `silence_frames` zero frames in a row end it."""

    def __init__(self, silence_frames: int = 5):
        super().__init__(capabilities=vad.VADCapabilities(update_interval=0.02))
        self.silence_frames = silence_frames

    def stream(self) -> vad.VADStream:
        return _FakeVADStream(self)


class _FakeVADStream(vad.VADStream):
    async def _main_task(self) -> None:
        speaking, speech, silent, buf = False, 0.0, 0, []
        async for f in self._input_ch:
            if isinstance(f, self._FlushSentinel):
                continue
            voiced = bool(np.frombuffer(f.data, np.int16).any())
            d = f.samples_per_channel / f.sample_rate
            if voiced and not speaking:
                speaking, speech, silent, buf = True, d, 0, [f]
                self._event_ch.send_nowait(vad.VADEvent(type=vad.VADEventType.START_OF_SPEECH, samples_index=0,
                                                        timestamp=0.0, speech_duration=speech, silence_duration=0.0,
                                                        frames=[f], speaking=True))
            elif speaking:
                buf.append(f)
                speech, silent = (speech + d, 0) if voiced else (speech, silent + 1)
                self._event_ch.send_nowait(vad.VADEvent(type=vad.VADEventType.INFERENCE_DONE, samples_index=0,
                                                        timestamp=0.0, speech_duration=speech,
                                                        silence_duration=silent * d, frames=[f], speaking=True,
                                                        raw_accumulated_silence=silent * d))
                if silent >= self._vad.silence_frames:
                    speaking = False
                    self._event_ch.send_nowait(vad.VADEvent(type=vad.VADEventType.END_OF_SPEECH, samples_index=0,
                                                            timestamp=0.0, speech_duration=speech,
                                                            silence_duration=silent * d, frames=list(buf)))


class WordsSTT(stt.STT):
    """Hears one word per 100 ms of voiced audio."""

    def __init__(self):
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))
        self.calls = 0

    async def _recognize_impl(self, buffer, *, language=NOT_GIVEN, conn_options):
        self.calls += 1
        f = utils.merge_frames(buffer)
        n = int(round(np.count_nonzero(np.frombuffer(f.data, np.int16)) / f.sample_rate / 0.1))
        await asyncio.sleep(0.005)
        return stt.SpeechEvent(type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                               alternatives=[stt.SpeechData(language="en", text=" ".join(["word"] * n))])


def test_live_stt_sends_growing_interim_words_then_one_final():
    async def go():
        stream = LiveSTT(stt=WordsSTT(), vad=FakeVAD(silence_frames=5), interim_every=0.1).stream()
        events = []

        async def read():
            async for ev in stream:
                events.append(ev)

        reader = asyncio.create_task(read())
        for _ in range(30):  # 0.6 s of voice, in small real-time steps so the interim calls can finish
            stream.push_frame(frame(1000))
            await asyncio.sleep(0.02)
        for _ in range(6):
            stream.push_frame(frame(0))
            await asyncio.sleep(0.02)
        stream.end_input()
        await asyncio.wait_for(reader, 5)
        return events

    events = asyncio.run(go())
    kinds = [e.type for e in events]
    assert kinds[0] == stt.SpeechEventType.START_OF_SPEECH
    assert kinds[-2:] == [stt.SpeechEventType.END_OF_SPEECH, stt.SpeechEventType.FINAL_TRANSCRIPT]
    interim = [len(e.alternatives[0].text.split()) for e in events if e.type == stt.SpeechEventType.INTERIM_TRANSCRIPT]
    assert len(interim) >= 2 and interim[-1] > interim[0]  # the words grow while the person speaks
    assert all(k != stt.SpeechEventType.INTERIM_TRANSCRIPT for k in kinds[kinds.index(stt.SpeechEventType.END_OF_SPEECH):])
    assert len(events[-1].alternatives[0].text.split()) == 6  # the final hears the whole piece: 0.6 s


async def live_run(pattern, preflight_after=0.06, silence_frames=8):
    """Pushes 20 ms frames in real time: `pattern` is a list of (voiced, frames). Returns the events and the STT."""
    words = WordsSTT()
    stream = LiveSTT(stt=words, vad=FakeVAD(silence_frames=silence_frames), interim_every=10.0,
                     preflight_after=preflight_after).stream()
    events = []

    async def read():
        async for ev in stream:
            events.append(ev)

    reader = asyncio.create_task(read())
    for voiced, n in pattern:
        for _ in range(n):
            stream.push_frame(frame(1000 if voiced else 0))
            await asyncio.sleep(0.02)
    stream.end_input()
    await asyncio.wait_for(reader, 5)
    return events, words


def texts(events, kind):
    return [e.alternatives[0].text for e in events if e.type == kind]


def test_a_preflight_at_a_pause_becomes_the_final():
    events, words = asyncio.run(live_run([(True, 20), (False, 10)]))  # 0.4 s of voice, then the pause ends it
    kinds = [e.type for e in events]
    pre = texts(events, stt.SpeechEventType.PREFLIGHT_TRANSCRIPT)
    assert pre == ["word word word word"]  # 0.4 s: four words, sent at the start of the pause
    assert kinds.index(stt.SpeechEventType.PREFLIGHT_TRANSCRIPT) < kinds.index(stt.SpeechEventType.END_OF_SPEECH)
    assert texts(events, stt.SpeechEventType.FINAL_TRANSCRIPT) == pre  # the same text keeps the early reply
    assert words.calls == 1  # the final did not transcribe again


def test_a_preflight_is_dropped_when_the_person_goes_on():
    events, words = asyncio.run(live_run([(True, 15), (False, 5), (True, 15), (False, 10)]))
    assert texts(events, stt.SpeechEventType.PREFLIGHT_TRANSCRIPT)[0] == "word word word"
    assert texts(events, stt.SpeechEventType.FINAL_TRANSCRIPT) == ["word word word word word word"]
    assert words.calls == 2  # one per pause; the second one is the final


def test_no_preflight_when_it_is_off():
    events, words = asyncio.run(live_run([(True, 20), (False, 10)], preflight_after=0.0))
    assert texts(events, stt.SpeechEventType.PREFLIGHT_TRANSCRIPT) == []
    assert texts(events, stt.SpeechEventType.FINAL_TRANSCRIPT) == ["word word word word"] and words.calls == 1


async def sentences(pieces, **kw):
    """The pieces that the first-clause tokenizer sends, for `pieces` pushed one by one (as LLM tokens)."""
    from orb_voice.plugins import FirstClauseTokenizer

    stream = FirstClauseTokenizer(**kw).stream()
    out = []

    async def read():
        async for ev in stream:
            out.append(ev.token.strip())

    reader = asyncio.create_task(read())
    for p in pieces:
        stream.push_text(p)
        await asyncio.sleep(0)
    stream.end_input()
    await asyncio.wait_for(reader, 5)
    await stream.aclose()
    return out


def words_of(text):
    return [w + " " for w in text.split(" ")]


def test_first_clause_goes_out_before_its_sentence_ends():
    reply = ("If you need strong transactional integrity across many tables, Postgres is the safer pick. "
             "What data will you store?")
    out = asyncio.run(sentences(words_of(reply)))
    assert out[0] == "If you need strong transactional integrity across many tables,"
    assert " ".join(out) == reply.strip()


def test_a_short_first_sentence_or_a_short_clause_is_not_cut():
    reply = "Postgres is the safer pick here. Yes, because it has transactions, and it is well known."
    out = asyncio.run(sentences(words_of(reply)))
    assert out[0] == "Postgres is the safer pick here."  # a sentence came first: no clause cut
    out = asyncio.run(sentences(words_of("Yes, it is. But what about the cost of twelve, 500 rupees per month?")))
    assert out[0].startswith("Yes, it is.")  # "Yes," is too short to be a first piece
    assert asyncio.run(sentences(["The total was 12,500 rupees and it was paid on time today."]))[0] == \
        "The total was 12,500 rupees and it was paid on time today."  # no cut inside a number


def test_first_clause_off():
    reply = "If you need strong transactional integrity across many tables, Postgres is the safer pick."
    assert asyncio.run(sentences(words_of(reply), first_clause=0, first_max=0)) == [reply]


def test_a_long_first_sentence_without_a_comma_is_cut_at_a_word():
    reply = "The choice depends entirely on the data structure and the query patterns that you expect. Which is it?"
    out = asyncio.run(sentences(words_of(reply)))
    assert out[0] == "The choice depends entirely on the data structure and the"  # whole words, at most 60 chars
    assert len(out[0]) <= 60 and " ".join(out) == reply.strip()
    assert asyncio.run(sentences(words_of(reply), first_max=0))[0] == reply.split(" Which")[0]  # off: the sentence


def test_the_worker_learns_which_stt_engines_are_cloud_services():
    from orb_voice.plugins import stt_engines_info

    async def ask(handler):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await stt_engines_info(http, "http://speech")

    ok = lambda r: httpx.Response(200, json={"stt": {"model": "whistle", "engines": ["whistle", "cartesia"],
                                                     "cloud": ["cartesia"], "unavailable": {}}, "tts": {}})
    assert asyncio.run(ask(ok)) == ({"cartesia"}, "whistle")
    assert asyncio.run(ask(lambda r: httpx.Response(200, json={"stt": {"model": "whistle"}}))) == (set(), "whistle")  # an older service
    assert asyncio.run(ask(lambda r: httpx.Response(200, text="not json"))) == (set(), "")
    assert asyncio.run(ask(lambda r: httpx.Response(200, json={}))) == (set(), "")

    def down(request):
        raise httpx.ConnectError("refused", request=request)

    assert asyncio.run(ask(down)) == (set(), "")  # the speech service is down: the call goes on with local settings
