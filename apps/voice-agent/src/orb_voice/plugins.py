"""LiveKit Agents plugins that connect the voice worker to orb's own services.

- OrbSTT: one HTTP call to the speech service per piece of speech (batch).
- LiveSTT: a streaming STT made from a batch STT and a VAD. While the person speaks, the growing piece is
  transcribed again every `interim_every` seconds, so the session has interim words. LiveKit counts those words for
  `min_words`: a real barge-in stops the agent within about a second, and a one-word backchannel ("Sure.") never cuts
  it. Whistle takes about 10 to 20 ms per call, so this costs little.
- OrbTTS: one HTTP call per sentence. The speech service streams PCM16 mono at 24 kHz.
- OrbLLM: the orb agent service is the LLM. Only the person's newest words are sent, because the agent service owns
  the history. A turn whose reply never played is sent again with replace_last (see TurnTracker).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import wave
from pathlib import Path
from typing import Any, Callable

import httpx
import numpy as np
from livekit import rtc
from livekit.agents import APIConnectionError, APIConnectOptions, APIStatusError, llm, stt, tokenize, tts, utils, vad
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr

log = logging.getLogger("orb.voice")

TTS_SAMPLE_RATE = 24000
_Capabilities = stt.STTCapabilities
# LiveKit joins the pieces of one turn with spaces, and Whistle ends each piece with a period:
# "use Go for the back end. because the team". A period before a lowercase word is such a join.
PIECE_JOIN = re.compile(r"(?<=[a-z0-9])[.]\s+(?=[a-z])")


def service_error(body: bytes | str, status: int) -> str:
    """The {"error"} message of an orb service, or the HTTP status. It never holds audio or a transcript."""
    try:
        data = json.loads(body)
        if isinstance(data, dict) and data.get("error"):
            return str(data["error"])
    except (ValueError, TypeError):
        pass
    return f"HTTP {status}"


def mono_pcm16(frame: rtc.AudioFrame) -> bytes:
    data = np.frombuffer(frame.data, dtype=np.int16)
    if frame.num_channels > 1:
        data = data.reshape(-1, frame.num_channels).mean(axis=1).astype(np.int16)
    return data.tobytes()


class OrbSTT(stt.STT):
    """Batch speech to text through the orb speech service: POST /v1/stt."""

    def __init__(self, http: httpx.AsyncClient, base_url: str, *, engine: str = "", language: str = "en",
                 debug_dir: str = "", vocabulary: str = ""):
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))
        self._http, self._url, self._engine, self._language = http, base_url, engine, language
        # The session's own terms (ORB_STT_VOCABULARY). The same header goes with every call of a session, so the
        # interim, preflight and final texts of one piece are corrected alike.
        self._headers = {"X-Orb-Vocabulary": vocabulary.encode("utf-8")} if vocabulary.strip() else {}
        self._debug_dir = Path(debug_dir) if debug_dir else None
        self._dumped = 0

    @property
    def model(self) -> str:
        return self._engine or "default"

    @property
    def provider(self) -> str:
        return "orb-speech"

    async def _recognize_impl(self, buffer: utils.AudioBuffer, *, language: NotGivenOr[str] = NOT_GIVEN,
                              conn_options: APIConnectOptions) -> stt.SpeechEvent:
        frame = utils.merge_frames(buffer)
        pcm = mono_pcm16(frame)
        lang = language if utils.is_given(language) and language else self._language
        params: dict[str, Any] = {"sample_rate": frame.sample_rate}
        if lang:
            params["language"] = lang
        if self._engine:
            params["engine"] = self._engine
        try:
            r = await self._http.post(f"{self._url}/v1/stt", params=params, content=pcm, headers=self._headers,
                                      timeout=conn_options.timeout)
        except httpx.HTTPError as e:
            raise APIConnectionError(f"speech service: {type(e).__name__}") from e
        if r.status_code != 200:
            raise APIStatusError(service_error(r.content, r.status_code), status_code=r.status_code,
                                 retryable=r.status_code >= 500)
        data = r.json()
        text = (data.get("text") or "").strip()
        if self._debug_dir is not None:
            self._dump(pcm, frame.sample_rate, text)
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT, request_id=utils.shortuuid("stt_"),
            alternatives=[stt.SpeechData(language=data.get("language") or lang or "en", text=text,
                                         confidence=1.0 if text else 0.0,
                                         end_time=float(data.get("audio_ms") or 0) / 1000)])

    def _dump(self, pcm: bytes, rate: int, text: str) -> None:
        """Debug only (ORB_DEBUG_AUDIO_DIR): every recognized piece as WAV + text. It saves real voices."""
        self._debug_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._dumped += 1
        base = self._debug_dir / f"{time.strftime('%H%M%S')}-{self._dumped:04d}"
        with wave.open(str(base.with_suffix(".wav")), "wb") as w:
            w.setnchannels(1), w.setsampwidth(2), w.setframerate(rate)
            w.writeframes(pcm)
        base.with_suffix(".txt").write_text(text + "\n")
        for p in (base.with_suffix(".wav"), base.with_suffix(".txt")):
            os.chmod(p, 0o600)

    async def warm(self) -> None:
        """0.2 s of silence: opens the speech service's keep-warm window (it matters for Parakeet on the GPU)."""
        frame = rtc.AudioFrame(data=bytes(6400), sample_rate=16000, num_channels=1, samples_per_channel=3200)
        try:
            await self.recognize(buffer=frame)
        except Exception as e:  # a warm-up must never stop the job
            log.warning("speech-to-text warm-up failed: %s", e)


async def stt_engines_info(http: httpx.AsyncClient, base_url: str) -> tuple[set[str], str]:
    """(The STT engines that send the audio to another company, the default STT engine), from the speech service.
    When the service does not answer, every engine counts as local and the default is unknown."""
    try:
        data = (await http.get(f"{base_url}/v1/engines", timeout=5)).json()["stt"]
        return set(data.get("cloud") or []), str(data.get("model") or "")
    except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError):
        return set(), ""


class LiveSTT(stt.STT):
    """A streaming STT from a batch STT and a VAD, with interim transcripts while the person speaks.

    Preflight: after `preflight_after` seconds of silence in a piece of speech (the start of a pause; the VAD ends
    the piece only after its min_silence_duration), the speech so far is transcribed and sent as a preflight
    transcript. LiveKit then starts the reply early (preemptive generation). When the pause ends the piece, the final
    transcript is that same text, so LiveKit keeps the early reply. When the person speaks again, the preflight is
    dropped, and LiveKit replaces the early reply. 0 turns preflight off.
    """

    def __init__(self, *, stt: stt.STT, vad: vad.VAD, interim_every: float = 0.3, preflight_after: float = 0.0):
        # the keyword names match LiveKit's StreamAdapter; inside, they hide the modules, hence _Capabilities
        super().__init__(capabilities=_Capabilities(streaming=True, interim_results=True))
        self._stt, self._vad, self._every, self._preflight_after = stt, vad, interim_every, preflight_after
        self._stt.on("metrics_collected", self._forward_metrics)

    @property
    def wrapped(self) -> stt.STT:
        return self._stt

    @property
    def model(self) -> str:
        return self._stt.model

    @property
    def provider(self) -> str:
        return self._stt.provider

    def _forward_metrics(self, *args: Any, **kwargs: Any) -> None:
        self.emit("metrics_collected", *args, **kwargs)

    async def _recognize_impl(self, buffer: utils.AudioBuffer, *, language: NotGivenOr[str] = NOT_GIVEN,
                              conn_options: APIConnectOptions) -> stt.SpeechEvent:
        return await self._stt.recognize(buffer=buffer, language=language, conn_options=conn_options)

    def stream(self, *, language: NotGivenOr[str] = NOT_GIVEN,
               conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS) -> stt.RecognizeStream:
        return _LiveStream(self, language=language, conn_options=conn_options)

    async def aclose(self) -> None:
        self._stt.off("metrics_collected", self._forward_metrics)


class _LiveStream(stt.RecognizeStream):
    def __init__(self, live: LiveSTT, *, language: NotGivenOr[str], conn_options: APIConnectOptions):
        # recognize() already retries, so the stream itself does not
        super().__init__(stt=live, conn_options=APIConnectOptions(max_retry=0, timeout=conn_options.timeout))
        self._live, self._language, self._inner_opts = live, language, conn_options

    async def _recognize(self, frames: list[rtc.AudioFrame]) -> stt.SpeechData | None:
        ev = await self._live.wrapped.recognize(buffer=utils.merge_frames(frames), language=self._language,
                                                conn_options=self._inner_opts)
        return ev.alternatives[0] if ev.alternatives and ev.alternatives[0].text else None

    async def _run(self) -> None:
        vad_stream = self._live._vad.stream()
        segment = 0          # counts pieces of speech; an interim result of an old piece is dropped
        speaking = False
        preflight: asyncio.Task | None = None  # the text of the speech before the current pause

        async def forward() -> None:
            async for item in self._input_ch:
                if isinstance(item, self._FlushSentinel):
                    vad_stream.flush()
                    continue
                vad_stream.push_frame(item)
            vad_stream.end_input()

        async def interim(seg: int, frames: list[rtc.AudioFrame]) -> None:
            try:
                data = await self._recognize(frames)
            except Exception as e:  # an interim failure is not a turn failure; the final call decides
                log.debug("interim transcription failed: %s", e)
                return
            if data and speaking and seg == segment:
                self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                                                           alternatives=[data]))

        async def early(seg: int, frames: list[rtc.AudioFrame]) -> stt.SpeechData | None:
            data = await self._recognize(frames)
            if data and speaking and seg == segment and preflight is asyncio.current_task():
                self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.PREFLIGHT_TRANSCRIPT,
                                                           alternatives=[data]))
                log.info("stt preflight sent", extra={"chars": len(data.text)})
            return data

        async def final(frames: list[rtc.AudioFrame]) -> stt.SpeechData | None:
            """The text of the whole piece. The preflight text when the pause that began with it ended the piece:
            the frames after it are silence, and the same text keeps LiveKit's early reply."""
            nonlocal preflight
            task, preflight = preflight, None
            if task is not None:
                try:
                    data = await task
                    log.info("stt final is the preflight", extra={"chars": len(data.text) if data else 0})
                    return data
                except Exception as e:  # a failed preflight is not a failed turn: transcribe again
                    log.debug("preflight transcription failed: %s", e)
            return await self._recognize(frames)

        async def recognize() -> None:
            nonlocal segment, speaking, preflight
            frames: list[rtc.AudioFrame] = []
            next_interim = self._live._every
            pending: asyncio.Task | None = None
            try:
                async for ev in vad_stream:
                    if ev.type == vad.VADEventType.START_OF_SPEECH:
                        segment += 1
                        speaking, frames, next_interim = True, list(ev.frames), self._live._every
                        self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH))
                    elif ev.type == vad.VADEventType.INFERENCE_DONE and speaking:
                        frames.extend(ev.frames)
                        if ev.raw_accumulated_silence <= 0:  # speech: a preflight of the last pause is out of date
                            if preflight is not None:
                                preflight.cancel()
                                preflight = None
                            if ev.speech_duration >= next_interim and (pending is None or pending.done()):
                                next_interim = ev.speech_duration + self._live._every
                                pending = asyncio.create_task(interim(segment, list(frames)))
                        elif (preflight is None and self._live._preflight_after > 0
                              and ev.raw_accumulated_silence >= self._live._preflight_after):
                            if pending is not None and not pending.done():
                                pending.cancel()
                            preflight = asyncio.create_task(early(segment, list(frames)))
                    elif ev.type == vad.VADEventType.END_OF_SPEECH:
                        speaking = False
                        if pending is not None and not pending.done():
                            pending.cancel()
                        end_time = time.time() - ev.silence_duration - ev.inference_duration
                        self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH,
                                                                   speech_end_time=end_time))
                        data = await final(list(ev.frames))
                        if data:
                            self._event_ch.send_nowait(stt.SpeechEvent(
                                type=stt.SpeechEventType.FINAL_TRANSCRIPT, alternatives=[data],
                                speech_end_time=end_time))
            finally:
                for task in (pending, preflight):
                    if task is not None and not task.done():
                        task.cancel()

        tasks = [asyncio.create_task(forward(), name="live_stt_forward"),
                 asyncio.create_task(recognize(), name="live_stt_recognize")]
        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.cancel_and_wait(*tasks)
            await vad_stream.aclose()


class OrbTTS(tts.TTS):
    """Text to speech through the orb speech service: POST /v1/tts, which streams PCM16 mono at 24 kHz."""

    def __init__(self, http: httpx.AsyncClient, base_url: str, *, engine: str = "", voice: str = "",
                 style: str = "", lang: str = ""):
        super().__init__(capabilities=tts.TTSCapabilities(streaming=False), sample_rate=TTS_SAMPLE_RATE,
                         num_channels=1)
        self._http, self._url = http, base_url
        self.options = {k: v for k, v in {"engine": engine, "voice": voice, "style": style, "lang": lang}.items() if v}

    @property
    def model(self) -> str:
        return self.options.get("engine", "default")

    @property
    def provider(self) -> str:
        return "orb-speech"

    def synthesize(self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS) -> tts.ChunkedStream:
        return _OrbChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class _OrbChunkedStream(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        orb: OrbTTS = self._tts  # type: ignore[assignment]
        output_emitter.initialize(request_id=utils.shortuuid("tts_"), sample_rate=TTS_SAMPLE_RATE, num_channels=1,
                                  mime_type="audio/pcm")
        body = {"text": self.input_text, **orb.options}
        try:
            async with orb._http.stream("POST", f"{orb._url}/v1/tts", json=body,
                                        timeout=httpx.Timeout(60.0, connect=self._conn_options.timeout)) as r:
                if r.status_code != 200:
                    raise APIStatusError(service_error(await r.aread(), r.status_code), status_code=r.status_code,
                                         retryable=False)
                rate = int(r.headers.get("x-sample-rate", TTS_SAMPLE_RATE))
                if rate != TTS_SAMPLE_RATE:
                    raise APIStatusError(f"speech service sent {rate} Hz, not {TTS_SAMPLE_RATE}", status_code=500,
                                         retryable=False)
                async for chunk in r.aiter_bytes():
                    output_emitter.push(chunk)
        except httpx.HTTPError as e:
            raise APIConnectionError(f"speech service: {type(e).__name__}") from e
        output_emitter.flush()


# The first piece of a reply: a sentence end (. ! ? ।) after 20 characters, or a clause end (, ; :) after 30.
SENTENCE_END = re.compile(r"[.!?।](?=\s)")
CLAUSE_END = re.compile(r"[,;:](?=\s)")


class FirstClauseTokenizer(tokenize.SentenceTokenizer):
    """LiveKit's sentence tokenizer, but the first piece of a reply can end at a clause.

    Kokoro makes a whole piece before its first audio (about 0.1 s per second of speech), and the tokenizer sends a
    sentence only when it ends. So a long first sentence delays the voice twice. The first piece may end at a comma,
    a semicolon or a colon after `first_clause` characters, or, with no such end in its first `first_max`
    characters, at the last word boundary before them. The rest goes through the normal tokenizer.
    """

    def __init__(self, first_clause: int = 30, min_sentence: int = 20, first_max: int = 60):
        self._inner = tokenize.blingfire.SentenceTokenizer(retain_format=True)
        self._first_clause, self._min_sentence, self._first_max = first_clause, min_sentence, first_max

    def tokenize(self, text: str, *, language: str | None = None) -> list[str]:
        return self._inner.tokenize(text, language=language)

    def stream(self, *, language: str | None = None) -> tokenize.SentenceStream:
        return _FirstClauseStream(self._inner.stream(language=language), self._first_clause, self._min_sentence,
                                  self._first_max)


class _FirstClauseStream(tokenize.SentenceStream):
    def __init__(self, inner: tokenize.SentenceStream, first_clause: int, min_sentence: int, first_max: int = 0):
        super().__init__()
        self._inner, self._first_clause, self._min_sentence = inner, first_clause, min_sentence
        self._first_max = first_max
        self._head: str | None = ""  # the text before the first piece; None once the first piece is decided
        self._pump = asyncio.create_task(self._forward())

    async def _forward(self) -> None:
        async for ev in self._inner:
            self._event_ch.send_nowait(ev)
        if not self._event_ch.closed:
            self._event_ch.close()

    def _cut(self) -> int:
        """The end of the first piece in the head, 0 when a sentence end comes first, -1 when it is too early."""
        for m in SENTENCE_END.finditer(self._head):
            if len(self._head[:m.end()].strip()) >= self._min_sentence:
                return 0  # a sentence comes first: the normal tokenizer sends it as soon as it can
        for m in CLAUSE_END.finditer(self._head) if self._first_clause > 0 else ():
            if len(self._head[:m.end()].strip()) >= self._first_clause:
                return m.end()
        if 0 < self._first_max < len(self._head):  # a long start with no clause end: cut at the last whole word
            at = self._head.rfind(" ", 0, self._first_max + 1)
            if len(self._head[:at].strip()) >= self._min_sentence:
                return at
        return -1

    def _release(self) -> None:
        head, self._head = self._head, None
        if head:
            self._inner.push_text(head)

    def push_text(self, text: str) -> None:
        self._check_not_closed()
        if self._head is None:
            self._inner.push_text(text)
            return
        self._head += text
        cut = self._cut()
        if cut > 0:
            piece, rest = self._head[:cut], self._head[cut:]
            self._head = None
            self._event_ch.send_nowait(tokenize.TokenData(token=piece))
            if rest:
                self._inner.push_text(rest)
        elif cut == 0:
            self._release()

    def flush(self) -> None:
        self._release()
        self._inner.flush()
        self._head = ""  # after a flush, a new reply starts

    def end_input(self) -> None:
        self._release()
        self._inner.end_input()

    async def aclose(self) -> None:
        await self._inner.aclose()
        await utils.aio.cancel_and_wait(self._pump)
        if not self._event_ch.closed:
            self._event_ch.close()


class TurnTracker:
    """Knows whether the last turn sent to the agent service has had any reply audio played.

    When it has not, the next turn says the earlier words again with more words: a preemptive reply that was dropped,
    or a reply cut before playback. That turn goes with replace_last, and the agent service undoes the unheard turn,
    but only when the old words start the new text (orb_discussion.agent.continues).
    """

    def __init__(self) -> None:
        self._pending = False

    def sent(self) -> bool:
        """A turn goes out now. Returns replace_last for it."""
        replace, self._pending = self._pending, True
        return replace

    def played(self) -> None:
        """Reply audio reached the room: the person heard the agent."""
        self._pending = False


def turn_text(chat_ctx: llm.ChatContext) -> str:
    """The person's words since the agent last spoke. LiveKit adds an agent message only when its audio played, so
    after a reply that never played, the earlier words and the new words are joined into one turn."""
    parts: list[str] = []
    for item in reversed(chat_ctx.items):
        if not isinstance(item, llm.ChatMessage):
            continue
        if item.role == "assistant":
            break
        if item.role == "user" and (text := (item.text_content or "").strip()):
            parts.append(text)
    return PIECE_JOIN.sub(" ", " ".join(reversed(parts)))


class OrbLLM(llm.LLM):
    """The orb agent service as the LLM: POST /v1/sessions/{id}/turns, which streams NDJSON events."""

    def __init__(self, http: httpx.AsyncClient, base_url: str, session_id: str, *, tracker: TurnTracker,
                 on_event: Callable[[dict], None] | None = None):
        super().__init__()
        self._http, self._url, self._sid, self._tracker, self._on_event = http, base_url, session_id, tracker, on_event

    @property
    def model(self) -> str:
        return "orb-agent"

    @property
    def provider(self) -> str:
        return "orb"

    def chat(self, *, chat_ctx: llm.ChatContext, tools: list | None = None,
             conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS, parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
             tool_choice: NotGivenOr[Any] = NOT_GIVEN, extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN) -> llm.LLMStream:
        return _OrbLLMStream(self, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options)


class _OrbLLMStream(llm.LLMStream):
    async def _run(self) -> None:
        orb: OrbLLM = self._llm  # type: ignore[assignment]
        text = turn_text(self._chat_ctx)
        if not text:
            return
        body = {"text": text[:8000], "channel": "voice", "replace_last": orb._tracker.sent()}
        rid = utils.shortuuid("orb_")
        try:
            # a turn changes the session (a Jev read, the ledger): it is never retried
            async with orb._http.stream("POST", f"{orb._url}/v1/sessions/{orb._sid}/turns", json=body,
                                        timeout=httpx.Timeout(None, connect=5.0)) as r:
                if r.status_code != 200:
                    raise APIStatusError(service_error(await r.aread(), r.status_code), status_code=r.status_code,
                                         retryable=False)
                async for line in r.aiter_lines():
                    if not line.strip():
                        continue
                    ev = json.loads(line)
                    kind, data = ev.get("type"), ev.get("data")
                    if kind == "token":
                        if data:
                            self._event_ch.send_nowait(llm.ChatChunk(
                                id=rid, delta=llm.ChoiceDelta(role="assistant", content=data)))
                    elif kind == "error":
                        raise APIStatusError(f"agent service: {data}", status_code=500, retryable=False)
                    elif kind != "state" and orb._on_event is not None:
                        orb._on_event(ev)
        except httpx.HTTPError as e:
            raise APIConnectionError(f"agent service: {type(e).__name__}", retryable=False) from e
