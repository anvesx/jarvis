"""The speech service: the voice worker's STT and TTS, plus the bench, on 127.0.0.1 only.

    python -m orb_speech.server            # port 8200, or ORB_SPEECH_PORT
    ORB_STT_ENGINE=parakeet python -m orb_speech.server  # another default STT; any engine can still be asked for
    ORB_STT_FORMAT=raw python -m orb_speech.server       # the default of format=: written (default) or raw
    ORB_STT_VOCABULARY_FILE=team.txt python -m orb_speech.server  # the team's terms, one per line

POST /v1/stt?sample_rate=16000[&language=en][&engine=whistle|parakeet][&format=written|raw][&vocab=on|off]
               header X-Orb-Vocabulary: the request's own terms, comma separated
                                             body: PCM16 mono  ->  {"text", "raw", "stages", "language", "engine", "ms", "audio_ms"}
POST /v1/tts   {"text", "engine"?: kokoro|maya1|indic-speak, "voice"?, "speed"?, "style"?, "lang"?}
                                             ->  a stream of PCM16 mono at the engine's rate (header X-Sample-Rate)
GET  /v1/engines                             ->  the STT and TTS engines, which are loaded, and the voices
GET  /health

Engines load on first use (the bench switches them without a restart); the default ones load at start.
Audio and text stay on this machine. The log has only sizes and timings, never words.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.requests import ClientDisconnect
from pydantic import BaseModel, Field

from orb_speech import audio, vocab
from orb_speech.engines import EngineError, Registry
from orb_speech.textfmt import written

log = logging.getLogger("orb.speech")
STT_MAX_SECONDS = 60
STT_FORMATS = ("written", "raw")
VOCAB_SWITCH = ("on", "off")
VOCAB_HEADER = "X-Orb-Vocabulary"
CHUNK_BYTES = 4800  # 100 ms of 24 kHz PCM16


class SpeakRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    engine: str | None = Field(default=None, max_length=32)
    voice: str | None = Field(default=None, max_length=400)  # a voice name, or a voice description (maya1)
    speed: float | None = Field(default=None, ge=0.5, le=2.0)
    style: str | None = Field(default=None, max_length=64)
    lang: str | None = Field(default=None, max_length=16)


async def read_pcm(request: Request, sample_rate: int, max_seconds: int, model_rate: int):
    """The request body as float32 at the model's sample rate. 400 or 413 with a clear reason when it breaks a rule."""
    if sample_rate not in (8000, 16000, 24000, 48000):
        raise HTTPException(400, "sample_rate must be 8000, 16000, 24000 or 48000.")
    limit = sample_rate * 2 * max_seconds
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        raise HTTPException(413, f"The audio is longer than {max_seconds} s.")
    body = bytearray()
    try:
        async for part in request.stream():
            body.extend(part)
            if len(body) > limit:
                raise HTTPException(413, f"The audio is longer than {max_seconds} s.")
    except ClientDisconnect:  # the voice worker drops an interim request when the call ends or the person goes on
        raise HTTPException(499, "The client closed the request.")
    try:
        samples = audio.pcm16_to_float(bytes(body))
    except audio.AudioError as e:
        raise HTTPException(400, str(e))
    return audio.resample(samples, sample_rate, model_rate)


def as_registry(engines) -> Registry | None:
    if engines is None or isinstance(engines, Registry):
        return engines
    return Registry.single(engines)


async def engine_of(reg: Registry, name: str | None, kind: str):
    try:
        name = reg.resolve(name)
    except ValueError as e:
        raise HTTPException(400, f"{kind}: {e}")
    try:
        return await asyncio.to_thread(reg.get, name)  # a first use loads the model (seconds for a large one)
    except (FileNotFoundError, ValueError, ImportError, RuntimeError) as e:
        log.exception("%s engine %s did not load", kind, name)
        raise HTTPException(503, f"{kind} engine {name} did not load: {e}")


def check_format(value: str | None, default: str = "written") -> str:
    """The STT output format of a request: written (the default) or raw. A bad value is a 400 with the choices."""
    fmt = default if value is None or value == "" else value
    if fmt not in STT_FORMATS:
        raise ValueError(f"format must be one of {', '.join(STT_FORMATS)}, not {fmt!r}.")
    return fmt


def check_vocab(value: str | None) -> bool:
    """The `vocab` switch of a request: on (the default) or off."""
    if value in (None, "", "on"):
        return True
    if value == "off":
        return False
    raise ValueError(f"vocab must be one of {', '.join(VOCAB_SWITCH)}, not {value!r}.")


def create_app(stt=None, tts=None, stt_format: str = "written", vocabulary: tuple[str, ...] = ()) -> FastAPI:
    """The app with the given engines: a Registry, one engine, or None. A missing kind gives 503 on its endpoint.

    `stt_format` is the default of `format` for /v1/stt (setting ORB_STT_FORMAT). `vocabulary` is the team's term list
    (setting ORB_STT_VOCABULARY_FILE); a request adds its own terms in the X-Orb-Vocabulary header.
    """
    stt, tts = as_registry(stt), as_registry(tts)
    stt_format = check_format(stt_format)
    vocabulary = vocab.check_terms(vocabulary, "vocabulary file")
    app = FastAPI(title="orb speech", docs_url=None, redoc_url=None)

    def describe(reg: Registry | None) -> dict | None:
        if not reg:
            return None
        meta = {name: reg.meta.get(name, {}) for name in reg.names}
        return {"model": reg.default, "engines": reg.names, "loaded": reg.loaded,
                "cloud": [n for n, m in meta.items() if m.get("cloud")],  # engines that send the audio to another company
                "unavailable": {n: m["unavailable"] for n, m in meta.items() if m.get("unavailable")}}

    @app.get("/health")
    async def health():
        return {"ok": stt is not None and tts is not None, "stt": describe(stt), "tts": describe(tts)}

    @app.get("/v1/engines")
    async def engines():
        out = {"voices": {}, "styles": {}, "langs": {}}
        for name in (tts.names if tts else []):
            engine, meta = tts.peek(name), tts.meta.get(name, {})
            for key in out:
                v = getattr(engine, key, None) or meta.get(key)
                if v:
                    out[key][name] = sorted(v) if isinstance(v, (set, frozenset)) else list(v)
        return {"stt": describe(stt), "tts": {**(describe(tts) or {}), **out}}

    @app.post("/v1/stt")
    async def transcribe(request: Request, sample_rate: int = Query(16000), language: str | None = Query(None),
                         engine: str | None = Query(None, max_length=32),
                         fmt: str | None = Query(None, alias="format", max_length=16),
                         vocab_switch: str | None = Query(None, alias="vocab", max_length=8)):
        if stt is None:
            raise HTTPException(503, "Speech-to-text is not loaded.")
        try:
            fmt = check_format(fmt, stt_format)
            terms = vocab.merge(vocabulary, vocab.parse_header(request.headers.get(VOCAB_HEADER))) \
                if check_vocab(vocab_switch) else ()
        except ValueError as e:  # also a VocabularyError: checked before the body is read and before any model call
            raise HTTPException(400, str(e))
        samples = await read_pcm(request, sample_rate, STT_MAX_SECONDS, 16000)
        model = await engine_of(stt, engine, "STT")
        args = {"keywords": terms} if terms and getattr(model, "takes_keywords", False) else {}
        try:
            out = await asyncio.to_thread(model.transcribe, samples, language, **args)
        except ValueError as e:
            raise HTTPException(400, str(e))
        except EngineError as e:  # an engine that works through another service: it is not reachable, or it refused
            raise HTTPException(502, str(e))
        name = stt.resolve(engine)
        raw, stages = out["text"], {"asr": out["ms"]}  # the engine's time includes clean()
        text = raw
        if terms:
            t0 = time.perf_counter()
            text = vocab.correct(text, terms)
            stages["vocab"] = round((time.perf_counter() - t0) * 1000, 3)
        if fmt == "written":
            t0 = time.perf_counter()
            text = written(text)
            stages["written"] = round((time.perf_counter() - t0) * 1000, 3)
        log.info("stt %s %d ms audio -> %d chars (%s, %d terms) in %.1f ms", name, out["audio_ms"], len(text), fmt,
                 len(terms), out["ms"])
        return {**out, "text": text, "raw": raw, "stages": stages, "engine": name}

    @app.post("/v1/tts")
    async def speak(req: SpeakRequest):
        if tts is None:
            raise HTTPException(503, "Text-to-speech is not loaded.")
        model = await engine_of(tts, req.engine, "TTS")
        args = dict(voice=req.voice, speed=req.speed, style=req.style, lang=req.lang)
        try:
            model.check(req.text, **args)
        except ValueError as e:
            raise HTTPException(400, str(e))
        name, rate, t0 = tts.resolve(req.engine), model.sample_rate, time.perf_counter()

        async def body():
            first, total = None, 0
            async for chunk in model.stream(req.text, **args):
                pcm = audio.float_to_pcm16(chunk)
                if first is None:
                    first = (time.perf_counter() - t0) * 1000
                total += len(pcm)
                for i in range(0, len(pcm), CHUNK_BYTES):
                    yield pcm[i:i + CHUNK_BYTES]
            log.info("tts %s %d chars -> %d ms audio, first audio %.0f ms", name, len(req.text),
                     total // 2 * 1000 // rate, first or 0)

        headers = {"X-Sample-Rate": str(rate), "X-Channels": "1", "X-Engine": name}
        return StreamingResponse(body(), media_type=f"audio/L16; rate={rate}; channels=1", headers=headers)

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

    return app


STT_ENGINES = ("whistle", "parakeet", "cartesia")
CARTESIA_NO_KEY = ("CARTESIA_API_KEY is not set. Add the line CARTESIA_API_KEY=... to the .env file of the repo "
                   "(it is not in git), then restart the speech service.")
TTS_ENGINES = ("kokoro", "maya1", "indic-speak")


def make_stt(engine: str, language: str | None):
    """A speech-to-text engine: whistle (the default), parakeet (decision 0005) or cartesia (a cloud service)."""
    from orb_speech import models

    if engine not in STT_ENGINES:
        raise ValueError(f"ORB_STT_ENGINE must be one of {', '.join(STT_ENGINES)}, not {engine!r}.")
    if engine == "whistle":
        from orb_speech.stt import WhistleSTT
        return WhistleSTT(weights=str(models.whistle_weights()), language=language)
    if engine == "cartesia":
        from orb_speech.stt import CARTESIA_VERSION, CartesiaSTT
        key = os.environ.get("CARTESIA_API_KEY", "").strip()
        if not key:
            raise RuntimeError(CARTESIA_NO_KEY)
        return CartesiaSTT(key, language=language, version=os.environ.get("ORB_CARTESIA_VERSION") or CARTESIA_VERSION)
    from orb_speech.stt import ParakeetSTT
    return ParakeetSTT(models.parakeet_dir(fetch_missing=False), language=language)


def build() -> FastAPI:
    """The registries, with the default engines loaded now (Whistle and Kokoro: about 2 s on an Apple M5)."""
    from orb_speech import models
    from orb_speech.tts import LANGS as tts_langs, KokoroTTS
    from orb_speech.tts_lm import IndicSpeakTTS, Maya1TTS  # numpy only; MLX loads with an engine

    t0 = time.perf_counter()
    language = os.environ.get("ORB_STT_LANGUAGE") or None
    stt_meta = {"cartesia": {"cloud": True}}  # the audio of a call goes to Cartesia
    if not os.environ.get("CARTESIA_API_KEY", "").strip():
        stt_meta["cartesia"]["unavailable"] = CARTESIA_NO_KEY
    stt = Registry({name: (lambda n=name: make_stt(n, language)) for name in STT_ENGINES},
                   os.environ.get("ORB_STT_ENGINE", "whistle"), heavy=frozenset({"parakeet"}), meta=stt_meta)
    kokoro = lambda: KokoroTTS(models.path(os.environ.get("ORB_TTS_MODEL", "kokoro-v1.0.int8.onnx"), verify=True),
                               models.path("voices-v1.0.bin", verify=True),
                               voice=os.environ.get("ORB_TTS_VOICE", "af_heart"))
    maya1 = lambda: Maya1TTS(models.maya1_dir(fetch_missing=False), models.snac_dir(fetch_missing=False))
    indic_speak = lambda: IndicSpeakTTS(models.indic_speak_dir(fetch_missing=False), models.snac_dir(fetch_missing=False))
    meta = {"kokoro": {"langs": tts_langs},
            "maya1": {"voices": Maya1TTS.voices, "langs": ("en-us", "en-gb")},
            "indic-speak": {"voices": IndicSpeakTTS.voices, "styles": IndicSpeakTTS.styles, "langs": IndicSpeakTTS.langs}}
    tts = Registry({"kokoro": kokoro, "maya1": maya1, "indic-speak": indic_speak},
                   os.environ.get("ORB_TTS_ENGINE", "kokoro"), heavy=frozenset({"maya1", "indic-speak"}), meta=meta)
    stt_format = check_format(os.environ.get("ORB_STT_FORMAT"))  # a bad value stops the start, not the first call
    vocab_file = os.environ.get("ORB_STT_VOCABULARY_FILE", "").strip()
    vocabulary = vocab.load_file(vocab_file) if vocab_file else ()  # a missing or bad file stops the start too
    stt.get(), tts.get()  # the defaults load now, so the first call is fast
    log.info("models loaded in %.1f s (stt %s, tts %s, format %s, %d vocabulary terms)", time.perf_counter() - t0,
             stt.default, tts.default, stt_format, len(vocabulary))
    return create_app(stt, tts, stt_format, vocabulary)


def main():
    import uvicorn

    from orb_core.config import load_env_prefixes

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    os.environ.setdefault("NEEDLE_TELEMETRY", "0")
    load_env_prefixes(("CARTESIA_", "ORB_CARTESIA_"))  # the Cartesia key only: other secrets in .env stay out
    uvicorn.run(build(), host="127.0.0.1", port=int(os.environ.get("ORB_SPEECH_PORT", "8200")), log_level="warning")


if __name__ == "__main__":
    main()
