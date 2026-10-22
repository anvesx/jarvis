"""The speech service: three endpoints for the Go voice worker, on 127.0.0.1 only.

    python -m orb_speech.server            # port 8200, or ORB_SPEECH_PORT

POST /v1/stt?sample_rate=16000[&language=en]   body: PCM16 mono  ->  {"text", "language", "ms", "audio_ms"}
POST /v1/turn?sample_rate=16000                body: PCM16 mono  ->  {"probability", "complete", "threshold", "ms"}
POST /v1/tts   {"text", "voice"?, "speed"?}    ->  a stream of PCM16 mono at 24000 Hz (header X-Sample-Rate)
GET  /health

Audio and text stay on this machine. The log has only sizes and timings, never words.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from orb_speech import audio

log = logging.getLogger("orb.speech")
STT_MAX_SECONDS = 60
TURN_MAX_SECONDS = 30
CHUNK_BYTES = 4800  # 100 ms of 24 kHz PCM16


class SpeakRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    voice: str | None = None
    speed: float | None = Field(default=None, ge=0.5, le=2.0)


async def read_pcm(request: Request, sample_rate: int, max_seconds: int, model_rate: int):
    """The request body as float32 at the model's sample rate. 400 or 413 with a clear reason when it breaks a rule."""
    if sample_rate not in (8000, 16000, 24000, 48000):
        raise HTTPException(400, "sample_rate must be 8000, 16000, 24000 or 48000.")
    limit = sample_rate * 2 * max_seconds
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        raise HTTPException(413, f"The audio is longer than {max_seconds} s.")
    body = bytearray()
    async for part in request.stream():
        body.extend(part)
        if len(body) > limit:
            raise HTTPException(413, f"The audio is longer than {max_seconds} s.")
    try:
        samples = audio.pcm16_to_float(bytes(body))
    except audio.AudioError as e:
        raise HTTPException(400, str(e))
    return audio.resample(samples, sample_rate, model_rate)


def create_app(stt=None, tts=None, turn=None) -> FastAPI:
    """The app with the given engines. A missing engine gives 503 on its endpoint, so tests can use fakes."""
    app = FastAPI(title="orb speech", docs_url=None, redoc_url=None)

    @app.get("/health")
    async def health():
        return {"ok": all(x is not None for x in (stt, tts, turn)),
                "stt": {"model": "whistle", "sample_rate": 16000} if stt else None,
                "tts": {"model": "kokoro", "sample_rate": tts.sample_rate, "voice": tts.voice} if tts else None,
                "turn": {"model": "smart-turn-v3.2", "threshold": turn.threshold} if turn else None}

    @app.post("/v1/stt")
    async def transcribe(request: Request, sample_rate: int = Query(16000), language: str | None = Query(None)):
        if stt is None:
            raise HTTPException(503, "Speech-to-text is not loaded.")
        samples = await read_pcm(request, sample_rate, STT_MAX_SECONDS, 16000)
        try:
            out = await asyncio.to_thread(stt.transcribe, samples, language)
        except ValueError as e:
            raise HTTPException(400, str(e))
        log.info("stt %d ms audio -> %d chars in %.1f ms", out["audio_ms"], len(out["text"]), out["ms"])
        return out

    @app.post("/v1/turn")
    async def end_of_turn(request: Request, sample_rate: int = Query(16000)):
        if turn is None:
            raise HTTPException(503, "Turn detection is not loaded.")
        samples = await read_pcm(request, sample_rate, TURN_MAX_SECONDS, 16000)
        out = await asyncio.to_thread(turn.predict, samples)
        log.info("turn %d ms audio -> p=%.3f in %.1f ms", out["audio_ms"], out["probability"], out["ms"])
        return out

    @app.post("/v1/tts")
    async def speak(req: SpeakRequest):
        if tts is None:
            raise HTTPException(503, "Text-to-speech is not loaded.")
        try:
            tts.check(req.text, req.voice, req.speed)
        except ValueError as e:
            raise HTTPException(400, str(e))
        t0 = time.perf_counter()

        async def body():
            first, total = None, 0
            async for chunk in tts.stream(req.text, req.voice, req.speed):
                pcm = audio.float_to_pcm16(chunk)
                if first is None:
                    first = (time.perf_counter() - t0) * 1000
                total += len(pcm)
                for i in range(0, len(pcm), CHUNK_BYTES):
                    yield pcm[i:i + CHUNK_BYTES]
            log.info("tts %d chars -> %d ms audio, first audio %.0f ms", len(req.text),
                     total // 2 * 1000 // tts.sample_rate, first or 0)

        headers = {"X-Sample-Rate": str(tts.sample_rate), "X-Channels": "1"}
        return StreamingResponse(body(), media_type=f"audio/L16; rate={tts.sample_rate}; channels=1", headers=headers)

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

    return app


def build() -> FastAPI:
    """Load the real models from ORB_MODELS_DIR. Slow: about 3 s on an Apple M5."""
    from orb_speech import models
    from orb_speech.stt import WhistleSTT
    from orb_speech.tts import KokoroTTS
    from orb_speech.turn import SmartTurn

    t0 = time.perf_counter()
    stt = WhistleSTT(weights=str(models.whistle_weights()), language=os.environ.get("ORB_STT_LANGUAGE") or None)
    tts = KokoroTTS(models.path(os.environ.get("ORB_TTS_MODEL", "kokoro-v1.0.int8.onnx"), verify=True),
                    models.path("voices-v1.0.bin", verify=True), voice=os.environ.get("ORB_TTS_VOICE", "af_heart"))
    turn = SmartTurn(models.path("smart-turn-v3.2-cpu.onnx", verify=True),
                     threshold=float(os.environ.get("ORB_TURN_THRESHOLD", "0.5")))
    log.info("models loaded in %.1f s", time.perf_counter() - t0)
    return create_app(stt, tts, turn)


def main():
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    os.environ.setdefault("NEEDLE_TELEMETRY", "0")
    uvicorn.run(build(), host="127.0.0.1", port=int(os.environ.get("ORB_SPEECH_PORT", "8200")), log_level="warning")


if __name__ == "__main__":
    main()
