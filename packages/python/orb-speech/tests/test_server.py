"""The speech service with fake engines: routes, limits and error messages."""
import numpy as np
from fastapi.testclient import TestClient

from orb_speech import audio
from orb_speech.server import STT_MAX_SECONDS, create_app


class FakeSTT:
    sample_rate = 16000

    def __init__(self):
        self.got = None

    def transcribe(self, samples, language=None):
        self.got = (len(samples), language)
        if language == "xx":
            raise ValueError("Whistle supports en, de, fr, es, it, nl, pl, not 'xx'.")
        return {"text": "hello there", "language": language or "en", "ms": 1.0, "audio_ms": len(samples) // 16}


class FakeTTS:
    sample_rate = 24000
    voice = "af_heart"

    def check(self, text, voice, speed):
        if voice not in (None, "af_heart"):
            raise ValueError(f"Unknown voice {voice!r}.")
        return text, voice or self.voice, speed or 1.0

    async def stream(self, text, voice=None, speed=None):
        for _ in range(2):
            yield np.full(12000, 0.25, dtype=np.float32)  # 0.5 s each


class FakeTurn:
    threshold = 0.5

    def predict(self, samples):
        p = 0.9 if len(samples) > 16000 else 0.1
        return {"probability": p, "complete": p >= 0.5, "threshold": 0.5, "ms": 1.0, "audio_ms": len(samples) // 16}


def client(**kw):
    engines = {"stt": FakeSTT(), "tts": FakeTTS(), "turn": FakeTurn()}
    engines.update(kw)
    return TestClient(create_app(**engines)), engines


def pcm(seconds, rate=16000):
    return audio.float_to_pcm16(np.zeros(int(seconds * rate), dtype=np.float32))


def test_health():
    c, _ = client()
    assert c.get("/health").json()["ok"] is True
    c, _ = client(stt=None)
    assert c.get("/health").json()["ok"] is False


def test_stt_converts_and_resamples():
    c, e = client()
    r = c.post("/v1/stt?sample_rate=48000&language=de", content=pcm(1, 48000))
    assert r.status_code == 200 and r.json()["text"] == "hello there"
    assert e["stt"].got == (16000, "de")  # 1 s at 48 kHz becomes 16000 samples


def test_stt_rejects_bad_input_with_reasons():
    c, _ = client()
    assert c.post("/v1/stt?sample_rate=22050", content=pcm(1)).json()["error"].startswith("sample_rate must be")
    r = c.post("/v1/stt", content=b"\x00\x01\x02")
    assert r.status_code == 400 and "even number of bytes" in r.json()["error"]
    r = c.post("/v1/stt", content=pcm(STT_MAX_SECONDS + 1))
    assert r.status_code == 413
    r = c.post("/v1/stt?language=xx", content=pcm(1))
    assert r.status_code == 400 and "not 'xx'" in r.json()["error"]


def test_turn():
    c, _ = client()
    assert c.post("/v1/turn", content=pcm(2)).json()["complete"] is True
    assert c.post("/v1/turn", content=pcm(0.5)).json()["complete"] is False


def test_tts_streams_pcm_with_its_rate():
    c, _ = client()
    r = c.post("/v1/tts", json={"text": "Hello."})
    assert r.status_code == 200 and r.headers["x-sample-rate"] == "24000"
    assert len(r.content) == 2 * 12000 * 2  # two chunks of 12000 samples, 2 bytes each
    assert c.post("/v1/tts", json={"text": "Hi", "voice": "nope"}).status_code == 400
    assert c.post("/v1/tts", json={"text": ""}).status_code == 422
    assert c.post("/v1/tts", json={"text": "Hi", "speed": 5}).status_code == 422


def test_missing_engines_give_503():
    c, _ = client(stt=None, tts=None, turn=None)
    assert c.post("/v1/stt", content=pcm(1)).status_code == 503
    assert c.post("/v1/turn", content=pcm(1)).status_code == 503
    assert c.post("/v1/tts", json={"text": "Hi"}).status_code == 503
