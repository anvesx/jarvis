"""The speech service with fake engines: routes, limits and error messages."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from orb_speech import audio
from orb_speech.server import STT_MAX_SECONDS, create_app


class FakeSTT:
    sample_rate = 16000
    name = "fake-stt"

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
    name = "fake-tts"
    voices = {"af_heart"}

    def __init__(self):
        self.got = None

    def check(self, text, voice=None, speed=None, style=None, lang=None):
        if voice not in (None, "af_heart"):
            raise ValueError(f"Unknown voice {voice!r}.")
        return text, voice or self.voice, speed or 1.0, lang or "en-us"

    async def stream(self, text, voice=None, speed=None, style=None, lang=None):
        self.got = {"voice": voice, "style": style, "lang": lang}
        for _ in range(2):
            yield np.full(12000, 0.25, dtype=np.float32)  # 0.5 s each


def client(**kw):
    engines = {"stt": FakeSTT(), "tts": FakeTTS()}
    engines.update(kw)
    return TestClient(create_app(**engines)), engines


def pcm(seconds, rate=16000):
    return audio.float_to_pcm16(np.zeros(int(seconds * rate), dtype=np.float32))


def test_health():
    c, _ = client()
    h = c.get("/health").json()
    assert h["ok"] is True and h["stt"]["model"] == "fake-stt"
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


def test_tts_streams_pcm_with_its_rate():
    c, _ = client()
    r = c.post("/v1/tts", json={"text": "Hello."})
    assert r.status_code == 200 and r.headers["x-sample-rate"] == "24000"
    assert len(r.content) == 2 * 12000 * 2  # two chunks of 12000 samples, 2 bytes each
    assert c.post("/v1/tts", json={"text": "Hi", "voice": "nope"}).status_code == 400
    assert c.post("/v1/tts", json={"text": ""}).status_code == 422
    assert c.post("/v1/tts", json={"text": "Hi", "speed": 5}).status_code == 422


def test_missing_engines_give_503():
    c, _ = client(stt=None, tts=None)
    assert c.post("/v1/stt", content=pcm(1)).status_code == 503
    assert c.post("/v1/tts", json={"text": "Hi"}).status_code == 503


def test_unknown_stt_engine_is_refused_before_any_model_loads():
    import pytest
    from orb_speech.server import make_stt

    with pytest.raises(ValueError, match="ORB_STT_ENGINE must be one of whistle, parakeet"):
        make_stt("whisper", "en")


def test_stt_engine_per_request_loads_on_first_use():
    from orb_speech.engines import Registry

    built = []

    def factory(name):
        def make():
            built.append(name)
            stt = FakeSTT()
            stt.name = name
            return stt
        return make

    reg = Registry({"whistle": factory("whistle"), "parakeet": factory("parakeet")}, "whistle")
    reg.get()  # the default is loaded at start
    c, _ = client(stt=reg)
    assert c.post("/v1/stt", content=pcm(1)).json()["engine"] == "whistle"
    assert built == ["whistle"]  # parakeet is not loaded until someone asks for it
    assert c.post("/v1/stt?engine=parakeet", content=pcm(1)).json()["engine"] == "parakeet"
    assert built == ["whistle", "parakeet"]
    r = c.post("/v1/stt?engine=whisper", content=pcm(1))
    assert r.status_code == 400 and "Unknown engine 'whisper'" in r.json()["error"]
    e = c.get("/v1/engines").json()["stt"]
    assert e["model"] == "whistle" and e["engines"] == ["whistle", "parakeet"] and set(e["loaded"]) == {"whistle", "parakeet"}


def test_tts_passes_style_and_lang_and_refuses_unknown_engines():
    c, e = client()
    r = c.post("/v1/tts", json={"text": "Namaste.", "lang": "hi", "style": "HAPPY"})
    assert r.status_code == 200 and r.headers["x-engine"] == "fake-tts"
    assert e["tts"].got == {"voice": None, "style": "HAPPY", "lang": "hi"}
    r = c.post("/v1/tts", json={"text": "Hello.", "engine": "maya1"})
    assert r.status_code == 400 and "Unknown engine 'maya1'" in r.json()["error"]
    assert c.get("/v1/engines").json()["tts"]["voices"] == {"fake-tts": ["af_heart"]}


def test_heavy_engines_unload_the_oldest():
    from orb_speech.engines import Registry

    closed = []

    class Heavy:
        def __init__(self, name):
            self.name = name

        def close(self):
            closed.append(self.name)

    reg = Registry({"kokoro": lambda: Heavy("kokoro"), "maya1": lambda: Heavy("maya1"),
                    "indic-speak": lambda: Heavy("indic-speak")}, "kokoro", heavy=frozenset({"maya1", "indic-speak"}))
    reg.get("kokoro"), reg.get("maya1"), reg.get("indic-speak")
    assert closed == ["maya1"] and set(reg.loaded) == {"kokoro", "indic-speak"}  # one heavy model at a time


def test_engines_lists_voices_of_engines_that_are_not_loaded_and_load_errors_are_503():
    from orb_speech.engines import Registry

    def broken():
        raise FileNotFoundError("0xhb/maya1-mlx-4bit is not fetched. Run: python -m orb_speech.models --all")

    tts = Registry({"fake-tts": FakeTTS, "maya1": broken}, "fake-tts", heavy=frozenset({"maya1"}),
                   meta={"fake-tts": {"langs": ("en-us", "hi")}, "maya1": {"voices": ("Calm voice.", "Warm voice."),
                                                                         "langs": ("en-us",)}})
    tts.get()
    c, _ = client(tts=tts)
    e = c.get("/v1/engines").json()["tts"]
    assert e["loaded"] == ["fake-tts"]
    assert e["voices"] == {"fake-tts": ["af_heart"], "maya1": ["Calm voice.", "Warm voice."]}  # the order is kept
    assert e["langs"] == {"fake-tts": ["en-us", "hi"], "maya1": ["en-us"]} and e["styles"] == {}
    r = c.post("/v1/tts", json={"text": "Hello.", "engine": "maya1"})
    assert r.status_code == 503 and "did not load" in r.json()["error"] and "--all" in r.json()["error"]


def test_a_client_that_hangs_up_during_the_upload_is_not_a_server_error():
    import asyncio

    from fastapi import HTTPException
    from starlette.requests import ClientDisconnect

    from orb_speech.server import read_pcm

    class Hangup:
        headers = {}

        async def stream(self):
            yield b"\x00\x00" * 100
            raise ClientDisconnect()

    with pytest.raises(HTTPException) as e:
        asyncio.run(read_pcm(Hangup(), 16000, 60, 16000))
    assert e.value.status_code == 499


def test_cartesia_is_a_cloud_engine_that_needs_a_key(monkeypatch):
    from orb_speech.engines import EngineError, Registry
    from orb_speech.server import CARTESIA_NO_KEY, make_stt

    monkeypatch.delenv("CARTESIA_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="CARTESIA_API_KEY is not set"):
        make_stt("cartesia", None)
    monkeypatch.setenv("CARTESIA_API_KEY", "sk_car_test")
    cartesia = make_stt("cartesia", "en")  # builds the engine only: no call, no audio leaves
    assert cartesia.name == "cartesia-ink-whisper" and cartesia.cloud is True

    class Broken:
        name = "cartesia"

        def transcribe(self, samples, language=None):
            raise EngineError("Cartesia refused the API key. Check CARTESIA_API_KEY (HTTP 401).")

    def no_key():
        raise RuntimeError(CARTESIA_NO_KEY)

    stt = Registry({"fake-stt": FakeSTT, "cartesia": Broken, "cartesia-nokey": no_key}, "fake-stt",
                   meta={"cartesia": {"cloud": True}, "cartesia-nokey": {"cloud": True, "unavailable": CARTESIA_NO_KEY}})
    c, _ = client(stt=stt)
    e = c.get("/v1/engines").json()["stt"]
    assert e["cloud"] == ["cartesia", "cartesia-nokey"] and list(e["unavailable"]) == ["cartesia-nokey"]
    assert "CARTESIA_API_KEY" in e["unavailable"]["cartesia-nokey"]
    r = c.post("/v1/stt?engine=cartesia", content=pcm(1))
    assert r.status_code == 502 and "refused the API key" in r.json()["error"]  # the service failed, not the caller
    r = c.post("/v1/stt?engine=cartesia-nokey", content=pcm(1))
    assert r.status_code == 503 and "CARTESIA_API_KEY is not set" in r.json()["error"] and "restart" in r.json()["error"]
