"""The Cartesia STT engine against a fake Cartesia (httpx.MockTransport): nothing leaves the machine in these tests."""
import io
import wave

import httpx
import numpy as np
import pytest

from orb_speech.engines import EngineError
from orb_speech.stt import CARTESIA_URL, CARTESIA_VERSION, CartesiaSTT, wav_bytes

KEY = "sk_car_test_0123456789"


def engine(handler, **kw):
    return CartesiaSTT(KEY, client=httpx.Client(transport=httpx.MockTransport(handler)), **kw)


def speech(seconds=1.0):
    return (np.sin(np.linspace(0, 400 * seconds, int(16000 * seconds))) * 0.3).astype(np.float32)


def parts(request: httpx.Request) -> tuple[dict, bytes]:
    """The text fields and the file of a multipart request."""
    boundary = request.headers["content-type"].split("boundary=")[1].encode()
    fields, file = {}, b""
    for chunk in request.content.split(b"--" + boundary):
        head, _, body = chunk.partition(b"\r\n\r\n")
        body = body.removesuffix(b"\r\n")
        if b'name="file"' in head:
            file = body
        elif b'name="' in head:
            fields[head.split(b'name="')[1].split(b'"')[0].decode()] = body.decode()
    return fields, file


def test_a_call_sends_one_wav_with_the_key_in_the_header_only():
    seen = {}

    def handler(request):
        seen["request"] = request
        return httpx.Response(200, json={"type": "transcript", "text": " We should hash the passwords. ",
                                         "language": "en", "duration": 1.0, "request_id": "r1"})

    x = speech(1.0)
    out = engine(handler).transcribe(x)
    req = seen["request"]
    assert req.method == "POST" and str(req.url) == CARTESIA_URL
    assert req.headers["authorization"] == f"Bearer {KEY}" and req.headers["cartesia-version"] == CARTESIA_VERSION
    fields, file = parts(req)
    assert fields == {"model": "ink-whisper", "language": "en"}
    with wave.open(io.BytesIO(file)) as w:  # the audio is a WAV: 16 kHz, mono, 16 bits, every sample
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()) == (16000, 1, 2, len(x))
    assert KEY.encode() not in req.content and KEY not in str(req.url)  # the key is in no body and no URL
    assert out["text"] == "We should hash the passwords." and out["language"] == "en" and out["audio_ms"] == 1000
    assert out["ms"] >= 0 and set(out) == {"text", "language", "ms", "audio_ms"}


def test_the_language_is_passed_on_and_checked():
    sent = []

    def handler(request):
        sent.append(parts(request)[0]["language"])
        return httpx.Response(200, json={"type": "transcript", "text": "नमस्ते", "language": "hi"})

    e = engine(handler, language="hi")
    assert e.transcribe(speech())["language"] == "hi" and e.transcribe(speech(), "de")["text"] == "नमस्ते"
    assert sent == ["hi", "de"]
    for bad in ("english", "EN", "e", "en;x"):
        with pytest.raises(ValueError, match="language code"):
            e.transcribe(speech(), bad)
    with pytest.raises(ValueError, match="language code"):
        CartesiaSTT(KEY, language="klingon!")


def test_fillers_and_quotes_are_cleaned_like_the_other_engines():
    e = engine(lambda r: httpx.Response(200, json={"text": '"Uh, we should use Go for the backend."'}))
    assert e.transcribe(speech())["text"] == "We should use Go for the backend."


def test_less_than_100_ms_of_audio_makes_no_call_and_costs_nothing():
    def handler(request):
        raise AssertionError("a call that is too short must not reach Cartesia")

    out = engine(handler).transcribe(np.zeros(1500, np.float32))
    assert out == {"text": "", "language": "en", "ms": 0.0, "audio_ms": 94}


@pytest.mark.parametrize("status, body, words", [
    (401, {"title": "unauthorized"}, ["refused the API key", "HTTP 401", "CARTESIA_API_KEY"]),
    (403, {}, ["refused the API key", "HTTP 403"]),
    (402, {}, ["no credits left"]),
    (429, {}, ["too many requests", "HTTP 429"]),
    (422, {"message": "language is not supported"}, ["HTTP 422", "language is not supported"]),
    (500, "oops", ["HTTP 500"]),
])
def test_each_failure_has_a_plain_reason_and_never_the_key(status, body, words):
    def handler(request):
        return httpx.Response(status, json=body) if isinstance(body, dict) else httpx.Response(status, text=body)

    with pytest.raises(EngineError) as e:
        engine(handler).transcribe(speech())
    for w in words:
        assert w in str(e.value)
    assert KEY not in str(e.value)


def test_a_network_error_is_an_engine_error_without_the_request_in_it():
    def handler(request):
        raise httpx.ConnectError("no route to api.cartesia.ai", request=request)

    with pytest.raises(EngineError) as e:
        engine(handler).transcribe(speech())
    assert "not reachable (ConnectError)" in str(e.value) and KEY not in str(e.value) and e.value.__cause__ is None


def test_an_answer_that_is_not_json_is_an_engine_error():
    with pytest.raises(EngineError, match="not JSON"):
        engine(lambda r: httpx.Response(200, text="<html>")).transcribe(speech())


def test_an_empty_key_is_refused_at_once():
    with pytest.raises(RuntimeError, match="empty"):
        CartesiaSTT("  ")


def test_wav_bytes_round_trip():
    x = speech(0.5)
    with wave.open(io.BytesIO(wav_bytes(x))) as w:
        back = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32767
    assert np.max(np.abs(back - x)) < 1e-4
    assert len(wav_bytes(np.array([2.0, -2.0], np.float32))) == 44 + 4  # clipped, not wrapped: 2 samples of 2 bytes
