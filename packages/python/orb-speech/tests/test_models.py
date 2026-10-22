"""The real models together. Skipped when the model files are not on this machine.

Kokoro speaks a sentence, Whistle must hear the same words, and Smart Turn must tell a full sentence from a
sentence that stops in the middle. Kokoro reads every fragment with a final, falling tone, so only a cut in the
middle of a phrase works as an "unfinished" test with synthetic speech.
"""
import os
import re

import pytest

from orb_speech import audio, models

needed = [models.models_dir() / n for n in models.REQUIRED]
pytestmark = pytest.mark.skipif(not all(p.is_file() for p in needed), reason="model files are not fetched")


@pytest.fixture(scope="module")
def engines():
    os.environ.setdefault("NEEDLE_TELEMETRY", "0")
    from orb_speech.stt import WhistleSTT
    from orb_speech.tts import KokoroTTS
    from orb_speech.turn import SmartTurn

    tts = KokoroTTS(models.path("kokoro-v1.0.int8.onnx"), models.path("voices-v1.0.bin"))
    return WhistleSTT(weights=str(models.whistle_weights())), tts, SmartTurn(models.path("smart-turn-v3.2-cpu.onnx"))


def words(text):
    return re.sub(r"[^a-z ]", "", text.lower()).split()


def test_tts_then_stt_gives_the_same_words(engines):
    stt, tts, _ = engines
    sentence = "We should store user passwords in a plain JSON file so the assistant can log in for them."
    speech = audio.resample(tts.synthesize(sentence, voice="am_michael"), 24000, 16000)
    out = stt.transcribe(speech)
    assert words(out["text"]) == words(sentence)
    assert out["ms"] < 500


def test_smart_turn_tells_full_from_cut(engines):
    _, tts, turn = engines
    full = audio.resample(tts.synthesize("Can you summarize where we are?"), 24000, 16000)
    cut = audio.resample(tts.synthesize("We should store user passwords in a plain JSON file so the"), 24000, 16000)
    assert turn.predict(full)["complete"] is True
    assert turn.predict(cut)["complete"] is False


def test_checksums_of_the_fetched_files():
    for name in models.REQUIRED:
        assert models.sha256(models.path(name)) == models.FILES[name].sha256
