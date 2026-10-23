"""The real models together. Skipped when the model files are not on this machine.

Kokoro speaks a sentence, and each speech-to-text engine must hear the same words. Kokoro also speaks Hindi.

The accuracy on real accented speech is measured by scripts/stt_bench.py, not here (decision 0005).
"""
import os
import re

import pytest

from orb_speech import audio, models

needed = [models.models_dir() / n for n in models.REQUIRED]
pytestmark = pytest.mark.skipif(not all(p.is_file() for p in needed), reason="model files are not fetched")


def parakeet_fetched():
    try:
        models.parakeet_dir(fetch_missing=False)
        return True
    except (FileNotFoundError, ValueError):
        return False


@pytest.fixture(scope="module")
def kokoro():
    os.environ.setdefault("NEEDLE_TELEMETRY", "0")
    from orb_speech.tts import KokoroTTS

    return KokoroTTS(models.path("kokoro-v1.0.int8.onnx"), models.path("voices-v1.0.bin"))


@pytest.fixture(scope="module", params=["parakeet", "whistle"])
def stt(request):
    from orb_speech.stt import ParakeetSTT, WhistleSTT

    if request.param == "parakeet":
        if not parakeet_fetched():
            pytest.skip("Parakeet is not fetched")
        return ParakeetSTT(models.parakeet_dir(fetch_missing=False), language="en")
    return WhistleSTT(weights=str(models.whistle_weights()))


def words(text):
    return re.sub(r"[^a-z ]", "", text.lower()).split()


def test_tts_then_stt_gives_the_same_words(kokoro, stt):
    tts = kokoro
    sentence = "We should store user passwords in a plain JSON file so the assistant can log in for them."
    speech = audio.resample(tts.synthesize(sentence, voice="am_michael"), 24000, 16000)
    out = stt.transcribe(speech)
    assert words(out["text"]) == words(sentence)
    assert out["ms"] < 500


def test_checksums_of_the_fetched_files():
    for name in models.REQUIRED:
        assert models.sha256(models.path(name)) == models.FILES[name].sha256
    if parakeet_fetched():  # parakeet_dir checks the SHA-256 of every file at the pinned commit
        assert models.parakeet_dir(fetch_missing=False).name == models.PARAKEET_REVISION


def test_kokoro_speaks_hindi_without_language_marks(kokoro):
    tts = kokoro
    text = "मैं कल office जा रहा हूँ, meeting है।"  # Hindi with English words
    phonemes = tts.hindi_phonemes(text)
    assert "(" not in phonemes and len(phonemes) > 10  # no "(en)" / "(hi)" marks for Kokoro to speak
    speech = tts.synthesize(text, lang="hi")
    assert len(speech) > 0.8 * 24000  # about one second of speech or more
    assert tts.check(text, lang="hi")[1] == "hf_alpha"  # a Hindi voice when none is given
