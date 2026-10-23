"""Metrics and audio math."""
import numpy as np
import pytest

from orb_bench import audio
from orb_bench.metrics import edits, normalize, percentile, score, wer


def test_english_normalizer_drops_fillers_punctuation_and_case():
    assert normalize("Um, We should STORE passwords!") == "we should store passwords"
    assert score("We should store user passwords.", "we should store the user passwords") == (1, 5)


def test_hindi_normalizer_keeps_vowel_signs_and_drops_the_danda():
    assert normalize("मैं कल office जा रहा हूँ, meeting है।", "hi") == "मैं कल office जा रहा हूँ meeting है"
    assert score("नमस्ते, आप कैसे हैं?", "नमस्ते आप कैसे है", "hi") == (1, 4)


def test_edits_wer_and_percentile():
    assert edits("a b c".split(), "a x c d".split()) == 2
    assert wer([(1, 4), (1, 6)]) == 0.2 and wer([]) is None
    assert percentile([1, 2, 3, 4, 5], 50) == 3 and percentile([10, 20], 90) == pytest.approx(19)


def test_noise_is_unit_rms_and_mix_hits_the_snr():
    rng = np.random.default_rng(1)
    speech = (0.1 * np.sin(np.linspace(0, 400 * np.pi, 16000))).astype(np.float32)
    for kind in ("white", "pink", "brown", "hum"):
        n = audio.noise(kind, 16000, rng)
        assert audio.rms(n) == pytest.approx(1.0, rel=1e-3)
        mixed = audio.mix(speech, n, 10)
        measured = 20 * np.log10(audio.rms(speech) / audio.rms(mixed - speech))
        assert measured == pytest.approx(10, abs=0.1)


def test_pad_adds_silence_at_both_ends():
    x = np.ones(1600, np.float32)
    y = audio.pad(x, 0.3, 0.2)
    assert len(y) == 1600 + 4800 + 3200 and y[:4800].max() == 0 and y[-3200:].max() == 0
