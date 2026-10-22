import numpy as np
import pytest

from orb_speech import audio
from orb_speech.turn import last_window


def test_pcm_round_trip_is_close():
    x = np.sin(np.linspace(0, 20, 1600)).astype(np.float32) * 0.8
    y = audio.pcm16_to_float(audio.float_to_pcm16(x))
    assert y.dtype == np.float32 and len(y) == 1600
    assert np.max(np.abs(x - y)) < 1e-4


def test_pcm_clips_and_rejects_odd_bytes():
    assert audio.pcm16_to_float(audio.float_to_pcm16(np.array([2.0, -2.0]))).tolist() == pytest.approx([0.99997, -0.99997], abs=1e-4)
    with pytest.raises(audio.AudioError):
        audio.pcm16_to_float(b"\x00\x01\x02")


def test_resample_lengths():
    x = np.zeros(48000, dtype=np.float32)
    assert len(audio.resample(x, 48000, 16000)) == 16000
    assert len(audio.resample(x, 16000, 16000)) == 48000
    with pytest.raises(audio.AudioError):
        audio.resample(x, 0, 16000)


def test_turn_window_pads_at_the_start_and_keeps_the_end():
    short = np.ones(16000, dtype=np.float32)
    w = last_window(short)
    assert len(w) == 8 * 16000 and w[0] == 0.0 and w[-1] == 1.0 and w.sum() == 16000
    long = np.arange(10 * 16000, dtype=np.float32)
    w = last_window(long)
    assert len(w) == 8 * 16000 and w[-1] == long[-1] and w[0] == long[2 * 16000]


def test_stt_clean_removes_stray_quotes_only():
    from orb_speech.stt import clean

    assert clean(' "No, I don\'t agree with you. ') == "No, I don't agree with you."
    assert clean('"What would you pick for the database?"') == "What would you pick for the database?"
    assert clean("“Quoted”") == "Quoted"
    assert clean("") == "" and clean(None) == ""
