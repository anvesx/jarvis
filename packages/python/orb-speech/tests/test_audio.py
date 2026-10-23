import numpy as np
import pytest

from orb_speech import audio


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


def test_stt_clean_removes_stray_quotes_only():
    from orb_speech.stt import clean

    assert clean(' "No, I don\'t agree with you. ') == "No, I don't agree with you."
    assert clean('"What would you pick for the database?"') == "What would you pick for the database?"
    assert clean("“Quoted”") == "Quoted"
    assert clean("") == "" and clean(None) == ""


def test_stt_clean_removes_fillers_only():
    from orb_speech.stt import clean

    assert clean("Uh it's called uh Petticoat Market Lane.") == "It's called Petticoat Market Lane."
    assert clean("We were, uh, late") == "We were late"
    assert clean("sweet, uh something every meal uh.") == "sweet, something every meal."
    assert clean("Um.") == "" and clean("Uh?") == ""
    assert clean("UM, okay") == "Okay"
    assert clean("The umbrella and the hum stay. Uhura too.") == "The umbrella and the hum stay. Uhura too."


def test_parakeet_keep_warm_runs_only_inside_the_window(monkeypatch):
    import threading
    import time

    from orb_speech import stt

    monkeypatch.setattr(stt, "KEEP_WARM_IDLE", 0.05)
    monkeypatch.setattr(stt, "KEEP_WARM_WINDOW", 0.6)
    m = stt.ParakeetSTT.__new__(stt.ParakeetSTT)  # no model: _run is a stub
    runs = []
    m._run = lambda x: runs.append(len(x)) or "hello"
    m.language, m._lock, m._stop = None, threading.Lock(), threading.Event()
    m._silence = np.zeros(3200, dtype=np.float32)
    m._last_call = m._last_real = time.monotonic()
    threading.Thread(target=m._keep_warm, daemon=True).start()
    time.sleep(0.5)
    assert runs and set(runs) == {3200}  # silent 0.2 s calls inside the window
    time.sleep(0.6)
    stopped = len(runs)
    time.sleep(0.6)
    assert len(runs) == stopped  # no calls after the window
    assert m.transcribe(np.zeros(16000, dtype=np.float32))["text"] == "hello"  # a real call opens it again
    time.sleep(0.5)
    m.close()
    assert len(runs) > stopped + 1
