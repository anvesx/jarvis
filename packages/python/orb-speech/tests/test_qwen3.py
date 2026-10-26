"""Qwen3STT with a fake mlx_qwen3_asr Session: no model is loaded."""
import sys
import types

import numpy as np
import pytest

from orb_speech.server import STT_ENGINES
from orb_speech.stt import Qwen3STT


class FakeSession:
    calls = []

    def __init__(self, path):
        self.path = path

    def transcribe(self, audio, **kw):
        FakeSession.calls.append((len(audio), kw))
        return types.SimpleNamespace(text="  um मैंने nested ifelse dot sh फ़ाइल खोली ")


@pytest.fixture
def qwen3(monkeypatch):
    FakeSession.calls = []
    monkeypatch.setitem(sys.modules, "mlx_qwen3_asr", types.SimpleNamespace(Session=FakeSession))
    return Qwen3STT("/models/qwen3")


def test_qwen3_lets_the_model_find_the_language_and_passes_the_terms_as_context(qwen3):
    out = qwen3.transcribe(np.zeros(16000, np.float32), "en", keywords=("useEffect", "Redis"))
    assert out["text"] == "मैंने nested ifelse dot sh फ़ाइल खोली" and out["language"] == "en"
    assert FakeSession.calls[-1] == (16000, {"context": "useEffect, Redis"})  # no language: Hinglish stays mixed
    assert len(FakeSession.calls) == 2  # the warm-up call at load


def test_qwen3_checks_the_language_and_skips_tiny_audio(qwen3):
    with pytest.raises(ValueError):
        qwen3.transcribe(np.zeros(16000, np.float32), "fr")
    assert qwen3.transcribe(np.zeros(800, np.float32))["text"] == ""
    assert "qwen3" in STT_ENGINES
