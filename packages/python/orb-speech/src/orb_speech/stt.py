"""Speech to text with Whistle (Cactus Compute), 16 kHz mono, on the CPU.

Whistle loads one model per process and is not thread-safe, so every call takes a lock. A call for a sentence
takes about 12 to 35 ms on an Apple M5, so the lock is rarely a wait.
"""
from __future__ import annotations

import os
import threading
import time

import numpy as np

SAMPLE_RATE = 16000
LANGUAGES = ("en", "de", "fr", "es", "it", "nl", "pl")
QUOTES = "\"“”„«»"  # Whistle sometimes opens or closes a transcript with a quote mark


def clean(text: str) -> str:
    """Whitespace and stray quote marks at the ends removed. Apostrophes stay ("don't")."""
    return (text or "").strip().strip(QUOTES).strip()


class WhistleSTT:
    sample_rate = SAMPLE_RATE

    def __init__(self, weights: str | None = None, language: str | None = None):
        os.environ.setdefault("NEEDLE_TELEMETRY", "0")  # the package counts calls unless this is 0
        from needle import Whistle

        if language is not None and language not in LANGUAGES:
            raise ValueError(f"Whistle supports {', '.join(LANGUAGES)}, not {language!r}.")
        self.language = language
        self._model = Whistle(weights=weights)
        self._lock = threading.Lock()

    def transcribe(self, samples: np.ndarray, language: str | None = None) -> dict:
        """{"text", "language", "ms", "audio_ms"}. An empty or near-silent input gives empty text."""
        lang = language or self.language
        if lang is not None and lang not in LANGUAGES:
            raise ValueError(f"Whistle supports {', '.join(LANGUAGES)}, not {lang!r}.")
        audio_ms = round(len(samples) / SAMPLE_RATE * 1000)
        if len(samples) < SAMPLE_RATE // 10:  # less than 100 ms: nothing to say
            return {"text": "", "language": lang, "ms": 0.0, "audio_ms": audio_ms}
        with self._lock:
            t0 = time.perf_counter()
            r = self._model.transcribe(np.asarray(samples, dtype=np.float32), language=lang)
            ms = (time.perf_counter() - t0) * 1000
        return {"text": clean(r.get("text")), "language": r.get("language"), "ms": round(ms, 1),
                "audio_ms": audio_ms}
