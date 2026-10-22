"""Text to speech with Kokoro-82M through ONNX Runtime. Output: 24 kHz mono float32.

The int8 model makes about 10 seconds of speech per second of compute on an Apple M5 (real-time factor 0.09).
The full model sounds slightly cleaner and is about 2 times slower.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

import numpy as np

SAMPLE_RATE = 24000
MAX_TEXT = 2000


class KokoroTTS:
    sample_rate = SAMPLE_RATE

    def __init__(self, model_path: str, voices_path: str, voice: str = "af_heart", speed: float = 1.0, lang: str = "en-us"):
        from kokoro_onnx import Kokoro

        self._k = Kokoro(str(model_path), str(voices_path))
        self.voices = set(self._k.get_voices())
        if voice not in self.voices:
            raise ValueError(f"Unknown voice {voice!r}. Some voices: {', '.join(sorted(self.voices)[:8])}.")
        self.voice, self.speed, self.lang = voice, speed, lang

    def check(self, text: str, voice: str | None, speed: float | None) -> tuple[str, str, float]:
        text = " ".join((text or "").split())
        if not text:
            raise ValueError("The text is empty.")
        if len(text) > MAX_TEXT:
            raise ValueError(f"The text has {len(text)} characters. The limit is {MAX_TEXT}.")
        voice = voice or self.voice
        if voice not in self.voices:
            raise ValueError(f"Unknown voice {voice!r}.")
        speed = self.speed if speed is None else float(speed)
        if not 0.5 <= speed <= 2.0:
            raise ValueError("The speed must be from 0.5 to 2.0.")
        return text, voice, speed

    def synthesize(self, text: str, voice: str | None = None, speed: float | None = None) -> np.ndarray:
        text, voice, speed = self.check(text, voice, speed)
        audio, rate = self._k.create(text, voice=voice, speed=speed, lang=self.lang)
        assert rate == SAMPLE_RATE
        return audio.astype(np.float32)

    async def stream(self, text: str, voice: str | None = None, speed: float | None = None) -> AsyncIterator[np.ndarray]:
        """Yields audio chunks as Kokoro makes them. Stopping the iteration stops the synthesis."""
        text, voice, speed = self.check(text, voice, speed)
        async for audio, rate in self._k.create_stream(text, voice=voice, speed=speed, lang=self.lang):
            yield audio.astype(np.float32)
