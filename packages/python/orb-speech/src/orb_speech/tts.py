"""Text to speech with Kokoro-82M through ONNX Runtime. Output: 24 kHz mono float32.

The int8 model makes about 10 seconds of speech per second of compute on an Apple M5 (real-time factor 0.09).
The full model sounds slightly cleaner and is about 2 times slower.

Every TTS engine of the speech service has the same contract (see server.py): `name`, `sample_rate`, `voices`,
`check(text, voice, speed, style, lang)` and an async `stream(...)` of float32 chunks.

Languages: en-us (default), en-gb and hi. Kokoro's phonemizer marks every switch between Hindi and English with
"(en)" and "(hi)", and Kokoro would speak those marks. So Hindi text is phonemized here with the marks removed, and
Kokoro gets the phonemes. Kokoro has no style control: `style` is accepted and ignored.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

import numpy as np

SAMPLE_RATE = 24000
MAX_TEXT = 2000
LANGS = ("en-us", "en-gb", "hi")
HINDI_VOICES = ("hf_alpha", "hf_beta", "hm_omega", "hm_psi")


class KokoroTTS:
    sample_rate = SAMPLE_RATE
    name = "kokoro"

    def __init__(self, model_path: str, voices_path: str, voice: str = "af_heart", speed: float = 1.0, lang: str = "en-us"):
        from kokoro_onnx import Kokoro

        self._k = Kokoro(str(model_path), str(voices_path))
        self.voices = set(self._k.get_voices())
        if voice not in self.voices:
            raise ValueError(f"Unknown voice {voice!r}. Some voices: {', '.join(sorted(self.voices)[:8])}.")
        if lang not in LANGS:
            raise ValueError(f"Kokoro here supports {', '.join(LANGS)}, not {lang!r}.")
        self.voice, self.speed, self.lang = voice, speed, lang
        self._hindi = None  # the espeak backend for Hindi, made on first use

    def check(self, text: str, voice: str | None = None, speed: float | None = None, style: str | None = None,
              lang: str | None = None) -> tuple[str, str, float, str]:
        text = " ".join((text or "").split())
        if not text:
            raise ValueError("The text is empty.")
        if len(text) > MAX_TEXT:
            raise ValueError(f"The text has {len(text)} characters. The limit is {MAX_TEXT}.")
        lang = lang or self.lang
        if lang not in LANGS:
            raise ValueError(f"Kokoro here supports {', '.join(LANGS)}, not {lang!r}.")
        voice = voice or (HINDI_VOICES[0] if lang == "hi" else self.voice)
        if voice not in self.voices:
            raise ValueError(f"Unknown voice {voice!r}.")
        speed = self.speed if speed is None else float(speed)
        if not 0.5 <= speed <= 2.0:
            raise ValueError("The speed must be from 0.5 to 2.0.")
        return text, voice, speed, lang

    def hindi_phonemes(self, text: str) -> str:
        """Hindi (and Hindi with English words) as Kokoro phonemes, with no language-switch marks."""
        from kokoro_onnx.tokenizer import _espeak_lock
        from phonemizer.backend import EspeakBackend

        with _espeak_lock:
            if self._hindi is None:
                self._hindi = EspeakBackend("hi", preserve_punctuation=True, with_stress=True,
                                            language_switch="remove-flags")
            phonemes = self._hindi.phonemize([text])[0]
        return " ".join(phonemes.split())

    def _args(self, text: str, voice, speed, style, lang) -> dict:
        text, voice, speed, lang = self.check(text, voice, speed, style, lang)
        if lang == "hi":
            return {"text": self.hindi_phonemes(text), "voice": voice, "speed": speed, "lang": lang, "is_phonemes": True}
        return {"text": text, "voice": voice, "speed": speed, "lang": lang}

    def synthesize(self, text: str, voice: str | None = None, speed: float | None = None, style: str | None = None,
                   lang: str | None = None) -> np.ndarray:
        audio, rate = self._k.create(**self._args(text, voice, speed, style, lang))
        assert rate == SAMPLE_RATE
        return audio.astype(np.float32)

    async def stream(self, text: str, voice: str | None = None, speed: float | None = None, style: str | None = None,
                     lang: str | None = None) -> AsyncIterator[np.ndarray]:
        """Yields audio chunks as Kokoro makes them. Stopping the iteration stops the synthesis."""
        async for audio, rate in self._k.create_stream(**self._args(text, voice, speed, style, lang)):
            yield audio.astype(np.float32)
