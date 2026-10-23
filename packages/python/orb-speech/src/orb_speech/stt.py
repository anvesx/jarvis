"""Speech to text, 16 kHz mono. Two engines with one contract: transcribe(samples, language) -> dict.

- ParakeetSTT (default): NVIDIA Parakeet TDT 0.6B v3 through transformers, on the Apple GPU (MPS) or the CPU.
  On real accented conversational speech it makes less than half the errors of Whistle (decision 0005).
- WhistleSTT: Whistle (Cactus Compute), a 17 MB model on the CPU. Fast (about 20 ms), but weak on real speech.

- CartesiaSTT: Cartesia Ink-Whisper through its batch HTTP API. A CLOUD engine: every call sends the audio to Cartesia.

Neither local model is thread-safe, so every call takes a lock. The voice worker sends one piece or one turn at a
time, so the lock is rarely a wait.
"""
from __future__ import annotations

import io
import os
import re
import threading
import time
import wave
from pathlib import Path

import httpx
import numpy as np

from orb_speech import gpu
from orb_speech.engines import EngineError

SAMPLE_RATE = 16000
LANGUAGES = ("en", "de", "fr", "es", "it", "nl", "pl")  # Whistle
PARAKEET_LANGUAGES = ("bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hr", "hu", "it", "lt", "lv", "mt",
                      "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "uk")
QUOTES = "\"“”„«»"  # Whistle sometimes opens or closes a transcript with a quote mark
# Keep-warm: after a few seconds without work, the next Parakeet call on the Apple GPU often costs 0.5 to 0.8 s instead
# of about 0.09 s (measured 2026-10-22: 6 of 9 calls after 5 to 20 s idle). A short silent call after each idle second
# keeps it at about 0.08 s, but only during a conversation, so the GPU can rest between calls.
KEEP_WARM_IDLE = 1.0      # seconds without a call before a silent call
KEEP_WARM_WINDOW = 120.0  # seconds after the last real call; then the keep-warm stops
FILLERS = re.compile(r"(?i)(?<![\w'])(?:uh|uhm|um|umm|erm)(?![\w']),?\s*")  # Parakeet writes "uh" and "um"
ENCLOSED = re.compile(r"(?i),\s*(?:uh|uhm|um|umm|erm),\s*")  # "We were, uh, late": both commas go with it


def clean(text: str) -> str:
    """The words for the agent: stray quote marks at the ends and the fillers "uh", "um", "erm" removed.

    Apostrophes stay ("don't"). A sentence that started with a filler starts with a capital letter again.
    """
    text = (text or "").strip().strip(QUOTES).strip()
    if not FILLERS.search(text):
        return text
    capital = text[:1].isupper()
    text = FILLERS.sub("", ENCLOSED.sub(" ", text))
    text = re.sub(r"\s+([,.?!])", r"\1", re.sub(r"\s{2,}", " ", text)).strip(" ,")
    if text in (".", "?", "!"):
        return ""
    return text[:1].upper() + text[1:] if capital else text


def _check(language: str | None, known: tuple[str, ...], model: str) -> None:
    if language is not None and language not in known:
        raise ValueError(f"{model} supports {', '.join(known)}, not {language!r}.")


class ParakeetSTT:
    """NVIDIA Parakeet TDT 0.6B v3 (CC-BY-4.0) from a local folder, in float32.

    The model finds the language itself. A `language` argument is checked against the 25 languages that it knows
    and is returned, but it does not steer the model. float16 gives the same words on the Apple GPU, but some calls
    take up to 1 s while new kernels compile, so float32 stays (decision 0005).
    """
    sample_rate = SAMPLE_RATE
    name = "parakeet-tdt-0.6b-v3"

    def __init__(self, model_dir: str | Path, language: str | None = None, device: str | None = None,
                 keep_warm: bool = True):
        import torch
        from transformers import AutoProcessor, ParakeetForTDT

        _check(language, PARAKEET_LANGUAGES, "Parakeet")
        self.language = language
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self._torch = torch
        self._proc = AutoProcessor.from_pretrained(str(model_dir))
        self._lock = gpu.MPS  # one torch user of the GPU at a time in this process (gpu.py)
        self._silence = np.zeros(SAMPLE_RATE // 5, dtype=np.float32)  # 0.2 s: about 45 ms of GPU work
        with self._lock:
            self._model = ParakeetForTDT.from_pretrained(str(model_dir), dtype=torch.float32).to(self.device).eval()
            for warm_up in (self._silence, np.zeros(SAMPLE_RATE, dtype=np.float32)):
                self._run(warm_up)  # the first call of each length compiles GPU kernels (up to 0.9 s)
        self._last_call = self._last_real = time.monotonic()  # the keep-warm window opens at start
        self._stop = threading.Event()
        if keep_warm and self.device != "cpu":
            threading.Thread(target=self._keep_warm, name="parakeet-keep-warm", daemon=True).start()

    def _keep_warm(self):
        while not self._stop.wait(0.25):
            now = time.monotonic()
            if now - self._last_real > KEEP_WARM_WINDOW or now - self._last_call < KEEP_WARM_IDLE:
                continue
            with self._lock:
                self._run(self._silence)
                self._last_call = time.monotonic()

    def close(self):
        """Stops the keep-warm thread."""
        self._stop.set()

    def _run(self, samples: np.ndarray) -> str:
        inputs = self._proc(samples, sampling_rate=SAMPLE_RATE, return_tensors="pt").to(self.device)
        with self._torch.inference_mode():
            out = self._model.generate(**inputs)
        ids = out.sequences if hasattr(out, "sequences") else out
        return self._proc.batch_decode(ids, skip_special_tokens=True)[0]

    def transcribe(self, samples: np.ndarray, language: str | None = None) -> dict:
        """{"text", "language", "ms", "audio_ms"}. Less than 100 ms of audio gives empty text."""
        lang = language or self.language
        _check(lang, PARAKEET_LANGUAGES, "Parakeet")
        audio_ms = round(len(samples) / SAMPLE_RATE * 1000)
        if len(samples) < SAMPLE_RATE // 10:
            return {"text": "", "language": lang, "ms": 0.0, "audio_ms": audio_ms}
        with self._lock:
            t0 = time.perf_counter()
            text = self._run(np.asarray(samples, dtype=np.float32))
            ms = (time.perf_counter() - t0) * 1000
            self._last_call = self._last_real = time.monotonic()
        return {"text": clean(text), "language": lang, "ms": round(ms, 1), "audio_ms": audio_ms}


class WhistleSTT:
    sample_rate = SAMPLE_RATE
    name = "whistle-2.0.0"
    takes_keywords = True  # the vocabulary also goes into the model as keyword biasing

    def __init__(self, weights: str | None = None, language: str | None = None):
        os.environ.setdefault("NEEDLE_TELEMETRY", "0")  # the package counts calls unless this is 0
        from needle import Whistle

        _check(language, LANGUAGES, "Whistle")
        self.language = language
        self._model = Whistle(weights=weights)
        self._lock = threading.Lock()

    def transcribe(self, samples: np.ndarray, language: str | None = None,
                   keywords: tuple[str, ...] = ()) -> dict:
        """{"text", "language", "ms", "audio_ms"}. An empty or near-silent input gives empty text. `keywords` are the
        vocabulary terms for Whistle's keyword biasing."""
        lang = language or self.language
        _check(lang, LANGUAGES, "Whistle")
        audio_ms = round(len(samples) / SAMPLE_RATE * 1000)
        if len(samples) < SAMPLE_RATE // 10:  # less than 100 ms: nothing to say
            return {"text": "", "language": lang, "ms": 0.0, "audio_ms": audio_ms}
        with self._lock:
            t0 = time.perf_counter()
            r = self._model.transcribe(np.asarray(samples, dtype=np.float32), language=lang,
                                       keywords=list(keywords) or None)
            ms = (time.perf_counter() - t0) * 1000
        return {"text": clean(r.get("text")), "language": r.get("language"), "ms": round(ms, 1),
                "audio_ms": audio_ms}


CARTESIA_URL = "https://api.cartesia.ai/stt"
# The only version that the batch page of Cartesia's docs allows (read 2026-10-22). ORB_CARTESIA_VERSION overrides it.
CARTESIA_VERSION = "2026-08-14"
CARTESIA_MODEL = "ink-whisper"
LANGUAGE_CODE = re.compile(r"^[a-z]{2,3}$")  # an ISO-639 code; Cartesia checks it against its own list of about 100


def wav_bytes(samples: np.ndarray, rate: int = SAMPLE_RATE) -> bytes:
    """16-bit mono PCM in a WAV container, from float32 samples in [-1, 1]."""
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class CartesiaSTT:
    """Cartesia Ink-Whisper through its batch API: POST /stt with one WAV file for each call.

    **A cloud engine:** every call sends the audio to Cartesia. Cartesia bills 1 credit for each 2 seconds of audio,
    silence included (its pricing page, read 2026-10-22). A call that sends a growing piece again and again (interim
    transcripts) pays for the whole piece each time, so the voice worker calls less often for a cloud engine.

    The key is sent only in the Authorization header. It is in no log, no error message and no result.
    """
    sample_rate = SAMPLE_RATE
    name = "cartesia-ink-whisper"
    cloud = True

    def __init__(self, api_key: str, language: str | None = None, *, url: str = CARTESIA_URL,
                 version: str = CARTESIA_VERSION, timeout: float = 20.0, client: httpx.Client | None = None):
        if not api_key.strip():
            raise RuntimeError("CARTESIA_API_KEY is empty.")
        if language is not None and not LANGUAGE_CODE.match(language):
            raise ValueError(f"Cartesia needs a language code such as en or hi, not {language!r}.")
        self.language, self._key, self._url, self._version = language, api_key.strip(), url, version
        self._client = client or httpx.Client(timeout=timeout)  # one connection kept open: a call does not pay for TLS

    def close(self) -> None:
        self._client.close()

    def transcribe(self, samples: np.ndarray, language: str | None = None) -> dict:
        """{"text", "language", "ms", "audio_ms"}. Less than 100 ms of audio gives empty text without a call."""
        lang = language or self.language or "en"
        if not LANGUAGE_CODE.match(lang):
            raise ValueError(f"Cartesia needs a language code such as en or hi, not {lang!r}.")
        audio_ms = round(len(samples) / SAMPLE_RATE * 1000)
        if len(samples) < SAMPLE_RATE // 10:
            return {"text": "", "language": lang, "ms": 0.0, "audio_ms": audio_ms}
        t0 = time.perf_counter()
        try:
            r = self._client.post(self._url, data={"model": CARTESIA_MODEL, "language": lang},
                                  files={"file": ("audio.wav", wav_bytes(samples), "audio/wav")},
                                  headers={"Authorization": f"Bearer {self._key}", "Cartesia-Version": self._version})
        except httpx.HTTPError as e:
            raise EngineError(f"Cartesia is not reachable ({type(e).__name__}).") from None
        ms = (time.perf_counter() - t0) * 1000
        if r.status_code != 200:
            raise EngineError(self._why(r))
        try:
            body = r.json()
        except ValueError:
            raise EngineError("Cartesia sent an answer that is not JSON.") from None
        return {"text": clean(body.get("text")), "language": body.get("language") or lang, "ms": round(ms, 1),
                "audio_ms": audio_ms}

    @staticmethod
    def _why(r: httpx.Response) -> str:
        """A plain reason for a failed call. It uses the status and Cartesia's short message, never the request."""
        known = {401: "refused the API key. Check CARTESIA_API_KEY", 403: "refused the API key. Check CARTESIA_API_KEY",
                 402: "says the account has no credits left", 429: "says there are too many requests. Try again soon"}
        if r.status_code in known:
            return f"Cartesia {known[r.status_code]} (HTTP {r.status_code})."
        try:
            body = r.json()
            detail = str(body.get("message") or body.get("title") or body.get("error") or "")[:160]
        except (ValueError, AttributeError):
            detail = ""
        return f"Cartesia returned HTTP {r.status_code}" + (f": {detail}" if detail else ".")
