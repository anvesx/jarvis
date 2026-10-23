"""Audio for the bench: the call path, noise, mixing at a signal-to-noise ratio, WAV files."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

RATE = 16000


def read(path: str | Path) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    return x if sr == RATE else soxr.resample(x, sr, RATE, quality="VHQ")


def write(path: str | Path, x: np.ndarray, rate: int = RATE) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(x, -1, 1), rate, subtype="PCM_16")


def pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()


def call_path(x16: np.ndarray) -> np.ndarray:
    """16 kHz -> 48 kHz -> Opus 32 kbps (voip) -> 48 kHz -> soxr "LQ" -> 16 kHz, as in a LiveKit call."""
    pcm = pcm16(soxr.resample(x16, RATE, 48000, quality="VHQ"))
    ogg = subprocess.run(["ffmpeg", "-v", "error", "-f", "s16le", "-ar", "48000", "-ac", "1", "-i", "-",
                          "-c:a", "libopus", "-b:a", "32k", "-application", "voip", "-frame_duration", "20",
                          "-f", "ogg", "-"], input=pcm, capture_output=True, check=True).stdout
    dec = subprocess.run(["ffmpeg", "-v", "error", "-i", "-", "-f", "s16le", "-ar", "48000", "-ac", "1", "-"],
                         input=ogg, capture_output=True, check=True).stdout
    return soxr.resample(np.frombuffer(dec, "<i2").astype("float32") / 32768, 48000, RATE, quality="LQ")


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0


def noise(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """Noise with an RMS of 1: white, pink (1/f), brown (1/f^2) or hum (a fan: 50 Hz harmonics plus soft noise)."""
    if kind == "white":
        x = rng.standard_normal(n)
    elif kind in ("pink", "brown"):
        f = np.fft.rfft(rng.standard_normal(n))
        k = np.arange(1, len(f) + 1)
        f /= np.sqrt(k) if kind == "pink" else k
        x = np.fft.irfft(f, n)
    elif kind == "hum":
        t = np.arange(n) / RATE
        x = sum(np.sin(2 * np.pi * 50 * h * t + h) / h for h in range(1, 8)) + 0.2 * rng.standard_normal(n)
    else:
        raise ValueError(f"Unknown noise {kind!r}.")
    return (x / (rms(x) or 1.0)).astype(np.float32)


def mix(speech: np.ndarray, noise_unit: np.ndarray, snr_db: float) -> np.ndarray:
    """Speech plus noise, the noise scaled so that RMS(speech) / RMS(noise) is snr_db decibels."""
    gain = rms(speech) / (10 ** (snr_db / 20))
    return np.clip(speech + noise_unit[: len(speech)] * gain, -1, 1).astype(np.float32)


def pad(x: np.ndarray, before: float, after: float) -> np.ndarray:
    return np.concatenate([np.zeros(int(before * RATE), np.float32), x, np.zeros(int(after * RATE), np.float32)])
