"""PCM helpers. On the wire, audio is 16-bit signed little-endian mono PCM. Inside, it is float32 in [-1, 1]."""
from __future__ import annotations

import numpy as np

try:
    import soxr
except ImportError:  # pragma: no cover - soxr is a declared dependency
    soxr = None


class AudioError(ValueError):
    pass


def pcm16_to_float(data: bytes) -> np.ndarray:
    if len(data) % 2:
        raise AudioError("PCM16 audio must have an even number of bytes.")
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def float_to_pcm16(samples: np.ndarray) -> bytes:
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767.0).round().astype("<i2").tobytes()


def resample(samples: np.ndarray, src: int, dst: int) -> np.ndarray:
    """High-quality resampling with soxr. The output has round(len * dst / src) samples."""
    if src == dst:
        return np.asarray(samples, dtype=np.float32)
    if src <= 0 or dst <= 0:
        raise AudioError("Sample rates must be positive.")
    if soxr is not None:
        return soxr.resample(np.asarray(samples, dtype=np.float32), src, dst).astype(np.float32)
    n = int(round(len(samples) * dst / src))  # fallback: linear, with some aliasing
    return np.interp(np.linspace(0, len(samples) - 1, n), np.arange(len(samples)), samples).astype(np.float32)


def seconds(samples: np.ndarray, rate: int) -> float:
    return len(samples) / float(rate)
