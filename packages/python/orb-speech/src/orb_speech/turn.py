"""End-of-turn detection with Smart Turn v3.2 (Pipecat, BSD-2-Clause), from the audio of the user's turn.

The model reads the last 8 seconds of 16 kHz audio as 80 Whisper log-mel bins x 800 frames, and gives the
probability that the speaker has finished. Shorter audio is padded with zeros at the start, so the speech is
always at the end. About 12 to 16 ms on the CPU of an Apple M5.

Why not the LiveKit turn detector: its license allows use only with the LiveKit Agents framework, and orb uses
its own Go SDK. See docs/decisions/0003-local-speech-models.md.
"""
from __future__ import annotations

import time

import numpy as np

SAMPLE_RATE = 16000
MAX_SECONDS = 8
THRESHOLD = 0.5


def last_window(samples: np.ndarray, seconds: int = MAX_SECONDS, rate: int = SAMPLE_RATE) -> np.ndarray:
    """The last `seconds` of audio. Shorter audio is padded with zeros at the start."""
    n = seconds * rate
    x = np.asarray(samples, dtype=np.float32)
    if len(x) >= n:
        return x[-n:]
    return np.pad(x, (n - len(x), 0))


class SmartTurn:
    sample_rate = SAMPLE_RATE

    def __init__(self, model_path: str, threshold: float = THRESHOLD):
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor

        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(str(model_path), sess_options=so)
        self._features = WhisperFeatureExtractor(chunk_length=MAX_SECONDS)
        self.threshold = threshold

    def probability(self, samples: np.ndarray) -> float:
        x = last_window(samples)
        f = self._features(x, sampling_rate=SAMPLE_RATE, return_tensors="np", padding="max_length",
                           max_length=MAX_SECONDS * SAMPLE_RATE, truncation=True, do_normalize=True)
        out = self._session.run(None, {"input_features": f.input_features.astype(np.float32)})
        return float(out[0][0].item())  # the model ends in a sigmoid

    def predict(self, samples: np.ndarray) -> dict:
        t0 = time.perf_counter()
        p = self.probability(samples) if len(samples) else 0.0
        return {"probability": round(p, 4), "complete": p >= self.threshold, "threshold": self.threshold,
                "ms": round((time.perf_counter() - t0) * 1000, 1), "audio_ms": round(len(samples) / SAMPLE_RATE * 1000)}
