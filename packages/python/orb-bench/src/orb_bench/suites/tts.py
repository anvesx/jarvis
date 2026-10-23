"""TTS suite: each engine and voice speaks each text of its language through POST /v1/tts.

Measures the time to the first audio, the real-time factor (making time / audio time) and intelligibility: a judge STT
hears the speech back (Parakeet for English through the speech service, Whisper large-v3-turbo for Hindi, local).
The audio is saved next to the results, so a person can listen in the bench.
"""
from __future__ import annotations

import time

import httpx
import numpy as np
import soxr

from orb_bench import audio, report
from orb_bench.corpus import manifest
from orb_bench.metrics import score
from orb_bench.paths import results_dir

CONFIGS = [  # an engine, a voice, and the languages it speaks
    {"engine": "kokoro", "voice": "af_heart", "langs": ["en-us"]},
    {"engine": "kokoro", "voice": "hf_alpha", "langs": ["hi"]},
    {"engine": "maya1", "langs": ["en-us"]},  # the voice: the default description of the engine
    {"engine": "indic-speak", "voice": "Kavya", "langs": ["hi", "en-us"]},
]
WARM_TEXT = {"hi": "नमस्ते।", "en-us": "Hello."}


class HindiJudge:
    """Whisper large-v3-turbo on the Apple GPU, loaded on first use (1.6 GB, in the Hugging Face cache)."""

    def __init__(self):
        self._model = None

    def __call__(self, x16: np.ndarray) -> str:
        import torch
        from transformers import AutoProcessor, WhisperForConditionalGeneration

        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        if self._model is None:
            repo = "openai/whisper-large-v3-turbo"
            self._proc = AutoProcessor.from_pretrained(repo)
            self._model = WhisperForConditionalGeneration.from_pretrained(repo, dtype=torch.float16).to(dev).eval()
        f = self._proc(x16, sampling_rate=16000, return_tensors="pt").input_features.to(dev, torch.float16)
        with torch.inference_mode():
            ids = self._model.generate(f, language="hi", task="transcribe", num_beams=1, do_sample=False)
        return self._proc.batch_decode(ids, skip_special_tokens=True)[0]


def run(configs=CONFIGS, speech_url="http://127.0.0.1:8200", judge: bool = True, progress=None,
        http: httpx.Client | None = None, hindi_judge=None) -> dict:
    http = http or httpx.Client(timeout=300)
    texts, rows, rid = manifest("tts"), [], report.run_id()
    hindi_judge = hindi_judge or (HindiJudge() if judge else None)
    out_dir = results_dir("tts") / rid
    for cfg in configs:
        # The first request loads the engine (seconds for a speech LM) and compiles its GPU kernels. Not measured.
        lang = cfg["langs"][0]
        warm = {"text": WARM_TEXT.get(lang, "Hello."), "engine": cfg["engine"], "voice": cfg.get("voice"), "lang": lang}
        http.post(f"{speech_url}/v1/tts", json={k: v for k, v in warm.items() if v})
        for t in texts:
            if t["lang"] not in cfg["langs"]:
                continue
            body = {"text": t["text"], "engine": cfg["engine"], "voice": cfg.get("voice"), "lang": t["lang"],
                    "style": cfg.get("style")}
            t0, first, pcm = time.perf_counter(), None, bytearray()
            with http.stream("POST", f"{speech_url}/v1/tts", json={k: v for k, v in body.items() if v}) as r:
                if r.status_code != 200:
                    rows.append({**cfg, "text_id": t["id"], "error": r.read().decode()[:200]})
                    continue
                rate = int(r.headers.get("x-sample-rate", 24000))
                for chunk in r.iter_bytes():
                    first = first or (time.perf_counter() - t0)
                    pcm.extend(chunk)
            total = time.perf_counter() - t0
            x = np.frombuffer(bytes(pcm), "<i2").astype(np.float32) / 32768
            sec = len(x) / rate
            name = f"{cfg['engine']}-{cfg.get('voice', 'default')}-{t['id']}.wav"
            audio.write(out_dir / name, x, rate)
            row = {"engine": cfg["engine"], "voice": cfg.get("voice"), "text_id": t["id"], "lang": t["lang"],
                   "chars": len(t["text"]), "first_audio_s": round(first or total, 3), "total_s": round(total, 3),
                   "audio_s": round(sec, 2), "rtf": round(total / sec, 3) if sec else None, "wav": f"{rid}/{name}"}
            if judge and sec:
                x16 = soxr.resample(x, rate, 16000, quality="VHQ")
                if t["lang"].startswith("en"):
                    j = http.post(f"{speech_url}/v1/stt", content=audio.pcm16(x16),
                                  params={"engine": "parakeet", "sample_rate": 16000, "language": "en"})
                    heard = j.json().get("text", "") if j.status_code == 200 else ""
                    lang = "en"
                else:
                    heard, lang = hindi_judge(x16), t["lang"]
                e, n = score(t["text"], heard, lang)
                row.update({"heard": heard, "wer": round(e / n, 3) if n else None})
            rows.append(row)
            if progress:
                progress(f"tts {cfg['engine']} {cfg.get('voice')} {t['id']}: first audio {row['first_audio_s']} s")
    result = {"suite": "tts", "run": rid, "configs": configs, "rows": rows}
    report.save(result)
    return result
