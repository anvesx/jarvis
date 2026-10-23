"""Speech-to-text accuracy on real accented conversational speech (EdAcc), through the same codec as a call.

    python scripts/stt_bench.py prepare              # about 1 GB of EdAcc test data -> 182 clips
    python scripts/stt_bench.py run parakeet         # or: whistle, whisper, parakeet-ctc
    python scripts/stt_bench.py summary

The clips: up to 40 per speaker for the four South Asian speakers of the EdAcc test split (two Indian English,
two with Sinhalese as their first language), and 40 Mainstream US English clips as a control. Each clip has 5 to 40
words, about the length of a spoken turn. The random choice uses a fixed seed, so every run uses the same clips.

Every engine hears the clips as a call delivers them: 48 kHz -> Opus 32 kbps (voip) -> 48 kHz -> soxr "LQ" -> 16 kHz,
the same resampler quality as LiveKit's media-sdk. `run <engine> clean` uses the original clips.

WER = (substitutions + deletions + insertions) / reference words, after the Whisper English normalizer on both
sides (lower case, no punctuation, no fillers such as "um", numbers as digits).

EdAcc: Sanabria et al., "The Edinburgh International Accents of English Corpus", 2023; the clips stay in
~/.cache/orb/stt_bench and never go into git.
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import os
import random
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

WORK = Path(os.environ.get("ORB_STT_BENCH_DIR", "~/.cache/orb/stt_bench")).expanduser()
SPEAKERS = ("EDACC-C36-A", "EDACC-C05-B", "EDACC-C10-A", "EDACC-C10-B")  # the South Asian speakers
PARTS = ("test-00000", "test-00001", "test-00005")


def prepare():
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem, hf_hub_download

    fs = HfFileSystem()
    files = {f.rsplit("/", 1)[1][:10]: f.split("edacc/", 1)[1]
             for f in fs.glob("datasets/edinburghcstr/edacc/data/test-*.parquet")}
    cols = ["speaker", "text", "accent", "l1"]
    rows = []
    for part in PARTS:
        local = hf_hub_download("edinburghcstr/edacc", files[part], repo_type="dataset")
        for i, r in enumerate(pq.read_table(local, columns=cols).to_pylist()):
            rows.append({**r, "part": part, "row": i, "local": local})
    ok = lambda r: 5 <= len(r["text"].split()) <= 40 and "IGNORE" not in r["text"]
    rng = random.Random(7)
    pick = []
    for spk in SPEAKERS:
        pick += rng.sample(c := [r for r in rows if r["speaker"] == spk and ok(r)], min(40, len(c)))
    us = [r for r in rows if r["accent"] == "Mainstream US English" and ok(r) and r["part"] == "test-00000"]
    pick += rng.sample(us, 40)
    (WORK / "clean").mkdir(parents=True, exist_ok=True)
    (WORK / "opus").mkdir(exist_ok=True)
    tables, manifest = {}, []
    for r in pick:
        t = tables.setdefault(r["local"], pq.read_table(r["local"], columns=["audio"]))
        x, sr = sf.read(io.BytesIO(t.column("audio")[r["row"]].as_py()["bytes"]), dtype="float32", always_2d=True)
        x = soxr.resample(x.mean(axis=1), sr, 16000, quality="VHQ")
        name = f"{r['speaker']}_{r['row']:05d}.wav"
        sf.write(WORK / "clean" / name, x, 16000, subtype="PCM_16")
        sf.write(WORK / "opus" / name, call_path(x), 16000, subtype="PCM_16")
        manifest.append({"name": name, "ref": r["text"], "speaker": r["speaker"], "sec": round(len(x) / 16000, 2),
                         "group": "south_asian" if r["speaker"] in SPEAKERS else "us_control"})
    json.dump(manifest, open(WORK / "manifest.json", "w"), indent=1)
    print(f"{len(manifest)} clips in {WORK}:", dict(collections.Counter(m["group"] for m in manifest)))


def call_path(x16: np.ndarray) -> np.ndarray:
    """16 kHz -> 48 kHz -> Opus 32 kbps -> 48 kHz -> soxr LQ -> 16 kHz, as in a LiveKit call."""
    pcm = (np.clip(soxr.resample(x16, 16000, 48000, quality="VHQ"), -1, 1) * 32767).astype("<i2").tobytes()
    ogg = subprocess.run(["ffmpeg", "-v", "error", "-f", "s16le", "-ar", "48000", "-ac", "1", "-i", "-",
                          "-c:a", "libopus", "-b:a", "32k", "-application", "voip", "-frame_duration", "20",
                          "-f", "ogg", "-"], input=pcm, capture_output=True, check=True).stdout
    dec = subprocess.run(["ffmpeg", "-v", "error", "-i", "-", "-f", "s16le", "-ar", "48000", "-ac", "1", "-"],
                         input=ogg, capture_output=True, check=True).stdout
    return soxr.resample(np.frombuffer(dec, "<i2").astype("float32") / 32768, 48000, 16000, quality="LQ")


def engine(name):
    """A function from 16 kHz float samples to text."""
    if name in ("parakeet", "whistle"):
        from orb_speech import models
        from orb_speech.stt import ParakeetSTT, WhistleSTT
        m = (ParakeetSTT(models.parakeet_dir()) if name == "parakeet"
             else WhistleSTT(weights=str(models.whistle_weights()), language="en"))
        return lambda x: m.transcribe(x)["text"]
    import torch
    from transformers import AutoProcessor
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "whisper":
        from transformers import WhisperForConditionalGeneration
        repo = "openai/whisper-large-v3-turbo"
        proc = AutoProcessor.from_pretrained(repo)
        model = WhisperForConditionalGeneration.from_pretrained(repo, dtype=torch.float16).to(dev).eval()

        def run(x):
            f = proc(x, sampling_rate=16000, return_tensors="pt").input_features.to(dev, torch.float16)
            with torch.inference_mode():
                ids = model.generate(f, language="en", task="transcribe", num_beams=1, do_sample=False)
            return proc.batch_decode(ids, skip_special_tokens=True)[0]
        return run
    if name == "parakeet-ctc":
        from transformers import ParakeetForCTC
        repo = "nvidia/parakeet-ctc-0.6b"
        proc = AutoProcessor.from_pretrained(repo)
        model = ParakeetForCTC.from_pretrained(repo, dtype=torch.float32).to(dev).eval()

        def run(x):
            with torch.inference_mode():
                ids = model.generate(**proc(x, sampling_rate=16000, return_tensors="pt").to(dev))
            return proc.batch_decode(ids, skip_special_tokens=True)[0]
        return run
    raise SystemExit(f"unknown engine {name!r}: parakeet, whistle, whisper or parakeet-ctc")


def edits(ref: list[str], hyp: list[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def run(name, version):
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

    norm = EnglishTextNormalizer({})
    clips = json.load(open(WORK / "manifest.json"))
    stt = engine(name)
    stt(sf.read(WORK / version / clips[0]["name"], dtype="float32")[0])  # warm-up, not timed
    (WORK / "results").mkdir(exist_ok=True)
    with open(WORK / "results" / f"{name}_{version}.jsonl", "w") as out:
        for c in clips:
            x = sf.read(WORK / version / c["name"], dtype="float32")[0]
            t0 = time.perf_counter()
            text = stt(x)
            ms = (time.perf_counter() - t0) * 1000
            ref, hyp = norm(c["ref"]).split(), norm(text).split()
            out.write(json.dumps({**c, "hyp": text, "n": len(ref), "e": edits(ref, hyp), "ms": round(ms, 1)}) + "\n")
    summary()


def summary():
    print(f"{'engine (audio)':24s} {'group':12s} {'WER':>6s} {'median ms':>9s} {'p90 ms':>7s}  per speaker")
    for f in sorted((WORK / "results").glob("*.jsonl")):
        rs = [json.loads(line) for line in open(f)]
        for g in ("south_asian", "us_control"):
            s = [r for r in rs if r["group"] == g]
            ms = sorted(r["ms"] for r in s)
            spk = collections.defaultdict(lambda: [0, 0])
            for r in s:
                spk[r["speaker"][6:]][0] += r["e"]
                spk[r["speaker"][6:]][1] += r["n"]
            per = {k: f"{e / n:.0%}" for k, (e, n) in spk.items()} if g == "south_asian" else ""
            print(f"{f.stem:24s} {g:12s} {sum(r['e'] for r in s) / sum(r['n'] for r in s):6.1%} "
                  f"{statistics.median(ms):9.0f} {ms[int(0.9 * len(ms))]:7.0f}  {per}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["prepare", "run", "summary"])
    ap.add_argument("engine", nargs="?", default="parakeet")
    ap.add_argument("audio", nargs="?", default="opus", choices=["opus", "clean"])
    a = ap.parse_args()
    {"prepare": prepare, "summary": summary}.get(a.command, lambda: run(a.engine, a.audio))()
