"""The bench's test data in data/testbench/corpus (git-ignored). `python -m orb_bench corpus` builds it.

Groups (each in corpus/<group>/ with a manifest.json; every audio clip went through the call path):
- stt/synthetic: 12 sentences with orb's topic words x 6 Kokoro voices (two of them Hindi voices reading English,
  a stand-in for an Indian accent) x 4 versions: clean, and pink noise at 20, 10 and 5 dB SNR.
- stt/dictation: 40 written-form sentences (dev and office dictation: numbers, times, emails, code names, commands,
  self-corrections, lists) x the same 6 voices x clean and pink noise at 10 dB. Each has the spoken text that Kokoro
  reads, the written `ref`, and `keys` (entities that must be exact). See dictation.py.
- stt/edacc: the 182 EdAcc clips of decision 0005 (4 South Asian speakers and 40 US English clips; seed 7).
- stt/librispeech: 100 LibriSpeech test-clean utterances of 5 to 40 words (read speech; seed 11).
- noise: no speech at all (white, pink, brown, hum) at three levels and two lengths. Any word is invented.
- echo: the agent's own voice at a low level, as from laptop speakers into the microphone.
- llm/cases.json and tts/texts.json: text cases.

Licenses: EdAcc and LibriSpeech are CC-BY licensed research sets; their audio stays in data/testbench, never in git.
"""
from __future__ import annotations

import io
import json
import random
from pathlib import Path

import numpy as np

from orb_bench import audio
from orb_bench.dictation import DICTATION
from orb_bench.paths import corpus_dir

SENTENCES = [
    "We should store user passwords in a plain JSON file so the assistant can log in for them.",
    "What would you pick for the database, Postgres or a document store?",
    "The GST invoice for October is still pending with the client.",
    "Can you summarize where we are?",
    "I think we should use Go for the backend, because the team already knows it.",
    "Our budget is twelve thousand five hundred rupees for three months.",
    "No, I don't agree with you.",
    "Please send the meeting notes to Priya before Friday.",
    "The latency of the voice agent must stay under one second.",
    "Let's deploy the new build on Monday morning.",
    "Which input tax credit can a small firm claim on a company car?",
    "Sure, that sounds good to me.",
]
VOICES = [("af_heart", "en-us"), ("am_michael", "en-us"), ("bf_emma", "en-gb"), ("bm_george", "en-gb"),
          ("hf_alpha", "en-us"), ("hm_omega", "en-us")]
SNRS = (None, 20, 10, 5)
DICTATION_SNRS = (None, 10)

LLM_CASES = [
    {"id": "passwords", "turns": ["We should store user passwords in a plain JSON file so the assistant can log in for them."],
     "expect": {"persona": "Rival"}},
    {"id": "disagree", "turns": ["We should store user passwords in a plain JSON file so the assistant can log in for them.",
                                 "No, I don't agree with you."], "expect": {"persona": "Examiner"}},
    {"id": "summary", "turns": ["Let's design the login flow for our app.", "Can you summarize where we are?"], "expect": {}},
    {"id": "gst", "turns": ["Our client asks whether they can claim GST input credit on a company car."], "expect": {}},
    {"id": "backchannel", "turns": ["Okay."], "expect": {}},
    # Lines of a real call (2026-10-22): short, and about the call itself. On such lines gemma4:e2b writes 200 to 370
    # hidden tokens before its first word, and an agent with a small token limit stayed silent. The hidden text changes
    # from run to run, so each line runs 3 times.
    {"id": "mic_check", "turns": ["Can you hear me clearly? I don't know you are listening or not."], "expect": {},
     "repeat": 3},
    {"id": "hello", "turns": ["Hello, can you listen me? Then reply past."], "expect": {}, "repeat": 3},
    {"id": "are_you_there", "turns": ["Hello? Are you there?"], "expect": {}, "repeat": 3},
]
TTS_TEXTS = [
    {"id": "short", "text": "Sure, let's do that.", "lang": "en-us"},
    {"id": "question", "text": "What would happen to the passwords if the file were copied by mistake?", "lang": "en-us"},
    {"id": "long", "text": "Storing passwords in a plain file means anyone who reads the file can log in as your users. "
                           "A safer way is to keep only a salted hash, and to check the hash at login.", "lang": "en-us"},
    {"id": "numbers", "text": "The invoice of twelve thousand five hundred rupees is due on the fifteenth of March.",
     "lang": "en-us"},
    {"id": "hindi", "text": "नमस्ते, आप कैसे हैं? आज हम आपके प्रोजेक्ट पर बात करेंगे।", "lang": "hi"},
    {"id": "hinglish", "text": "मैं कल office जा रहा हूँ, meeting है।", "lang": "hi"},
]


def _save(group: str, items: list[dict]) -> Path:
    path = corpus_dir(group) / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, indent=1, ensure_ascii=False))
    return path


def manifest(group: str) -> list[dict]:
    path = corpus_dir(group) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"The bench corpus has no {group!r}. Build it: python -m orb_bench corpus")
    return json.loads(path.read_text())


def _kokoro():
    from orb_speech import models
    from orb_speech.tts import KokoroTTS
    return KokoroTTS(models.path("kokoro-v1.0.int8.onnx"), models.path("voices-v1.0.bin"))


def stt_synthetic(log=print) -> int:
    import soxr

    tts, rng, items = _kokoro(), np.random.default_rng(3), []
    for si, text in enumerate(SENTENCES):
        for voice, lang in VOICES:
            clean = soxr.resample(tts.synthesize(text, voice=voice, lang=lang), 24000, audio.RATE, quality="VHQ")
            clean = audio.pad(clean, 0.3, 0.3)
            for snr in SNRS:
                x = clean if snr is None else audio.mix(clean, audio.noise("pink", len(clean), rng), snr)
                variant = "clean" if snr is None else f"snr{snr}"
                rel = f"{voice}/{si:02d}-{variant}.wav"
                audio.write(corpus_dir("stt/synthetic") / rel, audio.call_path(x))
                items.append({"id": f"{voice}-{si:02d}-{variant}", "path": rel, "ref": text, "lang": "en",
                              "group": "synthetic", "voice": voice, "variant": variant, "written": True,
                              "sec": round(len(clean) / audio.RATE, 2)})
        log(f"stt/synthetic: sentence {si + 1} of {len(SENTENCES)}")
    _save("stt/synthetic", items)
    return len(items)


def stt_dictation(log=print) -> int:
    """Like stt_synthetic, but Kokoro reads `spoken` and the reference is the written form."""
    import soxr

    tts, rng, items = _kokoro(), np.random.default_rng(4), []
    for si, d in enumerate(DICTATION):
        for voice, lang in VOICES:
            clean = soxr.resample(tts.synthesize(d["spoken"], voice=voice, lang=lang), 24000, audio.RATE, quality="VHQ")
            clean = audio.pad(clean, 0.3, 0.3)
            for snr in DICTATION_SNRS:
                x = clean if snr is None else audio.mix(clean, audio.noise("pink", len(clean), rng), snr)
                variant = "clean" if snr is None else f"snr{snr}"
                rel = f"{voice}/{si:02d}-{variant}.wav"
                audio.write(corpus_dir("stt/dictation") / rel, audio.call_path(x))
                items.append({"id": f"{voice}-{si:02d}-{variant}", "path": rel, "ref": d["ref"], "spoken": d["spoken"],
                              "keys": d["keys"], "cls": d["cls"], "lang": "en", "group": "dictation", "voice": voice,
                              "variant": variant, "written": True, "sec": round(len(clean) / audio.RATE, 2)})
        log(f"stt/dictation: sentence {si + 1} of {len(DICTATION)}")
    _save("stt/dictation", items)
    return len(items)


def stt_edacc(log=print) -> int:
    """The clips of decision 0005 (scripts/stt_bench.py prepare): the same speakers, filter and seed."""
    import pyarrow.parquet as pq
    import soundfile as sf
    import soxr
    from huggingface_hub import HfFileSystem, hf_hub_download

    speakers = ("EDACC-C36-A", "EDACC-C05-B", "EDACC-C10-A", "EDACC-C10-B")
    parts = ("test-00000", "test-00001", "test-00005")
    files = {f.rsplit("/", 1)[1][:10]: f.split("edacc/", 1)[1]
             for f in HfFileSystem().glob("datasets/edinburghcstr/edacc/data/test-*.parquet")}
    rows = []
    for part in parts:
        local = hf_hub_download("edinburghcstr/edacc", files[part], repo_type="dataset")
        for i, r in enumerate(pq.read_table(local, columns=["speaker", "text", "accent", "l1"]).to_pylist()):
            rows.append({**r, "part": part, "row": i, "local": local})
    ok = lambda r: 5 <= len(r["text"].split()) <= 40 and "IGNORE" not in r["text"]
    rng, pick = random.Random(7), []
    for spk in speakers:
        pick += rng.sample(c := [r for r in rows if r["speaker"] == spk and ok(r)], min(40, len(c)))
    us = [r for r in rows if r["accent"] == "Mainstream US English" and ok(r) and r["part"] == "test-00000"]
    pick += rng.sample(us, 40)
    tables, items = {}, []
    for r in pick:
        t = tables.setdefault(r["local"], pq.read_table(r["local"], columns=["audio"]))
        x, sr = sf.read(io.BytesIO(t.column("audio")[r["row"]].as_py()["bytes"]), dtype="float32", always_2d=True)
        x = soxr.resample(x.mean(axis=1), sr, audio.RATE, quality="VHQ")
        rel = f"{r['speaker']}_{r['row']:05d}.wav"
        audio.write(corpus_dir("stt/edacc") / rel, audio.call_path(x))
        items.append({"id": rel[:-4], "path": rel, "ref": r["text"], "lang": "en", "speaker": r["speaker"],
                      "group": "south_asian" if r["speaker"] in speakers else "us_control",
                      "sec": round(len(x) / audio.RATE, 2)})
    log(f"stt/edacc: {len(items)} clips")
    _save("stt/edacc", items)
    return len(items)


def stt_librispeech(log=print, n: int = 100) -> int:
    import pyarrow.parquet as pq
    import soundfile as sf
    import soxr
    from huggingface_hub import hf_hub_download

    local = hf_hub_download("openslr/librispeech_asr", "clean/test/0000.parquet", repo_type="dataset")
    t = pq.read_table(local)
    texts = t.column("text").to_pylist()
    ids = [i for i, x in enumerate(texts) if 5 <= len(x.split()) <= 40]
    items = []
    for i in sorted(random.Random(11).sample(ids, n)):
        a = t.column("audio")[i].as_py()
        x, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32", always_2d=True)
        x = soxr.resample(x.mean(axis=1), sr, audio.RATE, quality="VHQ") if sr != audio.RATE else x.mean(axis=1)
        rel = f"{i:05d}.wav"
        audio.write(corpus_dir("stt/librispeech") / rel, audio.call_path(x))
        items.append({"id": f"ls-{i:05d}", "path": rel, "ref": texts[i], "lang": "en", "group": "librispeech",
                      "sec": round(len(x) / audio.RATE, 2)})
    log(f"stt/librispeech: {len(items)} clips")
    _save("stt/librispeech", items)
    return len(items)


def noise_only(log=print) -> int:
    rng, items = np.random.default_rng(5), []
    for kind in ("white", "pink", "brown", "hum"):
        for level in (0.003, 0.01, 0.03):
            for sec in (0.8, 2.5):
                x = audio.noise(kind, int(sec * audio.RATE), rng) * level
                rel = f"{kind}-{level}-{sec}.wav"
                audio.write(corpus_dir("noise") / rel, audio.call_path(x))
                items.append({"id": rel[:-4], "path": rel, "ref": "", "lang": "en", "group": "noise", "kind": kind,
                              "level": level, "sec": sec})
    log(f"noise: {len(items)} clips")
    _save("noise", items)
    return len(items)


def echo(log=print) -> int:
    import soxr

    tts, items = _kokoro(), []
    lines = ["That sounds like a major security risk.", "What makes you think that storing passwords is acceptable?",
             "Let me summarize the three points we agreed on."]
    for li, text in enumerate(lines):
        x = soxr.resample(tts.synthesize(text), 24000, audio.RATE, quality="VHQ")  # the agent's voice, af_heart
        for level in (0.005, 0.02):
            y = audio.pad(x * (level / (audio.rms(x) or 1.0)), 0.3, 0.3)
            rel = f"echo-{li}-{level}.wav"
            audio.write(corpus_dir("echo") / rel, audio.call_path(y))
            items.append({"id": rel[:-4], "path": rel, "ref": text, "lang": "en", "group": "echo", "level": level,
                          "sec": round(len(y) / audio.RATE, 2)})
    log(f"echo: {len(items)} clips")
    _save("echo", items)
    return len(items)


def texts(log=print) -> int:
    _save("llm", LLM_CASES)
    _save("tts", TTS_TEXTS)
    log(f"llm: {len(LLM_CASES)} cases; tts: {len(TTS_TEXTS)} texts")
    return len(LLM_CASES) + len(TTS_TEXTS)


BUILDERS = {"texts": texts, "stt/synthetic": stt_synthetic, "stt/dictation": stt_dictation, "noise": noise_only, "echo": echo,
            "stt/edacc": stt_edacc, "stt/librispeech": stt_librispeech}


def build(only: list[str] | None = None, log=print) -> dict[str, int]:
    counts = {}
    for name, fn in BUILDERS.items():
        if only and name not in only:
            continue
        counts[name] = fn(log=log)
    return counts
