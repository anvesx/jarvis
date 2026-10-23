"""STT suite: every clip of the chosen groups through POST /v1/stt of the speech service, once per engine.

Speech clips get a word error rate. Noise clips have no words, so any text there is invented: the suite counts it.
"""
from __future__ import annotations

import time
from collections import defaultdict

import httpx

from orb_bench import audio, report
from orb_bench.corpus import manifest
from orb_bench.metrics import percentile, score, wer
from orb_bench.paths import corpus_dir

GROUPS = ("stt/synthetic", "stt/edacc", "stt/librispeech", "noise")


def run(engines=("whistle", "parakeet"), groups=GROUPS, speech_url="http://127.0.0.1:8200", limit: int | None = None,
        progress=None, http: httpx.Client | None = None) -> dict:
    http = http or httpx.Client(timeout=120)
    rows = []
    for engine in engines:
        for group in groups:
            items = manifest(group)[:limit] if limit else manifest(group)
            for k, it in enumerate(items):
                x = audio.read(corpus_dir(group) / it["path"])
                t0 = time.perf_counter()
                r = http.post(f"{speech_url}/v1/stt", content=audio.pcm16(x),
                              params={"engine": engine, "sample_rate": audio.RATE, "language": "en"})
                rtt = (time.perf_counter() - t0) * 1000
                if r.status_code != 200:
                    rows.append({"engine": engine, "corpus": group, "id": it["id"], "error": r.text[:200]})
                    continue
                d = r.json()
                e, n = score(it["ref"], d["text"], it.get("lang", "en")) if it["ref"] else (0, 0)
                rows.append({"engine": engine, "corpus": group, "group": it.get("group"), "variant": it.get("variant"),
                             "id": it["id"], "ref": it["ref"], "hyp": d["text"], "e": e, "n": n, "ms": d.get("ms"),
                             "rtt_ms": round(rtt, 1), "sec": it.get("sec")})
                if progress and (k + 1) % 20 == 0:
                    progress(f"stt {engine} {group}: {k + 1} of {len(items)}")
    result = {"suite": "stt", "run": report.run_id(), "engines": list(engines), "groups": list(groups),
              "summary": summarize(rows), "rows": rows}
    report.save(result)
    return result


def summarize(rows: list[dict]) -> list[dict]:
    cells = defaultdict(list)
    for r in rows:
        if "error" in r:
            continue
        key = (r["engine"], r["corpus"], r.get("variant") or r.get("group") or "")
        cells[key].append(r)
    out = []
    for (engine, corpus, part), rs in sorted(cells.items()):
        ms = [r["ms"] for r in rs if r.get("ms") is not None]
        row = {"engine": engine, "corpus": corpus, "part": part, "clips": len(rs),
               "ms_p50": percentile(ms, 50), "ms_p90": percentile(ms, 90)}
        if corpus == "noise":
            row["invented"] = sum(1 for r in rs if r["hyp"].strip()) / len(rs)
        else:
            row["wer"] = wer([(r["e"], r["n"]) for r in rs])
        out.append(row)
    for engine in sorted({r["engine"] for r in rows if "error" not in r}):  # one total per engine and corpus
        for corpus in sorted({r["corpus"] for r in rows if "error" not in r and r["corpus"] != "noise"}):
            rs = [r for r in rows if "error" not in r and r["engine"] == engine and r["corpus"] == corpus]
            if rs:
                out.append({"engine": engine, "corpus": corpus, "part": "all", "clips": len(rs),
                            "wer": wer([(r["e"], r["n"]) for r in rs]),
                            "ms_p50": percentile([r["ms"] for r in rs if r.get("ms") is not None], 50),
                            "ms_p90": percentile([r["ms"] for r in rs if r.get("ms") is not None], 90)})
    return out
