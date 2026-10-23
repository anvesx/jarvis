"""STT suite: every clip of the chosen groups through POST /v1/stt of the speech service, once per engine.

Speech clips get a word error rate. Noise clips have no words, so any text there is invented: the suite counts it.
Clips with a written-form reference (`written` in the manifest: stt/synthetic, stt/dictation) also get fwer, entity and
punct_f1 (metrics.py). Each row keeps the raw text of the engine, so the format suite can re-score it offline.
"""
from __future__ import annotations

import time
from collections import defaultdict

import httpx

from orb_bench import audio, report
from orb_bench.corpus import manifest
from orb_bench.metrics import f1, percentile, row_scores, wer
from orb_bench.paths import corpus_dir

GROUPS = ("stt/synthetic", "stt/dictation", "stt/edacc", "stt/librispeech", "noise")


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
                row = {"engine": engine, "corpus": group, "group": it.get("group"), "variant": it.get("variant"),
                       "cls": it.get("cls"), "id": it["id"], "ref": it["ref"], "keys": it.get("keys") or [],
                       "written": bool(it.get("written")), "lang": it.get("lang", "en"),
                       "hyp": d["text"], "raw": d.get("raw", d["text"]), "ms": d.get("ms"),
                       "rtt_ms": round(rtt, 1), "sec": it.get("sec")}
                rows.append(row | row_scores(it["ref"], d["text"], row["lang"], row["written"], row["keys"]))
                if progress and (k + 1) % 20 == 0:
                    progress(f"stt {engine} {group}: {k + 1} of {len(items)}")
    result = {"suite": "stt", "run": report.run_id(), "engines": list(engines), "groups": list(groups),
              "summary": summarize(rows), "rows": rows}
    report.save(result)
    return result


def _stats(rs: list[dict]) -> dict:
    """The numbers of one cell: WER, and for written-form clips fwer, entity and punct_f1."""
    ms = [r["ms"] for r in rs if r.get("ms") is not None]
    out = {"clips": len(rs), "ms_p50": percentile(ms, 50), "ms_p90": percentile(ms, 90)}
    if any("stage_ms_total" in r for r in rs):  # a format-suite cell: time of the stages, and words they added
        st = [r["stage_ms_total"] for r in rs]
        out |= {"stage_ms_p50": percentile(st, 50), "stage_ms_p95": percentile(st, 95),
                "added": sum(len(r["added"]) for r in rs)}
    if all(r["corpus"] == "noise" for r in rs):
        out["invented"] = sum(1 for r in rs if r["hyp"].strip()) / len(rs)
        return out
    out["wer"] = wer([(r["e"], r["n"]) for r in rs])
    w = [r for r in rs if "fw" in r]
    if w:
        out["fwer"] = wer([tuple(r["fw"]) for r in w])
        ent = [r["entity"] for r in w if "entity" in r]
        out["entity"] = sum(f for f, _ in ent) / sum(t for _, t in ent) if ent and sum(t for _, t in ent) else None
        out["punct_f1"] = f1(*(sum(r["punct"][k] for r in w) for k in range(3)))
    return out


def summarize(rows: list[dict]) -> list[dict]:
    """One row per engine (and stage combination, in the format suite), corpus and part, then a total per corpus.
    A dictation clip also counts in the part `cls:<class>`."""
    ok = [r for r in rows if "error" not in r]
    cells = defaultdict(list)
    for r in ok:
        base = (r["engine"], r.get("stages", ""), r["corpus"])
        cells[base + (r.get("variant") or r.get("group") or "",)].append(r)
        if r.get("cls"):
            cells[base + (f"cls:{r['cls']}",)].append(r)
        if r["corpus"] != "noise":
            cells[base + ("all",)].append(r)
    out = []
    order = lambda k: (k[0], k[1], k[2], k[3] == "all", k[3].startswith("cls:"), k[3])
    for (engine, stages, corpus, part), rs in sorted(cells.items(), key=lambda kv: order(kv[0])):
        out.append({"engine": engine, **({"stages": stages} if stages else {}), "corpus": corpus, "part": part,
                    **_stats(rs)})
    return out
