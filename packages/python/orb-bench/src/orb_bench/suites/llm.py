"""LLM suite: scripted voice turns through the agent service, one session per case and model.

Checks each reply: it exists; at most 3 sentences and 400 characters (the voice channel's limit); no leaked prompt
notes (persona labels, "NOTE", brackets); the expected persona where the case names one. It also records the
times that the agent service reports: the Jev read, the LLM's first token and the whole reply.
"""
from __future__ import annotations

import json
import re
import time

import httpx

from orb_bench import report
from orb_bench.corpus import manifest
from orb_bench.metrics import percentile

LEAK = re.compile(r"(?i)\b(persona|need|debt|stage|why|note)\s*:|\[|NOTE_END|\A\s*(assistant|model|user|system)\s*:?\s*\n")
SENTENCE_END = re.compile(r"[.!?](?:\s|$)")


def checks(reply: str, persona: str | None, expect: dict) -> dict:
    sentences = len(SENTENCE_END.findall(reply.strip())) or (1 if reply.strip() else 0)
    out = {"has_reply": bool(reply.strip()), "short": sentences <= 3 and len(reply) <= 400,
           "no_leak": not LEAK.search(reply)}
    if expect.get("persona"):
        out["persona"] = persona == expect["persona"]
    return out


def _turn(http: httpx.Client, agent_url: str, sid: str, text: str) -> tuple:
    """One voice turn. Returns (first-token seconds, reply, persona, timing, error)."""
    t0, first, reply, persona, timing, err = time.perf_counter(), None, [], None, {}, None
    with http.stream("POST", f"{agent_url}/v1/sessions/{sid}/turns", json={"text": text, "channel": "voice"}) as s:
        for line in s.iter_lines():
            if not line.strip():
                continue
            ev = json.loads(line)
            if ev["type"] == "token":
                first = first or time.perf_counter() - t0
                reply.append(ev["data"])
            elif ev["type"] == "decision":
                persona = ev["data"].get("persona")
            elif ev["type"] == "reply":
                timing = ev["data"].get("timing") or {}
            elif ev["type"] == "error":
                err = ev["data"]
    return first, "".join(reply), persona, timing, err


def run(models: list[str] | None = None, agent_url="http://127.0.0.1:8100", progress=None,
        http: httpx.Client | None = None) -> dict:
    http = http or httpx.Client(timeout=300)
    if not models:
        info = http.get(f"{agent_url}/v1/models").json()["llm"]
        models = [info["default"]]
    rows = []
    for model in models:
        # Ollama keeps one model in memory: the first turn after a switch also loads the model. Not measured.
        r = http.post(f"{agent_url}/v1/sessions", json={"topic": "Bench: warm-up", "config": {"llm_model": model}})
        if r.status_code == 201:
            _turn(http, agent_url, r.json()["id"], "Hello.")
            if progress:
                progress(f"llm {model}: loaded")
        for case in manifest("llm"):
            for n in range(case.get("repeat", 1)):  # a case with "repeat" runs again: the model does not answer alike
                name = case["id"] + (f"#{n + 1}" if case.get("repeat", 1) > 1 else "")
                r = http.post(f"{agent_url}/v1/sessions", json={"topic": f"Bench: {name}", "config": {"llm_model": model}})
                if r.status_code != 201:
                    rows.append({"model": model, "case": name, "error": r.text[:200]})
                    continue
                sid = r.json()["id"]
                for turn in case["turns"]:
                    first, text, persona, timing, err = _turn(http, agent_url, sid, turn)
                row = {"model": model, "case": name, "reply": text, "persona": persona,
                       "first_token_s": round(first, 3) if first else None, "timing": timing,
                       "checks": checks(text, persona, case.get("expect", {}))}
                if err:
                    row["error"] = err
                rows.append(row)
                if progress:
                    progress(f"llm {model} {name}: {'ok' if all(row['checks'].values()) else 'check failed'}")
    result = {"suite": "llm", "run": report.run_id(), "models": models, "rows": rows,
              "summary": [summarize(m, [r for r in rows if r["model"] == m]) for m in models]}
    report.save(result)
    return result


def summarize(model: str, rows: list[dict]) -> dict:
    """The cases of a model: how many passed, how many replies were silent, and how long the first word took."""
    first = [r["first_token_s"] for r in rows if r.get("first_token_s") is not None]
    return {"model": model, "cases": len(rows),
            "passed": sum(1 for r in rows if r.get("checks") and all(r["checks"].values()) and "error" not in r),
            "silent": sum(1 for r in rows if r.get("checks") and not r["checks"]["has_reply"]),
            "first_token_p50": percentile(first, 50), "first_token_max": max(first) if first else None}
