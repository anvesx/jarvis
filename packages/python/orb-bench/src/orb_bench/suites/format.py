"""Format suite: re-score a saved `stt` run through text stages, offline.

It reads the stored rows of an `stt` run, applies each stage combination to the raw text of every row, and scores the
result again: WER, fwer, entity, punct_f1, invented text on noise, added words, and the time of the stages. It never
calls the speech service and never loads a model, so it is cheap enough to run after every change of a formatter.

A stage is a function `str -> str`. A combination is stage names joined by `+`, applied left to right: `raw`,
`written`, `vocab+written`. `raw` is the identity and is always registered.

How a ticket adds a stage (the bench imports it only when a run names it):
- STT-2 (`written`): `register("written", "orb_speech.textfmt:written")`, or on the command line
  `python -m orb_bench run format --stages raw,written --define written=orb_speech.textfmt:written`.
- STT-3 (`vocab`): the same with its own function. A stage that needs a vocabulary is a function that already holds
  it, for example a module-level `def vocab(text): return correct(text, TERMS)`. Do not read `keys` from the corpus:
  that would hand the answer to the stage.
- A stage given as a `"module:function"` string is imported on first use. A stage that raises stops the run.
"""
from __future__ import annotations

import importlib
import time
from typing import Callable

from orb_bench import report
from orb_bench.metrics import added_words, row_scores
from orb_bench.suites import stt

Stage = Callable[[str], str]

STAGES: dict[str, Stage | str] = {"raw": lambda text: text}


def register(name: str, stage: Stage | str) -> None:
    """Add a stage: a function `str -> str`, or the string "package.module:function" (imported on first use)."""
    STAGES[name] = stage


def resolve(name: str) -> Stage:
    if name not in STAGES:
        raise KeyError(f"There is no stage {name!r}. Known: {', '.join(sorted(STAGES))}. "
                       "Register it, or pass --define name=package.module:function.")
    stage = STAGES[name]
    if isinstance(stage, str):
        module, _, attr = stage.partition(":")
        stage = STAGES[name] = getattr(importlib.import_module(module), attr)
    return stage


def parse(spec: str) -> list[str]:
    """"raw,written,vocab+written" -> ["raw", "written", "vocab+written"]."""
    return [c.strip() for c in spec.split(",") if c.strip()]


def apply(combo: str, text: str) -> tuple[str, dict[str, float]]:
    """The text after the stages of `combo`, and the milliseconds each stage took."""
    ms = {}
    for name in combo.split("+"):
        t0 = time.perf_counter()
        text = resolve(name)(text)
        ms[name] = (time.perf_counter() - t0) * 1000
    return text, ms


def run(source: str | None = None, stages=("raw",), engines=None, groups=None, progress=None) -> dict:
    """Score the rows of the stt run `source` (the newest when None) through each stage combination."""
    src = report.load("stt", source)
    if src is None:
        raise FileNotFoundError("There is no stt run to re-score. Run: python -m orb_bench run stt")
    rows = []
    for combo in stages:
        for r in src["rows"]:
            if "error" in r or (engines and r["engine"] not in engines) or (groups and r["corpus"] not in groups):
                continue
            raw = r.get("raw", r["hyp"])
            out, ms = apply(combo, raw)
            row = {k: v for k, v in r.items() if k not in ("e", "n", "fw", "punct", "entity")}
            row |= {"stages": combo, "raw": raw, "hyp": out, "stage_ms": ms, "stage_ms_total": sum(ms.values())}
            row |= row_scores(r["ref"], out, r.get("lang", "en"), r.get("written", False), r.get("keys"))
            row["added"] = added_words(raw, out, r["ref"])
            rows.append(row)
        if progress:
            progress(f"format {combo}: {sum(1 for r in rows if r['stages'] == combo)} rows")
    result = {"suite": "format", "run": report.run_id(), "source": src["run"], "stages": list(stages),
              "summary": stt.summarize(rows), "rows": rows}
    report.save(result)
    return result
