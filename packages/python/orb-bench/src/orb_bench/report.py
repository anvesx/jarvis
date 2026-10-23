"""Results of bench runs: data/testbench/results/<suite>/<time>.json, newest last."""
from __future__ import annotations

import json
import time
from pathlib import Path

from orb_bench.paths import results_dir


def run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def save(result: dict) -> Path:
    path = results_dir(result["suite"]) / f"{result['run']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str))
    return path


def runs(suite: str) -> list[str]:
    d = results_dir(suite)
    return sorted(p.stem for p in d.glob("*.json")) if d.is_dir() else []


def load(suite: str, run: str | None = None) -> dict | None:
    names = runs(suite)
    if not names:
        return None
    name = run or names[-1]
    if name not in names:
        raise FileNotFoundError(f"There is no {suite} run {name!r}.")
    return json.loads((results_dir(suite) / f"{name}.json").read_text())
