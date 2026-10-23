"""Where the bench keeps its data: data/testbench in the repo (git-ignored), or ORB_TESTBENCH_DIR."""
from __future__ import annotations

import os
from pathlib import Path

from orb_core.config import repo_root


def testbench_dir() -> Path:
    return Path(os.environ.get("ORB_TESTBENCH_DIR") or repo_root() / "data" / "testbench").expanduser()


def corpus_dir(group: str = "") -> Path:
    return testbench_dir() / "corpus" / group if group else testbench_dir() / "corpus"


def results_dir(suite: str = "") -> Path:
    return testbench_dir() / "results" / suite if suite else testbench_dir() / "results"
