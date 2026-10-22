"""Build the page that shows every Jev call of a debt-engine run.

    python -m orb_discussion.view data/debt_runs/20261012-152327.json

Writes docs/explainers/debt_run.html: a standalone page with the run data inside it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from orb_discussion.rules import DUE, KINDS, PERSONA, PROMISES, STAGES, TIE, YES


def render(run: dict) -> str:
    run = {**run, "rulebook": {"yes": YES, "stages": STAGES, "promises": PROMISES, "due": DUE, "kinds": KINDS,
                               "persona": PERSONA, "tie": TIE}}
    data = json.dumps(run, ensure_ascii=False).replace("</", "<\\/")  # the data cannot close the <script> tag
    return (Path(__file__).parent / "view_template.html").read_text().replace("__RUN_DATA__", data)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="a run file, data/debt_runs/<chat id>.json")
    ap.add_argument("--out", default="docs/explainers/debt_run.html")
    args = ap.parse_args()
    run = json.loads(Path(args.run).read_text())
    page = render(run)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(page)
    print(f"Wrote {args.out} ({len(page) // 1024} KB, {len(run['steps'])} messages)")


if __name__ == "__main__":
    main()
