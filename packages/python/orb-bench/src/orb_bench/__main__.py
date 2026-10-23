"""The orb test bench.

    python -m orb_bench corpus [--only stt/synthetic,noise,...]   # build data/testbench/corpus (git-ignored)
    python -m orb_bench run stt [--engines whistle,parakeet] [--groups stt/synthetic,noise] [--limit N]
    python -m orb_bench run format [--source <stt run>] [--stages raw,written] [--define written=pkg.mod:func]
                                   [--engines parakeet] [--groups stt/dictation]    # offline, no service needed
    python -m orb_bench run tts [--no-judge]
    python -m orb_bench run llm [--models gemma4:e2b,qwen3:4b]
    python -m orb_bench run turn                                 # needs every service of scripts/dev.sh
    python -m orb_bench show <suite> [run]

Results go to data/testbench/results/<suite>/<run>.json (data/testbench is ORB_TESTBENCH_DIR when set); the bench
page shows them too.
"""
from __future__ import annotations

import argparse
import json
import os

from orb_bench import corpus, report

AGENT = os.environ.get("ORB_AGENT_URL", "http://127.0.0.1:8100")
SPEECH = os.environ.get("ORB_SPEECH_URL", "http://127.0.0.1:8200")


def pct(v) -> str:
    return "-" if v is None else f"{v:.1%}"


def num(v, digits: int = 0) -> str:
    return "-" if v is None else f"{v:.{digits}f}"


def show(result: dict) -> None:
    suite = result["suite"]
    print(f"{suite} run {result['run']}")
    if suite in ("stt", "format"):
        show_scores(result)
    elif suite == "tts":
        print(f"{'engine':8s} {'voice':10s} {'text':9s} {'first audio':>11s} {'RTF':>6s} {'audio':>6s} {'WER':>6s}")
        for r in result["rows"]:
            if "error" in r:
                print(f"{r['engine']:8s} {str(r.get('voice')):10s} {r['text_id']:9s} ERROR {r['error'][:60]}")
                continue
            print(f"{r['engine']:8s} {str(r['voice']):10s} {r['text_id']:9s} {r['first_audio_s']:10.2f}s "
                  f"{r['rtf']:6.2f} {r['audio_s']:5.1f}s {pct(r.get('wer')):>6s}")
    elif suite == "llm":
        for s in result["summary"]:
            print(f"{s['model']}: {s['passed']} of {s['cases']} cases pass, {s.get('silent', 0)} silent, first word "
                  f"p50 {num(s.get('first_token_p50'), 2)} s, slowest {num(s.get('first_token_max'), 2)} s")
        for r in result["rows"]:
            failed = [k for k, v in (r.get("checks") or {}).items() if not v]
            t = r.get("timing") or {}
            print(f"  {r['model']:18s} {r['case']:12s} first token {num(r.get('first_token_s'), 2)}s "
                  f"(LLM {t.get('llm_first_token')}s) {'FAILED ' + ','.join(failed) if failed else 'ok'}"
                  f"{' ERROR ' + str(r['error'])[:60] if r.get('error') else ''}")
    else:
        print(json.dumps(result.get("summary"), indent=1))
        for r in result["rows"]:
            print("  ", r)


def show_scores(result: dict) -> None:
    """The summary of an stt or format run: one line per engine (and stages), corpus and part."""
    fmt = result["suite"] == "format"
    if fmt:
        print(f"re-scored from stt run {result['source']}")
    print(f"{'engine':9s} {'stages':14s} {'corpus':15s} {'part':16s} {'clips':>5s} {'WER':>7s} {'fwer':>7s} {'entity':>7s} "
          f"{'punctF1':>8s} {'invented':>9s} {'added':>5s} {'p50 ms':>7s} {'p90 ms':>7s}" + (f" {'stage p95':>9s}" if fmt else ""))
    for r in result["summary"]:
        print(f"{r['engine']:9s} {r.get('stages', ''):14s} {r['corpus']:15s} {r['part']:16s} {r['clips']:5d} "
              f"{pct(r.get('wer')):>7s} {pct(r.get('fwer')):>7s} {pct(r.get('entity')):>7s} {pct(r.get('punct_f1')):>8s} "
              f"{pct(r.get('invented')):>9s} {str(r.get('added', '-')):>5s} {num(r['ms_p50']):>7s} {num(r['ms_p90']):>7s}"
              + (f" {num(r.get('stage_ms_p95'), 3):>9s}" if fmt else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("corpus")
    c.add_argument("--only", default="")
    r = sub.add_parser("run")
    r.add_argument("suite", choices=["stt", "format", "tts", "llm", "turn"])
    r.add_argument("--engines", default="whistle,parakeet")
    r.add_argument("--source", default=None, help="format: the stt run to re-score (default: the newest)")
    r.add_argument("--stages", default="raw", help="format: stage combinations, e.g. raw,written,vocab+written")
    r.add_argument("--define", action="append", default=[], help="format: name=package.module:function")
    r.add_argument("--groups", default="")
    r.add_argument("--models", default="")
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--no-judge", action="store_true")
    s = sub.add_parser("show")
    s.add_argument("suite", choices=["stt", "format", "tts", "llm", "turn"])
    s.add_argument("run", nargs="?")
    a = ap.parse_args()
    if a.cmd == "corpus":
        print(corpus.build([x for x in a.only.split(",") if x] or None))
    elif a.cmd == "show":
        res = report.load(a.suite, a.run)
        if res is None:
            print(f"There are no {a.suite} runs yet.")
        else:
            show(res)
    else:
        from orb_bench.suites import format as fmt, llm, stt, tts, turn
        if a.suite == "format":
            for d in a.define:
                name, _, path = d.partition("=")
                fmt.register(name, path)
            res = fmt.run(a.source, fmt.parse(a.stages), [x for x in a.engines.split(",") if x],
                          [x for x in a.groups.split(",") if x] or None, progress=print)
        elif a.suite == "stt":
            groups = [x for x in a.groups.split(",") if x] or list(stt.GROUPS)
            res = stt.run(a.engines.split(","), groups, SPEECH, a.limit, progress=print)
        elif a.suite == "tts":
            res = tts.run(speech_url=SPEECH, judge=not a.no_judge, progress=print)
        elif a.suite == "llm":
            res = llm.run([m for m in a.models.split(",") if m] or None, AGENT, progress=print)
        else:
            res = turn.run(AGENT, SPEECH, progress=print)
        show(res)


if __name__ == "__main__":
    main()
