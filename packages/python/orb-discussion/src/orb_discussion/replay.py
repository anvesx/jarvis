"""Replay a saved chat through the debt engine, one message at a time, with a live Jev call for each message.

    python -m orb_discussion.replay data/chats/20261012-152327.json

The engine sees only the messages so far, exactly as in a live chat. It reads both formats: the old chat app
(data/chats/*.json) and orb sessions (data/sessions/*.json). Writes data/debt_runs/<chat id>.json and prints one
line per message. Build the page with `python -m orb_discussion.view data/debt_runs/<chat id>.json`.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from orb_core.config import bootstrap
from orb_core.jev import cost_usd, make_reader
from orb_discussion.engine import DebtEngine


def load_chat(path: Path) -> tuple[str, list[dict], dict]:
    """(topic, messages as {seq, speaker, text}, old personas by seq) for both saved formats."""
    saved = json.loads(path.read_text())
    if "conv" in saved:  # the old chat app
        conv = saved["conv"]
        old = {t["human"]: t["plan"]["persona"].capitalize() for t in saved.get("state", {}).get("turns", [])
               if isinstance(t.get("plan"), dict)}
        return conv["title"], [{"seq": m["seq"], "speaker": m["speaker"], "text": m["text"]} for m in conv["messages"]], old
    old = {t["human"]: t["decision"]["persona"] for t in saved.get("turns", []) if t.get("decision")}
    return saved.get("title") or saved["topic"], saved["messages"], old


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("chat", help="a saved chat: data/chats/<id>.json or data/sessions/<id>.json")
    ap.add_argument("--out", default="data/debt_runs")
    args = ap.parse_args()

    bootstrap()
    topic, msgs, old = load_chat(Path(args.chat))
    engine = DebtEngine(make_reader(), topic)
    t0 = time.time()
    for m in msgs:
        e = engine.step(m["speaker"], m["text"])
        if m["seq"] in old and "need" in e:
            e["need"]["old_persona"] = old[m["seq"]]
        started = [ev["debt"] for ev in e["events"] if ev["type"] == "start"]
        paid = [ev["debt"] for ev in e["events"] if ev["type"] == "pay"]
        tail = (f"need {e['need']['debt']} -> {e['need']['persona']} (before: {e['need'].get('old_persona', '-')})" if "need" in e
                else f"check: {e['check']['text']}" if "check" in e else "")
        print(f"{e['id']:>4} {e['speaker']:5} {e['call']['seconds']:5.2f}s {e['call']['tokens'] or 0:6d} tok "
              f"{len(e['call']['questions']):2d} q | {e['stage']:5} | start {','.join(started) or '-':12} "
              f"paid {','.join(paid) or '-':12} | {tail}")
    run = {"chat": Path(args.chat).stem, "topic": topic, "model": engine.trace[0]["call"]["model"] if engine.trace else None,
           "ran_at": time.strftime("%Y-%m-%d %H:%M"), "seconds": round(time.time() - t0, 1), "tokens": engine.tokens,
           "cost_usd": cost_usd(engine.tokens), "steps": engine.trace}
    dest = Path(args.out) / f"{run['chat']}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(run, indent=1, ensure_ascii=False))
    print(f"\n{len(engine.trace)} messages, {run['seconds']} s, {engine.tokens} tokens, about ${run['cost_usd']}. Wrote {dest}")


if __name__ == "__main__":
    main()
