"""Turn suite: a synthetic person in a real LiveKit room, against the running voice worker (VAD + turn detector).

Cases and what they measure:
- finished: whole sentences; the time from the end of speech to the first agent audio.
- unfinished: a sentence that stops mid-phrase for 1.2 s, then goes on. Agent audio in the pause is an early reply.
- noise: noise only (no words) while the agent listens. A new person message or agent audio is a false turn.
- backchannel: "Sure." while the agent speaks. The agent must keep talking, and "sure" must not become a message.
- echo: the agent's own voice at a low level while it speaks (laptop speakers). Reported, not scored.
- barge-in: a full sentence while the agent speaks. The time until the agent's audio stops, and from the worker's
  events: the first transcript with 2 words (the interruption needs them) and the agent's pause.
"""
from __future__ import annotations

import asyncio
import time

import httpx
import numpy as np

from orb_bench import audio as baudio, report
from orb_bench.corpus import manifest
from orb_bench.metrics import percentile
from orb_bench.paths import corpus_dir
from orb_bench.room import Person, at_48k, kokoro, send, silence

FINISHED = ["What would you pick for the database, Postgres or a document store?",
            "Our budget is twelve thousand five hundred rupees for three months.",
            "Can you summarize where we are?",
            "The latency of the voice agent must stay under one second."]
UNFINISHED = [("I think the main problem is that", "the team has no time for reviews."),
              ("We could store the keys in", "a vault that only the server can read."),
              ("Before we decide, can you tell me whether", "the GST rules allow this?")]
NOISE = ["pink-0.03-2.5", "hum-0.03-2.5", "white-0.01-2.5", "brown-0.03-2.5"]
PROMPT = "Explain in three sentences why hashing passwords is safer than storing them."
BARGE = "Wait, stop, I want to talk about the database instead of passwords."


async def _speaking(p: Person, limit: float = 30.0) -> float | None:
    """Asks a question that gives a long answer, and returns when the agent's audio started."""
    a = await kokoro(p.client, p.speech_url, PROMPT)
    await send(p.source, a)
    t_end, deadline = time.monotonic(), time.monotonic() + limit
    while time.monotonic() < deadline:
        await silence(p.source, 0.1)
        if (t := p.ear.first_loud_after(t_end)) is not None:
            return t
    return None


def _committed(p: Person, after: float) -> float | None:
    """Seconds from `after` to the worker's turn commit (its end-of-utterance metrics event), or None."""
    t = p.first_event(after, "metrics", lambda ev: (ev.get("metrics") or {}).get("type") == "eou_metrics")
    return round(t - after, 3) if t else None


async def _run(agent_url: str, speech_url: str, config: dict | None, progress) -> list[dict]:
    rows = []
    async with httpx.AsyncClient(timeout=60) as client:
        p = Person(client, agent_url, speech_url)
        joined = await p.join("Turn suite", config)
        rows.append({"case": "join", "seconds": round(joined, 2)})
        noise = {it["id"]: it for it in manifest("noise")}

        for text in FINISHED:
            a = await kokoro(client, speech_url, text)
            await send(p.source, a)
            t_end = time.monotonic()
            first = await p.wait_reply(t_end)
            rows.append({"case": "finished", "text": text, "first_audio_s": round(first - t_end, 3) if first else None,
                         "commit_s": _committed(p, t_end)})
            if progress:
                progress(f"turn finished: {rows[-1]['first_audio_s']} s")

        for first_part, rest in UNFINISHED:
            await send(p.source, await kokoro(client, speech_url, first_part))
            t_pause = time.monotonic()
            await silence(p.source, 1.2)
            early = p.ear.first_loud_after(t_pause, until=time.monotonic()) is not None
            await send(p.source, await kokoro(client, speech_url, rest))
            t_end = time.monotonic()
            first = await p.wait_reply(t_end)
            rows.append({"case": "unfinished", "text": f"{first_part} … {rest}", "early_reply": early,
                         "first_audio_s": round(first - t_end, 3) if first else None, "commit_s": _committed(p, t_end)})
            if progress:
                progress(f"turn unfinished: early reply {early}")

        for nid in NOISE:
            persons = sum(1 for m in await p.messages() if m["speaker"] == "human")
            x = baudio.read(corpus_dir("noise") / noise[nid]["path"])
            t0 = time.monotonic()
            await send(p.source, at_48k(x))
            await silence(p.source, 3.0)
            after = sum(1 for m in await p.messages() if m["speaker"] == "human")
            rows.append({"case": "noise", "clip": nid, "new_turn": after > persons,
                         "agent_audio": p.ear.first_loud_after(t0) is not None})

        # while the agent speaks: a backchannel, then an echo, then a real barge-in
        t_speak = await _speaking(p)
        if t_speak is not None:
            await silence(p.source, 1.0)
            persons = [m["text"].lower() for m in await p.messages() if m["speaker"] == "human"]
            sure = await kokoro(client, speech_url, "Sure.")
            t_s = time.monotonic()
            await send(p.source, sure)
            await silence(p.source, 2.5)
            kept = p.ear.first_loud_after(t_s + len(sure) / 48000 + 1.0) is not None
            new = [m["text"].lower() for m in await p.messages() if m["speaker"] == "human"][len(persons):]
            rows.append({"case": "backchannel", "agent_kept_talking": kept,
                         "became_turn": any("sure" in m for m in new)})
            await p.wait_reply(t_s, limit=20)

        t_speak = await _speaking(p)
        if t_speak is not None:
            await silence(p.source, 0.8)
            echo = [it for it in manifest("echo") if it["level"] == 0.02][0]
            persons = sum(1 for m in await p.messages() if m["speaker"] == "human")
            t_e = time.monotonic()
            await send(p.source, at_48k(baudio.read(corpus_dir("echo") / echo["path"])))
            await silence(p.source, 1.5)
            gap = [x for x in p.ear.loud_times if t_e < x]
            continued = bool(gap and gap[-1] > time.monotonic() - 1.0)
            await p.wait_reply(t_e, limit=20)
            new = sum(1 for m in await p.messages() if m["speaker"] == "human") - persons
            rows.append({"case": "echo", "agent_audio_continued": continued, "became_turn": new > 0})

        t_speak = await _speaking(p)
        if t_speak is not None:
            await silence(p.source, 0.8)
            barge = await kokoro(client, speech_url, BARGE)  # made first, so the clock starts with the voice
            t_b = time.monotonic()
            await send(p.source, barge)
            during = [x for x in p.ear.loud_times if x > t_b]
            stop = (during[-1] - t_b) if during else 0.0
            words = p.first_event(t_b, "transcript", lambda ev: len((ev.get("text") or "").split()) >= 2)
            paused = p.first_event(t_b, "agent_state", lambda ev: ev.get("state") != "speaking")
            rows.append({"case": "barge_in", "stop_s": round(stop, 3),
                         "two_words_s": round(words - t_b, 3) if words else None,
                         "agent_paused_s": round(paused - t_b, 3) if paused else None})
            await p.wait_reply(time.monotonic(), limit=20)
        await p.leave()
    return rows


def summarize(rows: list[dict]) -> dict:
    fin = [r["first_audio_s"] for r in rows if r["case"] in ("finished", "unfinished") and r.get("first_audio_s")]
    return {"first_audio_p50": percentile(fin, 50), "first_audio_p90": percentile(fin, 90),
            "early_replies": sum(1 for r in rows if r.get("early_reply")),
            "unfinished": sum(1 for r in rows if r["case"] == "unfinished"),
            "noise_false_turns": sum(1 for r in rows if r["case"] == "noise" and (r["new_turn"] or r["agent_audio"])),
            "noise_clips": sum(1 for r in rows if r["case"] == "noise"),
            "backchannel_ok": next((r["agent_kept_talking"] and not r["became_turn"] for r in rows
                                    if r["case"] == "backchannel"), None),
            "barge_in_stop_s": next((r["stop_s"] for r in rows if r["case"] == "barge_in"), None),
            "commit_p50": percentile([r["commit_s"] for r in rows if r.get("commit_s")], 50),
            "echo_became_turn": next((r["became_turn"] for r in rows if r["case"] == "echo"), None)}


def run(agent_url="http://127.0.0.1:8100", speech_url="http://127.0.0.1:8200", config: dict | None = None,
        progress=None) -> dict:
    rows = asyncio.run(_run(agent_url, speech_url, config, progress))
    result = {"suite": "turn", "run": report.run_id(), "config": config or {}, "rows": rows,
              "summary": summarize(rows)}
    report.save(result)
    return result
