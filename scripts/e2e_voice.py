"""End-to-end voice test: a synthetic person joins a room, speaks, and listens to the agent.

    scripts/dev.sh                                   # in another terminal: every service runs
    ~/miniconda3/envs/orb/bin/python scripts/e2e_voice.py

What it does:
1. Creates a session in the agent service and gets a LiveKit token for its room.
2. Joins the room, publishes a microphone track, and speaks sentences made with Kokoro (through the speech service).
3. Measures, for each turn: the time from the end of the person's speech to the first audio of the agent, and how
   long the agent speaks. Then it reads the session to compare the transcript and the persona.
4. A turn with pauses in a noisy room (three clauses, 450 ms pauses, pink noise at 20 dB SNR) must become one message.
5. A short "Sure." while the agent speaks must not stop the agent and must not become a turn.
6. Last: it starts to speak while the agent speaks, and checks that the agent stops (barge-in).

It prints one line per turn and exits with 1 when a check fails.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time

import httpx
import numpy as np
from livekit import rtc

AGENT = "http://127.0.0.1:8100"
SPEECH = "http://127.0.0.1:8200"
RATE = 48000
FRAME = RATE // 50  # 20 ms
TURNS = [
    ("We should store user passwords in a plain JSON file so the assistant can log in for them.", "Rival"),
    ("No, I don't agree with you.", "Examiner"),
]


def words(text):
    return re.sub(r"[^a-z ]", "", text.lower()).split()


def word_error(ref, hyp):
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(1, len(r))


def pink(n, rms, seed=3):
    w = np.random.default_rng(seed).standard_normal(n)
    f = np.fft.rfft(w)
    f /= np.maximum(1, np.sqrt(np.arange(len(f))))
    w = np.fft.irfft(f, n)
    return (w / (np.sqrt(np.mean(w ** 2)) + 1e-9) * rms).astype(np.float32)


async def speak_audio(client: httpx.AsyncClient, text: str) -> np.ndarray:
    """Kokoro speech for `text`, at 48 kHz float32."""
    r = await client.post(f"{SPEECH}/v1/tts", json={"text": text, "voice": "am_michael"})
    r.raise_for_status()
    pcm = np.frombuffer(r.content, dtype="<i2").astype(np.float32) / 32768.0
    import soxr
    return soxr.resample(pcm, int(r.headers["x-sample-rate"]), RATE).astype(np.float32)


class Ear:
    """Listens to the agent's audio track and records when it is loud."""

    def __init__(self):
        self.loud_times: list[float] = []
        self.frames = 0

    async def listen(self, track):
        async for ev in rtc.AudioStream(track, sample_rate=RATE, num_channels=1):
            data = np.frombuffer(ev.frame.data, dtype=np.int16).astype(np.float32) / 32768.0
            self.frames += 1
            if len(data) and float(np.sqrt(np.mean(data ** 2))) > 0.01:
                self.loud_times.append(time.monotonic())

    def first_loud_after(self, t):
        return next((x for x in self.loud_times if x > t), None)

    def quiet_since(self, t, quiet=1.2):
        """True when the agent spoke after t and has been quiet for `quiet` seconds."""
        after = [x for x in self.loud_times if x > t]
        return bool(after) and time.monotonic() - after[-1] > quiet


async def send(source: rtc.AudioSource, samples: np.ndarray):
    """Send audio in real time, 20 ms per frame."""
    pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    t0 = time.monotonic()
    for i in range(0, len(pcm), FRAME):
        chunk = pcm[i:i + FRAME]
        if len(chunk) < FRAME:
            chunk = np.pad(chunk, (0, FRAME - len(chunk)))
        await source.capture_frame(rtc.AudioFrame(chunk.tobytes(), RATE, 1, FRAME))
        delay = t0 + (i + FRAME) / RATE - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)


async def silence(source, seconds):
    await send(source, np.zeros(int(seconds * RATE), dtype=np.float32))


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-barge-in", action="store_true")
    args = ap.parse_args()
    ok = True
    async with httpx.AsyncClient(timeout=60) as client:
        s = (await client.post(f"{AGENT}/v1/sessions", json={"topic": "Should our personal assistant store user passwords?"})).json()
        t = (await client.post(f"{AGENT}/v1/voice/token", json={"session_id": s["id"]})).json()
        print(f"session {s['id']}, room {t['room']}")
        room, ear = rtc.Room(), Ear()

        @room.on("track_subscribed")
        def _on_track(track, publication, participant):
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                asyncio.ensure_future(ear.listen(track))

        await room.connect(t["url"], t["token"])
        source = rtc.AudioSource(RATE, 1)
        mic = rtc.LocalAudioTrack.create_audio_track("mic", source)
        await room.local_participant.publish_track(mic, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
        t_join = time.monotonic()
        while not room.remote_participants and time.monotonic() - t_join < 15:
            await silence(source, 0.2)
        if not room.remote_participants:
            print("FAIL: the voice worker did not join the room in 15 s. Is apps/voice running?")
            return 1
        print(f"agent joined after {time.monotonic() - t_join:.1f} s")
        await silence(source, 1.0)

        for i, (sentence, persona) in enumerate(TURNS):
            audio = await speak_audio(client, sentence)
            await send(source, audio)
            t_end = time.monotonic()
            t_first, deadline = None, t_end + 30
            while time.monotonic() < deadline:
                await silence(source, 0.1)
                t_first = t_first or ear.first_loud_after(t_end)
                if t_first and ear.quiet_since(t_end):
                    break
            state = (await client.get(f"{AGENT}/v1/sessions/{s['id']}")).json()
            turn = state["turns"][-1] if len(state["turns"]) > i else None
            heard = next((m["text"] for m in reversed(state["messages"]) if m["speaker"] == "human"), "")
            got = turn["decision"]["persona"] if turn and turn.get("decision") else None
            spoke = (ear.loud_times[-1] - t_first) if t_first else 0
            same = words(heard) == words(sentence)
            line = (f"turn {i + 1}: heard {'the same words' if same else repr(heard)} | persona {got} (want {persona}) | "
                    f"first agent audio {t_first - t_end:.2f} s after the person stopped | agent spoke {spoke:.1f} s"
                    if t_first else f"turn {i + 1}: FAIL no agent audio in 30 s (heard {heard!r})")
            if turn and turn.get("timing"):
                tm = turn["timing"]
                line += f" | Jev {tm.get('read_human')} s, LLM first word {tm.get('llm_first_token')} s"
            print(line)
            ok &= bool(t_first) and same and got == persona

        # 4. a turn with pauses in a noisy room must stay one message
        clauses = ["I think we should use Go for the backend", "because the team already knows it",
                   "and it handles concurrency well"]
        pieces = []
        for c in clauses:
            pieces += [await speak_audio(client, c), np.zeros(int(RATE * 0.45), np.float32)]
        speech = np.concatenate(pieces)
        level = float(np.sqrt(np.mean(speech[np.abs(speech) > 0.01] ** 2)))
        noisy = np.concatenate([np.zeros(RATE, np.float32), speech, np.zeros(RATE, np.float32)])
        noisy = noisy + pink(len(noisy), level / 10 ** (20 / 20))  # 20 dB SNR, also in the pauses and around the turn
        before = (await client.get(f"{AGENT}/v1/sessions/{s['id']}")).json()
        await send(source, noisy)
        t_end = time.monotonic()
        while time.monotonic() - t_end < 30:
            await silence(source, 0.1)
            if ear.first_loud_after(t_end) and ear.quiet_since(t_end):
                break
        after = (await client.get(f"{AGENT}/v1/sessions/{s['id']}")).json()
        new_human = [m for m in after["messages"][len(before["messages"]):] if m["speaker"] == "human"]
        heard = new_human[-1]["text"] if new_human else ""
        err = word_error(" ".join(clauses), heard)
        # a merged turn undoes the unheard first part, so one message means one turn
        print(f"noisy turn with pauses: {len(new_human)} new person message(s), word error {err:.0%} | heard {heard!r}")
        ok &= len(new_human) == 1 and err <= 0.15

        # 5. a short "Sure." while the agent speaks (a backchannel): the agent goes on, and it is not a turn
        audio = await speak_audio(client, "What would you pick for the database?")
        await send(source, audio)
        t_end = time.monotonic()
        while not ear.first_loud_after(t_end) and time.monotonic() - t_end < 30:
            await silence(source, 0.1)
        await silence(source, 0.8)
        n_before = len((await client.get(f"{AGENT}/v1/sessions/{s['id']}")).json()["messages"])
        okay = await speak_audio(client, "Sure.")
        t_ok = time.monotonic()
        await send(source, okay)
        await silence(source, 1.5)
        kept_talking = any(x > t_ok + len(okay) / RATE + 0.3 for x in ear.loud_times)
        while time.monotonic() - t_ok < 30 and not ear.quiet_since(t_ok):
            await silence(source, 0.1)
        msgs = (await client.get(f"{AGENT}/v1/sessions/{s['id']}")).json()["messages"]
        okay_turn = any(m["speaker"] == "human" and "sure" in m["text"].lower() for m in msgs[n_before:])
        print(f"short 'Sure.' ({len(okay) / RATE:.2f} s) while the agent speaks: the agent "
              f"{'kept talking' if kept_talking else 'STOPPED'}; it {'became a turn' if okay_turn else 'was not a turn'}")
        ok &= kept_talking and not okay_turn

        if not args.no_barge_in:
            audio = await speak_audio(client, "Encryption needs a key, and that key has to live on the same machine as the file.")
            await send(source, audio)
            t_end = time.monotonic()
            while not ear.first_loud_after(t_end) and time.monotonic() - t_end < 30:
                await silence(source, 0.1)
            await silence(source, 0.6)  # let the agent talk for a moment
            barge = await speak_audio(client, "Wait, stop, let me add one more thing.")
            t_cut = time.monotonic()
            await send(source, barge)
            t_said = time.monotonic()
            during = [x for x in ear.loud_times if t_cut < x < t_said]
            stop_after = (during[-1] - t_cut) if during else 0.0
            # the guard needs about 550 ms of speech; then the audio already in the room drains
            stopped = stop_after <= 1.5 and not [x for x in ear.loud_times if t_cut + 1.5 < x < t_said]
            await silence(source, 6)
            state = (await client.get(f"{AGENT}/v1/sessions/{s['id']}")).json()
            cut = [m for m in state["messages"] if m["speaker"] == "ai" and m["interrupted"]]
            print(f"barge-in: the agent's last audio came {stop_after:.2f} s after the person started "
                  f"({'stopped' if stopped else 'did NOT stop'} within 1.5 s); {len(cut)} AI message(s) marked as cut off")
            ok &= stopped

        await room.disconnect()
        print("PASS" if ok else "FAIL")
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
