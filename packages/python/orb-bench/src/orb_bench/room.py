"""A synthetic person in a LiveKit room: it publishes a microphone track, speaks in real time and hears the agent.

Shared by scripts/e2e_voice.py and the turn suite.
"""
from __future__ import annotations

import asyncio
import json
import time

import httpx
import numpy as np
import soxr
from livekit import rtc

RATE = 48000
FRAME = RATE // 50  # 20 ms
BENCH_TOPIC = "orb.bench"  # the worker's events: states, transcripts, metrics (orb_voice.main.Bench)


class Ear:
    """Listens to the agent's audio track and records when it is loud."""

    def __init__(self, loud: float = 0.01):
        self.loud_times: list[float] = []
        self.frames, self.loud = 0, loud

    async def listen(self, track):
        async for ev in rtc.AudioStream(track, sample_rate=RATE, num_channels=1):
            data = np.frombuffer(ev.frame.data, dtype=np.int16).astype(np.float32) / 32768.0
            self.frames += 1
            if len(data) and float(np.sqrt(np.mean(data ** 2))) > self.loud:
                self.loud_times.append(time.monotonic())

    def first_loud_after(self, t: float, until: float | None = None) -> float | None:
        return next((x for x in self.loud_times if x > t and (until is None or x <= until)), None)

    def quiet_since(self, t: float, quiet: float = 1.2) -> bool:
        """True when the agent spoke after t and has been quiet for `quiet` seconds."""
        after = [x for x in self.loud_times if x > t]
        return bool(after) and time.monotonic() - after[-1] > quiet


async def send(source: rtc.AudioSource, samples: np.ndarray) -> None:
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


async def silence(source: rtc.AudioSource, seconds: float) -> None:
    await send(source, np.zeros(int(seconds * RATE), dtype=np.float32))


async def kokoro(client: httpx.AsyncClient, speech_url: str, text: str, voice: str = "am_michael") -> np.ndarray:
    """Kokoro speech for `text`, at 48 kHz float32 (the person's voice; the agent speaks with af_heart)."""
    r = await client.post(f"{speech_url}/v1/tts", json={"text": text, "voice": voice})
    r.raise_for_status()
    pcm = np.frombuffer(r.content, dtype="<i2").astype(np.float32) / 32768.0
    return soxr.resample(pcm, int(r.headers["x-sample-rate"]), RATE).astype(np.float32)


def at_48k(x16: np.ndarray) -> np.ndarray:
    return soxr.resample(x16, 16000, RATE).astype(np.float32)


class Person:
    """Joins the room of a new session, publishes a microphone and listens to the agent."""

    def __init__(self, client: httpx.AsyncClient, agent_url: str, speech_url: str = "http://127.0.0.1:8200"):
        self.client, self.agent_url, self.speech_url = client, agent_url, speech_url
        self.room, self.ear, self.source, self.session_id = rtc.Room(), Ear(), None, None
        self.events: list[tuple[float, dict]] = []  # (monotonic time of the event at the worker, the event)

    async def join(self, topic: str, config: dict | None = None, wait: float = 15.0) -> float:
        """Returns the seconds until the agent joined; raises when it did not join in `wait` seconds."""
        body = {"topic": topic, **({"config": config} if config else {})}
        s = (await self.client.post(f"{self.agent_url}/v1/sessions", json=body)).json()
        self.session_id = s["id"]
        t = (await self.client.post(f"{self.agent_url}/v1/voice/token", json={"session_id": s["id"]})).json()

        @self.room.on("track_subscribed")
        def _on_track(track, publication, participant):
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                asyncio.ensure_future(self.ear.listen(track))

        @self.room.on("data_received")
        def _on_data(packet):
            if packet.topic != BENCH_TOPIC:
                return
            try:
                ev = json.loads(packet.data)
            except ValueError:
                return
            # the worker runs on this machine: its wall-clock time maps onto this monotonic clock
            self.events.append((ev.get("t", time.time()) - (time.time() - time.monotonic()), ev))

        await self.room.connect(t["url"], t["token"])
        self.source = rtc.AudioSource(RATE, 1)
        mic = rtc.LocalAudioTrack.create_audio_track("mic", self.source)
        await self.room.local_participant.publish_track(
            mic, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
        t0 = time.monotonic()
        while not self.room.remote_participants and time.monotonic() - t0 < wait:
            await silence(self.source, 0.2)
        if not self.room.remote_participants:
            raise RuntimeError(f"the voice worker did not join the room in {wait:.0f} s")
        return time.monotonic() - t0

    def first_event(self, after: float, kind: str, test=lambda ev: True) -> float | None:
        """The time of the first worker event of `kind` after `after` that passes `test`, or None."""
        return next((t for t, ev in self.events if t > after and ev.get("type") == kind and test(ev)), None)

    async def messages(self) -> list[dict]:
        return (await self.client.get(f"{self.agent_url}/v1/sessions/{self.session_id}")).json()["messages"]

    async def wait_reply(self, after: float, limit: float = 30.0, quiet: float = 1.2) -> float | None:
        """Sends silence until the agent spoke after `after` and went quiet. Returns its first audio time or None."""
        deadline = time.monotonic() + limit
        while time.monotonic() < deadline:
            await silence(self.source, 0.1)
            if self.ear.quiet_since(after, quiet):
                break
        return self.ear.first_loud_after(after)

    async def leave(self) -> None:
        await self.room.disconnect()
