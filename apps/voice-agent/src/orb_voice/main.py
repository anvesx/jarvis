"""The orb voice worker on LiveKit Agents.

    python -m orb_voice dev --no-reload --log-level info

It joins every new room on the LiveKit server (automatic dispatch). For each room:
Silero VAD (LiveKit plugin) -> LiveSTT (the speech service: interim words while the person speaks, and a preflight
transcript at each pause) -> LiveKit's turn detector v1-mini (local, on the CPU) -> the orb agent service (the reply)
-> FirstClauseTokenizer -> OrbTTS (Kokoro). The design is in LLD.md.
The VAD and the turn detector are fixed. STT, LLM and TTS are switchable (the bench). Every stage reports timing to
the log and to the room's data channel (topic "orb.bench"), with no words in the log unless ORB_LOG_TEXT=1.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path

import httpx
from livekit import agents, rtc
from livekit.agents import Agent, AgentServer, AgentSession, APIConnectOptions, JobContext, JobProcess, inference
from livekit.agents import tts as lk_tts
from livekit.agents.voice.agent_session import SessionConnectOptions
from livekit.plugins import silero

from orb_voice.config import Settings, check_livekit, load_repo_env, session_for
from orb_voice.plugins import FirstClauseTokenizer, LiveSTT, OrbLLM, OrbSTT, OrbTTS, TurnTracker, stt_engines_info

log = logging.getLogger("orb.voice")
BENCH_TOPIC = "orb.bench"


def setup(proc: JobProcess) -> None:
    """Once per worker process: the settings and the Silero model."""
    load_repo_env()
    s = Settings.from_env()
    proc.userdata["settings"] = s
    proc.userdata["vad"] = silero.VAD.load(min_speech_duration=s.vad_min_speech, min_silence_duration=s.vad_min_silence,
                                           prefix_padding_duration=s.vad_prefix, activation_threshold=s.vad_threshold)


def make_server() -> AgentServer:
    load_repo_env()
    check_livekit()
    # the health server binds 127.0.0.1 only (the default is every interface); one process waits warm for the next job
    return AgentServer(setup_fnc=setup, host="127.0.0.1", port=int(os.environ.get("ORB_VOICE_PORT", "8081")),
                       num_idle_processes=1, initialize_process_timeout=60.0)


server = make_server()


class Bench:
    """Sends small JSON events to the room's data channel, for the bench page. A failed send is only logged."""

    def __init__(self, room: rtc.Room):
        self._room, self._tasks = room, set()

    def send(self, kind: str, **data) -> None:
        if not self._room.isconnected():
            return
        payload = json.dumps({"type": kind, "t": round(time.time(), 3), **data}, default=str)
        task = asyncio.create_task(self._room.local_participant.publish_data(payload, reliable=True, topic=BENCH_TOPIC))
        self._tasks.add(task)
        task.add_done_callback(self._done)

    def _done(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            log.debug("bench event not sent: %s", task.exception())


async def ensure_session(http: httpx.AsyncClient, agent_url: str, sid: str) -> dict:
    """Opens the session when it is missing. Returns its config: the models that the bench chose, or {}."""
    r = await http.put(f"{agent_url}/v1/sessions/{sid}", json={"topic": "Open discussion"}, timeout=30.0)
    if r.status_code != 200:
        raise RuntimeError(f"the agent service did not open session {sid}: HTTP {r.status_code}")
    return r.json().get("config") or {}


def agent_event_logger(sid: str, s: Settings, bench: Bench):
    def on_event(ev: dict) -> None:
        kind, data = ev.get("type"), ev.get("data") or {}
        if kind == "decision":
            log.info("agent decision", extra={"session": sid, "persona": data.get("persona"), "debt": data.get("debt"),
                                              "stage": data.get("stage")})
            bench.send("agent_decision", persona=data.get("persona"), debt=data.get("debt"), stage=data.get("stage"))
        elif kind == "reply":
            timing = data.get("timing") or {}
            log.info("agent reply", extra={"session": sid, "chars": len(data.get("text") or ""), **timing})
            if not data.get("text"):  # the person hears nothing for this turn
                log.warning("agent reply is empty: the LLM wrote no words", extra={"session": sid, **timing})
            bench.send("agent_reply", chars=len(data.get("text") or ""), timing=timing,
                       **({"text": data.get("text")} if s.log_text else {}))
        elif kind == "read":
            bench.send("agent_read", seconds=data.get("seconds"), error=bool(data.get("error")))
        elif kind == "check":
            bench.send("agent_check", debt=data.get("debt"), done=data.get("done"))
    return on_event


@server.rtc_session()
async def entry(ctx: JobContext) -> None:
    s: Settings = ctx.proc.userdata["settings"]
    vad = ctx.proc.userdata["vad"]
    room = ctx.job.room.name
    sid = session_for(room)
    t0 = time.perf_counter()
    http = httpx.AsyncClient(limits=httpx.Limits(max_keepalive_connections=8))
    ctx.add_shutdown_callback(http.aclose)
    cfg = await ensure_session(http, s.agent_url, sid)
    stt_engine = cfg.get("stt_engine") or s.stt_engine
    cloud_engines, default_stt = await stt_engines_info(http, s.speech_url)
    stt_name = stt_engine or default_stt or "default"
    is_cloud = stt_name in cloud_engines
    # a cloud STT bills each call for the whole piece that it hears: fewer interim calls, and no silent warm-up call
    interim_every = max(s.interim_every, s.cloud_interim_every) if is_cloud else s.interim_every
    if is_cloud:
        log.warning("the speech-to-text engine is a cloud service: the audio of this call goes to it",
                    extra={"session": sid, "engine": stt_name})
    tts_opts = {"engine": cfg.get("tts_engine", ""), "voice": cfg.get("tts_voice") or s.tts_voice,
                "style": cfg.get("tts_style", ""), "lang": cfg.get("tts_lang", "")}
    log.info("agent session ready", extra={"session": sid, "ms": round((time.perf_counter() - t0) * 1000),
                                           "stt": stt_name, "tts": tts_opts["engine"] or "default",
                                           "llm": cfg.get("llm_model") or "default"})

    bench = Bench(ctx.room)
    tracker = TurnTracker()
    debug_dir = str(Path(s.debug_audio_dir) / sid) if s.debug_audio_dir else ""
    if debug_dir:
        log.warning("debug audio is on: every transcribed piece is saved", extra={"dir": debug_dir})
    batch = OrbSTT(http, s.speech_url, engine=stt_engine, language=s.stt_language, debug_dir=debug_dir,
                   vocabulary=s.stt_vocabulary)
    session = AgentSession(
        stt=LiveSTT(stt=batch, vad=vad, interim_every=interim_every, preflight_after=s.preflight_after),
        vad=vad,
        llm=OrbLLM(http, s.agent_url, sid, tracker=tracker, on_event=agent_event_logger(sid, s, bench)),
        tts=lk_tts.StreamAdapter(tts=OrbTTS(http, s.speech_url, **tts_opts),
                                 sentence_tokenizer=FirstClauseTokenizer(first_clause=s.first_clause,
                                                                         first_max=s.first_max)),
        turn_handling={
            # local and fixed: in dev mode the default is LiveKit's cloud model
            "turn_detection": inference.TurnDetector(version="v1-mini"),
            "endpointing": {"min_delay": s.min_delay, "max_delay": s.max_delay},
            # "vad", not "adaptive": the adaptive barge-in model is a billed cloud service
            "interruption": {"mode": "vad", "min_duration": s.min_interruption, "min_words": s.min_interruption_words,
                             "false_interruption_timeout": 2.0, "resume_false_interruption": True},
            "preemptive_generation": {"enabled": s.preemptive, "preemptive_tts": s.preemptive_tts},
        },
        conn_options=SessionConnectOptions(llm_conn_options=APIConnectOptions(max_retry=0, timeout=30.0)),
    )

    @session.on("agent_state_changed")
    def _agent_state(ev) -> None:
        if ev.new_state == "speaking":
            tracker.played()
        bench.send("agent_state", state=ev.new_state)

    @session.on("user_state_changed")
    def _user_state(ev) -> None:
        bench.send("user_state", state=ev.new_state)

    @session.on("user_input_transcribed")
    def _transcribed(ev) -> None:
        if ev.is_final:
            log.info("user words", extra={"session": sid, "chars": len(ev.transcript),
                                          **({"text": ev.transcript} if s.log_text else {})})
        bench.send("transcript", final=ev.is_final, text=ev.transcript)

    @session.on("metrics_collected")
    def _metrics(ev) -> None:
        m = ev.metrics.model_dump(mode="json", exclude_none=True)
        if m.get("type") not in ("vad_metrics",):
            log.info("metrics", extra={"session": sid, **{k: v for k, v in m.items() if not isinstance(v, (dict, list))}})
        bench.send("metrics", metrics=m)

    @session.on("conversation_item_added")
    def _item(ev) -> None:
        item = ev.item
        metrics = getattr(item, "metrics", None) or {}
        if getattr(item, "role", None) == "assistant" and metrics.get("e2e_latency") is not None:
            log.info("turn latency", extra={"session": sid, "e2e_s": round(metrics["e2e_latency"], 3),
                                            "interrupted": getattr(item, "interrupted", False)})
            bench.send("turn_latency", e2e=metrics["e2e_latency"], interrupted=getattr(item, "interrupted", False))

    @session.on("agent_false_interruption")
    def _false(ev) -> None:
        log.info("false interruption: the agent resumes", extra={"session": sid})
        bench.send("false_interruption")

    @session.on("error")
    def _error(ev) -> None:
        log.error("session error", extra={"session": sid, "error": str(getattr(ev, "error", ev))})
        bench.send("error", error=str(getattr(ev, "error", ev)))

    if stt_name == "parakeet":
        asyncio.create_task(batch.warm())  # Parakeet runs on the GPU: open the keep-warm window of the speech service
    await session.start(room=ctx.room, agent=Agent(instructions="The orb agent service writes every reply."))
    bench.send("session", session=sid, stt=stt_name + (" (cloud)" if is_cloud else ""), llm=cfg.get("llm_model") or "default",
               tts=tts_opts["engine"] or "default", voice=tts_opts["voice"] or "default", vad="silero",
               turn="livekit-v1-mini")


def main() -> None:
    agents.cli.run_app(server)


if __name__ == "__main__":
    main()
