"""The orb agent service: one HTTP API for every channel.

    python -m orb_agent.server            # http://127.0.0.1:8100, or ORB_AGENT_PORT

POST /v1/sessions                 {"topic", "agent"?}       -> the session state (201)
GET  /v1/sessions                                           -> [{id, topic, agent}]
GET  /v1/sessions/{id}[?since=v]                            -> the state, or {"version": v, "unchanged": true}
PUT  /v1/sessions/{id}            {"topic", "agent"?}       -> the state; creates the session when it is missing
POST /v1/sessions/{id}/turns      {"text", "channel"?, "replace_last"?}
                                  -> NDJSON events: read, decision, token..., reply, check, state
POST /v1/voice/token              {"session_id"}            -> {"url", "token", "room", "identity"}
GET  /health
GET  /                                                      -> the web chat

It listens on 127.0.0.1 only. It has no user accounts yet, so it must not be reachable from a network.
"""
from __future__ import annotations

import asyncio
import logging
import os
import secrets
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from orb_core.agent import CHANNELS, Event, InputError, SessionStore, check_session_id
from orb_agent.voice import VoiceConfig, room_for

log = logging.getLogger("orb.agent")
STATIC = Path(__file__).parent / "static"


class NewSession(BaseModel):
    topic: str = Field(min_length=1, max_length=2000)
    agent: str | None = None


class TurnIn(BaseModel):
    text: str = Field(min_length=1, max_length=8000)
    channel: str = "web"
    replace_last: bool = False  # voice: the last turn was never heard, and `text` says that sentence further


class VoiceIn(BaseModel):
    session_id: str


def create_app(store: SessionStore, voice: VoiceConfig | None = None) -> FastAPI:
    app = FastAPI(title="orb agent", docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    background: set[asyncio.Task] = set()

    def later(coro):
        task = asyncio.create_task(coro)
        background.add(task)
        task.add_done_callback(background.discard)

    def session_or_404(sid: str):
        try:
            check_session_id(sid)
        except InputError as e:
            raise HTTPException(400, str(e))
        s = store.get(sid)
        if s is None:
            raise HTTPException(404, f"There is no session {sid}. Start one with POST /v1/sessions.")
        return s

    def started(s):
        if hasattr(s, "prepare"):
            later(s.prepare())
        return s

    async def save_after(s):
        await s.settle()
        try:
            store.save(s)
        except OSError as e:
            log.warning("could not save session %s: %s", s.id, e)

    @app.get("/")
    async def page():
        return FileResponse(STATIC / "index.html")

    @app.get("/health")
    async def health():
        agents = {}
        for name, agent in store.agents.items():
            try:
                agents[name] = await agent.health()
            except Exception as e:
                agents[name] = {"error": f"{type(e).__name__}: {e}"}
        return {"ok": True, "agents": agents, "sessions": len(store.list()),
                "voice": {"url": voice.url, "dev": voice.dev} if voice else None}

    @app.post("/v1/sessions", status_code=201)
    async def create(body: NewSession):
        try:
            s = started(store.create(body.topic, body.agent))
        except InputError as e:
            raise HTTPException(400, str(e))
        return s.state()

    @app.get("/v1/sessions")
    async def listing():
        return store.list()

    @app.get("/v1/sessions/{sid}")
    async def read(sid: str, since: int | None = None):
        s = session_or_404(sid)
        state = s.state()
        if since is not None and state.get("version") == since:
            return {"version": since, "unchanged": True}
        return state

    @app.put("/v1/sessions/{sid}")
    async def ensure(sid: str, body: NewSession):
        try:
            s = store.get(check_session_id(sid))
            if s is None:
                s = started(store.create(body.topic, body.agent, sid))
        except InputError as e:
            raise HTTPException(400, str(e))
        return s.state()

    @app.post("/v1/sessions/{sid}/turns")
    async def turn(sid: str, body: TurnIn):
        s = session_or_404(sid)
        if body.channel not in CHANNELS:
            raise HTTPException(400, f"channel must be one of {', '.join(CHANNELS)}.")

        async def stream():
            try:
                async for ev in s.turn(body.text, body.channel, replace_last=body.replace_last):
                    yield ev.to_json() + "\n"
            except InputError as e:
                yield Event("error", str(e)).to_json() + "\n"
            finally:
                later(save_after(s))

        return StreamingResponse(stream(), media_type="application/x-ndjson")

    @app.post("/v1/voice/token")
    async def voice_token(body: VoiceIn):
        if voice is None:
            raise HTTPException(503, "Voice is off: set LIVEKIT_URL, LIVEKIT_API_KEY and LIVEKIT_API_SECRET.")
        s = session_or_404(body.session_id)
        identity = "user-" + secrets.token_hex(4)
        room = room_for(s.id)
        return {"url": voice.url, "room": room, "identity": identity, "token": voice.token(room, identity)}

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

    return app


def build() -> FastAPI:
    from orb_core import config
    from orb_core.jev import make_reader
    from orb_core.ollama import Ollama
    from orb_discussion.agent import DiscussionAgent

    config.bootstrap()
    root = config.repo_root()
    llm = Ollama(base_url=os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"),
                 model=os.environ.get("ORB_LLM_MODEL", "gemma4:e2b"))
    store = SessionStore({"discussion": DiscussionAgent(make_reader(), llm)}, save_dir=root / "data" / "sessions")
    return create_app(store, VoiceConfig.from_env())


def main():
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    uvicorn.run(build(), host="127.0.0.1", port=int(os.environ.get("ORB_AGENT_PORT", "8100")), log_level="warning")


if __name__ == "__main__":
    main()
