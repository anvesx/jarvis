"""The bench API behind static/bench.html: speech engines (proxied), corpus clips, checks, suite runs and results.

Everything stays on 127.0.0.1. The speech service is proxied so the page needs no cross-origin calls. One suite runs
at a time (a run uses the same models as live calls). File paths are checked to stay inside their folders.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

log = logging.getLogger("orb.agent.bench")
CORPUS_GROUPS = ("stt/synthetic", "stt/dictation", "stt/edacc", "stt/librispeech", "noise", "echo", "llm", "tts")
SUITES = ("stt", "format", "tts", "llm", "turn")


class ClipCheck(BaseModel):
    group: str = Field(max_length=32)
    id: str = Field(max_length=120)
    engine: str | None = Field(default=None, max_length=32)


class RunIn(BaseModel):
    suite: str
    engines: list[str] | None = None
    groups: list[str] | None = None
    models: list[str] | None = None
    stages: list[str] | None = Field(default=None, max_length=8)  # format suite: "raw", "written", "vocab+written" ...
    limit: int | None = Field(default=None, ge=1, le=1000)
    judge: bool = True


def inside(base: Path, rel: str) -> Path:
    """`base / rel`, refused when it leaves `base` (for example with ..)."""
    path = (base / rel).resolve()
    if not path.is_relative_to(base.resolve()) or not path.is_file():
        raise HTTPException(404, "There is no such file.")
    return path


def make_router(speech_url: str, agent_url: str) -> APIRouter:
    router = APIRouter()
    jobs: dict[str, dict] = {}
    busy = threading.Lock()

    @router.get("/v1/speech/engines")
    async def speech_engines():
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{speech_url}/v1/engines")
        return Response(r.content, status_code=r.status_code, media_type="application/json")

    @router.post("/v1/speech/stt")
    async def speech_stt(request: Request):
        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(f"{speech_url}/v1/stt", params=dict(request.query_params), content=await request.body())
        return Response(r.content, status_code=r.status_code, media_type="application/json")

    @router.post("/v1/speech/tts")
    async def speech_tts(request: Request):
        body = await request.body()
        client = httpx.AsyncClient(timeout=httpx.Timeout(120, connect=5))
        req = client.build_request("POST", f"{speech_url}/v1/tts", content=body,
                                   headers={"content-type": "application/json"})
        r = await client.send(req, stream=True)
        if r.status_code != 200:
            content = await r.aread()
            await r.aclose()
            await client.aclose()
            return Response(content, status_code=r.status_code, media_type="application/json")

        async def relay():
            try:
                async for chunk in r.aiter_bytes():
                    yield chunk
            finally:
                await r.aclose()
                await client.aclose()

        headers = {k: v for k, v in r.headers.items() if k.lower() in ("x-sample-rate", "x-channels", "x-engine")}
        return StreamingResponse(relay(), media_type=r.headers.get("content-type", "audio/L16"), headers=headers)

    @router.get("/v1/bench/corpus/{group:path}")
    async def corpus_manifest(group: str):
        from orb_bench.corpus import manifest
        if group not in CORPUS_GROUPS:
            raise HTTPException(404, f"There is no corpus group {group!r}.")
        try:
            return manifest(group)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))

    @router.get("/v1/bench/clip/{group:path}/{name}")
    async def corpus_clip(group: str, name: str):
        from orb_bench.paths import corpus_dir
        if group not in CORPUS_GROUPS:
            raise HTTPException(404, f"There is no corpus group {group!r}.")
        return FileResponse(inside(corpus_dir(group), name), media_type="audio/wav")

    @router.post("/v1/bench/stt")
    async def check_clip(body: ClipCheck):
        """One corpus clip through one STT engine, scored against its reference."""
        from orb_bench import audio
        from orb_bench.corpus import manifest
        from orb_bench.metrics import score
        from orb_bench.paths import corpus_dir
        if body.group not in CORPUS_GROUPS:
            raise HTTPException(404, f"There is no corpus group {body.group!r}.")
        item = next((it for it in manifest(body.group) if it["id"] == body.id), None)
        if item is None:
            raise HTTPException(404, f"There is no clip {body.id!r} in {body.group}.")
        x = audio.read(inside(corpus_dir(body.group), item["path"]))
        params = {"sample_rate": audio.RATE, "language": "en", **({"engine": body.engine} if body.engine else {})}
        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(f"{speech_url}/v1/stt", params=params, content=audio.pcm16(x))
        if r.status_code != 200:
            raise HTTPException(r.status_code, r.json().get("error", "speech service error"))
        d = r.json()
        e, n = score(item["ref"], d["text"], item.get("lang", "en")) if item["ref"] else (0, 0)
        return {**d, "ref": item["ref"], "wer": e / n if n else None, "rtt_ms": round((time.perf_counter() - t0) * 1000, 1)}

    def run_job(job: dict, body: RunIn) -> None:
        from orb_bench.suites import format as fmt, llm, stt, tts, turn
        progress = lambda line: job["progress"].append(line)
        try:
            if body.suite == "format":
                res = fmt.run(None, body.stages or ["raw"], body.engines, body.groups, progress=progress)
            elif body.suite == "stt":
                res = stt.run(body.engines or ["whistle", "parakeet"], body.groups or list(stt.GROUPS), speech_url,
                              body.limit, progress=progress)
            elif body.suite == "tts":
                res = tts.run(speech_url=speech_url, judge=body.judge, progress=progress)
            elif body.suite == "llm":
                res = llm.run(body.models, agent_url, progress=progress)
            else:
                res = turn.run(agent_url, speech_url, progress=progress)
            job.update(status="done", run=res["run"], summary=res.get("summary"))
        except Exception as e:  # the page shows it; the service keeps running
            log.exception("bench run %s failed", job["id"])
            job.update(status="failed", error=f"{type(e).__name__}: {e}")
        finally:
            job["ended"] = time.time()
            busy.release()

    @router.post("/v1/bench/runs", status_code=202)
    async def start_run(body: RunIn):
        if body.suite not in SUITES:
            raise HTTPException(400, f"suite must be one of {', '.join(SUITES)}.")
        if not busy.acquire(blocking=False):
            raise HTTPException(409, "A bench run is still going. Wait for it to end.")
        job = {"id": uuid.uuid4().hex[:10], "suite": body.suite, "status": "running", "progress": [],
               "started": time.time()}
        jobs[job["id"]] = job
        threading.Thread(target=run_job, args=(job, body), name=f"bench-{job['id']}", daemon=True).start()
        return job

    @router.get("/v1/bench/runs/{job_id}")
    async def run_status(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, f"There is no bench run {job_id}.")
        return {**job, "progress": job["progress"][-20:]}

    @router.get("/v1/bench/results/{suite}")
    async def results(suite: str):
        from orb_bench import report
        if suite not in SUITES:
            raise HTTPException(404, f"There is no suite {suite!r}.")
        return {"suite": suite, "runs": report.runs(suite)}

    @router.get("/v1/bench/results/{suite}/{run}")
    async def result(suite: str, run: str):
        from orb_bench import report
        if suite not in SUITES:
            raise HTTPException(404, f"There is no suite {suite!r}.")
        try:
            res = await asyncio.to_thread(report.load, suite, run)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        return res

    @router.get("/v1/bench/audio/{run}/{name}")
    async def tts_audio(run: str, name: str):
        from orb_bench.paths import results_dir
        return FileResponse(inside(results_dir("tts"), f"{run}/{name}"), media_type="audio/wav")

    return router
