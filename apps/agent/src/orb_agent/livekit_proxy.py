"""A forwarder for the LiveKit signal channel, so that one address (a tunnel to this service) carries the page and the voice signal.

The LiveKit server listens on 127.0.0.1, which a device that is not this computer cannot reach. The browser client opens
`wss://<host>/rtc/v1?...` and, when that fails, asks `GET /rtc/v1/validate` for the reason. This module forwards both to the local
server, so a device that opened the page through a tunnel needs no second address.

It carries signal messages only (small). The audio does not pass here: it goes straight between the device and the LiveKit server,
so a device needs a network path to this computer for it (decision 0009). The routes sit behind the access gate (access.py), and
LiveKit still checks the token in every request, so this is not an open proxy: the target is fixed by LIVEKIT_URL.
"""
from __future__ import annotations

import asyncio
import logging

import anyio
import httpx
import websockets
from fastapi import APIRouter, Request, WebSocket
from fastapi.responses import Response

log = logging.getLogger("orb.agent")
VALIDATE = ("validate", "v1/validate")  # the only HTTP paths that the browser client uses
NOT_A_CODE = (None, 1005, 1006, 1015)   # close codes that a peer must not send


def make_router(server_url: str) -> APIRouter:
    """Routes that forward /rtc to the LiveKit server at `server_url` (ws://, wss://, http:// or https://)."""
    http_base = ("http" + server_url[2:] if server_url.startswith("ws") else server_url).rstrip("/")
    ws_base = "ws" + http_base[4:]  # http -> ws, https -> wss
    router = APIRouter()

    async def forward(client: WebSocket, path: str) -> None:
        query = client.url.query
        target = f"{ws_base}/{path}" + (f"?{query}" if query else "")  # the query holds the token: never log it
        try:
            upstream = await websockets.connect(target, max_size=None, ping_interval=None, open_timeout=5)
        except (OSError, TimeoutError, websockets.WebSocketException) as e:
            log.warning("LiveKit signal: %s cannot be reached (%s)", ws_base, type(e).__name__)
            await client.close(code=1011)  # before accept(): the browser sees a refused handshake, then asks /validate
            return
        await client.accept()

        async def to_client() -> None:
            async for message in upstream:
                if isinstance(message, bytes):
                    await client.send_bytes(message)
                else:
                    await client.send_text(message)

        async def to_upstream() -> None:
            while True:
                message = await client.receive()
                if message["type"] == "websocket.disconnect":
                    return
                if message.get("bytes") is not None:
                    await upstream.send(message["bytes"])
                elif message.get("text") is not None:
                    await upstream.send(message["text"])

        tasks = [asyncio.create_task(to_client()), asyncio.create_task(to_upstream())]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in tasks:
                task.cancel()
            with anyio.move_on_after(3, shield=True):  # the clean-up runs also when the request itself is cancelled
                await asyncio.gather(*tasks, return_exceptions=True)
                code = upstream.close_code
                await upstream.close()
                try:
                    await client.close(code=1000 if code in NOT_A_CODE else code)
                except RuntimeError:  # the browser closed first
                    pass

    @router.websocket("/rtc")
    async def signal(client: WebSocket):
        await forward(client, "rtc")

    @router.websocket("/rtc/{rest:path}")
    async def signal_path(client: WebSocket, rest: str):
        await forward(client, f"rtc/{rest}")

    @router.get("/rtc/{rest:path}")
    async def validate(rest: str, request: Request):
        """The client asks this after a failed connection, to show the reason (LiveKit's own text) to the person."""
        if rest not in VALIDATE:
            return Response("Not found", status_code=404, media_type="text/plain")
        query = request.url.query
        try:
            async with httpx.AsyncClient(timeout=5) as http:
                r = await http.get(f"{http_base}/rtc/{rest}" + (f"?{query}" if query else ""))
        except httpx.HTTPError:
            return Response("The voice server does not answer.", status_code=502, media_type="text/plain")
        return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "text/plain"))

    return router
