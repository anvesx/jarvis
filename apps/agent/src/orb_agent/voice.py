"""LiveKit room tokens for the browser. The agent service only signs tokens for rooms of sessions that exist.

With no LIVEKIT_API_KEY and a loopback LIVEKIT_URL, it uses the public credentials of `livekit-server --dev`
(devkey / secret). Those are for this machine only. Any other server needs its own key and a secret of 32 or
more characters.
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import timedelta
from urllib.parse import urlparse

import httpx
from livekit import api as lkapi

ROOM_PREFIX = "orb-"
DEV_KEY, DEV_SECRET = "devkey", "secret"
LOOPBACK = ("127.0.0.1", "localhost", "::1")


def room_for(session_id: str) -> str:
    return ROOM_PREFIX + session_id


def is_local(url: str) -> bool:
    """True when the address is on this machine. A browser on another device cannot reach such an address."""
    return urlparse(url).hostname in LOOPBACK


def host_port(url: str) -> tuple[str, int]:
    u = urlparse(url)
    return u.hostname or "", u.port or (443 if u.scheme in ("wss", "https") else 80)


async def listening(host: str, port: int, timeout: float = 0.4) -> bool:
    """True when something accepts a TCP connection there. It says that a server runs, not that it works."""
    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout)
    except (OSError, asyncio.TimeoutError):
        return False
    writer.close()
    return True


async def livekit_answers(url: str, timeout: float = 1.5) -> bool:
    """True when the LiveKit server answers at this address: its health page (GET /) says OK. A tunnel with no server
    behind it answers with a page of its own, so a TCP connection would say yes when the answer is no."""
    http = "http" + url[2:] if url.startswith("ws") else url  # ws -> http, wss -> https
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(http, headers={"ngrok-skip-browser-warning": "1"})
    except httpx.HTTPError:
        return False
    return r.status_code == 200 and r.text.strip() == "OK"


@dataclass(frozen=True)
class VoiceConfig:
    url: str          # what the browser connects to
    api_key: str
    api_secret: str
    dev: bool
    server_url: str = ""  # where this service and the voice worker reach the LiveKit server (LIVEKIT_URL); empty: `url`

    @property
    def local(self) -> bool:
        return is_local(self.url)

    @property
    def server(self) -> str:
        return self.server_url or self.url

    @classmethod
    def from_env(cls) -> "VoiceConfig | None":
        server = os.environ.get("LIVEKIT_URL") or "ws://127.0.0.1:7880"
        url = os.environ.get("LIVEKIT_PUBLIC_URL") or server
        key, secret = os.environ.get("LIVEKIT_API_KEY"), os.environ.get("LIVEKIT_API_SECRET")
        loopback = is_local(url)
        if key and secret:
            if not loopback and len(secret) < 32:
                raise ValueError("LIVEKIT_API_SECRET must have 32 or more characters for a server that is not local.")
            return cls(url, key, secret, dev=False, server_url=server)
        if loopback:
            return cls(url, DEV_KEY, DEV_SECRET, dev=True, server_url=server)
        return None  # no credentials for a remote server: voice is off

    def token(self, room: str, identity: str, name: str = "You", ttl_minutes: int = 60) -> str:
        grants = lkapi.VideoGrants(room_join=True, room=room, can_publish=True, can_subscribe=True, can_publish_data=True)
        return (lkapi.AccessToken(self.api_key, self.api_secret).with_identity(identity).with_name(name)
                .with_grants(grants).with_ttl(timedelta(minutes=ttl_minutes)).to_jwt())
