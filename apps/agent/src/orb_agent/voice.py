"""LiveKit room tokens for the browser. The agent service only signs tokens for rooms of sessions that exist.

With no LIVEKIT_API_KEY and a loopback LIVEKIT_URL, it uses the public credentials of `livekit-server --dev`
(devkey / secret). Those are for this machine only. Any other server needs its own key and a secret of 32 or
more characters.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from urllib.parse import urlparse

from livekit import api as lkapi

ROOM_PREFIX = "orb-"
DEV_KEY, DEV_SECRET = "devkey", "secret"


def room_for(session_id: str) -> str:
    return ROOM_PREFIX + session_id


@dataclass(frozen=True)
class VoiceConfig:
    url: str          # what the browser connects to
    api_key: str
    api_secret: str
    dev: bool

    @classmethod
    def from_env(cls) -> "VoiceConfig | None":
        url = os.environ.get("LIVEKIT_PUBLIC_URL") or os.environ.get("LIVEKIT_URL") or "ws://127.0.0.1:7880"
        key, secret = os.environ.get("LIVEKIT_API_KEY"), os.environ.get("LIVEKIT_API_SECRET")
        loopback = urlparse(url).hostname in ("127.0.0.1", "localhost", "::1")
        if key and secret:
            if not loopback and len(secret) < 32:
                raise ValueError("LIVEKIT_API_SECRET must have 32 or more characters for a server that is not local.")
            return cls(url, key, secret, dev=False)
        if loopback:
            return cls(url, DEV_KEY, DEV_SECRET, dev=True)
        return None  # no credentials for a remote server: voice is off

    def token(self, room: str, identity: str, name: str = "You", ttl_minutes: int = 60) -> str:
        grants = lkapi.VideoGrants(room_join=True, room=room, can_publish=True, can_subscribe=True, can_publish_data=True)
        return (lkapi.AccessToken(self.api_key, self.api_secret).with_identity(identity).with_name(name)
                .with_grants(grants).with_ttl(timedelta(minutes=ttl_minutes)).to_jwt())
