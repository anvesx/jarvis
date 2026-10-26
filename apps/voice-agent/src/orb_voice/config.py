"""Settings of the voice worker, and the room-to-session rule.

Settings come from the environment. The repo .env fills in only ORB_* and LIVEKIT_* keys that are not set, so other
secrets in that file (for example a Cloudflare token) never enter the worker process.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from orb_core.config import load_env_prefixes

DEV_KEY, DEV_SECRET = "devkey", "secret"  # the public credentials of `livekit-server --dev`
ROOM_PREFIX = "orb-"
SESSION_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,47}$")  # the same rule as orb_core.agent.check_session_id
ENV_PREFIXES = ("ORB_", "LIVEKIT_")


def session_for(room: str) -> str:
    """Rooms that the web page makes are orb-<session id>. Any other room gets a stable session of its own."""
    if room.startswith(ROOM_PREFIX) and SESSION_ID.match(room[len(ROOM_PREFIX):]):
        return room[len(ROOM_PREFIX):]
    return "room-" + hashlib.sha256(room.encode()).hexdigest()[:12]


def check_livekit(env=os.environ) -> None:
    """The same rule as the agent service (orb_agent/voice.py): the public dev credentials, or a secret of fewer than
    32 characters, only with a LiveKit server on this machine. Raises ValueError otherwise."""
    url = env.get("LIVEKIT_URL", "")
    if urlparse(url).hostname in ("127.0.0.1", "localhost", "::1"):
        return
    key, secret = env.get("LIVEKIT_API_KEY", ""), env.get("LIVEKIT_API_SECRET", "")
    if key == DEV_KEY or secret == DEV_SECRET or len(secret) < 32:
        raise ValueError(f"LIVEKIT_URL {url!r} is not on this machine: it needs its own API key and a secret of "
                         "32 or more characters, not the dev credentials.")


def load_repo_env(root: Path | None = None) -> list[str]:
    """Copy ORB_* and LIVEKIT_* keys from the repo .env into os.environ when they are not set. Returns the names."""
    return load_env_prefixes(ENV_PREFIXES, root)


def _num(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else default


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    return default if not raw else raw in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    agent_url: str = "http://127.0.0.1:8100"
    speech_url: str = "http://127.0.0.1:8200"
    stt_engine: str = ""            # empty: the speech service's default (ORB_STT_ENGINE there); the final text
    stt_interim_engine: str = "whistle"  # the interim words (decision 0012)
    stt_language: str = "en"
    stt_vocabulary: str = ""        # comma-separated terms for the speech service (header X-Orb-Vocabulary)
    tts_voice: str = ""             # empty: the speech service's default voice
    interim_every: float = 0.3      # seconds of speech between interim transcripts
    cloud_interim_every: float = 1.0  # the same for a cloud STT: each interim call pays for the whole piece again
    vad_threshold: float = 0.5      # LiveKit's Silero defaults, kept fixed; the bench measures them
    vad_min_speech: float = 0.05
    vad_min_silence: float = 0.55
    vad_prefix: float = 0.5
    min_delay: float = 0.3          # LiveKit's endpointing defaults with the audio turn detector
    max_delay: float = 2.5
    min_interruption: float = 0.5   # seconds of speech before the agent pauses
    min_interruption_words: int = 2  # fewer words never cut the agent: "Sure.", "Okay.", "Mm-hmm."
    preemptive: bool = True
    preemptive_tts: bool = True     # the early reply also gets its audio early (Kokoro is local and cheap)
    preflight_after: float = 0.15   # seconds of silence before a preflight transcript; 0 is off (plugins.LiveSTT)
    first_clause: int = 30          # the first TTS piece may end at a comma after this many characters; 0 is off
    first_max: int = 60             # with no comma before this many characters, it ends at a word boundary; 0 is off
    log_text: bool = False
    debug_audio_dir: str = ""

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            agent_url=os.environ.get("ORB_AGENT_URL", cls.agent_url).rstrip("/"),
            speech_url=os.environ.get("ORB_SPEECH_URL", cls.speech_url).rstrip("/"),
            stt_engine=os.environ.get("ORB_VOICE_STT_ENGINE", "").strip(),
            stt_interim_engine=os.environ.get("ORB_VOICE_STT_INTERIM_ENGINE", cls.stt_interim_engine).strip(),
            stt_language=os.environ.get("ORB_STT_LANGUAGE", cls.stt_language).strip(),
            stt_vocabulary=os.environ.get("ORB_STT_VOCABULARY", "").strip(),
            tts_voice=os.environ.get("ORB_TTS_VOICE", "").strip(),
            interim_every=_num("ORB_INTERIM_EVERY_S", cls.interim_every),
            cloud_interim_every=_num("ORB_CLOUD_INTERIM_EVERY_S", cls.cloud_interim_every),
            vad_threshold=_num("ORB_VAD_THRESHOLD", cls.vad_threshold),
            vad_min_speech=_num("ORB_VAD_MIN_SPEECH_S", cls.vad_min_speech),
            vad_min_silence=_num("ORB_VAD_MIN_SILENCE_S", cls.vad_min_silence),
            vad_prefix=_num("ORB_VAD_PREFIX_S", cls.vad_prefix),
            min_delay=_num("ORB_MIN_ENDPOINTING_S", cls.min_delay),
            max_delay=_num("ORB_MAX_ENDPOINTING_S", cls.max_delay),
            min_interruption=_num("ORB_MIN_INTERRUPTION_S", cls.min_interruption),
            min_interruption_words=int(_num("ORB_MIN_INTERRUPTION_WORDS", cls.min_interruption_words)),
            preemptive=_flag("ORB_PREEMPTIVE", cls.preemptive),
            preemptive_tts=_flag("ORB_PREEMPTIVE_TTS", cls.preemptive_tts),
            preflight_after=_num("ORB_PREFLIGHT_S", cls.preflight_after),
            first_clause=int(_num("ORB_FIRST_CLAUSE_CHARS", cls.first_clause)),
            first_max=int(_num("ORB_FIRST_PIECE_MAX_CHARS", cls.first_max)),
            log_text=_flag("ORB_LOG_TEXT", False),
            debug_audio_dir=os.environ.get("ORB_DEBUG_AUDIO_DIR", "").strip(),
        )
