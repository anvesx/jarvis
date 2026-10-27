"""Gemini Live: one Google model hears, thinks and speaks (docs/design/gemini-live.md, ticket GL-2).

A session whose config says `voice_mode: gemini-live` gets `AgentSession(llm=RealtimeModel(...))` and nothing else:
no speech service, no orb agent turns, no Ollama. Gemini detects the turns on its server and sends both transcripts,
which LiveKit turns into the page's captions. The instructions are orb's system prompt for the session's topic, from
the agent service. The audio of the call goes to Google: the worker logs that at the start of every such call.
"""
from __future__ import annotations

import logging

import httpx
from google.genai import types
from livekit.plugins import google

from orb_core import gemini

log = logging.getLogger("orb.voice")

# Hints for the caption of the person's words (gemini-live.md section 9): without them a short sound was captioned in
# Italian or German. Other languages are still detected; the model itself hears the audio, not the caption.
CAPTION_LANGUAGES = ["en-IN", "hi-IN", "gu-IN"]


async def fetch_instructions(http: httpx.AsyncClient, agent_url: str, sid: str) -> str:
    """The session's system prompt for Gemini (agent service). Raises httpx.HTTPError when the service fails."""
    r = await http.get(f"{agent_url}/v1/sessions/{sid}/instructions", timeout=30.0)  # it may load the topic's source
    r.raise_for_status()
    return r.json()["instructions"]


def options(cfg: dict) -> dict:
    """The RealtimeModel arguments for a session config. Unknown values fall back to the defaults (the agent service
    already refused them, so this happens only with an older session file)."""
    model = cfg.get("gemini_model") if cfg.get("gemini_model") in gemini.MODELS else gemini.DEFAULT_MODEL
    voice = cfg.get("gemini_voice") if cfg.get("gemini_voice") in gemini.VOICES else gemini.DEFAULT_VOICE
    extra = {}
    if model.endswith("-extended-thinking"):  # it refuses a session without a level (1007); the plain model refuses one
        extra["thinking_config"] = types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW)
    return {
        **extra,
        "model": model,
        "voice": voice,
        # the page shows what each side said: Gemini sends both transcripts
        "input_audio_transcription": types.AudioTranscriptionConfig(language_codes=CAPTION_LANGUAGES),
        "output_audio_transcription": types.AudioTranscriptionConfig(),
        # one connection lasts about 10 minutes and an audio session 15: a sliding window and resumption go past both
        "context_window_compression": types.ContextWindowCompressionConfig(sliding_window=types.SlidingWindow()),
        "session_resumption": types.SessionResumptionConfig(),
        # turns (gemini-live.md section 9): at the default sensitivity a short "Mhm" cut Gemini off mid-sentence
        "realtime_input_config": types.RealtimeInputConfig(automatic_activity_detection=types.AutomaticActivityDetection(
            start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_LOW,
            prefix_padding_ms=200, silence_duration_ms=700)),
    }


def make_model(cfg: dict) -> google.realtime.RealtimeModel:
    """The model of a Gemini Live call. It reads GOOGLE_API_KEY from the environment; nothing connects until the
    session starts."""
    return google.realtime.RealtimeModel(**options(cfg))


def explain(error: object) -> str:
    """A message for the person from a Gemini error (a RealtimeModelError, an exception, or its chain). It never holds
    the key: Google's messages name a key as invalid, they do not repeat it."""
    parts, seen = [], set()
    e = error
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        parts.append(f"{type(e).__name__} {e}")
        e = getattr(e, "error", None) or getattr(e, "__cause__", None) or getattr(e, "__context__", None)
    text = " ".join(parts).lower()
    if "api key is required" in text:  # the plugin's ValueError: the worker's environment has no key
        return (f"Gemini Live has no API key in the voice worker. Set {gemini.KEY} in .env, then restart "
                "scripts/dev.sh.")
    if any(s in text for s in ("api key not valid", "api_key_invalid", "permission_denied", "unauthenticated",
                               " 401", " 403", "invalid api key")):
        return (f"Gemini Live refused the API key. Check {gemini.KEY} in .env (a key of aistudio.google.com), then "
                "restart scripts/dev.sh.")
    if any(s in text for s in ("resource_exhausted", "quota", " 429", "rate limit")):
        return "Gemini Live is over its quota or rate limit for this key. Wait a minute, or check the key's billing."
    if any(s in text for s in ("not found", "not supported", "1008", "policy violation")):
        return ("Gemini Live does not accept this model for this key. Choose gemini-3.8-live, or check that the key "
                "is a Gemini API key.")
    if "thinking level" in text:
        return "Gemini Live refused the model's settings (thinking level). Choose gemini-3.8-live and call again."
    if "1007" in text or "context exhausted" in text:
        return "The Gemini Live conversation got too long. Press New session and call again."
    if any(s in text for s in ("connecterror", "gaierror", "timeout", "connection refused", "network",
                               "name or service not known", "nodename nor servname")):
        return "Gemini Live cannot be reached: check the internet connection, then call again."
    return "Gemini Live stopped with an error. The voice worker log has the details; call again."
