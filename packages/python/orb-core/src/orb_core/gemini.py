"""Gemini Live: the facts that the agent service and the voice worker share (docs/design/gemini-live.md).

In the "gemini-live" voice mode one Google model hears, thinks and speaks; the orb pipeline (speech service, orb
agent, Ollama) is not used for the call. The lists below are the ones of livekit-plugins-google 1.8.9
(`realtime.api_proto`); the voice worker's tests check that the plugin still has them.
"""
from __future__ import annotations

import os

MODES = ("orb", "gemini-live")
MODELS = ("gemini-3.8-live", "gemini-3.8-live-extended-thinking")
DEFAULT_MODEL = MODELS[0]
VOICES = ("Achernar", "Achird", "Algenib", "Algieba", "Alnilam", "Aoede", "Autonoe", "Callirrhoe", "Charon", "Despina",
          "Enceladus", "Erinome", "Fenrir", "Gacrux", "Iapetus", "Kore", "Laomedeia", "Leda", "Orus", "Pulcherrima",
          "Puck", "Rasalgethi", "Sadachbia", "Sadaltager", "Schedar", "Sulafat", "Umbriel", "Vindemiatrix", "Zephyr",
          "Zubenelgenubi")
DEFAULT_VOICE = "Kore"  # a female voice (Google: "firm"); the owner asked for a female default
KEY = "GOOGLE_API_KEY"

# Added to orb's system prompt: Gemini speaks every word it writes, and no per-turn note comes (no personas).
VOICE_RULES = """
You are speaking, not writing: answer in one to three short sentences, then let the person talk.
Language: answer in the language of the person's latest sentence, not the first one. When they speak English, answer
in English; when they mix Hindi or Gujarati with English, mix them the same way, in Latin letters. When they ask for a
language ("speak in English"), use it for the rest of the call until they ask for another.
This is a voice call only: there is no chat, no typing, no files and no links. Never offer them. When the person wants
to share a document, say that they can read out the part they want to discuss.
When you did not hear the person clearly, ask them to say it again."""


def available() -> bool:
    """Whether a Gemini key is set. The value is never read out of this module."""
    return bool(os.environ.get(KEY, "").strip())


def check(mode: str | None, model: str | None, voice: str | None) -> str | None:
    """Why a session config cannot use these values, or None. Empty values mean the defaults."""
    if mode and mode not in MODES:
        return f"voice_mode must be one of {', '.join(MODES)}, not {mode!r}."
    if model and model not in MODELS:
        return f"gemini_model must be one of {', '.join(MODELS)}, not {model!r}."
    if voice and voice not in VOICES:
        return f"gemini_voice {voice!r} is not a Gemini voice. Voices: {', '.join(VOICES)}."
    return None
