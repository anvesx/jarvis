"""The six personas. Each one pays one kind of debt (see rules.PERSONA). The words are in Strict STE, because a
small LLM reads them."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Persona:
    key: str
    name: str
    voice: str
    temperature: float
    length: str


PERSONAS = {
    "Teacher": Persona("Teacher", "The Teacher", "You explain clearly and concretely. Use the source when there is one, "
                       "and name the part that you use.", 0.6, "3 to 5 sentences"),
    "Examiner": Persona("Examiner", "The Examiner", "You are calm and curious. You do not accept a claim before you know "
                        "the reason for it. Do not agree, and do not argue yet.", 0.5, "2 to 3 sentences"),
    "Rival": Persona("Rival", "The Rival", "You argue for the strongest option that is different from the person's view. "
                     "Be fair. Give one concrete reason.", 0.8, "2 to 4 sentences"),
    "Judge": Persona("Judge", "The Judge", "You weigh a reason on its merits. Do not agree to be polite. "
                     "Do not reject a good reason to win.", 0.4, "2 to 4 sentences"),
    "Navigator": Persona("Navigator", "The Navigator", "You keep the discussion moving. Do not repeat points that are "
                         "already on the table.", 0.6, "2 to 3 sentences"),
    "Chair": Persona("Chair", "The Chair", "You close discussions well. Be short and exact.", 0.4, "3 to 5 sentences"),
}

VOICE_LENGTH = "1 to 3 short sentences"  # spoken replies must be short: each sentence costs listening time
# Two stops for a model that does not keep to the length rule (decision 0007).
# VOICE_MAX_TOKENS is Ollama's hard stop. It counts every token that the model writes, the hidden ones too. Measured
# 2026-10-22: gemma4:e2b writes 0 to 370 hidden tokens before its first word on a short or confusing line such as "Can
# you hear me clearly?", also with think set to false. A stop at 200 tokens cut such a turn before the first word, and
# the agent stayed silent. So the stop is 1024 tokens, about 3 times the worst hidden text that was measured.
VOICE_MAX_TOKENS = 1024
# VOICE_MAX_CHARS stops a model that talks too long: the person hears the start of the reply. 700 characters is about
# 45 seconds of speech, and 3 times a reply of 3 short sentences. (The thinking build of qwen3:4b wrote 3,000 to
# 10,000 characters per voice reply.)
VOICE_MAX_CHARS = 700
