"""Test doubles for the reader (Jev) and the LLM. No test here calls a network service."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
PASSWORDS_RUN = ROOT / "data" / "debt_runs" / "20261012-152327.json"


class _Answer:
    def __init__(self, p):
        self.noul = p


class _Usage:
    def __init__(self, n):
        self.input_tokens = n


class _Response:
    def __init__(self, answers, tokens):
        self.answers = answers
        self.usage = _Usage(tokens)
        self.model = "scripted"


class ScriptedReader:
    """Answers each call from a list of {question name: probability}. A question that the script does not name
    gets 0.0 ("no"). An Exception in the script is raised for that call."""

    def __init__(self, script, delay=0.0):
        self.script = list(script)
        self.calls = []
        self.delay = delay

    def system_one(self, state, questions):
        if self.delay:
            import time
            time.sleep(self.delay)  # the agent calls the reader in a worker thread
        item = self.script.pop(0) if self.script else {}
        self.calls.append({"state": state, "questions": list(questions)})
        if isinstance(item, Exception):
            raise item
        return _Response({name: _Answer(item.get(name, 0.0)) for name in questions}, 1000)


class FakeLLM:
    """Streams a fixed reply, word by word. Records every prompt it gets."""

    def __init__(self, replies, fail=None):
        self.replies = list(replies)
        self.prompts = []
        self.fail = fail
        self.warmed = 0

    async def stream(self, messages, options=None):
        self.prompts.append({"messages": messages, "options": options})
        if self.fail:
            raise self.fail
        text = self.replies.pop(0) if self.replies else "Fine."
        for i, word in enumerate(text.split(" ")):
            yield word if i == 0 else " " + word

    async def warm(self, messages):
        self.warmed += 1

    async def health(self):
        return {"ok": True}


@pytest.fixture
def passwords_run():
    return json.loads(PASSWORDS_RUN.read_text())
