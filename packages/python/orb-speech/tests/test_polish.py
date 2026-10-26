"""The guarded LLM polish: the guard's rules, and the Polisher against a fake Ollama (no model is loaded)."""
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from orb_speech import polish
from orb_speech.polish import Polisher, guard
from orb_speech.server import create_app


@pytest.mark.parametrize("before, after", [
    ("Book the meeting for Tuesday, no wait, Wednesday at 4 PM.", "Book the meeting for Wednesday at 4 PM."),
    ("Send it to Priya, sorry, I mean Sanjay, before lunch.", "Send it to Sanjay before lunch."),
    ("Groceries. One, milk. Two, eggs. Three, rice.", "Groceries:\n1. Milk.\n2. Eggs.\n3. Rice."),
    ("Run git checkout dash b feature slash login and push.", "Run git checkout -b feature/login and push."),
    ("Update the read me and the use effect hook.", "Update the README and the useEffect hook."),
    ("um so I think we should ship it today", "I think we should ship it today."),
])
def test_the_guard_lets_a_dictation_cleanup_through(before, after):
    assert guard(before, after, (), "medium") is None


@pytest.mark.parametrize("before, after, reason", [
    ("Send it to Priya before lunch today.", "Send it to Sanjay before lunch today.", "new word"),
    ("The budget is 10,000 rupees for now.", "The budget is 15,000 rupees for now.", "new word"),
    ("What time is the standup tomorrow?", "The standup is at 10 AM tomorrow.", "new word"),
    ("I do not want the release on Friday.", "I do want the release on Friday.", "negation"),
    ("Book the meeting for Tuesday, no wait Wednesday at 4 PM.", "Book the meeting for Tuesday at 4 PM.",
     "lost correction"),
    ("Send it to Priya sorry, I mean Sanjay before lunch.", "Send it to Priya before lunch.", "lost correction"),
    ("We can't ship the build before Monday.", "We can ship the build before Monday.", "negation"),
    ("Please check the logs and the alerts and the dashboards now.", "Check now.", "dropped"),
    ("Ask the team about the plan.", "", "empty"),
])
def test_the_guard_keeps_the_input_when_the_model_changes_the_meaning(before, after, reason):
    assert guard(before, after, (), "medium") == reason


def test_a_vocabulary_term_is_not_a_new_word():
    assert guard("Ask about the cash migration.", "Ask about the Redis migration.", (), "medium") == "new word"
    assert guard("Ask about the cash migration.", "Ask about the Redis migration.", ("Redis",), "medium") is None


def test_light_allows_fewer_changes_than_medium():
    before = "so we we should um maybe move the the call to next week ok"
    after = "So we should maybe move the call to next week, OK?"
    assert guard(before, after, (), "medium") is None
    rewrite = ("Move the release to Monday and tell the team about it.", "Tell the team: move the release to Monday.")
    assert guard(*rewrite, (), "light") == "changed"


def ollama(reply: str | None = None, status: int = 200, delay: Exception | None = None):
    """A fake Ollama chat endpoint, and the requests it got."""
    seen = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        if delay:
            raise delay
        return httpx.Response(status, json={"message": {"role": "assistant", "content": reply}})
    return httpx.Client(transport=httpx.MockTransport(handle)), seen


def polisher(client) -> Polisher:
    p = Polisher("fake-model")
    p.client = client
    return p


def test_polish_applies_a_clean_reply_and_sends_the_context_and_no_thinking():
    client, seen = ollama("Send it to Sanjay before lunch.")
    out = polisher(client).polish("Send it to Priya, sorry, I mean Sanjay, before lunch.", context="Who gets it?")
    assert (out.text, out.applied, out.reason) == ("Send it to Sanjay before lunch.", True, None)
    body = seen[0]
    assert body["think"] is False and body["options"]["temperature"] == 0 and body["model"] == "fake-model"
    assert "Who gets it?" in body["messages"][0]["content"]
    assert body["messages"][-1]["content"] == "Send it to Priya, sorry, I mean Sanjay, before lunch."


def test_polish_keeps_the_input_when_the_guard_rejects_or_ollama_fails():
    text = "What time is the standup tomorrow?"
    out = polisher(ollama("The standup is at 10 AM.")[0]).polish(text)
    assert (out.text, out.applied, out.reason) == (text, False, "new word")
    assert polisher(ollama(status=500)[0]).polish(text).reason == "failed"
    assert polisher(ollama(delay=httpx.ReadTimeout("slow"))[0]).polish(text).reason == "slow"


def test_polish_skips_verbatim_and_short_text_without_a_model_call():
    client, seen = ollama("x")
    p = polisher(client)
    assert p.polish("Sure, sounds good.").reason == "short"
    assert p.polish("Send it to Sanjay before lunch.", strength="verbatim").reason == "skipped"
    assert seen == []


def test_the_polish_endpoint_merges_the_vocabulary_and_checks_the_strength():
    client, seen = ollama("Ask about the Redis migration.")
    app = TestClient(create_app(vocabulary=("Datadog",), polisher=polisher(client)))
    r = app.post("/v1/polish", json={"text": "Ask about the cash migration."}, headers={"X-Orb-Vocabulary": "Redis"})
    assert r.status_code == 200
    assert r.json() | {"ms": 0} == {"text": "Ask about the Redis migration.", "applied": True, "reason": None, "ms": 0,
                                    "model": "fake-model"}
    assert app.post("/v1/polish", json={"text": "hi there you all", "strength": "max"}).status_code == 400
    assert TestClient(create_app()).post("/v1/polish", json={"text": "hello"}).status_code == 503
    assert polish.check_strength(None) == "medium"
