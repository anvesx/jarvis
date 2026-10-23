"""The discussion agent with a scripted reader and a fake LLM."""
import asyncio
import re

from conftest import FakeLLM, ScriptedReader

from orb_core.agent import Agent, Session
from orb_core.ollama import OllamaError
from orb_discussion.agent import DiscussionAgent


def no_source(url):
    raise AssertionError("no URL in these topics, so no page may be loaded")


def make(script, replies, fail=None):
    llm = FakeLLM(replies, fail=fail)
    agent = DiscussionAgent(ScriptedReader(script), llm, load_source=no_source)
    return agent, agent.new_session("Should our assistant store passwords?", "test-1"), llm


async def run_turn(session, text, channel="web"):
    return [e async for e in session.turn(text, channel)]


def test_agent_and_session_follow_the_interface():
    agent, session, _ = make([], [])
    assert isinstance(agent, Agent) and isinstance(session, Session)


def test_one_full_turn():
    script = [{"claims": 1, "new_option": 1},              # the person: a claim, the first option
              {"new_option": 1, "disagrees": 1}]           # the AI: a second option
    agent, s, llm = make(script, ["Hashing is safer, because a leaked file shows nothing."])
    events = asyncio.run(run_turn(s, "We should store passwords in a JSON file."))
    types = [e.type for e in events]
    assert types[:2] == ["read", "decision"] and types[-3:] == ["reply", "check", "state"]
    assert types.count("token") == 9
    decision = events[1].data
    assert decision["persona"] == "Rival" and decision["kind"] == "option"
    assert events[-2].data["done"] is True  # the second option paid the option debt
    note = llm.prompts[0]["messages"][-1]["content"]
    assert note.startswith("This note is for your next reply only.")
    assert not re.search(r"\b[DMO]\d+\b", note)
    assert llm.prompts[0]["options"] == {"temperature": 0.8}
    assert [m["speaker"] for m in s.messages] == ["human", "ai"]
    assert len(s.engine.ledger.messages) == 2
    assert s.state()["stage"] == "Test" and llm.warmed == 1


def test_interrupted_reply_is_kept_and_read():
    script = [{"claims": 1, "new_option": 1}, {"new_option": 1}]
    agent, s, llm = make(script, ["Hashing is safer for many reasons that take a while to say."])

    async def go():
        gen = s.turn("We should store passwords in a JSON file.", "voice")
        async for e in gen:
            if e.type == "token":
                break  # the person barges in after the first word
        await gen.aclose()
        await s.settle()

    asyncio.run(go())
    assert [m["speaker"] for m in s.messages] == ["human", "ai"]
    assert s.messages[1]["interrupted"] is True and s.messages[1]["text"] == "Hashing"
    assert len(s.engine.ledger.messages) == 2  # the ledger has the heard part of the reply
    assert "Length: 1 to 3 short sentences." in llm.prompts[0]["messages"][-1]["content"]


def test_reader_failure_on_the_person_changes_nothing():
    agent, s, llm = make([RuntimeError("Jev is down")], ["unused"])
    events = asyncio.run(run_turn(s, "Hello there."))
    assert [e.type for e in events] == ["error"] and "nothing changed" in events[0].data
    assert s.messages == [] and s.engine.ledger.messages == [] and llm.prompts == []


def test_reader_failure_on_the_reply_keeps_the_ledger_in_step():
    script = [{"claims": 1, "new_option": 1}, RuntimeError("Jev timed out"), {"claims": 1}]
    agent, s, llm = make(script, ["Hashing is safer.", "Fine."])
    events = asyncio.run(run_turn(s, "We should store passwords in a JSON file."))
    check = next(e for e in events if e.type == "check").data
    assert check["done"] is None and "unknown" in check["text"]
    assert len(s.engine.ledger.messages) == 2
    asyncio.run(run_turn(s, "Still JSON."))  # the next message gets seq 2 in both lists
    assert s.messages[2]["seq"] == 2 and s.engine.ledger.messages[2]["seq"] == 2


def test_llm_failure_ends_the_turn_with_an_error():
    agent, s, llm = make([{"claims": 1}], [], fail=OllamaError("model not found"))
    events = asyncio.run(run_turn(s, "We should store passwords in a JSON file."))
    assert [e.type for e in events] == ["read", "decision", "error"]
    assert "model not found" in events[-1].data
    assert [m["speaker"] for m in s.messages] == ["human"]


def test_snapshot_is_json_safe():
    import json

    script = [{"claims": 1, "new_option": 1}, {"new_option": 1}]
    agent, s, llm = make(script, ["Hashing is safer."])
    asyncio.run(run_turn(s, "We should store passwords in a JSON file."))
    data = json.loads(json.dumps(s.snapshot(), default=str))
    assert data["trace"][0]["id"] == "M1" and data["turns"][0]["decision"]["persona"] == "Rival"


def test_continues():
    from orb_discussion.agent import continues

    assert continues("We should store passwords in a JSON file.", "We should store passwords in a JSON file so it is easy.")
    assert continues("What is sharding?", "What is sharding?")  # a resumed turn: the same words
    # the whole turn transcribed again: the first words differ a little
    assert continues("We should store user passwords in a plain JSON file.",
                     "We should store the user passwords in plain JSON files so the assistant can log in.")
    assert not continues("We should store passwords in a JSON file.", "Which language should we use for the backend?")
    assert not continues("No.", "Yes, and more.")
    assert not continues("", "anything")


def test_cancel_during_the_read_still_records_the_message():
    llm = FakeLLM(["unused"])
    reader = ScriptedReader([{"claims": 1, "new_option": 1}], delay=0.2)
    s = DiscussionAgent(reader, llm, load_source=no_source).new_session("Passwords?", "test-2")

    async def go():
        async def consume():
            async for _ in s.turn("We should store passwords in a JSON file."):
                pass
        task = asyncio.create_task(consume())
        await asyncio.sleep(0.05)  # Jev is still reading
        task.cancel()               # the channel goes away (a barge-in)
        try:
            await task
        except asyncio.CancelledError:
            pass
        await s.settle()

    asyncio.run(go())
    assert [m["speaker"] for m in s.messages] == ["human"]
    assert len(s.engine.ledger.messages) == 1 and len(s.turns) == 1
    assert llm.prompts == []  # the LLM never ran


def test_replace_last_undoes_an_unanswered_turn():
    script = [{"claims": 1, "new_option": 1},                   # the first part
              {"claims": 1, "gives_reason": 1, "new_option": 1},  # the whole sentence
              {"new_option": 1}]                                  # the reply
    llm = FakeLLM(["Hashing is safer."])
    s = DiscussionAgent(ScriptedReader(script, delay=0.05), llm, load_source=no_source).new_session("Passwords?", "test-3")
    first = "We should store user passwords in a plain JSON file."
    whole = first + " So the assistant can log in for them."

    async def go():
        async def consume():
            async for _ in s.turn(first, "voice"):
                pass
        task = asyncio.create_task(consume())
        await asyncio.sleep(0.01)
        task.cancel()  # the person goes on before any reply
        try:
            await task
        except asyncio.CancelledError:
            pass
        return [e async for e in s.turn(whole, "voice", replace_last=True)]

    events = asyncio.run(go())
    assert [m["text"] for m in s.messages] == [whole, "Hashing is safer."]
    assert [m["text"] for m in s.engine.ledger.messages] == [whole, "Hashing is safer."]
    assert [e["text"] for e in s.engine.trace] == [whole, "Hashing is safer."]
    assert s.turns[0]["replaced"] == first and len(s.turns) == 1
    assert next(e for e in events if e.type == "decision").data["kind"] == "option"
    assert s.engine.tokens == 3000  # the undone read was still paid for


def test_a_voice_reply_has_a_token_limit_that_leaves_room_for_hidden_text():
    from orb_discussion.personas import VOICE_MAX_TOKENS

    agent, s, llm = make([{"claims": 1, "new_option": 1}, {"new_option": 1}], ["Hashing is safer."])
    asyncio.run(run_turn(s, "We should store passwords in a JSON file.", channel="voice"))
    assert llm.prompts[0]["options"] == {"temperature": 0.8, "num_predict": VOICE_MAX_TOKENS}
    # Measured 2026-10-22: gemma4:e2b writes up to 370 hidden tokens before its first word on a line such as "Can you
    # hear me clearly?". Ollama's limit counts them. A limit of 200 left such a turn with no word at all.
    assert VOICE_MAX_TOKENS > 2 * 370


class HiddenThinkerLLM(FakeLLM):
    """Like gemma4:e2b on a short line: it writes `hidden` tokens that never reach the text, then its reply (one token
    for each word). `num_predict` counts the hidden tokens too, as Ollama's limit does."""

    def __init__(self, hidden, reply):
        super().__init__([])
        self.hidden, self.reply = hidden, reply

    async def stream(self, messages, options=None):
        self.prompts.append({"messages": messages, "options": options})
        budget = (options or {}).get("num_predict", 10 ** 9) - self.hidden
        for i, word in enumerate(self.reply.split(" ")[:max(budget, 0)]):
            yield word if i == 0 else " " + word


def hidden_thinker_turn(hidden, channel="voice"):
    llm = HiddenThinkerLLM(hidden, "I can hear you clearly. What do you want to talk about first?")
    agent = DiscussionAgent(ScriptedReader([{"claims": 1, "new_option": 1}, {"new_option": 1}]), llm, load_source=no_source)
    s = agent.new_session("Should our assistant store passwords?", "test-hidden")
    return s, asyncio.run(run_turn(s, "Can you hear me clearly? I don't know you are listening or not.", channel))


def test_a_voice_turn_still_speaks_after_300_hidden_tokens():
    s, events = hidden_thinker_turn(300)
    reply = next(e for e in events if e.type == "reply").data["text"]
    assert reply.startswith("I can hear you clearly.") and [m["speaker"] for m in s.messages] == ["human", "ai"]


def test_an_empty_reply_is_logged_and_never_added_to_the_chat(monkeypatch, caplog):
    import logging

    monkeypatch.setattr("orb_discussion.agent.VOICE_MAX_TOKENS", 200)  # the old limit: the hidden text ate all of it
    with caplog.at_level(logging.WARNING, logger="orb.discussion"):
        s, events = hidden_thinker_turn(300)
    assert next(e for e in events if e.type == "reply").data["text"] == ""
    assert "wrote no words" in caplog.text and [m["speaker"] for m in s.messages] == ["human"]
    assert not any(e.type == "check" for e in events)  # nothing to check, and no empty message in the chat


def test_a_long_voice_reply_is_cut_at_the_character_limit_but_a_web_reply_is_not():
    from orb_discussion.personas import VOICE_MAX_CHARS

    long_reply = " ".join(["word"] * 600)  # 2,999 characters
    script = [{"claims": 1, "new_option": 1}, {"new_option": 1}]
    for channel, cut in (("voice", True), ("web", False)):
        llm = FakeLLM([long_reply])
        s = DiscussionAgent(ScriptedReader(script), llm, load_source=no_source).new_session("Passwords?", "test-long")
        events = asyncio.run(run_turn(s, "We should store passwords in a JSON file.", channel))
        reply = next(e for e in events if e.type == "reply").data["text"]
        if cut:
            assert VOICE_MAX_CHARS <= len(reply) < VOICE_MAX_CHARS + 20 and s.messages[-1]["interrupted"] is True
        else:
            assert reply == long_reply and not s.messages[-1].get("interrupted")


def test_replace_last_ignores_a_new_sentence():
    script = [{"claims": 1, "new_option": 1}, {"new_option": 1}, {"claims": 1}, {}]
    s = DiscussionAgent(ScriptedReader(script), FakeLLM(["One.", "Two."]), load_source=no_source).new_session("P?", "test-4")
    asyncio.run(run_turn(s, "We should store passwords in a JSON file."))
    asyncio.run(_turn_replace(s, "Something different entirely."))
    assert [m["speaker"] for m in s.messages] == ["human", "ai", "human", "ai"]


async def _turn_replace(s, text):
    return [e async for e in s.turn(text, "voice", replace_last=True)]


def test_a_copied_note_never_reaches_the_person():
    leaked = ("Persona: The Teacher. You explain clearly.\nYour job now: Answer the person's question.\n"
              "Could you tell me what you mean by that word?")
    agent, s, llm = make([{"asks": 1}], [leaked])
    events = asyncio.run(run_turn(s, "What kind of decomia?", "voice"))
    spoken = "".join(e.data for e in events if e.type == "token")
    assert spoken == "Could you tell me what you mean by that word?"
    assert s.messages[-1]["text"] == spoken  # the chat history is clean, so the leak cannot feed itself
    assert len(s.turns[0]["dropped_lines"]) == 2
