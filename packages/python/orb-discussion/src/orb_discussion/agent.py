"""The discussion agent: Jev reads, the debt rules decide, the LLM speaks.

One turn:
1. Jev reads the person's message (one request, yes/no questions). The rules update the debts.
2. The rules choose the need: the most urgent due debt. Its kind picks the persona.
3. The LLM writes the reply with the persona note. Tokens stream to the channel.
4. Jev reads the reply in the background. The rules check whether the reply paid the need.
   The next turn waits for step 4, so the ledger always has every message in order.

Steps 1 and 2 run as one task that a channel cannot cut: when the channel goes away during the read (a voice
barge-in, a closed browser tab), the message and its turn are still recorded, so the ledger and the chat never
disagree. A voice channel can then send the whole sentence again with replace_last=True: the agent undoes the
unanswered turn and reads the whole sentence as one message.

The agent works the same for every channel. Only the reply length changes for voice.
"""
from __future__ import annotations

import asyncio
import copy
import re
import time
from difflib import SequenceMatcher

from orb_core import web
from orb_core.agent import Event, check_text
from orb_core.jev import cost_usd
from orb_core.ollama import OllamaError
from orb_discussion import prompt
from orb_discussion.engine import DebtEngine
from orb_discussion.personas import PERSONAS
from orb_discussion.rules import mid


def summarize(entry: dict) -> dict:
    """The part of an engine entry that a channel shows: the Jev call, the events and the stage."""
    call = entry.get("call") or {}
    return {
        "id": entry["id"], "speaker": entry["speaker"], "stage": entry["stage"], "stage_why": entry["stage_why"],
        "events": entry["events"], "error": entry.get("error"),
        "seconds": call.get("seconds"), "tokens": call.get("tokens"), "model": call.get("model"),
        "questions": [{k: q.get(k) for k in ("name", "group", "label", "debt", "answer", "yes")} for q in call.get("questions", [])],
    }


def _words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]", " ", (text or "").lower()).split()


CONTINUES_MIN = 0.6  # how closely the old words must match the start of the new text


def continues(old: str, new: str) -> bool:
    """True when `new` says the sentence of `old` again, maybe with more words. The voice channel transcribes the
    whole turn again, so the first words can differ a little from the first transcript: the old words must match
    the start of the new text by CONTINUES_MIN or more (difflib ratio on words). A resumed turn has the same words."""
    o, n = _words(old), _words(new)
    if not o or not n:
        return False
    return SequenceMatcher(a=o, b=n[: len(o)], autojunk=False).ratio() >= CONTINUES_MIN


class DiscussionAgent:
    name = "discussion"

    def __init__(self, reader, llm, load_source=web.load):
        self.reader = reader
        self.llm = llm
        self.load_source = load_source

    def new_session(self, topic: str, session_id: str) -> "DiscussionSession":
        return DiscussionSession(self, topic, session_id)

    async def health(self) -> dict:
        return {"reader": type(self.reader).__name__, "llm": await self.llm.health()}


class DiscussionSession:
    def __init__(self, agent: DiscussionAgent, topic: str, session_id: str):
        self.agent = agent
        self.id = session_id
        self.topic = topic
        self.agent_name = agent.name
        self.title = " ".join(web.URL_RE.sub("", topic).split()) or "Open discussion"
        self.source: dict = {}
        self.engine = DebtEngine(agent.reader, self.title)
        self.messages: list[dict] = []
        self.turns: list[dict] = []
        self.version = 0
        self.created = time.time()
        self._lock = asyncio.Lock()
        self._prepare_lock = asyncio.Lock()
        self._prepared = False
        self._pending: asyncio.Task | None = None  # the read in progress: of a person message or of a reply
        self._checkpoint = None  # the state before the last person message, for replace_last

    # --- setup ---
    def system_prompt(self) -> str:
        return prompt.system_prompt(self.title, self.source)

    async def prepare(self) -> None:
        """Load the source named in the topic and warm the LLM with the system prompt. Safe to call twice."""
        async with self._prepare_lock:
            if self._prepared:
                return
            url = web.find_url(self.topic)
            if url:
                self.source = await asyncio.to_thread(self.agent.load_source, url)
                if self.source.get("title"):
                    self.engine.ledger.topic = f"{self.title} (source: {self.source['title']})"
            await self.agent.llm.warm([{"role": "system", "content": self.system_prompt()},
                                       {"role": "user", "content": "Ready?"}])
            self._prepared = True
            self.version += 1

    async def settle(self) -> None:
        """Wait for the read in progress. It never raises: a failed read already reported its error."""
        if self._pending is not None:
            try:
                await asyncio.shield(self._pending)
            except Exception:
                pass

    # --- one turn ---
    async def turn(self, text: str, channel: str = "web", replace_last: bool = False):
        text = check_text(text)
        await self.settle()
        async with self._lock:
            await self.prepare()
            replaced = self._undo_last(text) if replace_last else None
            read = asyncio.create_task(self._read_person(text, channel, replaced))
            self._pending = read
            try:
                turn = await asyncio.shield(read)  # a cancel here leaves the read running, so it is still recorded
            except asyncio.CancelledError:
                raise
            except Exception as e:  # the ledger is unchanged, so the person can send the message again
                yield Event("error", f"The reader failed on your message, so nothing changed. "
                                     f"Send it again. ({type(e).__name__}: {e})")
                return
            yield Event("read", turn["read"])
            yield Event("decision", turn["decision"])

            persona = PERSONAS[turn["need"]["persona"]]
            msgs = prompt.messages(self.system_prompt(), self.messages, turn["decision"]["note"])
            parts, completed, error = [], False, None
            cleaner = prompt.ReplyFilter()  # no copied note line may reach the person, the voice or the chat history
            t0, first = time.perf_counter(), None
            try:
                async for piece in self.agent.llm.stream(msgs, {"temperature": persona.temperature}):
                    piece = cleaner.feed(piece)
                    if piece:
                        first = first if first is not None else time.perf_counter() - t0
                        parts.append(piece)
                        yield Event("token", piece)
                    if cleaner.stopped:
                        break
                tail = cleaner.end()
                if tail:
                    parts.append(tail)
                    yield Event("token", tail)
                completed = True
            except OllamaError as e:
                error = str(e)
            finally:
                if cleaner.dropped:
                    turn["dropped_lines"] = cleaner.dropped
                reply = "".join(parts).strip()
                turn["timing"].update({"llm_first_token": round(first, 3) if first is not None else None,
                                       "llm_total": round(time.perf_counter() - t0, 3)})
                if reply:  # an interrupted reply may have been heard in part, so it stays in the chat
                    self._add("ai", reply, channel, interrupted=not completed and error is None)
                    turn["ai"] = self.messages[-1]["seq"]
                    self._pending = asyncio.create_task(self._read_reply(turn, reply))
                self.version += 1
            if error:
                yield Event("error", f"The LLM failed: {error}")
                return
            yield Event("reply", {"text": reply, "timing": turn["timing"]})
            if turn.get("ai") is not None:
                yield Event("check", await asyncio.shield(self._pending))
            yield Event("state", self.state())

    async def _read_person(self, text: str, channel: str, replaced: str | None) -> dict:
        """Jev reads the message, the rules choose the need, and the turn is recorded: all or nothing."""
        before = (copy.deepcopy(self.engine.ledger), len(self.engine.trace), len(self.messages), len(self.turns))
        entry = await asyncio.to_thread(self.engine.step, "human", text)
        self._add("human", text, channel)
        need = entry["need"]
        persona = PERSONAS[need["persona"]]
        note = prompt.note(self.engine.ledger, need, channel)
        turn = {"human": entry["seq"], "channel": channel, "read": summarize(entry), "need": need,
                "timing": {"read_human": entry["call"]["seconds"]},
                "decision": {"persona": persona.key, "name": persona.name, "temperature": persona.temperature,
                             "debt": need["debt"], "kind": need["kind"], "promise": need["promise"],
                             "move": need["move"], "why": need["why"], "stage": entry["stage"], "note": note}}
        if replaced:
            turn["replaced"] = replaced
        self.turns.append(turn)
        self._checkpoint = before
        self.version += 1
        return turn

    def _undo_last(self, new_text: str) -> str | None:
        """Undo the last person message and what came after it, when `new_text` says the same sentence further.
        A voice channel asks for this when the person went on talking before they heard any reply.
        Returns the undone text, or None when nothing is undone. Jev tokens that were spent stay counted."""
        if self._checkpoint is None or not self.turns:
            return None
        last = self.turns[-1]
        old = self.messages[last["human"]]["text"] if last["human"] < len(self.messages) else ""
        if not continues(old, new_text):
            return None
        ledger, n_trace, n_messages, n_turns = self._checkpoint
        self.engine.ledger = ledger
        del self.engine.trace[n_trace:]
        del self.messages[n_messages:]
        del self.turns[n_turns:]
        self._checkpoint = None
        self.version += 1
        return old

    async def _read_reply(self, turn: dict, reply: str) -> dict:
        try:
            entry = await asyncio.to_thread(self.engine.step, "ai", reply)
            turn["timing"]["read_ai"] = entry["call"]["seconds"]
        except Exception as e:
            entry = self.engine.record("ai", reply, f"{type(e).__name__}: {e}")
        turn["ai_read"] = summarize(entry)
        turn["check"] = entry.get("check") or {"debt": None, "done": None, "text": "There was no need to check."}
        self.version += 1
        return turn["check"]

    def _add(self, speaker: str, text: str, channel: str, interrupted: bool = False):
        self.messages.append({"seq": len(self.messages), "id": mid(len(self.messages)), "speaker": speaker,
                              "text": text, "channel": channel, "interrupted": interrupted, "time": time.time()})

    # --- views ---
    def state(self) -> dict:
        L = self.engine.ledger
        last = self.engine.trace[-1] if self.engine.trace else None
        return {
            "id": self.id, "agent": self.agent_name, "topic": self.topic, "title": self.title, "version": self.version,
            "source": {k: self.source.get(k) for k in ("url", "title", "words", "error") if k in self.source},
            "messages": self.messages, "turns": self.turns,
            "stage": L.stage, "stage_why": L.stage_why,
            "options": [{"id": o["id"], "from": mid(o["seq"]), "text": prompt.quote(o["text"], 120)} for o in L.options],
            "scores": L.scores(), "board": L.board(), "need": L.need.id if L.need else None,
            "last_call": summarize(last) if last else None,
            "tokens": self.engine.tokens, "cost_usd": cost_usd(self.engine.tokens),
            "busy": self._lock.locked() or (self._pending is not None and not self._pending.done()),
        }

    def snapshot(self) -> dict:
        return {**self.state(), "created": self.created, "trace": self.engine.trace}
