"""The runtime loop for one chat: each new message makes one Jev call, then the rules update the ledger.

    engine = DebtEngine(client, topic)
    entry = engine.step("human", "We should store passwords in a JSON file.")
    entry["need"]  ->  {"debt": "D1", "persona": "Rival", "move": ..., "why": ...}

Each entry keeps the whole Jev call (what Jev read, every question and its answer), the events that the
answers caused, the stage, the six promises (target and score), the board of debts, and the need.
"""
import time

from orb_discussion import questions
from orb_discussion.rules import MOVE, NO_NEED, PERSONA, STAGES, YES, Ledger, mid


class DebtEngine:
    def __init__(self, client, topic):
        self.client = client
        self.ledger = Ledger(topic)
        self.trace = []
        self.tokens = 0

    def step(self, speaker, text):
        L = self.ledger
        msg = {"seq": len(L.messages), "speaker": speaker, "text": text}
        need_before = L.need
        state, qs, meta = questions.request(L, msg)
        t0 = time.time()
        r = self.client.system_one(state=state, questions=qs)
        seconds = round(time.time() - t0, 2)
        ans = {name: round(a.noul, 3) for name, a in r.answers.items()}
        tokens = r.usage.input_tokens if r.usage else None
        self.tokens += tokens or 0
        for q in meta:
            q["answer"] = ans[q["name"]]
            q["yes"] = ans[q["name"]] >= YES

        events = L.apply(msg, ans)
        entry = {
            "seq": msg["seq"], "id": mid(msg["seq"]), "speaker": speaker, "text": text,
            "call": {"seconds": seconds, "tokens": tokens, "model": r.model, "state": state, "questions": meta},
            "events": events,
            "options": [{"id": o["id"], "from": mid(o["seq"])} for o in L.options],
            "stage": L.stage, "stage_index": STAGES.index(L.stage), "stage_why": L.stage_why,
        }
        if speaker == "human":
            d, why = L.choose(msg["seq"])
            entry["need"] = ({"debt": d.id, "kind": d.kind, "promise": d.promise, "persona": PERSONA[d.promise],
                              "move": MOVE[d.kind], "why": why} if d else
                             {"debt": None, "kind": None, "promise": "Progression", "persona": "Navigator", "move": NO_NEED, "why": why})
        elif need_before is not None:
            d = need_before
            if d.owner == "human" and d.paid is None:
                q = f"asks_for_{d.id}"
                done = ans.get(q, 0) >= YES
                entry["check"] = {"debt": d.id, "persona": PERSONA[d.promise], "q": q, "p": ans.get(q), "done": done,
                                  "text": f"The {PERSONA[d.promise]} had to ask for {d.id}. " + ("It did." if done else "It did not.")}
            else:
                done = d.paid == msg["seq"]
                entry["check"] = {"debt": d.id, "persona": PERSONA[d.promise], "q": d.paid_by, "p": d.paid_p, "done": done,
                                  "text": f"The {PERSONA[d.promise]} had to pay {d.id}. " + ("It did." if done else "It did not.")}
            L.need = None
        entry["scores"] = L.scores()
        entry["board"] = L.board()
        self.trace.append(entry)
        return entry

    def record(self, speaker, text, error):
        """Keep the ledger in step when the reader fails on a message that must stay in the chat (an AI reply that
        the person already heard). The message is applied with no answers, so no debt starts or ends because of it."""
        L = self.ledger
        msg = {"seq": len(L.messages), "speaker": speaker, "text": text}
        events = L.apply(msg, {})
        entry = {
            "seq": msg["seq"], "id": mid(msg["seq"]), "speaker": speaker, "text": text, "call": None, "error": error,
            "events": events, "options": [{"id": o["id"], "from": mid(o["seq"])} for o in L.options],
            "stage": L.stage, "stage_index": STAGES.index(L.stage), "stage_why": L.stage_why,
        }
        if speaker == "ai" and L.need is not None:
            d = L.need
            entry["check"] = {"debt": d.id, "persona": PERSONA[d.promise], "q": None, "p": None, "done": None,
                              "text": f"The reader failed, so the check of {d.id} is unknown."}
            L.need = None
        entry["scores"] = L.scores()
        entry["board"] = L.board()
        self.trace.append(entry)
        return entry
