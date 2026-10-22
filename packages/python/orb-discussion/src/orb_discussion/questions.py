"""What Jev reads for one new message, and the yes/no questions it answers.

Every question has one job in rules.py:
- questions about the new message decide which debts start and whether a new option is on the table;
- one "does it pay?" question for each open debt that the writer can pay decides which debts end;
- after a person message, one check question about the AI reply asks whether it did the job of its persona.
All questions go to Jev in one request and are answered in parallel.
"""
from typesafe_sdk import Noul, NoulCriteria

from orb_discussion.rules import mid

EARLIER_CAP = 600     # characters kept from each earlier message
MSG_CAP = 1500        # characters kept from the new message
RECENT = 4            # the last messages are always in view
MAX_EARLIER = 12      # at most this many earlier messages, so the request size stays bounded

# name -> (asked for whose messages, instructions, what "yes" means, what "no" means, short label for the page)
ABOUT = {
    "asks": ("all", "Does `new_message` ask the other side a question that needs a reply?",
             "It asks for information, an opinion, a reason, a plan or a decision, and it expects the other side to reply.",
             "It asks nothing, or only a polite or rhetorical question that needs no reply.",
             "asks a question"),
    "claims": ("human", "Does `new_message` state its writer's own view, proposal or plan, or reject a point of the other side?",
               "It says what its writer thinks is true or should be done, or it says no to a point that the other side made.",
               "It only asks, thanks, or answers with facts, and it takes no position of its own.",
               "makes a claim"),
    "gives_reason": ("human", "Does `new_message` give a reason, evidence or an example for a claim?",
                     "It explains why: a cause, an argument, data, a test result or an example.",
                     "It states things without any reason or evidence.",
                     "gives a reason"),
    "repeats": ("human", "Does `new_message` repeat a point that `new_message.speaker` already made in `earlier_messages`?",
                "It restates an earlier point of the same speaker and adds no new reason, example or evidence.",
                "It makes a new point, or it adds a new reason or evidence to an old point.",
                "repeats an old point"),
    "changes_view": ("human", "Does `new_message.speaker` accept a point that they argued against earlier in `earlier_messages`, or change their own position?",
                     "The speaker gives up or softens an earlier view: accepts the other side's point, drops a part of their proposal, or says they were wrong.",
                     "The speaker keeps their view, or had no earlier view on this point. Asking a question or asking to close is not a change of view.",
                     "changes view"),
    "asks_to_close": ("human", "Does `new_message` ask to conclude the discussion: to decide, to summarize or to stop?",
                      "It asks for a decision, a summary, a final answer, or to end the discussion.",
                      "It starts or continues the discussion. 'Let's discuss this' or 'tell me more' starts it and does not conclude it.",
                      "asks to close"),
    "disagrees": ("all", "Does `new_message` disagree with, reject or correct a point that the other side made in `earlier_messages`?",
                  "It argues against, rejects, corrects or doubts something that the other side said.",
                  "It agrees, asks a neutral question, or makes a point that the other side did not address.",
                  "disagrees"),
    "new_option": ("all", "Does `new_message` put a new option on the table: a way to settle `topic` that is not already in `options`?",
                   "It proposes a concrete approach, tool, method or answer for the topic, and that proposal differs from every entry in `options`. "
                   "When `options` is empty, the first concrete proposal counts.",
                   "It supports, attacks or repeats an option that is already in `options`, or it proposes nothing concrete.",
                   "puts a new option on the table"),
    "names_open_point": ("all", "Does `new_message` name a point that is still open and must be decided before the discussion can end?",
                         "It says what is still unresolved, open or undecided, for example 'the main open issue is ...'.",
                         "It names no open point. A question alone does not count.",
                         "names an open point"),
}

PAYS = {  # debt kind -> (instructions, what "yes" means, what "no" means)
    "answer": ("Does `new_message` give the answer that `open_debts.{id}` waits for?",
               "It answers that question: it gives the information, opinion, reason or decision that the question asks for.",
               "It does not answer that question, or it answers something else."),
    "reply": ("Does `new_message` reply to the option in `open_debts.{id}`?",
              "It accepts that option, rejects it with a reason, or compares it with another option.",
              "It ignores that option, or it only repeats its own earlier view."),
    "reason": ("Does `new_message` give the reason that `open_debts.{id}` waits for?",
               "It explains why that claim holds: a cause, an argument, evidence or an example. Withdrawing the claim also counts.",
               "It gives no reason for that claim. A repeat, a 'no' or a new claim does not count."),
    "weigh": ("Does `new_message` reply on its merits to the reason in `open_debts.{id}`?",
              "It accepts that reason, or it answers that reason with a concrete counter-argument.",
              "It ignores that reason, or it only repeats its own view."),
    "record": ("Does `new_message` state what is now agreed, as `open_debts.{id}` asks?",
               "It says clearly which point both sides now accept.",
               "It does not say what is agreed."),
    "move_on": ("Does `new_message` move the discussion to a new step, as `open_debts.{id}` asks?",
                "It names the repeat, or it brings a new question, a new option, a direct comparison or a concrete proposal.",
                "It only repeats points that are already on the table."),
    "end": ("Does `new_message` end the disagreement or settle the open point in `open_debts.{id}`?",
            "One side accepts the other side's point, a decision is made, or the point moves to another place or time.",
            "Both sides still hold their views, or the point is still undecided."),
    "summary": ("Does `new_message` give the summary or decision that `open_debts.{id}` asks for?",
                "It sums up where the discussion stands, or it states a decision.",
                "It does not sum up and it does not decide."),
}
ASKS_FOR = ("Does `new_message` ask the human for the reason that `open_debts.{id}` waits for?",
            "It asks why, or it asks for evidence or an example for that claim.",
            "It does not ask for that reason.")


def noul(instructions, yes, no):
    return Noul(instructions=instructions, criteria=NoulCriteria(true=yes, false=no))


def cut(text, n):
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def earlier_in_view(ledger):
    """The opening message, the last RECENT messages, and the messages that started an open debt."""
    msgs = ledger.messages
    if not msgs:
        return []
    why = {msgs[0]["seq"]: "opening message"}
    for e in msgs[-RECENT:]:
        why.setdefault(e["seq"], "recent")
    for d in sorted(ledger.open(), key=lambda d: -d.start):
        why.setdefault(d.start, f"started {d.id}")
    rank = lambda s: 0 if why[s] == "opening message" else 1 if why[s] == "recent" else 2
    keep = sorted(why, key=lambda s: (rank(s), -s))[:MAX_EARLIER]
    by_seq = {e["seq"]: e for e in msgs}
    return [(by_seq[s], why[s]) for s in sorted(keep)]


def request(ledger, msg):
    """The state and the questions for one new message, plus a description of each question for the trace."""
    who = msg["speaker"]
    state = {
        "topic": ledger.topic,
        "options": {o["id"]: f"{cut(o['text'], 160)} (first proposed in {mid(o['seq'])})" for o in ledger.options},
        "open_debts": {d.id: d.about for d in ledger.open()},
        "earlier_messages": {mid(e["seq"]): {"speaker": e["speaker"], "why_shown": w, "text": cut(e["text"], EARLIER_CAP)}
                             for e, w in earlier_in_view(ledger)},
        "new_message": {"id": mid(msg["seq"]), "speaker": who, "text": cut(msg["text"], MSG_CAP)},
    }
    questions, meta = {}, []
    for name, (scope, instr, yes, no, label) in ABOUT.items():
        if scope in ("all", who):
            questions[name] = noul(instr, yes, no)
            meta.append({"name": name, "group": "message", "label": label, "text": instr})
    for d in ledger.payable(who):
        instr, yes, no = PAYS[d.kind]
        name = f"pays_{d.id}"
        questions[name] = noul(instr.format(id=d.id), yes, no)
        meta.append({"name": name, "group": "pay", "debt": d.id, "label": f"pays {d.id}", "text": instr.format(id=d.id)})
    need = ledger.need
    if who == "ai" and need is not None and need.paid is None and need.owner == "human":
        instr, yes, no = ASKS_FOR
        name = f"asks_for_{need.id}"
        questions[name] = noul(instr.format(id=need.id), yes, no)
        meta.append({"name": name, "group": "check", "debt": need.id, "label": f"asks for {need.id}", "text": instr.format(id=need.id)})
    return state, questions, meta
