"""The debt rules: what a message starts, what it pays, the stage, the need and the persona.

First principles. A good discussion keeps six promises. When a message leaves something open, the talk owes
something: a debt. A later message pays it. The stage (Open, Test, Close) decides which debts are due now.
The AI pays the most urgent due debt, and the kind of that debt picks the persona.

Jev only answers yes/no questions (questions.py). These rules turn the answers into debts. The only
number here is YES, the line where a Jev answer counts as "yes". There are no weights and no guessed targets:
a promise is due (target 1) or not due (target 0), and its score is debts paid / debts owed.
"""
from dataclasses import asdict, dataclass

YES = 0.5  # a Jev answer at or above this counts as "yes"

STAGES = ["Open", "Test", "Close"]
PROMISES = ["Interaction", "Divergence", "Challenge", "Reflection", "Progression", "Resolution"]
DUE = {  # the target: the promises whose debts must be paid in each stage
    "Open": ["Interaction", "Divergence"],
    "Test": ["Interaction", "Divergence", "Challenge", "Reflection", "Progression"],
    "Close": PROMISES,
}
KINDS = {  # debt kind -> the promise it belongs to
    "answer": "Interaction",
    "reply": "Interaction",
    "option": "Divergence",
    "reason": "Challenge",
    "weigh": "Reflection",
    "record": "Reflection",
    "move_on": "Progression",
    "end": "Resolution",
    "summary": "Resolution",
}
PERSONA = {"Interaction": "Teacher", "Divergence": "Rival", "Challenge": "Examiner",
           "Reflection": "Judge", "Progression": "Navigator", "Resolution": "Chair"}
TIE = ["Resolution", "Reflection", "Progression", "Challenge", "Interaction", "Divergence"]  # one message starts several due debts
MOVE = {  # the job of the persona for each kind of debt
    "answer": "Answer the person's question.",
    "reply": "Reply to the person's new option on its merits.",
    "option": "Offer a second option, with one concrete reason.",
    "reason": "Ask for the reason. Do not agree, and do not argue yet.",
    "weigh": "Weigh the reason on its merits: accept it, or answer it with a better reason.",
    "record": "Say clearly what is now agreed.",
    "move_on": "Name the repeat, and move the talk to a new step.",
    "end": "Propose how to settle the open point.",
    "summary": "Sum up where the discussion stands, then propose a decision.",
}
NO_NEED = "Nothing is due that the AI can act on. Move the talk forward with one new question."


def mid(seq):
    return f"M{seq + 1}"  # message IDs count from 1, like the pages


@dataclass
class Debt:
    id: str
    kind: str
    owner: str             # who can pay it: "human", "ai" or "both"
    start: int             # seq of the message that started it
    about: str             # the sentence that Jev reads in `open_debts`
    rule: str              # the rule that started it, in words
    paid: int | None = None
    paid_by: str | None = None   # the question whose answer paid it
    paid_p: float | None = None

    @property
    def promise(self):
        return KINDS[self.kind]


class Ledger:
    """The memory of one chat: the messages, the options on the table, the debts and the stage."""

    def __init__(self, topic):
        self.topic = topic
        self.messages, self.debts, self.options = [], [], []
        self.stage, self.stage_why = "Open", "The talk starts. Fewer than 2 options are on the table."
        self.close_asked = False
        self.need = None  # the debt chosen after the latest person message

    # --- views ---
    def open(self):
        return [d for d in self.debts if d.paid is None]

    def payable(self, speaker):
        """Open debts that a message by this speaker can pay. The option debt is paid by counting options."""
        return [d for d in self.open() if d.owner in (speaker, "both") and d.kind != "option"]

    def status(self, d):
        if d.paid is not None:
            return "paid"
        return "due" if d.promise in DUE[self.stage] else "waiting"

    @staticmethod
    def actionable(d):
        """The AI cannot pay an answer or a reply that the person owes. It can only wait for it."""
        return not (d.kind in ("answer", "reply") and d.owner == "human")

    def scores(self):
        out = {}
        for p in PROMISES:
            ds = [d for d in self.debts if d.promise == p]
            out[p] = {"owed": len(ds), "paid": sum(d.paid is not None for d in ds), "due": p in DUE[self.stage]}
        return out

    def board(self):
        return [{**asdict(d), "promise": d.promise, "status": self.status(d)} for d in self.debts]

    # --- the rules ---
    def apply(self, msg, ans):
        """Apply one message. `ans` holds Jev's answers by question name. Returns the events in order."""
        seq, who = msg["seq"], msg["speaker"]
        other = "ai" if who == "human" else "human"
        m = mid(seq)
        yes = lambda q: ans.get(q, 0.0) >= YES
        events = []

        def start(kind, owner, about, rule, q):
            d = Debt(f"D{len(self.debts) + 1}", kind, owner, seq, about, rule)
            self.debts.append(d)
            events.append({"type": "start", "debt": d.id, "rule": rule, "q": q, "p": ans.get(q)})

        def pay(d, q, rule):
            d.paid, d.paid_by, d.paid_p = seq, q, ans.get(q)
            events.append({"type": "pay", "debt": d.id, "rule": rule, "q": q, "p": ans.get(q)})

        # 1. Pay: Jev answered one "does it pay?" question for each open debt that this writer can pay.
        for d in self.payable(who):
            q = f"pays_{d.id}"
            if yes(q):
                pay(d, q, f"Jev says that {m} pays {d.id}.")

        # 2. Options on the table. The second option pays the option debt.
        if yes("new_option"):
            self.options.append({"id": f"O{len(self.options) + 1}", "seq": seq, "text": msg["text"]})
            events.append({"type": "option", "option": self.options[-1]["id"], "rule": f"{m} puts a new option on the table.",
                           "q": "new_option", "p": ans["new_option"]})
            if len(self.options) >= 2:
                for d in self.open():
                    if d.kind == "option":
                        pay(d, "new_option", "A second option is on the table, so the option debt is paid.")

        # 3. Start new debts.
        if who == "human":
            if yes("asks_to_close"):
                self.close_asked = True
                start("summary", "ai", f"{m} (human) asks to close. The ai owes a summary and a decision.",
                      "The person asks to close, so the AI owes a summary and a decision.", "asks_to_close")
            if yes("changes_view"):
                start("record", "ai", f"{m} (human) changes view. The ai owes a clear record of what is now agreed.",
                      "The person changes view, so the AI owes a record of what is now agreed.", "changes_view")
            elif yes("claims") and not yes("gives_reason"):
                asked_once = any(d.kind == "reason" and d.owner == "human" for d in self.debts)
                if yes("repeats") and asked_once:
                    start("move_on", "ai", f"{m} (human) repeats an earlier point with nothing new. The ai owes a new step.",
                          "The person repeats a point with no reason, and a reason was already asked for once. "
                          "Asking again would go in a circle, so the AI owes a new step.", "repeats")
                else:
                    start("reason", "human", f"{m} (human) makes a claim with no reason. The human owes a reason for it.",
                          "The person makes a claim with no reason, so the person owes a reason.", "claims")
            if yes("gives_reason"):
                start("weigh", "ai", f"{m} (human) gives a reason. The ai owes a fair reply to that reason.",
                      "The person gives a reason, so the AI owes a fair reply to it.", "gives_reason")
        asker, payer = ("person", "AI") if who == "human" else ("AI", "person")
        if yes("asks") and not (who == "human" and yes("asks_to_close")):
            start("answer", other, f"{m} ({who}) asks a question. The {other} owes an answer.",
                  f"The {asker} asks a question, so the {payer} owes an answer.", "asks")
        elif who == "ai" and yes("new_option") and len(self.options) >= 2:
            # the person's own options are tested by the claim and reason rules; a new option from the AI needs the person's reply
            start("reply", other, f"{m} (ai) puts a new option on the table. The human owes a reply to it.",
                  "The AI puts a new option on the table, so the person owes a reply to it.", "new_option")
        if not any(d.kind == "end" for d in self.open()):
            if yes("disagrees"):
                start("end", "both", f"{m} ({who}) disagrees with the {other}. The disagreement must end.",
                      "A disagreement starts, so it must end before the talk closes.", "disagrees")
            elif yes("names_open_point"):
                start("end", "both", f"{m} ({who}) names a point that is still open. It must be settled.",
                      "An open point is named, so it must be settled before the talk closes.", "names_open_point")
        if len(self.options) == 1 and not any(d.kind == "option" for d in self.debts):
            start("option", "ai", "Only one option is on the table. The ai owes a second option.",
                  "Only one option is on the table, so a second option is owed.", "new_option")

        # 4. The stage. It never goes back.
        if self.stage != "Close":
            left = self.open()
            if self.close_asked:
                self.stage, self.stage_why = "Close", "The person asks to close."
            elif self.stage == "Test" and left and all(d.promise == "Resolution" for d in left):
                self.stage, self.stage_why = "Close", "Only closing debts are left, so the talk moves to Close."
            elif len(self.options) >= 2:
                self.stage, self.stage_why = "Test", f"{len(self.options)} options are on the table, so the ideas can be tested."
            else:
                self.stage_why = f"{len(self.options)} option{'s' if len(self.options) != 1 else ''} on the table. Test starts at 2."
        self.messages.append(msg)
        return events

    def choose(self, seq):
        """The need: the most urgent due debt that the AI can act on. Returns (debt or None, why)."""
        due = [d for d in self.open() if self.status(d) == "due" and self.actionable(d)]
        waiting_for_person = [d.id for d in self.open() if self.status(d) == "due" and not self.actionable(d)]
        tail = (f" The AI waits for {' and '.join(waiting_for_person)}, because the person owes "
                f"{'it' if len(waiting_for_person) == 1 else 'them'}." if waiting_for_person else "")
        if not due:
            self.need = None
            return None, "No due debt that the AI can act on." + tail
        newest = [d for d in due if d.start == seq]
        if newest:
            d = min(newest, key=lambda d: TIE.index(d.promise))
            why = f"{d.id} is due, and the newest message started it."
            if len(newest) > 1:
                why += f" {len(newest)} due debts started here. The fixed order puts {d.promise} first."
        else:
            d = min(due, key=lambda d: (d.start, TIE.index(d.promise)))
            why = f"The newest message started no due debt that the AI can act on. {d.id} is the oldest due debt."
        self.need = d
        return d, why + tail
