"""One rule at a time, with hand-made answers."""
from orb_discussion.rules import Ledger


def human(L, text, **yes):
    return L.apply({"seq": len(L.messages), "speaker": "human", "text": text}, {k: 1.0 for k in yes})


def ai(L, text, **yes):
    return L.apply({"seq": len(L.messages), "speaker": "ai", "text": text}, {k: 1.0 for k in yes})


def kinds(L):
    return [(d.kind, d.owner, d.paid is None) for d in L.debts]


def test_first_claim_owes_a_reason_and_a_second_option():
    L = Ledger("Passwords")
    human(L, "Store passwords in a JSON file.", claims=1, new_option=1)
    assert kinds(L) == [("reason", "human", True), ("option", "ai", True)]
    assert L.stage == "Open"
    d, why = L.choose(0)
    assert d.kind == "option"  # in Open, reasons are not due yet: the AI offers a second option first
    assert L.status(L.debts[0]) == "waiting"


def test_second_option_pays_the_option_debt_and_starts_test():
    L = Ledger("Passwords")
    human(L, "Store passwords in a JSON file.", claims=1, new_option=1)
    events = ai(L, "Use one-way hashing instead.", new_option=1, disagrees=1)
    assert any(e["type"] == "pay" and e["debt"] == "D2" for e in events)
    assert L.stage == "Test"
    assert L.status(L.debts[0]) == "due"  # the reason debt is due now


def test_repeat_after_a_reason_was_asked_means_move_on():
    L = Ledger("Passwords")
    human(L, "Store passwords in a JSON file.", claims=1, new_option=1)
    ai(L, "Use hashing.", new_option=1)
    human(L, "JSON is fine.", claims=1, repeats=1)
    assert L.debts[-1].kind == "move_on"  # a reason debt already existed, so asking again would go in a circle
    d, _ = L.choose(2)
    assert d.kind == "move_on"


def test_repeat_without_an_earlier_reason_debt_owes_a_reason():
    L = Ledger("Passwords")
    human(L, "JSON files are simple.", claims=1, gives_reason=1, new_option=1)
    ai(L, "Hashing is safer.", new_option=1)
    human(L, "JSON is fine.", claims=1, repeats=1)
    assert L.debts[-1].kind == "reason"


def test_ask_to_close_moves_to_close_and_calls_the_chair():
    L = Ledger("Passwords")
    human(L, "Store passwords in a JSON file.", claims=1, new_option=1)
    ai(L, "Use hashing.", new_option=1)
    human(L, "Let's decide now. Summarize.", asks_to_close=1, asks=1)
    assert L.stage == "Close"
    d, _ = L.choose(2)
    assert d.kind == "summary"
    assert not any(x.kind == "answer" and x.start == 2 for x in L.debts)  # a request to close is not a question


def test_the_ai_waits_for_an_answer_that_the_person_owes():
    L = Ledger("Passwords")
    human(L, "Store passwords in a JSON file.", claims=1, gives_reason=1, new_option=1)
    ai(L, "Why not hashing?", new_option=1, asks=1)
    human(L, "Hmm.")
    d, why = L.choose(2)
    answer = next(x for x in L.debts if x.kind == "answer")
    assert answer.owner == "human" and not L.actionable(answer)
    assert answer.id in why


def test_stage_never_goes_back():
    L = Ledger("Passwords")
    human(L, "Decide now.", asks_to_close=1)
    assert L.stage == "Close"
    human(L, "Another idea: use a vault.", claims=1, new_option=1)
    assert L.stage == "Close"


def test_a_change_of_view_owes_a_record_not_a_reason():
    L = Ledger("Passwords")
    human(L, "OK, fair point. No plain passwords.", changes_view=1, claims=1)
    assert [d.kind for d in L.debts] == ["record"]
