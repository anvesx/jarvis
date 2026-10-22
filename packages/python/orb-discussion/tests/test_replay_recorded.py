"""Regression test: the recorded Jev answers of the passwords chat must give the same questions, debts, stages,
needs and checks as on the day they were recorded. No network call."""
from conftest import ScriptedReader

from orb_discussion.engine import DebtEngine


def test_recorded_run_gives_the_same_decisions(passwords_run):
    steps = passwords_run["steps"]
    script = [{q["name"]: q["answer"] for q in s["call"]["questions"]} for s in steps]
    reader = ScriptedReader(script)
    engine = DebtEngine(reader, passwords_run["topic"])
    for i, rec in enumerate(steps):
        got = engine.step(rec["speaker"], rec["text"])
        assert reader.calls[i]["questions"] == [q["name"] for q in rec["call"]["questions"]], rec["id"]
        assert [(e["type"], e.get("debt"), e["q"]) for e in got["events"]] == \
               [(e["type"], e.get("debt"), e["q"]) for e in rec["events"]], rec["id"]
        assert got["stage"] == rec["stage"], rec["id"]
        assert got["scores"] == rec["scores"], rec["id"]
        if "need" in rec:
            assert (got["need"]["debt"], got["need"]["persona"]) == (rec["need"]["debt"], rec["need"]["persona"]), rec["id"]
        if "check" in rec:
            assert got["check"]["done"] == rec["check"]["done"], rec["id"]


def test_recorded_run_personas_match_the_old_app(passwords_run):
    """The day it was recorded, the new rules picked the same 7 personas as the old weighted app."""
    needs = [s["need"] for s in passwords_run["steps"] if "need" in s]
    assert [n["persona"] for n in needs] == ["Rival", "Judge", "Examiner", "Judge", "Navigator", "Judge", "Chair"]
    assert all(n["persona"] == n["old_persona"] for n in needs)
