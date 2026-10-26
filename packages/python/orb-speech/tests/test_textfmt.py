"""The written-form formatter: what it fixes, what it must leave alone, and the three properties of design 4.1."""
import collections
import json
import random
import re
import statistics
import time
from pathlib import Path

import pytest

from orb_speech.textfmt import written

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "synth_hyps.json").read_text())
# Parakeet and Whistle on the stt/dictation corpus (480 Kokoro clips, run 20261023-122740), one row for each distinct raw text.
DICTATION = json.loads((Path(__file__).parent / "fixtures" / "dictation_raw.json").read_text())

# Each class: (spoken form, written form). The first block is the ITN classes of design section 2.
FIXES = [
    # pull requests, issues, numbers
    ("Please review pull request number 4821 before the standoff.", "Please review pull request #4821 before the standoff."),
    ("Please review pull request number four thousand eight hundred and twenty-one before the stand-up.",
     "Please review pull request #4821 before the stand-up."),
    ("review PR 4821 now", "review PR #4821 now"),
    ("review pr number twelve now", "review PR #12 now"),
    ("Pull request four thousand eight hundred twenty one", "Pull request #4821"),
    ("fix issue 12 and ticket three", "fix issue #12 and ticket #3"),
    ("is it number 5 on the list", "is it #5 on the list"),
    ("Please review pool request number 4821 before the stand-up.", "Please review pool request #4821 before the stand-up."),
    # ids read in pairs, and Parakeet's split numbers, only after a marker
    ("I closed issue number three eighteen and opened PR number three twenty two for the fix.",
     "I closed issue #318 and opened PR #322 for the fix."),
    ("Can you link issue number one zero four two in the release notes?", "Can you link issue #1042 in the release notes?"),
    ("review pull request number forty eight twenty one now", "review pull request #4821 now"),
    ("file ticket three oh two please", "file ticket #302 please"),
    ("Please review pull request number 4800 and 21 before the stand-up.", "Please review pull request #4821 before the stand-up."),
    ("Please review pole request number 4800, and 21, before the standup.", "Please review pole request #4821, before the standup."),
    ("Please review pool request number four thousand eight hundred, and twenty one before", "Please review pool request #4821 before"),
    ("issue 4000 and 321 and PR 100 and 5 others", "issue #4321 and PR #100 and 5 others"),
    # times
    ("Wednesday at 3 30 pm.", "Wednesday at 3:30 PM."),
    ("Wednesday at 3 30 p.m. we go", "Wednesday at 3:30 PM we go"),
    ("Book it for three thirty p.m. tomorrow.", "Book it for 3:30 PM tomorrow."),
    ("at three oh five pm", "at 3:05 PM"),
    ("at ten forty-five am", "at 10:45 AM"),
    ("at ten am", "at 10 AM"),
    ("at 5 pm. Thanks", "at 5 PM. Thanks"),
    ("at 3.30pm.", "at 3:30 PM."),
    ("at 330 p.m.", "at 3:30 PM."),
    ("at 10am about it", "at 10 AM about it"),
    ("at 10:00 am about it", "at 10:00 AM about it"),
    ("come at three o'clock", "come at 3 o'clock"),
    # dates and years
    ("Let's meet on November fourteenth at ten am.", "Let's meet on November 14th at 10 AM."),
    ("due on March third.", "due on March 3rd."),
    ("the fourteenth of november", "the 14th of November"),
    ("the third of March", "the 3rd of March"),
    ("November fourteenth, twenty twenty six, is the day", "November 14th, 2026, is the day"),
    ("in twenty twenty-six we ship", "in 2026 we ship"),
    ("in nineteen ninety nine", "in 1999"),
    ("in twenty oh five", "in 2005"),
    ("in two thousand twenty six", "in 2026"),
    ("the twenty first century", "the 21st century"),
    # money, percent, units
    ("twelve thousand four hundred fifty dollars and thirty cents, due", "$12,450.30, due"),
    ("it costs five dollars and fifty cents.", "it costs $5.50."),
    ("12 dollars and 40 dollars", "$12 and $40"),
    ("grew by twenty five percent", "grew by 25%"),
    ("one hundred percent sure", "100% sure"),
    ("a 3 percent rise", "a 3% rise"),
    ("two point five million dollars", "2.5 million dollars"),
    ("version three point two", "version 3.2"),
    ("Llama three point two", "Llama 3.2"),
    ("two hundred megabytes in three days", "200 megabytes in 3 days"),
    ("eleven apples and twelve hundred rupees", "11 apples and 1,200 rupees"),
    ("between twenty and thirty percent", "between 20 and 30%"),
    ("twenty-one questions", "21 questions"),
    # email, domains, urls
    ("Send it to priya dot sharma at devxlabs dot ai by end of day.", "Send it to priya.sharma@devxlabs.ai by end of day."),
    ("Send it to priaya.sharma at devxlabs.ai by end of day.", "Send it to priaya.sharma@devxlabs.ai by end of day."),
    ("please email priya at devxlabs.ai today", "please email priya@devxlabs.ai today"),
    ("send it to bob at devxlabs dot ai please", "send it to bob@devxlabs.ai please"),
    ("My email is arjun at example.com in case you need it.", "My email is arjun@example.com in case you need it."),
    ("bob underscore smith at example dot co dot in", "bob_smith@example.co.in"),
    ("Check example dot com slash docs and then go home.", "Check example.com/docs and then go home."),
    ("Visit https colon slash slash example dot com slash docs slash api now.", "Visit https://example.com/docs/api now."),
    ("www dot example dot org", "www.example.org"),
    ("The docs are at docs.devslabs.ai slash orb slash setup.", "The docs are at docs.devslabs.ai/orb/setup."),
    ("Open GitHub.com slash devslabs slash orb and check the issues tab.", "Open github.com/devslabs/orb and check the issues tab."),
    ("see github.com slash dev then slash orb", "see github.com/dev then slash orb"),
    ("Edit it, then set settings dot Json and restart", "Edit it, then set settings dot Json and restart"),
    # casing
    ("ok so the api and ci are down, prs and urls too", "OK so the API and CI are down, PRs and URLs too"),
    ("update the readme for the sql and aws parts", "update the README for the SQL and AWS parts"),
    ("Update the Readme.", "Update the README."),
    # punctuation and spacing survive
    ("(five hundred dollars) and more", "($500) and more"),
    ("  twenty five   percent  ", "  25%  "),
    ("Two weeks ago, five minutes", "2 weeks ago, 5 minutes"),
]

# Text that is already written, or ordinary talk with number words in it: it must come back as it came.
UNCHANGED = [
    "one of them",
    "I have one of them and two of those.",
    "No one knows, and the one thing I want is one more try.",
    "At the first point, two of them, three or four people.",
    "wait a second, the second one",
    "Gemma 4 beats Llama 3.2 on one of the tests.",
    "The invoice total is $12,450.30, due on March 3rd.",
    "Book the meeting for Wednesday at 3:30 PM.",
    "Let's sync on Friday the 14th of November at 10 RM about it.",
    "Please review pull request #4821 before the standup.",
    "Send the report to priya.sharma@devxlabs.ai by end of day.",
    "Visit https://example.com/docs now.",
    "Open README.md and check ci/cd.",
    "Look at google.com for details.",
    "Send it to Bob at the office.",
    "The meeting is arjun at example.com",
    "Meet me at noon, then look at it.",
    "The dot com bubble burst and polka dot dresses stayed.",
    "the number of users and a phone number 5551234",
    "We have three eighteen apples and forty eight twenty one people, a three oh two chance.",
    "I paid 4800 and 21 people came, in 4800 and 21 days.",
    "run git slash status and then cd slash tmp, or go to the slash page",
    "Open github.com slash dev slabs slash orb and see",
    "The A. P. I. latency, the A.P.I. too",
    "number one priority and the number three bus",
    "the first thing, the third party, a second look",
    "one two three four",
    "twenty twenty five",
    "A hundred and one dalmatians.",
    "She said \"number five is alive, PR 7, at 3 30 pm\" and left.",
    "use `three thousand dollars` here",
    "I may first check it, you may go.",
    "In March or may first.",
    "Action items 1. Update the README 2. Fix the flaky test in CI. 3. Deploy to production.",
    "Run git checkout dash b feature slash login dash flow and then push to origin.",
    "wrap the call inside use effect and add userI to the dependency array.",
    "",
    "   ",
    "uh",
]


@pytest.mark.parametrize("spoken,want", FIXES)
def test_the_written_form(spoken, want):
    assert written(spoken) == want


@pytest.mark.parametrize("text", UNCHANGED)
def test_written_and_ordinary_text_is_not_damaged(text):
    assert written(text) == text


def test_none_and_empty():
    assert written("") == "" and written(None) == ""


def test_a_style_is_checked():
    with pytest.raises(ValueError, match="style must be one of default"):
        written("hello", style="loud")
    assert written("twenty percent", style="default") == "20%"


@pytest.mark.parametrize("row", [pytest.param(r, id=f"{r['id']}-{i}") for i, r in enumerate(FIXTURES)])
def test_the_recorded_model_outputs(row):
    """Real model outputs for the synthetic clips (bench/results, 9 engines): the golden written form."""
    assert written(row["hyp"]) == row["written"]


@pytest.mark.parametrize("row", [pytest.param(r, id=f"{r['cls']}-{i}") for i, r in enumerate(DICTATION)])
def test_the_recorded_dictation_outputs(row):
    assert written(row["raw"]) == row["written"]


def test_the_dictation_fixtures_fix_the_number_and_url_gaps():
    got = {r["raw"]: r["written"] for r in DICTATION}
    assert got["I closed issue number three eighteen, and opened PR number three twenty two for the fix."] == \
        "I closed issue #318, and opened PR #322 for the fix."
    assert got["Please review pull request number 4800 and 21 before the stand-up."] == \
        "Please review pull request #4821 before the stand-up."
    assert got["The docs are at docs.devslabs.ai slash orb slash setup."] == "The docs are at docs.devslabs.ai/orb/setup."
    assert len(DICTATION) >= 500 and sum(r["raw"] != r["written"] for r in DICTATION) >= 150


def test_the_fixtures_cover_the_classes():
    changed = [r for r in FIXTURES if r["written"] != r["hyp"]]
    assert len(FIXTURES) >= 90 and len(changed) >= 30
    assert any("#4821" in r["written"] for r in changed)
    assert any("3:30 PM" in r["written"] for r in changed)
    assert any(re.search(r"\w@devxlabs\.ai", r["written"]) for r in changed)
    assert any("README" in r["written"] for r in changed)


def words(text):
    return re.findall(r"[a-z]+", text.lower())


# The fixed table of design 4.1: the suffix of an ordinal, and the letters of AM and PM.
FIXED_TABLE = {"st", "nd", "rd", "th", "am", "pm"}


def added_words(spoken, out):
    pool = collections.Counter(words(spoken))
    extra = []
    for w in words(out):
        if pool[w] > 0:
            pool[w] -= 1
        elif w not in FIXED_TABLE:
            extra.append(w)
    return extra


CORPUS = ([r["raw"] for r in DICTATION] + [r["ref"] for r in DICTATION] + [r["hyp"] for r in FIXTURES] + [r["ref"] for r in FIXTURES] + [s for s, _ in FIXES] + [w for _, w in FIXES]
          + UNCHANGED)


@pytest.mark.parametrize("text", CORPUS)
def test_idempotent_deterministic_and_adds_no_word(text):
    once = written(text)
    assert written(once) == once, "written(written(x)) must equal written(x)"
    assert written(text) == once, "the same text must give the same text (preflight == final)"
    assert added_words(text, once) == []


def test_fuzzed_token_soup_keeps_the_three_rules():
    vocabulary = ("one two three four five six seven eight nine ten eleven twelve thirteen twenty thirty forty fifty "
                  "ninety hundred thousand million and point oh zero first second third fourteenth twentieth "
                  "twenty-one forty-five percent dollars dollar cents cent pm am p.m. a.m. 3 30 12 5 4821 3.30pm "
                  "10am 330 3.30 number pull request pr issue ticket at dot com ai slash colon https example priya "
                  "sharma devxlabs to email November March May friday the of in on a an ok api ci readme sql 14th "
                  "$5 #12 o'clock minutes days seconds times version step page , . ? ! ( ) - 12:30 1,200 2.5 50% "
                  'co in org uk twenty-first "quoted five" `code three` github.com docs.devslabs.ai 4800 21 4000 eighteen '
                  'twenty-two orb setup').split()
    vocabulary += ['"quoted', 'five"', "(five", "dollars)", "twenty,", "pm.", "a.m,"]
    rnd = random.Random(20261023)
    for _ in range(6000):
        spoken = " ".join(rnd.choice(vocabulary) for _ in range(rnd.randint(1, 14)))
        once = written(spoken)
        assert written(once) == once, (spoken, once)
        assert added_words(spoken, once) == [], (spoken, once)


def test_text_in_quotes_and_backticks_is_never_touched():
    assert written('He said "three thirty pm at ten dollars" and ten dollars') == \
        'He said "three thirty pm at ten dollars" and $10'
    assert written("run `five dollars` or “twenty percent” then twenty percent") == \
        "run `five dollars` or “twenty percent” then 20%"


def test_whitespace_and_line_breaks_stay():
    assert written("Tuesday.\n No wait.\n Wednesday at 3 30 p.m.") == "Tuesday.\n No wait.\n Wednesday at 3:30 PM."
    assert written("pull   request  number 5") == "pull   request  #5"


def test_it_is_fast_enough_for_the_turn_path():
    text = ("So the plan is to review pull request number four thousand eight hundred and twenty-one before the "
            "standup at three thirty p.m. and then send the report to priya dot sharma at devxlabs dot ai, because "
            "the invoice for twelve thousand four hundred fifty dollars is due on November fourteenth.")
    assert len(text.split()) >= 40
    for _ in range(20):
        written(text)
    runs = []
    for _ in range(200):
        t0 = time.perf_counter()
        written(text)
        runs.append((time.perf_counter() - t0) * 1000)
    runs.sort()
    assert runs[int(len(runs) * 0.95)] < 5.0, f"p95 {runs[int(len(runs) * 0.95)]:.2f} ms, median {statistics.median(runs):.2f} ms"
