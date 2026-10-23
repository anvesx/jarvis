"""The written-form metrics (fwer, entity, punct_f1, added_words) and the dictation sentences."""
import pytest

from orb_bench.dictation import DICTATION
from orb_bench.metrics import (added_words, entity_hits, f1, fscore, punct_counts, punct_f1, row_scores, score,
                               written_tokens)

CLASSES = {"number", "time", "date", "money", "percent", "ordinal", "email", "url", "identifier", "command",
           "domain", "correction", "list", "plain"}


def test_written_tokens_split_marks_but_keep_entities_whole():
    assert written_tokens("Hi, priya.sharma@devxlabs.ai at 3:30 PM (#4821).") == [
        "Hi", ",", "priya.sharma@devxlabs.ai", "at", "3:30", "PM", "(", "#", "4821", ")", "."]
    assert written_tokens("git checkout -b feature/login-flow --watch; $12,450.30") == [
        "git", "checkout", "-b", "feature/login-flow", "--watch", ";", "$", "12,450.30"]


def test_fwer_sees_case_punctuation_and_number_format_that_wer_ignores():
    ref = "Book it for 3:30 PM."
    assert score(ref, "book it for 3:30 pm")[0] == 0
    assert fscore(ref, "book it for 3:30 pm") == (3, 6)  # Book, PM and "." differ
    assert fscore(ref, ref) == (0, 6)
    assert score("Review #4821 now.", "Review number four thousand eight hundred and twenty-one now.")[0] > 0
    assert fscore("Review #4821 now.", "Review #4821 now.") == (0, 5)


def test_entity_counts_exact_keys_not_parts_of_longer_words():
    hyp = "Review #4821 at 3:30 PM, run git checkout -b x and ask about Redis's cache."
    assert entity_hits(["#4821", "3:30 PM", "-b", "Redis"], hyp) == (4, 4)
    assert entity_hits(["redis"], hyp) == (0, 1)  # the case must match
    assert entity_hits(["4821"], "ticket 14821") == (0, 1)
    assert entity_hits(["3:30"], "at 13:30") == (0, 1)
    assert entity_hits(["-b"], "run --build") == (0, 1)
    assert entity_hits([], "anything") == (0, 0)


def test_punct_f1_aligns_words_first_so_a_wrong_word_does_not_move_a_mark():
    ref = "Hey team, quick update. The API dropped."
    assert punct_counts(ref, "Hey team, quick update. The API dropped.") == (3, 0, 0)
    assert punct_counts(ref, "Hey team quick update. The API dropped,") == (1, 1, 2)
    assert punct_counts(ref, "Hey team, quick updates. The AP dropped.") == (3, 0, 0)  # wrong words, right marks
    assert punct_f1(ref, ref) == 1.0 and punct_f1("no marks here", "none either") is None
    assert f1(1, 1, 2) == pytest.approx(0.4)


def test_added_words_are_content_words_in_neither_the_input_nor_the_reference():
    assert added_words("use effect and user id", "useEffect and userId and bananas", "Fix useEffect") == ["bananas"]
    assert added_words("number four thousand", "#4821", "pull request #4821") == []  # digits and symbols are not content
    assert added_words("poster's migration", "Postgres migration", "The Postgres migration") == []  # the reference vouches
    assert added_words("the third time", "the 3rd time", "the 3rd time") == []
    assert added_words("hello world", "hello world", "") == []


def test_a_change_of_case_or_joining_is_not_an_added_word():
    assert added_words("Send it to DevSlabs.aI", "Send it to devslabs.ai", "Send it to devxlabs.ai") == []
    assert added_words("docs.devslabs.ai slash orb", "docs.devslabs.ai/orb", "docs.devxlabs.ai/orb") == []
    assert added_words("priya dot sharma", "priya.sharma", "") == []
    assert added_words("Send it to DevSlabs.aI", "devslabs.ai bananas", "") == ["bananas"]
    assert added_words("hello", "HELLO World", "") == ["world"]


def test_row_scores_gives_written_metrics_only_to_written_rows_with_a_reference():
    plain = row_scores("We should store passwords.", "we should store passwords", written=False)
    assert plain == {"e": 0, "n": 4}
    w = row_scores("Review #4821.", "Review #4821.", written=True, keys=["#4821"])
    assert w == {"e": 0, "n": 2, "fw": [0, 4], "punct": [1, 0, 0], "entity": [1, 1]}
    assert "entity" not in row_scores("Sure.", "Sure.", written=True, keys=[])
    assert row_scores("", "noise words", written=True) == {"e": 0, "n": 0}


def test_dictation_sentences_cover_every_class_with_exact_keys():
    assert len(DICTATION) == 40 and {d["cls"] for d in DICTATION} == CLASSES
    for d in DICTATION:
        assert d["spoken"].strip() and d["ref"].strip(), d
        assert entity_hits(d["keys"], d["ref"]) == (len(d["keys"]), len(d["keys"])), d  # every key is in its own ref
        assert d["spoken"] != d["ref"] or d["cls"] in ("plain", "domain"), d
    assert sum(1 for d in DICTATION if "\n" in d["ref"]) == 3
