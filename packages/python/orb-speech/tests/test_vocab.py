"""orb_speech.vocab: the user's own terms put back into a transcript, with no false replacements."""
import itertools
import random
import re
import time

import pytest

from orb_speech import vocab
from orb_speech.vocab import VocabularyError, correct

TERMS = ("useEffect", "userId", "fetchUserProfile", "parseConfig", "TypeScript", "Redis", "Postgres", "Datadog",
         "OAuth", "Kubernetes", "Terraform", "README", "GitHub", "Sanjay", "Priya Sharma", "devxlabs.ai", "orb.dev",
         "Grafana", "Jira", "Figma", "LiveKit", "Kokoro", "Parakeet", "Gemma", "Ollama")

# Real engine outputs (Parakeet and Whistle on stt/dictation and stt/synthetic, 2026-10-23) and what the terms make.
FIXES = [
    ("Wrap the fetch inside use effect and add user ID to the dependency array.",
     "Wrap the fetch inside useEffect and add userId to the dependency array."),
    ("Rename the function to fetch user profile and add a type script type.",
     "Rename the function to fetchUserProfile and add a TypeScript type."),
    ("The bug is in the parse config helper.", "The bug is in the parseConfig helper."),
    ("The API latency dropped after the Red's cache went live.", "The API latency dropped after the Redis cache went live."),
    ("What would you pick for the database, postgurs, or a document store?",
     "What would you pick for the database, Postgres, or a document store?"),
    ("The docs are at docs.devslubs.ai slash orb slash setup.", "The docs are at docs.devxlabs.ai slash orb slash setup."),
    ("Send the report to Priya dot Sharma at devslabs dot ai today.",
     "Send the report to priya.sharma at devxlabs.ai today."),
    ("Send it to Sanjey before lunch.", "Send it to Sanjay before lunch."),
    ("We run Kubernetis and Terra form in staging.", "We run Kubernetes and Terraform in staging."),
]

# Ordinary sentences with words that sound like terms. None of them may change.
PRECISION = """I read his note before the meeting.
Post the update to the channel.
Use it when you need it.
It says like it takes five working days.
The poster's colors were too bright.
She put the data on the dog's collar.
We should use the effect sparingly.
The user said hi.
Fetch the user a coffee.
Parse the config by hand if you must.
Type the script into the box.
The red cache of apples is gone.
He wore a red shirt to the post office.
The grass is greener on the other side.
Read me the riot act.
Give me a ring at nine.
I got hub caps for the car.
The kitchen has a parrot and a cat.
The parakeet sang all morning.
Coco and Kiko are her cats.
I'd like a gem or two.
Old llamas walk slowly.
Sanjay's sister is here.
My friend Priya is a teacher.
The terrace is on the third floor.
A jeer from the crowd.
He took a fig, then a banana.
Live, kit and caboodle.
The orb glowed in the dark.
Dev labs are open late.
Redo it before lunch.
Read is past tense here.
The graph and the fauna.
Post, guess and check.
Data, dog food and milk.
Use, effect, cause.
I said "use effect" twice.
Check `use effect` in the code.
Write to priya@example.com today.
Open https://example.com/post.
Issue #4821 is closed.
It costs $12,450 now.
Meet at 3:30 PM.
We hold our standup at nine.
Cooper ate a pasta salad.
Gamma rays are dangerous.
A red is a colour.
The oak tree fell.
Kiwi is a fruit.
This is fine.
Can you hear me clearly?
I think we should talk to the client first, and then decide.
What did you have for lunch today?
Thanks, I'll get back to you tomorrow.
The stand up comedian was funny.
Juror number three stood up.
The figure on page two.
The graph on the wall.
The liver kit from the lab.
Gem and Sanjay went home.
We need more data, not more dogs.
The leaf tree is green.
The leaves fell on the porch.""".splitlines()


@pytest.mark.parametrize("raw, want", FIXES)
def test_the_terms_fix_real_engine_outputs(raw, want):
    assert correct(raw, TERMS) == want


@pytest.mark.parametrize("sentence", PRECISION)
def test_ordinary_sentences_never_change(sentence):
    assert correct(sentence, TERMS) == sentence


# Known limits (decision 0011): two words that sound like a term, and are not both ordinary words, become the term.
KNOWN_LIMITS = [("Terra firma at last.", "Terraform at last."), ("Get hub access from the admin.", "GitHub access from the admin."),
                ("the oath of office", "the OAuth of office"),
                ("Their user ID is on the card.", "Their userId is on the card.")]  # the words of a camelCase term
# Precision first: two ordinary words that do not spell a term exactly stay, even when the speaker meant the term.
NOT_FIXED = ["What would you pick for the database, post curves, or a document store?", "look at the data dog dashboard"]


@pytest.mark.parametrize("raw, got", KNOWN_LIMITS)
def test_the_known_limits_are_what_they_are(raw, got):
    assert correct(raw, TERMS) == got


@pytest.mark.parametrize("raw", NOT_FIXED)
def test_ordinary_word_pairs_that_only_sound_like_a_term_stay(raw):
    assert correct(raw, TERMS) == raw


def test_the_precision_set_is_big_enough():
    assert len(PRECISION) >= 60


CORPUS = [raw for raw, _ in FIXES] + [want for _, want in FIXES] + PRECISION


@pytest.mark.parametrize("text", CORPUS)
def test_it_is_idempotent_and_the_only_new_words_are_terms(text):
    once = correct(text, TERMS)
    assert correct(once, TERMS) == once
    words = lambda s: {w.lower() for w in re.findall(r"[A-Za-z0-9]+", s)}
    term_words = words(" ".join(TERMS))
    assert words(once) - words(text) <= term_words


def test_no_terms_change_nothing_and_the_same_input_gives_the_same_output():
    for text in CORPUS:
        assert correct(text, ()) == text
    assert [correct(t, TERMS) for t in CORPUS] == [correct(t, TERMS) for t in CORPUS]


def test_a_term_already_in_the_text_stays():
    assert correct("Restart Redis and LiveKit.", TERMS) == "Restart Redis and LiveKit."


def test_the_term_list_limits():
    assert vocab.check_terms([" Redis ", "redis", "", "Live  Kit"]) == ("Redis", "Live Kit")
    with pytest.raises(VocabularyError, match="at most 200 terms"):
        vocab.check_terms(f"t{i}" for i in range(201))
    with pytest.raises(VocabularyError, match="longer than 64"):
        vocab.check_terms(["x" * 65])
    with pytest.raises(VocabularyError, match="4096 bytes"):
        vocab.check_terms(["y" * 60 + str(i) for i in range(70)])
    with pytest.raises(VocabularyError, match="control character"):
        vocab.check_terms(["a\x07b"])


def test_the_header_and_the_team_file(tmp_path):
    assert vocab.parse_header(None) == () and vocab.parse_header(" ") == ()
    assert vocab.parse_header("useEffect, Redis ,,LiveKit") == ("useEffect", "Redis", "LiveKit")
    f = tmp_path / "team.txt"
    f.write_text("# our words\nRedis\nC#\nLiveKit  # the voice server\n\n")
    assert vocab.load_file(f) == ("Redis", "C#", "LiveKit")
    with pytest.raises(VocabularyError, match="cannot be read"):
        vocab.load_file(tmp_path / "missing.txt")
    assert vocab.merge(("Redis",), ("redis", "LiveKit")) == ("Redis", "LiveKit")


def test_the_common_word_list_is_loaded_and_has_its_source():
    words = vocab.common_words()
    assert 12000 < len(words) < 13000 and {"use", "effect", "post", "like", "read"} <= words
    assert "redis" not in words and "kubernetes" not in words
    assert "SCOWL" in vocab.COMMON_WORDS_FILE.read_text().splitlines()[1]


def test_it_is_fast_enough_for_the_turn_path():
    rng = random.Random(7)
    terms = tuple(f"{a}{b}" for a, b in itertools.islice(itertools.product(
        ["data", "user", "fetch", "parse", "post", "red", "kube", "graf", "live", "orb"],
        ["Base", "Id", "Profile", "Config", "Gres", "Is", "Netes", "Ana", "Kit", "Dev", "Hub", "Flow",
         "Store", "Cache", "Queue", "Log", "Map", "Set", "Node", "Edge"]), 200))
    words = " ".join(PRECISION).split()
    texts = [" ".join(rng.choice(words) for _ in range(40)) for _ in range(200)]
    correct(texts[0], terms)  # the first call compiles the term list
    times = []
    for t in texts:
        t0 = time.perf_counter()
        correct(t, terms)
        times.append((time.perf_counter() - t0) * 1000)
    assert sorted(times)[int(len(times) * 0.95)] < 5.0
