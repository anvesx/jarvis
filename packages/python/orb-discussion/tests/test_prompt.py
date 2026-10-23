import re

from orb_discussion import prompt
from orb_discussion.rules import Ledger


def ledger_after_m5():
    L = Ledger("Should our assistant store passwords?")
    msgs = [("human", "We should store passwords in a JSON file.", {"claims": 1, "new_option": 1}),
            ("ai", "Hashing is safer. Why a plain file?", {"new_option": 1, "asks": 1, "disagrees": 1}),
            ("human", "No, I don't agree with you.", {"claims": 1, "disagrees": 1})]
    for who, text, yes in msgs:
        L.apply({"seq": len(L.messages), "speaker": who, "text": text}, yes)
    return L


def test_note_names_persona_job_and_quotes_the_person_without_ids():
    L = ledger_after_m5()
    d, why = L.choose(2)
    need = {"debt": d.id, "kind": d.kind, "promise": d.promise, "persona": "Examiner",
            "move": "Ask for the reason. Do not agree, and do not argue yet.", "why": why}
    text = prompt.note(L, need, "web")
    assert "Persona: The Examiner." in text
    assert "Your job now: Ask for the reason." in text
    assert '"No, I don\'t agree with you."' in text
    assert "Length: 2 to 3 sentences." in text
    assert not re.search(r"\b[DMO]\d+\b", text)


def test_voice_notes_are_shorter():
    L = ledger_after_m5()
    d, why = L.choose(2)
    need = {"debt": d.id, "kind": d.kind, "promise": d.promise, "persona": "Examiner", "move": "Ask.", "why": why}
    assert "Length: 1 to 3 short sentences." in prompt.note(L, need, "voice")


def test_system_prompt_adds_the_source_only_when_it_has_text():
    assert "<source" not in prompt.system_prompt("Topic", {"url": "https://x.test", "error": "HTTP 404"})
    s = prompt.system_prompt("Topic", {"url": "https://x.test", "title": "T", "text": "Body text."})
    assert '<source title="T" url="https://x.test">' in s and "Body text." in s


def test_messages_order_and_roles():
    out = prompt.messages("SYS", [{"speaker": "human", "text": "hi"}, {"speaker": "ai", "text": "hello"}], "NOTE")
    assert [m["role"] for m in out] == ["system", "user", "assistant", "system"]
    assert out[-1]["content"] == "NOTE"


def test_clean_piece():
    assert prompt.clean_piece("**Yes**, it is") == ("Yes, it is", False)
    assert prompt.clean_piece("ok [Persona") == ("ok ", True)
    assert prompt.clean_piece("## Head") == (" Head", False)


def stream(chunks):
    f = prompt.ReplyFilter()
    out = "".join(f.feed(c) for c in chunks) + f.end()
    return out, f.dropped


def test_note_ends_with_the_reply_instruction():
    L = ledger_after_m5()
    d, why = L.choose(2)
    need = {"debt": d.id, "kind": d.kind, "promise": d.promise, "persona": "Examiner", "move": "Ask.", "why": why}
    assert prompt.note(L, need, "voice").splitlines()[-1] == prompt.NOTE_END


def test_filter_drops_a_copied_note_block():
    # the real leak of 2026-10-21, cut into small chunks the way Ollama streams it
    leaked = ("Persona: The Discussion Partner. Your job is to keep the conversation moving.\n"
              "Why: The user asked a completely unrelated question.\n"
              "I think we should focus on the backend language choice instead. Have you considered the performance needs?")
    chunks = [leaked[i:i + 3] for i in range(0, len(leaked), 3)]
    out, dropped = stream(chunks)
    assert out == "I think we should focus on the backend language choice instead. Have you considered the performance needs?"
    assert [d.split(":")[0] for d in dropped] == ["Persona", "Why"]


def test_filter_drops_labels_with_blank_lines_and_odd_case():
    out, dropped = stream(["YOUR JOB NOW: Answer.\n", "Stage of the discussion: Open.\n", "Length: 1 to 3.\n\n", "When you say that, what do you mean?"])
    assert out == "When you say that, what do you mean?" and len(dropped) == 3


def test_filter_keeps_normal_replies_and_words_that_look_like_labels():
    for text in ["Why do you think a JSON file is safe enough?",
                 "Note that hashing cannot be undone.",
                 "Personally, I would use a secret manager.",
                 "Stage one is to list the options.\nThen we test them."]:
        out, dropped = stream([text[i:i + 4] for i in range(0, len(text), 4)])
        assert out == text and dropped == [], text


def test_filter_drops_a_first_line_that_is_only_a_chat_role():
    # the real leak of 2026-10-22 (llama3.2:3b)
    out, dropped = stream(["assis", "tant\n", "That is quick to set up, but is it safe?"])
    assert out == "That is quick to set up, but is it safe?" and dropped == ["assistant"]
    out, dropped = stream(["Model:\n\nYes."])
    assert out == "Yes." and dropped == ["Model:"]


def test_filter_keeps_replies_that_start_with_a_role_word():
    for text in ["User passwords must be hashed.", "Assistants can log in for them.", "System design comes first.",
                 "Models differ.\nassistant"]:
        out, dropped = stream([text[i:i + 3] for i in range(0, len(text), 3)])
        assert out == text and dropped == [], text


def test_filter_holds_only_a_short_start():
    f = prompt.ReplyFilter()
    assert f.feed("Why") == ""            # could still be "Why:"
    assert f.feed(" not") == "Why not"    # it is not: released at once
    assert f.feed(" use a vault?") == " use a vault?"


def test_filter_stops_at_a_bracket_and_strips_markdown():
    out, _ = stream(["**Yes**, it is", " safe [Persona: x]", " more"])
    assert out == "Yes, it is safe "
