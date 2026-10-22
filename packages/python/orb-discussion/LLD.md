# LLD: orb-discussion (the discussion agent)

Code: `src/orb_discussion/` · Tests: `tests/` (33) · Status: built · Last update: 2026-10-21

**The discussion agent is an AI partner that does not just agree.** Jev reads each message with yes/no
questions, fixed debt rules turn the answers into debts, the most urgent due debt picks one of six personas,
and a local LLM (Gemma) writes the reply with that persona.

| File | Job |
|---|---|
| `rules.py` | the debts, the stages, the need and the persona: pure code, no I/O |
| `questions.py` | what Jev reads for one message, and its yes/no questions |
| `engine.py` | one step: one Jev call, then the rules update the ledger |
| `personas.py`, `prompt.py` | the six personas and the LLM prompt |
| `agent.py` | `DiscussionAgent` and `DiscussionSession`: the live agent behind the `orb_core.agent` interface |
| `replay.py`, `view.py` | replay a saved chat with live Jev calls; build a page that shows every call |

## 1. Input
| Input | From | Limits | Trust |
|---|---|---|---|
| Topic | the person, at session start | 2,000 characters | untrusted; a URL in it is loaded through `orb_core.web` |
| Person message | a channel, per turn | 8,000 characters; Jev sees the first 1,500 | untrusted |
| `replace_last` flag | the voice channel | boolean | acted on only when the new text continues the last message |
| Jev answers | the TypeSafe API | one probability in [0, 1] per question | trusted as data, checked against the 0.5 line |
| LLM tokens | Ollama | a stream of text | untrusted; `ReplyFilter` drops copied note lines and markdown, and stops at "[" |

## 2. Algorithm
One turn of `DiscussionSession.turn(text, channel, replace_last)`:
1. Wait for the previous background read (`settle`). Take the session lock.
2. With `replace_last`, undo the last person turn when the new text continues it (see Maths).
3. **Read the person (one task that a cancel cannot split):** deep-copy the ledger as a checkpoint; one Jev
   request (`questions.request`); the rules apply the answers (`Ledger.apply`): pay debts, count options,
   start debts, set the stage; choose the need (`Ledger.choose`); write the message and the turn record.
4. Emit `read` and `decision` (persona, job, why, the note for the LLM).
5. Stream the reply from the LLM: system prompt, the chat, then the note, which ends with "Now reply to the
   person. Write only the words that you say to them." Every piece goes through `ReplyFilter`: a line that copies
   the note ("Persona: ...", "Why: ...") is dropped before anyone reads or hears it, and recorded in
   `turn["dropped_lines"]`. Emit `token` events.
6. Record the reply (marked `interrupted` when the stream was cut). Start the read of the reply in the
   background: one Jev request, the rules apply it, and the check says whether the reply paid the need.
7. Emit `reply`, `check` and `state`.

**What Jev reads** (`questions.earlier_in_view`): the topic, the options on the table, the open debts as
sentences, at most 12 earlier messages (the opening message, the last 4, and the messages that started an open
debt), and the new message.

**The questions:** 9 about a person message (asks, claims, gives_reason, repeats, changes_view, asks_to_close,
disagrees, new_option, names_open_point), 4 about an AI message (asks, disagrees, new_option,
names_open_point), one "does it pay D*n*?" per open debt that the writer can pay, and one "does it ask for the
reason?" after an Examiner need.

**The rules** (`rules.py`), in order for each message:
1. Pay: each `pays_Dn` answer of "yes" pays debt Dn.
2. Options: a "yes" to new_option adds an option. The second option pays the option debt.
3. Start, for a person message: asks to close → summary debt (AI owes). Changes view → record debt (AI owes).
   Else a claim with no reason → reason debt (person owes), or a move_on debt (AI owes) when the person repeats
   and a reason was already asked for once. Gives a reason → weigh debt (AI owes).
   For any message: a question → answer debt (the other side owes). A disagreement or an open point → one end
   debt (both owe). One option on the table → option debt (AI owes).
4. Stage: Open until 2 options; Test from 2 options; Close when the person asks to close, or when only closing
   debts are left in Test. The stage never goes back.
5. Need: the newest due debt that the AI can act on (the AI cannot pay an answer that the person owes), with a
   fixed order when one message starts several; else the oldest due debt; else no need (Navigator).

## 3. Maths
| Name | Value | Source |
|---|---|---|
| YES | 0.5 | chosen: the line where a Jev probability counts as "yes"; the only number in the rules |
| Stages | Open < 2 options ≤ Test; Close on request or when only Resolution debts are left | chosen (first principles: collect, test, decide) |
| Due promises | Open: Interaction, Divergence · Test: + Challenge, Reflection, Progression · Close: all six | chosen |
| Tie order | Resolution > Reflection > Progression > Challenge > Interaction > Divergence | chosen: close before you open new threads |
| Score of a promise | debts paid ÷ debts owed | first principles; the target is "due" (1) or "not due" (0) |
| Persona of a need | Interaction → Teacher, Divergence → Rival, Challenge → Examiner, Reflection → Judge, Progression → Navigator, Resolution → Chair | chosen |
| Persona temperature | Rival 0.8 · Teacher, Navigator 0.6 · Examiner 0.5 · Judge, Chair 0.4 | chosen |
| Reply length | per persona (2 to 5 sentences); voice: 1 to 3 short sentences | chosen: each spoken sentence costs listening time |
| Jev view | ≤ 12 earlier messages · 600 characters each · new message ≤ 1,500 characters | chosen: bounds the request size |
| Cost of a turn | about 1,600 + 1,200 input tokens × $0.042 per million = $0.00012 | measured 2026-10-21 (median tokens) |
| `continues(old, new)` | difflib ratio of the old words against the first len(old) words of `new` ≥ 0.6 | chosen 2026-10-22: the voice channel transcribes the whole turn again, so the first words can differ a little; a resumed turn has the same words |
| Note leak, before the fix | 4 of 20 replies copied the note on a garbled voice turn ("What kind of decomia?"); 0 of 10 on a normal turn | measured 2026-10-21, Gemma `gemma4:e2b`, temperature 0.6 |
| Note leak, with the closing line | 0 of 20 raw, 0 of 20 after the filter, on both turns; every reply still asks a question | measured 2026-10-21 through the real prompt code |
| Filter hold | a line is held while its start can still become a label: at most 24 characters ("stage of the discussion:") | the longest label |

## 4. Test cases
| Case | Expected result | Test |
|---|---|---|
| The recorded passwords run, replayed with its saved Jev answers | the same questions, events, stages, scores, needs and checks as on 2026-10-19 | `test_replay_recorded.py::test_recorded_run_gives_the_same_decisions` |
| The 7 recorded needs | Rival, Judge, Examiner, Judge, Navigator, Judge, Chair, as the old app | `test_recorded_run_personas_match_the_old_app` |
| First claim | a reason debt (waiting) and an option debt (due): the need is the option | `test_rules.py::test_first_claim_owes_a_reason_and_a_second_option` |
| Second option | pays the option debt; the stage becomes Test | `test_second_option_pays_the_option_debt_and_starts_test` |
| Repeat after a reason was asked | a move_on debt, not another reason debt | `test_repeat_after_a_reason_was_asked_means_move_on` |
| Repeat with no earlier reason debt | a reason debt | `test_repeat_without_an_earlier_reason_debt_owes_a_reason` |
| Ask to close | stage Close, a summary debt, the Chair; no answer debt | `test_ask_to_close_moves_to_close_and_calls_the_chair` |
| An answer that the person owes | the AI waits for it | `test_the_ai_waits_for_an_answer_that_the_person_owes` |
| Stage | never goes back | `test_stage_never_goes_back` |
| Change of view | a record debt, not a reason debt | `test_a_change_of_view_owes_a_record_not_a_reason` |
| The note | names the persona and the job, quotes the person, has no internal IDs | `test_prompt.py` (5 tests) |
| A full turn | events in order; the Rival; the check says the debt was paid; temperature 0.8 | `test_agent.py::test_one_full_turn` |
| Barge-in during the reply | the heard part stays, marked `interrupted`, and is read by Jev | `test_interrupted_reply_is_kept_and_read` |
| Jev fails on the person | an error event; nothing changes | `test_reader_failure_on_the_person_changes_nothing` |
| Jev fails on the reply | the reply is recorded with no answers; message numbers stay in step | `test_reader_failure_on_the_reply_keeps_the_ledger_in_step` |
| The LLM fails | an error event; the person's message stays | `test_llm_failure_ends_the_turn_with_an_error` |
| A cancel during the read | the message and its turn are still recorded; the LLM never runs | `test_cancel_during_the_read_still_records_the_message` |
| `replace_last` with a continuation | the unanswered turn is undone; one message holds the whole sentence | `test_replace_last_undoes_an_unanswered_turn` |
| `replace_last` with a new sentence | nothing is undone | `test_replace_last_ignores_a_new_sentence` |
| The note's last line | "Now reply to the person. Write only the words that you say to them." | `test_prompt.py::test_note_ends_with_the_reply_instruction` |
| The real leak of 2026-10-21, in 3-character chunks | only the real reply is left; 2 lines dropped | `test_filter_drops_a_copied_note_block` |
| Labels in capitals, with blank lines | dropped; the reply starts at its first word | `test_filter_drops_labels_with_blank_lines_and_odd_case` |
| "Why do you...", "Note that...", "Personally...", "Stage one..." | kept unchanged | `test_filter_keeps_normal_replies_and_words_that_look_like_labels` |
| A held start | "Why" is held, "Why not" is released at once | `test_filter_holds_only_a_short_start` |
| Markdown and "[" | removed; the stream stops at "[" | `test_filter_stops_at_a_bracket_and_strips_markdown` |
| A leaked reply in a full turn | the person hears only the real reply; the chat history is clean | `test_agent.py::test_a_copied_note_never_reaches_the_person` |

## 5. Security
- **Message text leaves the machine.** Each Jev request carries the new message and up to 12 earlier ones to
  the TypeSafe API. Do not use the agent on confidential text until that is cleared (see the architecture).
- **Prompt injection.** A person's text reaches the LLM as a chat message and, cut to 160 characters, as a
  quote in the note. The note says that the person does not see it and that the LLM must not mention it.
  A person can still try to steer the LLM; the debt rules do not depend on the LLM, so the persona choice and
  the checks stay correct.
- **Internal IDs** (D7, M5) are removed from the note, so the LLM cannot repeat them to the person.
- **Snapshots** hold the full chat and the Jev trace. They are written to `data/sessions/`, which is not in git.

## 6. Performance
Measured on an Apple M5, 2026-10-21, Gemma `gemma4:e2b` in Ollama, 9 turns:
| Step | Median | Max |
|---|---|---|
| Jev reads a person message (11 questions, 1,586 tokens) | 0.49 s | 0.61 s |
| Rules and prompt | < 1 ms | |
| Gemma first token | 0.34 s | 0.37 s |
| Gemma full reply | 0.95 s | 1.29 s |
| Jev reads the reply (5 questions, 1,183 tokens) | 0.35 s | 0.51 s |

The Jev read of the person is on the critical path: the persona depends on it. **Next:** start the read on
the final transcript, before the voice turn commits, and apply it only when the turn commits (saves up to the
200 ms commit wait plus the overlap with the end-of-turn decision).

## 7. Edge cases
| Case | What happens | Why |
|---|---|---|
| Two person messages in a row (the reply failed) | both are read; the ledger has both | the rules do not need turns to alternate |
| A reply cut before any token | no AI message; the need stays open | nothing was said, so nothing was paid |
| A reply cut after some tokens | the produced part is recorded as `interrupted` | the person may have heard part of it |
| The person goes on talking before any reply | the voice channel sends the whole sentence with `replace_last`; the old turn is undone | one sentence must be one message, or the rules see two claims |
| `replace_last` when the last message is not a prefix | nothing is undone | protects a real earlier turn |
| The undone read | its Jev tokens stay counted | they were paid for |
| A URL that does not load | the topic stays; `source.error` says why | the agent can still discuss the topic |
| Jev says "changes view" by mistake | a record debt starts | a known error rate of the reader (2026-10-19: 0.53 to 0.73 on "Let's decide now") |
| The LLM copies the note into its reply | the filter drops the copied lines; the person hears only the real reply | a small model copies a trailing system message, most on confusing input |
| The note is moved before the person's message, or into the system prompt | no leak, but Gemma 4 makes 300 to 600 hidden reasoning tokens on confusing input: 8 to 14 s to the first word | measured 2026-10-21; so the note stays last, with the closing line and the filter |
| A reply that is only copied note lines | the reply is empty; nothing is said | not seen in 40 samples; the next step would be one retry |

## 8. Expected results
- The web chat shows, for each person message: the persona, its job, the reason, the stage, the six promises,
  the debts and every Jev question with its answer. After each reply it shows the check.
- On the passwords chat (`data/chats/20261012-152327.json`), the personas are Rival, Judge, Examiner, Judge,
  Navigator, Judge, Chair.
- `make test` passes with no network.
