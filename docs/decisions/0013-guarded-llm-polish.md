# 0013 · Guarded LLM polish of the final transcript: qwen2.5 1.5B behind a word guard

Date: 2026-10-26 · Status: accepted · Ticket: STT-4 (docs/design/stt-accuracy.md, section 4.5)

## Why
The deterministic stages (decisions 0010 and 0011) cannot fix what needs the meaning of the sentence: a
self-correction ("Tuesday, no wait, Wednesday"), a spoken list, the symbols of a spoken command ("dash b"), false
starts. Wispr Flow does these with an LLM after the transcript. LLMs used to correct ASR text also invent words
(3–12 % in arXiv 2405.15216), so the output needs a guard that checks the words, not the model's good will.

## Decision
- **Where:** `orb_speech/polish.py`, served as `POST /v1/polish`. The voice worker calls it after each final
  transcript, off the turn path. The agent keeps the deterministic text. The polished text replaces the caption
  (worker event `polished`). The agent's history is not rewritten: it holds what the agent answered.
- **Model:** `qwen2.5:1.5b` through Ollama (`ORB_POLISH_MODEL`), 986 MB, temperature 0, no thinking, 4 short examples
  that are not from the bench corpus. Strength `ORB_POLISH_STRENGTH`: verbatim (off), light, medium (default).
- **The guard keeps the input when the model's text:**
  - has a content word that is in neither the input nor the vocabulary (joins such as "read me" → README pass);
  - replaces or inserts more than 35 % (medium) or 15 % (light) of the input's words, or drops more than half;
  - drops a negation (not, never, can't);
  - drops the corrected value of a self-correction (the first word after "no wait", "I mean", "make that");
  - is empty, or arrives after 800 ms (400 ms until the live test below).
- 3 words or fewer are never sent.

## Measured (Apple M4, 16 GB, 2026-10-26)
Input: Parakeet rows of the baseline `stt` run (20261023-122740) through vocab (team list, 31 terms) and written form,
as the speech service gives them to the agent. Script: `bench/scripts/polish_bench.py` in the research folder.

All 480 `stt/dictation` rows (6 voices, clean and 10 dB noise):

| Measure | Target (TDD) | Result |
|---|---|---|
| Self-corrections fixed (right value in, first try out) | — | 2 → 35 of 36 |
| Lists fixed (every numbered key exact) | — | 6 → 19 of 36 |
| Corrections + lists fixed | ≥ 70 % | **75 %** (54 of 72) |
| Added words after the guard | 0 | **0** |
| Guard rejects | < 20 % | **8.8 %** (new word 25, slow 10, dropped 5, changed 1, lost correction 1) |
| Time p50 / p95 | p95 ≤ 400 ms | **202 / 342 ms** |
| fwer (case and punctuation count) | — | 20.2 → 14.4 % |

- The 17 lists counted as not fixed are numbered lines too, but the model drops "the" ("1. Release plan" for "1. The
  release plan"), or noise broke the words before the polish.
- The guard earns its place. Without the lost-correction rule, the model wrote "Book the meeting for Tuesday at 4 PM"
  (the wrong day) in 4 of 6 clean voices. Its "new word" rejections were real inventions: "Then **run**
  docker-compose up", "**actually make that** Sanjay".
- Plain sentences only gained commas ("Sure, that sounds good to me.").

## Rejected
- **The agent LLM (gemma4:e2b) for the polish.** It writes up to 700 hidden tokens on short lines (roadmap), so it
  cannot keep p95 under 400 ms, and it is 7.9 GB.
- **Polish on the turn path.** The preflight transcript must equal the final one, and an LLM is not a pure function.
- **Measuring every candidate of the TDD** (qwen2.5:3b, llama3.2:3b, gemma3:1b/4b). The smallest one met every target,
  and on a 16 GB Mac that already runs the agent LLM and Parakeet, memory is the limit. The downloads were stopped to
  keep the work short; a bigger model is a change of `ORB_POLISH_MODEL` and one bench run.

## Consequences
- One more resident model: about 1 GB more memory while Ollama keeps it (keep_alive 30 min).
- Captions can change twice: Whistle's interim words, Parakeet's final, then the polish (about 0.2 s later).
- `scripts/check_stack.sh` checks that the polish model is installed.
- **Time limit raised to 800 ms (2026-10-26, owner's live test).** With gemma4:e2b, Parakeet and the polish model
  loaded on the 16 GB Mac (10 GB of swap), every live polish took 403–424 ms and was kept as "slow". The polish is
  off the turn path, so a longer limit delays only the caption, never the agent's reply. The bench target (p95 ≤
  400 ms with nothing else loaded) is unchanged.
