# TDD: speech-to-text accuracy and written-form output

Status: draft · Branch: `feat/stt-accuracy` · Last update: 2026-10-23 · Scope: speech to text only (TTS is out of scope)

**orb should hear a person as well as Wispr Flow does, and write what they meant in written form.** Today Whistle
is the default (26.8 % WER on South Asian English), Parakeet makes less than half its errors (11.5 %), and neither
writes "PR #4821", "3:30 PM", "useEffect" or "priya@devxlabs.ai". This document plans the work, ticket by ticket,
with a measurement for every step.

## 1. What Wispr Flow does, and what we take from it
From the static study of Wispr Flow 1.6.1209 (`voice-agent-research/ARCHITECTURE.md`):
- **The key is the endpoint, not a VAD.** Punctuation does not come from pauses.
- **The server runs `transcribe → align → format`.** An ASR ensemble writes the raw text. A formatting LLM then adds
  punctuation, written forms, lists and self-corrections ("Tuesday, no wait, Wednesday" → "Wednesday"). An alignment
  step checks the LLM output against the raw text and falls back to the raw text when they differ too much.
- **Context goes with the audio.** It includes the app and its category, about 500 characters around the cursor,
  words visible on screen, IDE symbols, a personal and team dictionary, and the previous utterance. The style is set
  per app (formal … casual) and `editing_strength` (verbatim, light, medium, heavy).
- **Deterministic fix-ups run on the client** after the LLM: snippets, spacing, casing at the cursor, and no final
  period on short chat messages.

**What we take:** the three stages (raw ASR, then a deterministic formatter, then a guarded LLM), context as an input
of every stage, and the fall-back-to-raw guard. **What we leave:** the multi-vendor ensemble, and screen OCR (for now).

## 2. Evidence: where the errors are
The local benchmark of 2026-10-23 (`voice-agent-research/BENCHMARK.md`, Apple M4) ran Parakeet TDT 0.6B v3 on
dev/office dictation. It gets the words right but writes them in spoken form:

| Error class | Reference → Parakeet v3 | Stage that fixes it |
|---|---|---|
| Spoken → written (ITN) | `#4821` → "number four thousand eight hundred and twenty-one"; `3:30 PM` → "3 30 pm" | deterministic formatter |
| Symbols in commands | `git checkout -b feature/login-flow` → "dash b feature slash login dash flow" | LLM (needs context) |
| Identifiers | `useEffect` → "use effect"; `userId` → "userI" | vocabulary |
| Domain words | Redis → "Red's", Postgres → "poster's", OAuth → "oath" | vocabulary |
| Emails | `priya.sharma@devxlabs.ai` → "priaya.sharma at devslabs.ie" | formatter + vocabulary |
| Self-corrections | "Tuesday, no wait, Wednesday" kept word for word | LLM |
| Lists | "Action items 1. Update the README 2. …" | LLM |

orb's roadmap reports the same pattern: topic words heard wrong ("day or base" for "database").

**The bench cannot see these errors today.** Its WER first applies Whisper's English normalizer, which removes case,
punctuation and number formatting. So "#4821" and "number four thousand…" score the same. Ticket STT-0 adds
metrics that see written form.

## 3. Research: ways to raise accuracy, ranked
| # | Lever | Expected gain | Cost and risk | Plan |
|---|---|---|---|---|
| 1 | **Parakeet instead of Whistle for the final transcript** | 57 % fewer errors on accented speech (decision 0005) | 89–173 ms vs 17–36 ms per call; 2.5 GB fp32 (1.25 GB as bf16 under MLX, same words: BENCHMARK.md) | STT-1 (owner decision) |
| 2 | **Vocabulary** (dictionary + topic words) | fixes identifiers and domain words, the biggest error class left | Parakeet has no prompt. NeMo's GPU-PB phrase boosting supports TDT since NeMo 2.5, but users report worse WER with Parakeet as alpha rises, and Apple-Silicon support is unanswered ([NeMo #14500](https://github.com/NVIDIA-NeMo/NeMo/issues/14500), [word boosting docs](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/asr_customization/word_boosting.html)). Whistle has a `keywords` option orb does not use yet | STT-3: Whistle `keywords`, plus a phonetic correction pass after any engine |
| 3 | **Deterministic written form (ITN)** | the ITN error classes above, at about 1 ms | NeMo's WFST ITN needs pynini; pip does not support macOS, but conda-forge does, and orb already uses a conda env ([NeMo-text-processing](https://github.com/NVIDIA/NeMo-text-processing)). Parakeet already writes many numbers as digits, so ITN must not damage text that is already formatted | STT-2 |
| 4 | **Guarded LLM cleanup** | self-corrections, lists, symbols in commands, fillers | LLMs used for ASR correction hallucinate 3–12 % of the time, while small specialised models hallucinate 0 % ([arXiv 2405.15216](https://arxiv.org/html/2405.15216)). Prompting a model on the error text alone can add new errors ([EMNLP 2025 Industry](https://aclanthology.org/2025.emnlp-industry.110.pdf)). So the guard is not optional. It also adds latency (target ≤ 300 ms) | STT-4 |
| 5 | **A newer ASR model** | Qwen3-ASR-1.7B (Apache-2.0, 30 languages including Hindi, a free-form context prompt for vocabulary, MLX ports). Granite Speech 4.1 2B (5.33 avg WER on the Open ASR set vs Parakeet v3's 6.32). Canary-Qwen 2.5B (English only) | 2–4× the size of Parakeet, slower, and Hindi was its weakest language on FLEURS (16.7–17.4 %). Leaderboard figures are from third-party roundups and are not yet checked here ([Qwen3-ASR report](https://arxiv.org/html/2601.21337v1), [mlx-qwen3-asr](https://github.com/moona3k/mlx-qwen3-asr/), [leaderboard roundup](https://www.codesota.com/speech/stt-leaderboard)) | STT-5 spike: measure, then decide |
| 6 | VAD gating | stops text from noise (Parakeet: 29 % of noise clips) | already done by Silero in the worker | none |

## 4. Design

```
audio piece ─▶ ASR engine (whistle | parakeet | …)              raw text           [today]
           ─▶ clean(): fillers, stray quotes                                         [today]
           ─▶ vocab.correct(text, vocabulary)                  deterministic        STT-3
           ─▶ textfmt.written(text, style)                     deterministic        STT-2
           ─▶ {"text", "raw", ...}  ──▶ agent (turn path, preflight == final)
                     │
                     └─(async, final only)─▶ polish (LLM, guarded) ─▶ captions / history / dictation   STT-4
```

### 4.1 Rules that every stage keeps
- **Deterministic stages are pure functions of `(text, vocabulary, style)`.** The worker's preflight transcript must
  equal the final one, or LiveKit drops the early reply. So the turn path uses only deterministic stages.
- **No stage adds a content word** that is not in the raw text, the vocabulary or a fixed table (digits, symbols,
  `@`, `#`, `:`). A test checks this on every corpus row.
- **Every stage is idempotent:** `f(f(x)) == f(x)`. That way text that is already formatted stays as it is.
- **The raw text is always kept.** The response carries `raw` next to `text`, and the bench scores both.
- **Off switch per stage.** `format=raw|written` and `vocabulary` are optional request parameters. The defaults come
  from settings, so the bench can measure every combination.

### 4.2 Speech-service API (orb-speech)
`POST /v1/stt?...&format=written|raw[&vocabulary_id=]` with header `X-Orb-Vocabulary: <comma list>`. The header is
used because the body is PCM. Limits: at most 200 terms and 4 KB, checked before any model call. The answer adds
`raw` and `stages` (the stages that ran, with ms each). Settings: `ORB_STT_FORMAT` (`written`) and
`ORB_STT_VOCABULARY_FILE` (one term per line, the team dictionary).

### 4.3 The deterministic formatter (`orb_speech/textfmt.py`, STT-2)
- **Engine: orb's own rules** (decision 0010). NeMo ITN installs from conda-forge, but on our classes it fixed none of
  the 6 fixture rows with an ITN class, it damaged two rows ("4800 and twenty-one", "checkout-b"), and it took 30 ms
  for 58 words. The rules take 0.13 ms for 58 words. There is no optional NeMo import.
- **API:** `written(text, style="default") -> str`, pure, idempotent, no content word added. The speech service adds
  `format=written|raw` and the answer's `raw` and `stages` (a map of stage name to ms: `asr`, `written`).
- **Classes:**
  - cardinals and ordinals, in digits for 10 and more and for numbers with units, money, time, a decimal point or a
    label word (`version 3`, `step 2`). "one of them" stays;
  - "number N" / "PR (number) N" / "pull request (number) N" / "issue N" / "ticket N" → `#N`;
  - times ("3 30 pm", "three thirty p.m.", "3.30pm" → "3:30 PM"). A time needs an am/pm marker;
  - dates ("November fourteenth" → "November 14th") and the year that follows;
  - money (`$12,450.30`) and percent (`25%`);
  - emails ("x dot y at z dot ai" → `x.y@z.ai`, only when the name is email-like or follows a cue word and a valid
    domain follows);
  - URLs and domains ("example dot com slash docs" → `example.com/docs`);
  - casing of AM/PM, OK and 52 acronyms.
- **Never touched:** text in quotes or backticks, numbers that are part of a name ("Gemma 4"), and text that is already
  written form. Symbols in spoken commands ("dash b") and identifiers ("use effect") stay for STT-3 and STT-4.
- **Measured** (LLD section 6, 2026-10-23, Apple M4): on the recorded outputs for the 16 synthetic clips of 9 research
  models, case- and punctuation-sensitive token error 32.8 % → 27.7 %, and for Parakeet v3 alone 31.6 % → 21.9 %.
  The exit check on `stt/dictation` still needs STT-0's metrics.

### 4.4 Vocabulary (`orb_speech/vocab.py`, STT-3)
- **Sources, merged:**
  - the team file (`ORB_STT_VOCABULARY_FILE`);
  - the request's terms (the worker sends the session's topic words and names);
  - later, a per-user list learned from edits.
- **Whistle:** passes the terms as `keywords`.
- **Every engine:** a correction pass over word n-grams (1–3 words) of the text. A span is replaced by a term when all
  of these hold:
  - its phonetic key (Double Metaphone) matches;
  - the normalised edit distance is ≤ 0.34;
  - the span is not itself a common word, unless the term is multi-word ("use effect" → `useEffect`, "poster's" →
    Postgres).
- **Case and joiners:** the term's own case is used. CamelCase and dotted terms join the span's words.

### 4.5 Guarded LLM polish (`orb_speech/polish.py`, STT-4)
- **Runs on the final transcript only, off the turn path.** The agent already has the deterministic text. The polished
  text updates the caption and the stored history, and becomes the output of a future dictation mode.
- **Inputs:**
  - the deterministic text;
  - the vocabulary;
  - the previous agent reply (≤ 500 chars);
  - style and `editing_strength` (verbatim | light | medium), as Wispr has them.
- **Jobs:** self-corrections, lists, symbols in spoken commands, false starts, fillers beyond `clean()`, and
  punctuation of long run-ons.
- **Model:** a small non-thinking instruct model, resident, through Ollama. STT-4 benchmarks the candidates: Qwen2.5
  1.5B/3B instruct, Llama 3.2 3B, Gemma 3 1B/4B, and the installed gemma4:e2b. gemma4:e2b is a weak fit, because it
  writes up to 700 hidden tokens on short lines (roadmap).
- **Guard (the align step):**
  1. Word-align the deterministic text with the polished text.
  2. Reject the polished text if any of these hold:
     - it has a content word that is in neither the input nor the vocabulary;
     - word-level divergence > 0.35 (medium) or > 0.15 (light);
     - the time is over 400 ms.

  A rejection keeps the deterministic text and logs the reason (no words), like Wispr's `format_skip_reason`.
- **Skipped** for 3 words or fewer, and for `editing_strength=verbatim`.

### 4.6 Engine choice (STT-1)
- **Default engine:** `ORB_STT_ENGINE=parakeet`, if the owner agrees. Decision 0007 chose Whistle for speed. The bench
  numbers from STT-0 go into a new decision record either way.
- **Option, an interim/final split:** Whistle for the interim captions every 0.3 s (cheap), Parakeet for the preflight
  and the final (accurate). This needs `LiveSTT` to take two engines. Preflight and final both use Parakeet, so they
  stay equal.
- **Option, a lighter Parakeet:** Parakeet through MLX in bf16 (1.25 GB instead of 2.5 GB, same words in
  BENCHMARK.md). It is measured here for latency, and in parallel with Kokoro on the GPU.

## 5. Measurement (STT-0 builds it first)
- **New corpus group `stt/dictation`.** It has 40 written-form sentences across the classes of section 2, read by 6
  Kokoro voices (2 Hindi voices reading English), clean and at 10 dB pink noise. Each sentence has `ref` (written
  form), a `keys` list (entities) and the call path codec, as the synthetic group has. It is git-ignored like the rest.
- **New metrics, beside the normalized WER:**
  - `fwer`: WER on case- and punctuation-sensitive tokens, with no normalizer;
  - `entity`: the share of `keys` written exactly;
  - `punct_f1`;
  - `added_words`: content words in the output that are in neither the raw ASR text nor the reference (the
    hallucination rate of a stage);
  - `ms` per stage.
- **The `format` suite is offline.** It re-scores the stored rows of an `stt` run through any stage combination,
  without running ASR again. That makes it cheap enough to run after every change.
- **Baseline:** `stt` runs for whistle and parakeet over all groups. These are the numbers each ticket must beat.
- **Status (STT-0, 2026-10-23):** built and measured. The test audio lives in `orb-data/testbench` next to the
  worktrees (`ORB_TESTBENCH_DIR`), so every ticket shares it. The commands are in orb-bench's LLD.

### Baseline (run 20261023-122740)
Measured on 2026-10-23, Apple M4 with 16 GB, speech service on 127.0.0.1:8200, `python -m orb_bench run stt`,
2148 clips, 5 min 25 s. No clip failed. Whistle and Parakeet are the engines as they ran on `feat/stt-accuracy`
(no formatter, no vocabulary). The `format` suite with `raw` re-scores the same run to the same numbers (run
20261023-123334).

**Normalized WER (%),** Whisper's English normalizer, as in decision 0005:

| Corpus | Whistle | Parakeet |
|---|---|---|
| EdAcc South Asian (142 clips) | 26.8 | 11.5 |
| EdAcc US control (40) | 33.4 | 14.8 |
| LibriSpeech (100) | 5.2 | 2.4 |
| synthetic, all (288) | 14.3 | 5.0 |
| synthetic, clean / 5 dB (72 each) | 6.2 / 27.2 | 3.0 / 9.3 |
| **dictation, all (480)** | 36.1 | 19.8 |
| dictation, clean / 10 dB (240 each) | 32.2 / 40.1 | 17.8 / 21.7 |

**Written-form metrics on `stt/dictation` (%).** `fwer` falls and `entity` and `punct_f1` rise when a stage helps.

| Metric | Whistle | Parakeet |
|---|---|---|
| fwer, all (clean / 10 dB) | 45.8 (44.8 / 46.9) | 30.2 (25.9 / 34.5) |
| entity, all (clean / 10 dB) | 15.8 (14.2 / 17.4) | 29.0 (35.8 / 22.2) |
| punct_f1, all (clean / 10 dB) | 68.1 (68.1 / 68.2) | 75.5 (78.7 / 72.2) |
| fwer / punct_f1 on `stt/synthetic` (its refs are written sentences, without keys) | 18.1 / 84.9 | 7.1 / 96.2 |

**Invented text on the 24 noise clips (%):** Whistle 45.8, Parakeet 29.2. The 29.2 matches the 29 % of section 3,
row 6.

**Time per call (ms, p50 / p90, as the service reports it):**

| Corpus | Whistle | Parakeet |
|---|---|---|
| dictation | 54 / 77 | 189 / 257 |
| EdAcc | 46 / 107 | 155 / 312 |
| LibriSpeech | 74 / 146 | 179 / 334 |
| synthetic | 45 / 79 | 204 / 321 |

These times are 1.5 to 2 times those of decision 0005 (Whistle 17 to 36 ms, Parakeet 89 to 173 ms). The machine was
shared with another agent's work and a user session during the run. This is the likely reason, and it is not tested.
Compare a later run only against a run made under the same load.

**Per class on `stt/dictation`: fwer % / entity %** (48, 36 or 24 clips per class and engine):

| Class | Whistle | Parakeet |
|---|---|---|
| command | 87.4 / 0.0 | 55.7 / 1.7 |
| correction | 86.0 / 56.2 | 75.4 / 70.8 |
| date | 17.9 / 29.2 | 17.9 / 25.0 |
| domain | 27.7 / 25.0 | 6.7 / 68.1 |
| email | 55.6 / 0.0 | 39.2 / 0.0 |
| identifier | 41.9 / 0.0 | 29.1 / 15.3 |
| list | 61.4 / 0.0 | 42.8 / 20.8 |
| money | 21.1 / 55.6 | 11.5 / 77.8 |
| number (`#4821`) | 49.5 / 0.0 | 25.7 / 0.0 |
| ordinal | 14.5 / 12.5 | 9.2 / 29.2 |
| percent | 16.7 / 66.7 | 10.1 / 66.7 |
| plain (control) | 10.9 / no keys | 2.4 / no keys |
| time | 51.0 / 0.0 | 35.9 / 0.0 |
| url | 98.2 / 0.0 | 80.4 / 0.0 |

**What the numbers say.**
- Parakeet writes the words right and the form wrong. On dictation its normalized WER is 19.8 % but its `fwer` is
  30.2 %, and `entity` is 29.0 %. The same gap on plain sentences is small (0.3 % WER, 2.4 % fwer).
- Four classes have `entity` of 0 % for Parakeet: number, time, email and url. These are the classes of the
  deterministic formatter (STT-2). Command and identifier are near 0 to 15 %: these are for STT-3 and STT-4.
- The `correction` class has 75.4 % `fwer`, because the engine keeps the text that the speaker took back. Only the LLM
  stage (STT-4) can fix this.
- The synthetic voices are Kokoro voices on a clean path. Section 8 already notes that real voices may differ.
- Limits of the metrics: the `entity` check is an exact, case-sensitive match. A key found in the wrong place still
  counts. `punct_f1` counts a dot inside "p.m." as a mark after a word.

### STT-2 result (format run 20261023-125403, on stt run 20261023-122740)
| Parakeet, `stt/dictation` | raw | written | Whistle raw → written |
|---|---|---|---|
| normalized WER | 19.8 % | 15.3 % | 36.1 → 32.2 % |
| `fwer` | 30.2 % | 23.1 % (−23.5 % relative) | 45.8 → 40.1 % |
| `entity` | 29.0 % | 51.2 % | 15.8 → 32.6 % |
| `entity`, number / time / date / email / url | 0 / 0 / 25 / 0 / 0 % | 97.9 / 91.7 / 50 / 36.1 / 0 % | |
| `added_words` (rows) | | 0 | 0 |
| normalized WER, edacc / librispeech | 12.3 / 2.4 % | 12.3 / 2.4 % (unchanged) | 28.3 / 5.2 % (unchanged) |

- **The exit check is met except the `fwer` target.** `fwer` fell by 23.5 % relative, not the 30 % planned.
- **The rest of the gap is outside the formatter's classes.** It is self-corrections, lists, commands and identifiers
  (STT-3, STT-4), and misheard names in emails and URLs ("devslabs.ai": STT-3's vocabulary).
- **Two misses are style choices, not errors** (decision 0010). The refs write "the 3rd time" and "March 3rd, 2027";
  the formatter keeps "third" and "March 3, 2027".

## 6. Tickets
**Process (changed 2026-10-23, the owner's rule):**
- All work is on one branch, `feat/stt-accuracy`, in the main checkout. There are no worktrees.
- Each ticket is one commit.
- Tickets run one at a time: the next starts only after the previous one is verified.
- A ticket is done when its exit check passes, `make test` passes, and its LLD sections are updated.
- There is one PR at the end, and only after the owner has verified the final output locally.

| ID | Ticket | Depends on | Exit check |
|---|---|---|---|
| STT-0 | Bench: `stt/dictation` group, the new metrics, the offline `format` suite, baseline runs. **Done 2026-10-23** (section 5) | none | baseline table for whistle and parakeet over all groups, saved in a run, numbers in this TDD |
| STT-2 | Deterministic formatter (`textfmt.py`) + API `format` + `raw`. **Done 2026-10-23**, `fwer` −23.5 % of the −30 % target (section 5) | STT-0 metrics (for the exit only) | on `stt/dictation`, parakeet: `fwer` falls by ≥ 30 % relative, `entity` rises; normalized WER on edacc/librispeech no worse than +0.2 points; `added_words` = 0; p95 ≤ 5 ms; idempotence and preflight==final tests |
| STT-3 | Vocabulary: Whistle `keywords`, phonetic correction, header + team file, worker sends topic words | STT-2 | `entity` on `stt/dictation` ≥ 80 % for parakeet with the vocabulary; no WER rise on edacc/librispeech **without** a vocabulary match; false-replacement test set (common words) passes |
| STT-1 | Engine choice: Parakeet default and/or the interim/final split; MLX bf16 Parakeet measured | STT-0 | decision record with WER, latency p50/p90 and memory; `make e2e` passes; turn suite p50 within +0.15 s of the Whistle baseline |
| STT-4 | Guarded LLM polish, off the turn path; LLM candidates benchmarked | STT-2, STT-3 | on `stt/dictation`: self-correction and list sentences fixed ≥ 70 %; `added_words` after the guard = 0; guard rejects < 20 %; p95 ≤ 400 ms; captions show the polished text |
| STT-5 | Spike: Qwen3-ASR-1.7B (MLX, with context prompt) and Granite Speech on the bench | STT-0 | decision record: keep Parakeet, or add an engine; numbers for edacc South Asian, dictation `entity`, latency and memory |

**Order, one at a time:** STT-0 → STT-2 → STT-3 → STT-1 → STT-4 → STT-5.

## 7. Open questions for the owner
1. **The default engine (STT-1).** Parakeet everywhere, the interim/final split, or keep Whistle as decision 0007
   chose?
2. **Hinglish.** Must `stt/dictation` include Hinglish speech now? Parakeet does not list Hindi. If yes, STT-5 moves
   up.
3. **Model downloads.** STT-4 and STT-5 need about 1–4 GB each of new weights over a slow network. Is that fine?
4. **Team vocabulary.** Is there a list of company names, repos and people to seed `ORB_STT_VOCABULARY_FILE`?

## 8. Risks
- **The `stt/dictation` voices are synthetic (Kokoro).** Real voices may err differently. Before the PR, the owner
  records 20–30 real clips (with consent) as a `stt/user` group.
- **Phonetic correction can replace a right word with a term.** The guardrails are the common-word rule, a precision
  test set, and the per-request off switch.
- **The LLM may still change meaning.** That is why it stays off the turn path, behind the guard, and opt-in by
  `editing_strength`.
- **Some figures are unchecked.** The third-party leaderboard numbers in section 3 are claims until STT-5 measures
  them here.
