# 0010 · The written-form formatter uses orb's own rules, not NeMo ITN

Date: 2026-10-23 · Status: accepted and built (ticket STT-2) · Extends: the TDD `docs/design/stt-accuracy.md`, section 4.3

## Why
Parakeet writes what it hears: "pull request number 4821", "3 30 pm", "priya dot sharma at devxlabs dot ai". A person
writing the same line would type `#4821`, `3:30 PM` and `priya.sharma@devxlabs.ai`. The TDD asks for a deterministic
formatter between the engine and the agent (inverse text normalization, ITN). Its first step was to try NVIDIA's
`nemo_text_processing` English ITN and keep it only if it is clearly better than rules that we write ourselves. The
formatter must also be pure, idempotent, add no content word and take at most 5 ms (p95) for 40 words.

## Measurement
Machine: Apple M4, 16 GB, macOS 26.5.2, 2026-10-23. NeMo-text-processing 1.2.0, pynini 2.1.7 (conda-forge, osx-arm64),
Python 3.10, in a throwaway conda env (now removed). Input: 16 Parakeet outputs and 2 more lines for the synthetic
clips (`bench/results`), and 28 sentences of about 10 words with the ITN classes of the TDD.

| | NeMo ITN | orb's own rules (`orb_speech/textfmt.py`) |
|---|---|---|
| Install | pynini has no macOS pip wheel; conda-forge has it (py3.10 to 3.14, 9 builds for py3.12). `pip install --no-deps` of the 3.3 MB wheel, because it pins `pynini==2.1.6.post1`. The env grew from 386 MB to 517 MB (+131 MB; the package is 22 MB, the rest is pandas, numpy and their dependencies) | none (`re` only) |
| First load | import 4.7 s, then 2.9 s to compile the grammar (no cache) | import 4.4 ms |
| Warm load | import 0.9 s, then 1.8 s to read the 2.1 MB grammar cache (FAR file) | 0 |
| Per call, 10 words | median 4.2 ms, p95 6.5 ms (28 sentences) | 0.02 ms (9 words) |
| Per call, 58 words | median 30.4 ms, p95 41.6 ms, max 58 ms (8 runs, cached) | median 0.13 ms, p95 0.14 ms, max 0.23 ms (5,000 runs) |
| Idempotent, pure | not tested | yes: 556 tests, among them a fuzz of 6,000 random sentences |

NeMo on our classes:

| Spoken | NeMo gives | Rules give |
|---|---|---|
| pull request number 4821 | unchanged | pull request #4821 |
| number four thousand eight hundred and twenty-one | `number 4800 and twenty-one` (wrong) | #4821 |
| 3 30 pm / 3 30 p.m. | unchanged | 3:30 PM |
| three thirty p.m. | `03:30 p.m.` | 3:30 PM |
| ten am | `10:00 a.m. .` (adds minutes, stray dot) | 10 AM |
| priya dot sharma at devxlabs dot ai | `priya.sharma at devxlabs.ai` (no `@`) | priya.sharma@devxlabs.ai |
| twelve thousand four hundred fifty dollars and thirty cents, due on March third. | `$12450.30 , due on March 3rd .` (no comma, spaces before punctuation) | $12,450.30, due on March 3rd. |
| twenty five percent | `25 %` | 25% |
| twenty twenty-six | `20 twenty-six` | 2026 (after a month or a preposition) |
| five minutes, two hundred megabytes | `5 min .`, `200 mb` (changes the words) | 5 minutes, 200 megabytes |
| api, ci, readme | `API`, but `ci` unchanged | API, CI, README |
| Look at google.com / the dot com bubble | `Look@google.com`, `The.com bubble` (damage) | unchanged |
| run git checkout dash b feature slash login dash flow | `checkout-b feature/login-flow` (joins words, a command for STT-3 and STT-4) | unchanged |
| already written: `$12,450.30`, `Gemma 4`, one of them, `3 30 pm` | unchanged | unchanged |

On the 16 Parakeet outputs of the synthetic clips, NeMo changed two rows, and both changes were damage ("4800 and
twenty-one", "checkout-b"). It fixed none of the 6 rows with an ITN class.

## Decision
- **Orb's own rules only.** There is no optional NeMo import. A second engine that runs first would need a guard
  against its own damage, would add 30 ms for a long turn, and would add a dependency that pip cannot install on macOS (pynini comes from conda-forge).
  The spike env was removed with the cache.
- **The rules** (`packages/python/orb-speech/src/orb_speech/textfmt.py`, LLD section 2) are one left-to-right scan
  over the words of a text. At each word the first matcher that fits wins: email, URL, date, label (`#N`), time, number,
  case. A number run that is not one valid number is left as it is.
- **Digits** for 10 and more, for any number with money, percent, a time or metric unit, a decimal or a label
  (`version`, `step`, `page`), and in `PR`, `pull request`, `issue`, `ticket`, `bug` and `number` plus a number.
  "one of them" and "a second look" stay words.
- **Properties of TDD 4.1 are tested:** pure, idempotent, and no content word added (only digits, the symbols
  `# @ : . / $ %`, the ordinal suffix and AM/PM). The tests run all three on every fixture and on a fuzz.

## Consequences
- **Latency:** 0.1 ms for 58 words, so the stage adds nothing to the turn path. The TDD target (p95 ≤ 5 ms for 40
  words) is met 40 times over.
- **We own the rules.** Each new class is a few lines and a test, but a language other than English is not covered.
  The formatter is English only, like the TDD asks. Hindi or Hinglish output is passed on unchanged.
- **Known limits** (LLD section 7): a small number after a name ("GPT four") stays a word; "a hundred" stays; "twenty
  twenty" outside a date or a preposition stays; "number one" stays; digits for "3 30" need `am` or `pm` after them.
- **Revisit when** a class appears that needs real grammar (languages, fractions, long measure phrases). Then NeMo
  can be tested again on that class, behind its own off switch.

## Follow-up (2026-10-23): measured on the dictation corpus
- **Added after the first corpus run:**
  - spoken ids read in pairs after a marker ("issue number three eighteen" → `#318`);
  - an id that Parakeet split ("number 4800 and 21" → `#4821`);
  - URL paths after a domain ("docs.example.ai slash orb slash setup" → `docs.example.ai/orb/setup`);
  - an email after "my email is".
- **Result, Parakeet on `stt/dictation`** (format run 20261023-125403): `fwer` 30.2 → 23.1 %; `entity` 29.0 → 51.2 %;
  number 0 → 97.9 %; time 0 → 91.7 %. No content word is added on any row.
- **Style, chosen:** ordinals under 10 stay words ("the third time"), and a date keeps the cardinal ("March 3, 2027").
  The corpus refs use "3rd"; both are common written forms, so this counts as a style difference, not an error.
