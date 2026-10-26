# 0011 · Vocabulary for speech to text: a correction pass, and Whistle's keywords

Date: 2026-10-23 · Status: accepted · Ticket: STT-3 (docs/design/stt-accuracy.md)

## Why
After the written-form formatter (decision 0010), the biggest class of errors left was the team's own words. Parakeet
writes `useEffect` as "use effect", Redis as "Red's", Postgres as "postgurs", and devxlabs.ai as "devslabs.ai".
Parakeet takes no vocabulary prompt. NeMo's phrase boosting needs NeMo and an NVIDIA GPU, and users report that it
makes Parakeet's WER worse (TDD section 3). Whistle has a `keywords` option that orb did not use.

## Decision
- **Every engine gets a correction pass** (`orb_speech/vocab.py`). It runs after `clean()` and before the written
  form. A span of 1 to 3 words becomes a term when it sounds like the term (a phonetic key) and is spelled close to
  it (≤ 34 % letter edits). It must also not be an ordinary English word, and no word of the span may be extra. The
  thresholds and the rules are in the orb-speech LLD, section 3.
- **Whistle also gets the terms as `keywords`.** It is measured below. Parakeet and Cartesia get only the pass.
- **Terms come from two places:**
  - the team file, `ORB_STT_VOCABULARY_FILE`;
  - the request's `X-Orb-Vocabulary` header. The voice worker sends `ORB_STT_VOCABULARY` on every call of a session.

  `vocab=off` turns both off for one call.
- **The ordinary-word list is SCOWL size 20.** That is 12,530 words, allowed for commercial use with its notice. A
  first draft used the google-10000-english list; it was dropped because its notes ask for an LDC license for
  commercial use.
- **Precision comes before recall.**
  - Two ordinary words join into a term only when they spell it exactly and match its capitals: "use effect" →
    `useEffect`. But "red is" does not become Redis, and "like it" does not become LiveKit.
  - A case-only change is made only for terms like `useEffect`.

## Measured (Apple M4, 16 GB, 2026-10-23)
Format run 20261023-130520 on stt run 20261023-122740. The team list has 31 terms: the 18 the corpus needs, plus 13
distractors.

| | Parakeet | Whistle |
|---|---|---|
| `stt/dictation` `entity`, written → vocab + written | 51.2 → 65.5 % | 32.6 → 41.8 % |
| identifier / domain / email / url `entity` | 15 → 90 / 68 → 79 / 36 → 69 / 0 → 33 % | 0 → 69 / 32 → 32 / 3 → 8 / 0 → 4 % |
| `stt/dictation` `fwer` | 23.1 → 20.2 % | 40.1 → 37.6 % |
| EdAcc, LibriSpeech | 0 sentences changed | 0 sentences changed |
| Time of the pass | p95 0.27 ms | p95 0.27 ms |

**Whistle `keywords`, in process, on `stt/dictation`:**
- WER 35.9 → 32.7 %.
- `entity` 15.8 → 21.2 %, and 29.3 % together with the pass.
- 2 ms more per call.

On EdAcc, keywords changed 21 of 182 outputs, with the same WER (28.2 %). On LibriSpeech, 5.0 → 4.9 %.

**Precision:** 63 ordinary sentences with sound-alike words give 0 changes (`test_vocab.py`).

## Consequences
- **The STT-3 exit check is met in part.** `entity` on `stt/dictation` is 65.5 %, not the 80 % planned. The gap is in
  classes this pass cannot fix: commands, lists and self-corrections (STT-4), and date and ordinal style (decision
  0010). On its own classes (identifier, domain, email) Parakeet reaches 69 to 90 %. "No WER rise without a match"
  is met.
- **Known limits:**
  - A term that sounds like a rare English word replaces it: "the oath of office" with OAuth, "terra firma" with
    Terraform.
  - The words of a camelCase term always join: "their user ID" with userId.
  - Ordinary word pairs that only sound like a term stay: "data dog", "post curves".

  Tests pin each case, so a change of behaviour shows up.
- **Open:** sending the session topic's words as terms. The worker's topic is fixed ("Open discussion") today. A
  term list per user, learned from edits, is also open.
