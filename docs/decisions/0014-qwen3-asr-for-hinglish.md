# 0014 · Qwen3-ASR 1.7B as a per-session engine for Hinglish; Parakeet stays the default

Date: 2026-10-26 · Status: accepted · Ticket: STT-5 (docs/design/stt-accuracy.md)

## Why
The owner needs Hinglish now (2026-10-24). Parakeet v3 knows 25 European languages and no Hindi. The TDD asked for a
spike: measure Qwen3-ASR 1.7B and Granite Speech on the bench, then keep Parakeet or add an engine.

## Measured (Apple M4, 16 GB, 2026-10-26; nothing else loaded)
Script: `bench/scripts/stt5_spike.py` in the research folder, one engine per process. Dictation: clean clips of
af_heart, hf_alpha and hm_omega (120), through orb's vocab (team list) and written form, as the service does; Qwen3
gets the team terms as its context prompt. EdAcc: the first 60 South Asian clips. Hinglish: 40 clips (31 speakers or
videos, 3–15 s, 389 s) of `agarwalayushi/hinglish` (CC BY 4.0), references in mixed Devanagari and Latin script.

| Engine | Dictation WER / fwer / entity | EdAcc South Asian WER | Hinglish CER / WER | ms p50 (English) | Weights |
|---|---|---|---|---|---|
| Parakeet TDT 0.6B v3 (torch, MPS fp32) | 13.5 / 24.6 / 57.6 % | 7.6 % | 90.1 / 96.7 % | 136 | 2.5 GB |
| **Qwen3-ASR 1.7B (MLX bf16)** | **7.9 / 20.2 / 65.3 %** | **6.7 %** | **47.8 / 51.7 %** | 795 | 4.1 GB |
| Granite Speech 4.1 2B and 5.0 2B | did not run | — | no Hindi | — | about 4 GB |

- **Parakeet cannot write Hinglish.** It writes the Hindi words as English-sounding nonsense ("Isha can may
  Abjukarnachate…").
- **Qwen3's Hinglish is good; most of its CER is script.** It writes many English words in Devanagari ("नेस्टेड"
  where the reference has "nested"), and at least one reference is cut short. A script-blind score needs
  transliteration and was not measured.
- **Qwen3 is slower:** about 0.19 s per second of English audio and 0.5 s per second of Hinglish (Devanagari needs
  more tokens): 4.4 s for an 8.8 s Hinglish piece, against Parakeet's 0.14–0.25 s.
- **Granite did not run.** Both 4.1 and 5.0 fail in the projector of transformers 5.20
  (`modeling_granite_speech.py:119`, a view of the Q-Former output) on real audio. It has no Hindi, and Qwen3 already
  beats Parakeet on English, so it was not chased further.

## Decision
- **Parakeet stays the default.** It is 6 times faster on English, and the turn's first audio depends on it.
- **Qwen3-ASR is a new engine, `qwen3`, chosen per session** (the call page's STT list, or `ORB_STT_ENGINE=qwen3`).
  `Qwen3STT` in `stt.py`, through `mlx-qwen3-asr` 0.4.4 (in the `speech-lm` extra), weights pinned and checked in
  `models.py`. It is a heavy engine: loading it unloads Parakeet.
- **No language is passed to the model.** The voice worker sends `en` for every engine, and forcing English would
  turn Hinglish into English. The vocabulary goes in as Qwen3's context prompt, then orb's correction pass runs as for
  every engine.
- Whistle still writes the interim words. On Hinglish they are rough, but the final text is Qwen3's.

## Rejected
- **Qwen3 as the default.** Better words everywhere, but 795 ms p50 on English against 136 ms, and 4.1 GB of weights.
- **A language router (Parakeet for English, Qwen3 for Hindi, per piece).** It needs a language-ID step before the
  engine, and both models resident (6.6 GB). Revisit when Hinglish users are many.
- **The 0.6B Qwen3-ASR or a 4-bit build** to cut the time. Not measured; a smaller build is a change of `QWEN3_REPO`
  and one bench run.

## Consequences
- A Hinglish session waits about 1–4 s more for each final transcript. The agent's reply starts that much later.
- The polish (decision 0013) gets Devanagari text; its guard compares words, so it keeps the input when the model
  transliterates. Not measured on Hinglish.
- `python -m orb_speech.models --all` fetches the 4.1 GB weights.
- **Qwen3 loads when the call starts (2026-10-26, owner's live test).** It loaded on the first words instead: 77 s with
  the Mac in swap, so the first final timed out and "Yeah." was lost. The worker now sends one silent call at the start
  for any local engine and waits up to 120 s; the call page shows "Loading speech model… wait to speak", and a clear
  message when it fails.
