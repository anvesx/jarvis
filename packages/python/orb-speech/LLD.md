# LLD: orb-speech (local speech service)

Code: `src/orb_speech/` · Tests: `tests/` (615: 612 by default, 3 more with `ORB_TEST_TTS_LM=1`) · Status: built ·
Last update: 2026-10-23

**One small HTTP service gives the voice worker and the bench their local speech models.** Each request can name its
engine, so the bench switches models without a restart. The text of a speech-to-text call is then put in written
form (`#4821`, `3:30 PM`, `priya.sharma@devxlabs.ai`) by a fixed set of rules, unless the call asks for `format=raw`.
No audio and no text leaves the machine (except with the cloud engine `cartesia`, below).

| Kind | Engines | Default | Loads |
|---|---|---|---|
| Speech to text | `whistle` (Whistle 2.0.0, 17 MB), `parakeet` (Parakeet TDT 0.6B v3, 2.5 GB), `cartesia` (Cartesia Ink-Whisper, **a cloud service**) | `whistle` (the user's choice; decision 0005 has the accuracy) | Whistle at start; Parakeet on first use; Cartesia on first use, and only with `CARTESIA_API_KEY` (decision 0008) |
| Text to speech | `kokoro` (Kokoro-82M int8), `maya1` and `indic-speak` (3B speech LMs, 4-bit MLX) | `kokoro` | Kokoro at start; a speech LM on first use, one at a time |

## 1. Input
| Endpoint | Body | Limits | Answer |
|---|---|---|---|
| `POST /v1/stt?sample_rate=16000[&language=en][&engine=][&format=written\|raw][&vocab=on\|off]`, header `X-Orb-Vocabulary: term, term, …` | PCM16 LE mono | ≤ 60 s; rate 8, 16, 24 or 48 kHz; a language of the engine (Whistle: en de fr es it nl pl; Parakeet: 25 European languages; Qwen3-ASR: `en` or `hi`, only checked, the model finds the language; Cartesia: any ISO-639 code, about 100); `format` is `written` or `raw`, at most 16 characters, and defaults to `ORB_STT_FORMAT`; `vocab` is `on` (default) or `off`; the header has at most 8,192 bytes and, after the split at commas, at most 200 terms of 1 to 64 characters and 4,096 bytes, with no control character | `{"text", "raw", "stages", "language", "engine", "ms", "audio_ms"}`. `text` is the formatted text. `raw` is the engine's text after `clean()`. `stages` maps `asr` (the engine's ms, with `clean()`) `vocab` (ms of the vocabulary pass, only when there are terms) and `written` (ms of the formatter, only with `format=written`) to milliseconds. 400 for a bad `format`, `vocab`, vocabulary header, `sample_rate` or `language`; 502 when the engine's own service fails; 503 when the engine cannot load |
| `POST /v1/polish`, header `X-Orb-Vocabulary` as for `/v1/stt` | `{"text", "context"?, "strength"?}` | text 1 to 4,000 characters, context ≤ 4,000 (the last 500 are used), strength `verbatim`, `light` or `medium` (default `ORB_POLISH_STRENGTH`) | `{"text", "applied", "reason", "ms", "model"}`: the polished text, or the input with the reason the guard kept it (`short`, `skipped`, `new word`, `changed`, `dropped`, `negation`, `lost correction`, `slow`, `empty`, `failed`). 400 for a bad strength or vocabulary header; 503 without a polish model |
| `POST /v1/tts` | `{"text", "engine"?, "voice"?, "speed"?, "style"?, "lang"?}` | text 1 to 2,000 characters; engine ≤ 32, voice ≤ 400, style ≤ 64, lang ≤ 16 characters; then the engine's own checks (below) | a stream of PCM16 mono; headers `X-Sample-Rate` (24000), `X-Engine` |
| `GET /v1/engines` | none | | per kind: the engines, the default, the loaded ones, `cloud` (engines that send the audio to another company) and `unavailable` (engine → the reason, for example a missing key); per TTS engine: `voices`, `styles`, `langs` (known without a load) |
| `GET /health` | none | | which models are loaded |

The engine checks of `/v1/tts`:
| Engine | `voice` | `style` | `lang` | `speed` |
|---|---|---|---|---|
| `kokoro` | one of its 54 voices; Hindi default `hf_alpha` | ignored | `en-us`, `en-gb`, `hi` | 0.5 to 2.0 |
| `maya1` | a description in words, ≤ 400 characters, without `"`, `<` or `>`; 3 presets | ignored (emotion tags such as `<laugh>` go in the text) | English only | none, or 1.0 |
| `indic-speak` | one of 44 names (case does not matter); default: the female voice of `lang` | one of 14 trained tags, or none | 22 Indian languages, `en`, `en-us`, `en-gb` | none, or 1.0 |

Settings: `ORB_STT_ENGINE` (`parakeet`, decision 0012; the voice worker asks `whistle` for interim words), `ORB_STT_FORMAT` (`written`; a bad value stops the start), `ORB_STT_VOCABULARY_FILE` (the team's terms, one per line, `#` comments; a missing or bad file stops the start), `ORB_POLISH_MODEL` (`qwen2.5:1.5b`, through Ollama at `OLLAMA_URL`), `ORB_POLISH_STRENGTH` (`medium`; a bad value stops the start), `ORB_STT_LANGUAGE`, `ORB_TTS_ENGINE` (`kokoro`), `ORB_TTS_MODEL`,
`ORB_TTS_VOICE`, `ORB_SPEECH_PORT`, `ORB_MODELS_DIR`, `CARTESIA_API_KEY` and `ORB_CARTESIA_VERSION` (default
`2026-08-14`). The service reads only `CARTESIA_*` and `ORB_CARTESIA_*` keys from the repo `.env`. End-of-turn detection is not here: the voice worker runs
LiveKit's turn detector in its own process (decision 0006).

The callers are the voice worker, the agent service's bench proxy and the bench suites, on the same machine. The
service checks every limit before it touches a model: `format` is checked before the body is read, so a bad value
costs nothing. The bench's `stt` suite sends no `format`, so it now scores the written text; set `ORB_STT_FORMAT=raw`
for the old numbers.

## 2. Algorithm
- **Engines** (`engines.py`): a `Registry` per kind maps a name to a factory. `get(name)` builds the engine on first
  use, under a lock. A heavy engine (Parakeet and Qwen3-ASR; maya1 and indic-speak) counts against a limit of one per registry:
  loading another heavy engine first calls `close()` on the oldest one. A load error gives 503 with the reason.
- **One GPU user at a time** (`gpu.py`): PyTorch has one MPS command stream per process, and two threads must not use
  it at once. Every torch call on MPS (Parakeet, the SNAC and Vocos decoders) takes one reentrant lock. MLX has
  its own command queues and does not take it.
- **STT:** check `format`, `vocab` and the vocabulary header → read the body in parts and stop at the limit → PCM16
  to float32 → resample to 16 kHz (soxr) → the engine in a worker thread (Whistle also gets the terms as `keywords`)
  → `clean()`: stray quote marks at the ends and the fillers "uh", "um", "erm" removed (this is `raw`) →
  `vocab.correct()` when there are terms → `written()` unless `format=raw` (this is `text`). Both texts and the ms
  of each stage go in the answer.
- **Polish** (`polish.py`, decision 0013): skip 3 words or fewer and `verbatim` → Ollama `/api/chat` with
  `ORB_POLISH_MODEL` (`qwen2.5:1.5b`), temperature 0, no thinking, an 800 ms timeout, the strength's instruction, 4
  examples and the agent's last reply as context → `guard()`: word tokens of input and output (list markers dropped);
  reject a content word that is not in the input, the vocabulary, or a join of 2–3 input words; (replaced + inserted)
  > 35 % (medium) or 15 % (light) of the input's words; dropped > 50 %; fewer negations; a lost corrected value (the
  first content word after "no wait", "I mean", "make that"…). A rejection returns the input. The model loads at
  start unless the strength is `verbatim`.
- **Vocabulary** (`vocab.py`, `correct(text, terms)`, decision 0011): the terms are the team file's, then the
  request's, without repeats (ignoring case; the first spelling stays). The list is compiled once and cached by its
  value. For each word, the longest span of 1 to 3 words that sounds like a term becomes the term, written as given
  (`use effect` → `useEffect`, `Red's` → `Redis`, `docs.devslabs.ai` → `docs.devxlabs.ai`). Text in quotes or
  backticks, emails, URLs, `#N`, money and times are not touched. The function is pure and idempotent.
  - **Whistle:** cactus-needle, on the CPU, with the language as an input.
  - **Qwen3-ASR** (decision 0014): `mlx-qwen3-asr` with `mlx-community/Qwen3-ASR-1.7B-bf16` (4.1 GB, pinned revision,
    every file checked), on the GPU through MLX, without the torch lock. No language is passed, so Hinglish stays
    mixed; the vocabulary goes in as the context prompt. One warm-up call at load. Fetched by
    `python -m orb_speech.models --all`.
  - **Parakeet:** the feature extractor (128 log-mel bins) → `ParakeetForTDT.generate` on MPS in float32. At load,
    0.2 s and 1 s of silence compile the GPU kernels. A keep-warm thread transcribes 0.2 s of silence after 1 s
    without a call, within 120 s of the last real call.
  - **Cartesia (cloud):** the samples go into a 16 kHz mono WAV, and one `POST https://api.cartesia.ai/stt` sends it
    as a multipart upload (`model=ink-whisper`, `language`) with the headers `Authorization: Bearer <key>` and
    `Cartesia-Version`. The answer's `text` goes through `clean()`. Less than 100 ms of audio makes no call. A failed
    call raises `EngineError` with a plain reason (HTTP 401 or 403: the key; 402: no credits; 429: too many requests;
    a network error: not reachable), and the route answers 502. A missing key makes the engine fail to load with
    the fix in the message (503), and `/v1/engines` shows it as `unavailable`.
- **Written form** (`textfmt.py`, `written(text, style="default")`, decision 0010): a pure function of the text. It
  never runs on audio, never calls a model and never touches the network.
  1. **Protect.** Text in "double quotes" or `backticks` is cut out and put back as it was.
  2. **Tokenize.** Split on white space. Each token keeps its leading gap, its opening marks, its core and its closing
     punctuation, so the spacing of the text comes back as it was.
  3. **Scan once from left to right.** At each token the first matcher that fits takes the tokens it needs and
     writes one replacement; the scan goes on after them. The order is: email, URL, date, label, time, number, case.
     A matcher only joins tokens with no punctuation between them.
  4. **Numbers.** The run of number words that starts at a token is parsed with a small grammar: units, teens, tens,
     `hundred`, `thousand` to `trillion`, `and` after `hundred` or a scale, ordinals (`twenty first`, `hundredth`),
     and `point` with digits for decimals. The run must be one valid number. `twenty twenty five` and `two thirty` are
     not, so the whole run stays as it is. A number that is already digits is only read, never rewritten.
  5. **What a number becomes.** Digits when the value is 10 or more, or the number has a decimal point, or it is
     followed by money, `percent`, a time or metric unit, or follows a label word (`version`, `step`, `page`, …).
     Under 10 with none of these it stays a word. A spelled ordinal becomes digits at 10th or more, and under 10 only
     in a date.
  6. **Classes.** The table in section 3 has each rule and its limit.
  7. **Idempotence by construction.** Whatever a rule writes is never read again as something else: a number
     that is written next to `am` already reads as a time, a year next to a unit is not a year, a domain label is never
     a number, a label number is never followed by a unit. The fuzz test checks this on random text.
- **TTS, Kokoro:** check → `create_stream` yields audio per phoneme batch. For Hindi, the text is phonemized with
  espeak (`hi`, language switch flags removed), and Kokoro gets the phonemes, so the "(en)" and "(hi)" marks are
  not spoken.
- **TTS, speech LMs** (`tts_lm.py`): both models are Llama 3.2 3B fine-tunes that write SNAC 24 kHz codes, with the
  same token layout.
  1. Load in the engine's own worker thread: the LM with `mlx_lm.load`; SNAC; for indic-speak the Vocos decoder
     (`vocos.py`, the EMA weights of `vocos/best.pt`, read with `weights_only=True`). One short sentence then
     compiles the GPU kernels.
  2. Split the text into sentences, and a sentence of more than 200 characters at a comma or a space in its second
     half. 0.2 s of silence goes between pieces.
  3. Prompt: `<start_of_human> <bos>` + the voice part + the text + `<eot> <end_of_human> <start_of_ai>
     <start_of_speech>`. The voice part is `<description="…">` for maya1, and `<|speaker>` name `<speaker|>` "\n"
     [`<|style>` tag `<style|>` "\n"] for indic-speak. The KV cache of the voice part is made once per voice (8 kept)
     and trimmed back after each piece.
  4. Sample tokens (`generate_step`): maya1 temperature 0.4, top-p 0.9, repetition penalty 1.1; indic-speak 0.6,
     0.9, top-k 50, 1.2. Stop at `<end_of_speech>`, `<end_of_ai>`, the end-of-text token, the token limit, a
     stopped stream or an unload.
  5. Frames: a code token's value gives its position in the 7-code frame. A token out of place drops its frame;
     the next frame starts at the next position-0 token. Other tokens are ignored.
  6. Stream: when `first` frames (2) and then `step` frames (6 for maya1, 8 for indic-speak) are ready beyond the
     `lookahead` (2 / 4 frames), decode them with up to `context` frames before them (4 / 12) and send only their own
     samples. Decode: SNAC quantizer → SNAC decoder (maya1) or Vocos (indic-speak). maya1 never sends its first
     frame (its build drops the first 2048 samples).
  7. The async `stream()` gets the chunks from the worker thread through a queue. When the consumer stops, an event
     stops the generation at the next token.

## 3. Maths
| Name | Value | Source |
|---|---|---|
| PCM16 ↔ float | x / 32768; round(clip(x, −1, 1) × 32767) | standard |
| Word error rate (WER) | (substitutions + deletions + insertions) / reference words, after the Whisper English normalizer (English) or with Unicode punctuation and symbols removed (other languages) | `orb_bench.metrics` |
| SNAC frame | 7 codes = 1 + 2 + 4 codes of 3 levels = 2048 samples = 85.3 ms at 24 kHz | SNAC 24 kHz; the model card's token contract |
| Code token | 128266 + position × 4096 + code, position 0 to 6, code 0 to 4095 | token contract |
| Tokens for real time | 7 × 24000 / 2048 ≈ 82 tokens per second | from the frame size |
| Token limit of a piece | min(2520, (2 × characters / 12 + 3) × 82) | about twice the expected length at 12 characters per second |
| Vocos length | 512 samples per latent step, 4 latent steps per frame | `vocos.py` (upsample 4 × hop 128) |
| First audio of a speech LM | prefill + (first + lookahead) × 7 tokens: (2 + 2) × 7 = 28 tokens for maya1, (2 + 4) × 7 = 42 for indic-speak | the chunk rule |
| Real-time limit of a 3B 4-bit LM on this Mac | 1.86 GB of weights read per token; measured 57 to 59 tokens per second, so real-time factor ≥ 82 / 58 ≈ 1.41 | `raw_speed.py` run, 2026-10-22 |
| TTS part size | 4,800 bytes = 100 ms at 24 kHz | chosen |
| Cartesia price | 1 credit for each 2 s of audio, silence included | Cartesia's pricing page, read 2026-10-22; the value of a credit depends on the plan |
| Audio that a growing piece costs | with an interim call every c s over T s of speech: about T² / (2c) + T s (the interim calls, and the preflight) | the worker sends the whole piece each time. For T = 5 s: c = 0.3 s gives 47 s (9.3 ×), c = 1.0 s gives 17.5 s (3.5 ×) |
| Written form: digits for numbers | ≥ 10; decimals; with money, percent, time and metric units (under 10 too; `one` only with percent, money or metric units, so "one day" stays); after `version step page line chapter section item port build release level row column slide question problem exercise figure table`; with `PR`, `pull request`, `issue`, `ticket`, `bug` | chosen: the rule of dictation tools (Wispr-like), checked on the fixtures |
| Written form: thousands | a comma from 1,000 (`4,821`); no comma in a year (2001 to 2099 spoken as "two thousand …"), nor after `#` | chosen |
| Written form: millions | `five million` → `5 million`; `five million two hundred thousand` → `5,200,000` | chosen |
| Written form: `#N` | `PR`, `pull request`, `issue`, `ticket`, `bug` (+ optional `number`) then a number: `#N`. A bare `number N`: digits, or a spelled N ≥ 10, unless the word before is a determiner or a name of a number (`the`, `phone`, `account`, `order`, `serial`, `of`, …). `number one` stays | chosen |
| Written form: `#N` read in pairs or split | only right after a `#` marker: pairs (`three eighteen` → `#318`, `forty eight twenty one` → `#4821`, `three oh two` → `#302`), digit runs (`one zero four two` → `#1042`), and a split id (`4800 and 21` → `#4821`, when the second part fits the zeros of the first). The same words in prose stay | chosen from the dictation corpus, 2026-10-23 |
| Written form: url path | `slash X` joins a path only right after a domain or a URL (`docs.x.ai slash orb slash setup` → `docs.x.ai/orb/setup`), and stops at the first word that is not a path part. A spoken `slash` elsewhere stays (commands are STT-4) | chosen from the dictation corpus, 2026-10-23 |
| Written form: time | hour 1 to 12 (digits or words) + optional minutes (`30`, `thirty`, `oh five`, `forty-five`, `330`, `3.30`) + `am`/`pm`/`a.m.`/`p.m.` → `H:MM AM`; `three o'clock` → `3 o'clock`. A time needs its marker: "two thirty" stays | chosen |
| Written form: date | month + spelled ordinal → `Month 3rd`; `the third of March` → `3rd`; a year after a month or a date: `November 14th, 2026`. `March`, `May` and `August` need a capital letter | chosen |
| Written form: year | `two thousand twenty six` → `2026`; `twenty twenty-six` and `nineteen oh five` only after a month or `in since until by from during before after through of around` | chosen |
| Written form: money, percent | `N dollars [and M cents]` → `$N[.MM]` (integers only); `N percent` → `N%` | chosen |
| Written form: email | `local at domain` → `local@domain` (lower case) when the domain has a valid TLD (or is already dotted) and the name has a `dot`, `underscore`, digit or dash, or the word before is a cue (`email`, `to`, `cc`, `send`, …). "Send it to Bob at the office" stays | chosen |
| Written form: url | `x dot com`, `https colon slash slash x dot com slash a slash b` → `https://x.com/a/b`. The first word is not a function word (`the`, `polka`, …) or a number, the last is one of 44 TLDs; `dot in` / `dot so` / `dot no` / `dot it` / `dot es` need three labels | chosen |
| Written form: case | `ok` → `OK`; `api ci pr readme url sql aws` and 45 more acronyms, and their plural `APIs PRs URLs`; `am` and `pm` only inside a time | chosen |
| Written form: work limit | at most 16 words in an email name, a domain or a path, so the scan is linear for real text | chosen |
| Vocabulary: when a span becomes a term | the phonetic keys (orb's own Metaphone-like key) match or differ by one edit (keys of 4 or more); the letters differ by at most 34 % of the longer side (`day or base` → database is 3 of 9 = 0.33), and terms of 4 letters or fewer must match exactly; dropping any one word of the span must not match as well (`parse the config` stays) | chosen; tuned on the dictation corpus, 2026-10-23 |
| Vocabulary: ordinary words | a one-word span that is an ordinary word (12,530 words of SCOWL size 20) never changes. A span of ordinary words only must spell the term exactly and match its spaces and capitals: `use effect` → `useEffect`, but `red is` stays (Redis), `read me` stays (README), `like it` stays (LiveKit) | chosen, from the precision set |
| Vocabulary: case only | a one-word span that differs from the term only in case changes only for a term with an inner capital or digit (`useeffect` → `useEffect`); `parakeet` stays. `Sanjay's` keeps its possessive | chosen |
| Vocabulary: limits | 200 terms and 4,096 bytes per source; terms of 1 to 64 characters; the header at most 8,192 bytes; the team file at most 65,536 bytes | chosen |
| STT body limit | 60 s × 16,000 × 2 = 1,920,000 bytes at 16 kHz | chosen |
| Shortest STT input | 100 ms; shorter gives empty text without a model call | chosen |

## 4. Test cases
| Case | Expected result | Test |
|---|---|---|
| PCM round trip; clipping; resampling | error below 1e-4; ±0.99997; 48,000 → 16,000 samples | `test_audio.py` |
| Fillers and stray quotes | removed; "umbrella" kept | `test_stt_clean_*` |
| Keep-warm | silent calls only inside the 120 s window | `test_parakeet_keep_warm_runs_only_inside_the_window` |
| STT engine per request | each engine is built on first use; an unknown engine gives 400 | `test_server.py` |
| TTS style and language | passed to the engine; an unknown engine gives 400 | `test_tts_passes_style_and_lang_and_refuses_unknown_engines` |
| Heavy engines | loading a second heavy engine unloads the first | `test_heavy_engines_unload_the_oldest` |
| Qwen3-ASR | no language passed, terms as context, `fr` refused, under 100 ms gives empty text | `test_qwen3.py` |
| Polish guard | cleanups pass; a new name or number, a lost negation, a lost corrected value, a big drop or an empty reply keep the input; vocabulary terms are known | `test_polish.py` |
| Polish call | no thinking, temperature 0, context sent; Ollama errors and timeouts keep the input; short and verbatim send no call | `test_polish_*` |
| Engines list and load errors | voices, styles and languages of engines that are not loaded; a missing model gives 503 with the fix | `test_engines_lists_voices_of_engines_that_are_not_loaded_and_load_errors_are_503` |
| Text pieces; token limit | sentences (also "।"); pieces ≤ 200 characters, cut in the second half; a cap of 2520 | `test_tts_lm.py::test_split_text_*`, `test_token_limit_*` |
| SNAC frames | the 3 levels in the right order; a token out of place drops its frame | `test_frames_*` |
| Stream chunks | every frame goes out once and in order, for 6 settings; the first chunk waits for its lookahead | `test_chunker_*` |
| Engine checks | the defaults; each bad voice, style, language and speed gives a clear error | `test_maya1_check`, `test_indic_speak_check` |
| Prompt layout | as the model cards give it, with and without a style | `test_prompt_layouts_follow_the_model_cards` |
| Vocos | 512 samples per step; the checkpoint's parameter names; a bad layout is refused | `test_vocos_layout_and_length` |
| Cartesia request | one WAV (16 kHz, mono, every sample), the key in the header only, the right model, language and version | `test_cartesia.py::test_a_call_sends_one_wav_with_the_key_in_the_header_only` |
| Cartesia language | passed on; a bad code is refused before any call | `test_the_language_is_passed_on_and_checked` |
| Cartesia failures | a plain reason for 401, 403, 402, 429, 422, 500, a network error and a non-JSON answer; the key is in none of them | `test_each_failure_has_a_plain_reason_and_never_the_key` and 2 more |
| Cartesia, short audio | no call, no cost | `test_less_than_100_ms_of_audio_makes_no_call_and_costs_nothing` |
| Cartesia in the service | `cloud` and `unavailable` in `/v1/engines`; 502 for an engine failure; 503 with the fix for a missing key | `test_server.py::test_cartesia_is_a_cloud_engine_that_needs_a_key` |
| A caller that hangs up during an upload | a quiet 499, no traceback | `test_a_client_that_hangs_up_during_the_upload_is_not_a_server_error` |
| Written form of each class | 57 spoken → written pairs: `#N`, times, dates, years, money, percent, units, email, domains, URLs, acronym case, spacing | `test_textfmt.py::test_the_written_form` |
| The dictation corpus outputs | every distinct raw text of Parakeet and Whistle on `stt/dictation` (run 20261023-122740) gives its recorded written form; pairs, split ids and URL paths are fixed | `test_textfmt.py::test_the_recorded_dictation_outputs`, `test_the_dictation_fixtures_fix_the_number_and_url_gaps` |
| Text that must not change | 33 lines: already written forms (`$12,450.30`, `Gemma 4`, `3:30 PM`, `#4821`), "one of them", "a second look", "number one", "Send it to Bob at the office", quotes and backticks, spoken CLI commands, identifiers | `test_written_and_ordinary_text_is_not_damaged` |
| Recorded model outputs | 104 distinct outputs of 9 research models (bench/results) for the 16 synthetic clips: the written form of each is as stored | `test_the_recorded_model_outputs[*]`, fixture `tests/fixtures/synth_hyps.json` |
| The three rules of design 4.1 | on every fixture and case: `written(written(x)) == written(x)`, the same text twice gives the same text, no content word added (only digits, symbols, the ordinal suffix, AM/PM); and on 6,000 random sentences of 1 to 14 spoken tokens | `test_idempotent_deterministic_and_adds_no_word[*]`, `test_fuzzed_token_soup_keeps_the_three_rules` |
| Quotes, spacing, style, empty text | text in quotes or backticks is never touched; line breaks and double spaces stay; an unknown style is a `ValueError`; `""` and `None` give `""` | `test_text_in_quotes_and_backticks_is_never_touched` and 3 more |
| Speed | p95 under 5 ms for 40 words | `test_it_is_fast_enough_for_the_turn_path` |
| Vocabulary fixes | 9 real engine outputs: `useEffect`, `userId`, `fetchUserProfile`, `TypeScript`, `parseConfig`, `Redis`, `Postgres`, a subdomain, an email name and domain, `Sanjay`, `Kubernetes`, `Terraform` | `test_vocab.py::test_the_terms_fix_real_engine_outputs` |
| Vocabulary precision | 63 ordinary sentences with words that sound like the 25 terms: no change | `test_vocab.py::test_ordinary_sentences_never_change`, `test_the_precision_set_is_big_enough` |
| Vocabulary properties | idempotent; the only new words are terms; no terms, no change; same input, same output; a term already there stays | `test_vocab.py::test_it_is_idempotent_and_the_only_new_words_are_terms`, `test_no_terms_change_nothing_and_the_same_input_gives_the_same_output`, `test_a_term_already_in_the_text_stays` |
| Vocabulary known limits | the cases of section 7 give what they give, so a change of behaviour is seen | `test_vocab.py::test_the_known_limits_are_what_they_are`, `test_ordinary_word_pairs_that_only_sound_like_a_term_stay` |
| Vocabulary limits and sources | each limit with its reason; the header split; the team file with comments; the merge order | `test_vocab.py::test_the_term_list_limits`, `test_the_header_and_the_team_file`, `test_the_common_word_list_is_loaded_and_has_its_source` |
| Vocabulary in the service | header and team file correct the text; Whistle gets the terms as keywords; `vocab=off` skips both; a bad header or `vocab` is 400 before any model call; preflight equals final with terms | `test_server.py::test_the_vocabulary_header_and_the_team_file_correct_the_text`, `test_the_terms_go_to_an_engine_that_takes_keywords_and_vocab_off_skips_them`, `test_an_engine_without_keywords_gets_only_the_correction_pass`, `test_a_bad_vocabulary_is_refused_before_any_model_call`, `test_the_preflight_text_equals_the_final_text_with_a_vocabulary` |
| Vocabulary speed | p95 < 5 ms for 40 words and 200 terms | `test_vocab.py::test_it_is_fast_enough_for_the_turn_path` |
| Formatted STT answer | `text` formatted, `raw` as the engine said it, `stages` has `asr` and `written`; `format=raw` skips the stage; the setting sets the default | `test_server.py::test_stt_gives_the_written_form_by_default_and_keeps_the_raw_text`, `test_stt_format_raw_skips_the_formatter_and_the_default_comes_from_the_setting` |
| Bad `format` | 400 with the choices, before any model call (the fake engine was not called); an over-long value gives 422; a bad setting stops `create_app` | `test_a_bad_format_is_refused_before_any_model_call` |
| Preflight equals final | the same raw text through audio of 1.5, 2 and 3 s, and again, gives one `text` | `test_the_preflight_text_equals_the_final_text` |
| Real models (`ORB_TEST_TTS_LM=1`) | maya1 and indic-speak speak 0.8 to 8 s, not silence | `test_the_model_speaks[*]` |
| The GPU crash of 2026-10-22 (`ORB_TEST_TTS_LM=1`) | Parakeet in a loop and maya1 together: no abort | `test_parakeet_and_a_speech_lm_share_the_gpu` |
| Real models: Kokoro, then each STT engine | the same words back | `test_models.py` |
| Model files | SHA-256 and size of every pinned file | `test_checksums_of_the_fetched_files`; `python -m orb_speech.models --check --all` |

Quality and speed are measured by the bench suites (`python -m orb_bench run stt|tts`), not by these tests.

## 5. Security
- Binds 127.0.0.1 only. No authentication, so it must never be reachable from a network.
- **The log has sizes and timings only**, never words: `tts maya1 70 chars -> 2048 ms audio, first audio 700 ms`.
- **No network at run time:** `dev.sh` starts it with `HF_HUB_OFFLINE=1`; `NEEDLE_TELEMETRY=0` stops the
  cactus-needle usage counts. A speech-LM factory reads the cache only (`fetch_missing=False`).
- **Model files are pinned** in `models.py`: by SHA-256 and size, and the Hugging Face ones also by commit. A changed
  file is refused. The weights are safetensors, except two pickle files: SNAC's `pytorch_model.bin` and
  indic-speak's `vocos/best.pt`. Both load with `weights_only=True`, so only tensors and plain data come out.
- **No downloaded code runs.** The repos ship Python files (`inference.py`, `vocos/*.py`, `tts.py`); they are not
  fetched. `vocos.py` is orb's own module, built from the Vocos paper. Its output matches the reference module to
  2 × 10⁻⁸ on the same weights (checked once, 2026-10-22).
- **A cloud engine sends audio away.** With `cartesia`, the audio of a call goes to Cartesia (api.cartesia.ai). It is
  opt-in: the engine is not the default, it is unchecked in the bench's engine list, and it loads only when
  `CARTESIA_API_KEY` is set. The page marks it "(cloud)" and shows a warning when it is chosen, and the voice worker
  logs a warning when a call starts with it. The log of the service has sizes, times and the Cartesia address, never
  words or audio.
- **The key** is read from the process environment or from the `CARTESIA_*` lines of the repo `.env` (not in git;
  other secrets in that file are not loaded). It goes only into the Authorization header. It is in no log line, no
  error message, no result and no test output (a test checks this).
- **The vocabulary header is user input.** Its size is checked before it is split, every term is checked, and a bad
  one is a 400 before the body is read. Terms never go into the log (only their count). A term changes only the text
  of the same call.
- **The common-word list** (`data/common_words.txt`) is SCOWL size 20 (Copyright 2000-2018 Kevin Atkinson; use,
  copy, modify, distribute and sell allowed with the notice, which the file keeps).
- **Licenses:** Parakeet CC-BY-4.0 (credit NVIDIA); Kokoro and Whistle Apache-2.0; SNAC MIT; maya1 Apache-2.0; indic-speak the Indic Open Model License (its text is pinned with the model). Both speech
  LMs are Llama 3.2 fine-tunes, so the Llama 3.2 Community License applies too.

## 6. Performance
Measured on an Apple M5 (16 GB), 2026-10-22. STT: `orb_bench` stt run 20261022-194658. TTS: tts run 20261022-201558
(each engine warmed first; the judge is Parakeet for English and Whisper large-v3-turbo for Hindi).

| Engine | Measured |
|---|---|
| Whistle | WER: LibriSpeech 5.2 %, synthetic 13.9 % (clean 6.0 %, 5 dB noise 28.5 %), EdAcc 28.3 % (South Asian 26.8 %); p50 17 to 36 ms |
| Parakeet | WER: LibriSpeech 2.4 %, synthetic 4.9 % (clean 3.0 %, 5 dB noise 7.7 %), EdAcc 12.3 % (South Asian 11.5 %); p50 89 to 173 ms |
| Cartesia (a small sample: the first 12 clips of each group, run 20261022-223513; not enough to rank the engines) | WER: LibriSpeech 0.5 %, synthetic 0.5 %, EdAcc 12.5 % (Parakeet 6.5 %, Whistle 12.5 % on the same clips); p50 144 to 167 ms, p90 up to 346 ms (an HTTPS call over a kept-open connection; the first call 406 ms) |
| Text from noise only (24 clips) | Whistle 45.8 % ("Thank you." from hum and brown noise), Parakeet 29.2 % ("Mm-hmm.", "Yeah.") |
| Text from noise only, the first 12 clips (same run as the Cartesia sample) | Whistle 0 of 12, Parakeet 3 of 12, **Cartesia 12 of 12** ("Thank you.") |
| Kokoro, English | first audio 0.14 to 0.99 s; real-time factor 0.09 to 0.12; WER 0 % on 4 texts |
| Kokoro, Hindi | first audio 0.33 to 0.47 s; real-time factor 0.10 to 0.12; WER 9.1 % (Hindi), 25 % (Hinglish) |
| maya1 | first audio 0.70 to 0.76 s; real-time factor 1.53 to 1.89; WER 0 to 7.7 %; load 6 to 10 s |
| indic-speak (Kavya) | first audio 0.78 to 0.88 s; real-time factor 1.42 to 1.64; WER Hindi 0 %, English 0 to 14.3 %, Hinglish 37.5 %; load 4 to 8 s |

Written form, measured on an Apple M4 (16 GB), 2026-10-23 (`orb_speech.textfmt.written`, one thread, after warm-up):

| What | Measured |
|---|---|
| Time, 58 words with 6 classes | median 0.13 ms, p95 0.14 ms, p99 0.14 ms, max 0.23 ms (5,000 runs) |
| Time, 37 words with no class | median 0.11 ms, p95 0.11 ms (5,000 runs) |
| Time, 9 words | median 0.02 ms |
| Import | 4.4 ms (`re` only) |
| Worst case, an adversarial text of 8,000 characters (a chain of `dot`) | 29 ms; a normal 60 s turn is about 1,300 characters |
| NeMo ITN for comparison | 4.2 ms median for 10 words and 30 ms for 58 words, and 2.6 to 7.6 s to load (decision 0010) |
| Accuracy on the 104 recorded outputs of 9 research models for the 16 synthetic clips | token error rate with case and punctuation kept (no normalizer): 32.8 % → 27.7 %; digit and symbol tokens of the reference found exactly: 55 of 108 → 78 of 108 |
| Same, Parakeet v3 (bf16) only, 16 clips | 31.6 % → 21.9 %; 9 of 16 → 13 of 16 |

The rest of the gap on those clips is not for this stage: the ASR hears "priaya" and "devslabs", self-corrections
("Tuesday, no wait, Wednesday"), lists, `dash b`, "use effect" and "10 RM" are for STT-3 and STT-4. The exit check on
`stt/dictation` (`fwer`, `entity`) waits for STT-0's metrics.

Vocabulary, measured on an Apple M4 (16 GB), 2026-10-23 (format run 20261023-130520 on stt run 20261023-122740, with
a team list of 31 terms: the 18 of the corpus and 13 that are not in it):

| What | Measured |
|---|---|
| Parakeet, `stt/dictation`, written → vocab + written | `entity` 51.2 → 65.5 %: identifier 15.3 → 90.3 %, domain 68.1 → 79.2 %, email 36.1 → 69.4 %, url 0 → 33.3 %; `fwer` 23.1 → 20.2 % |
| Whistle, same | `entity` 32.6 → 41.8 %; `fwer` 40.1 → 37.6 % |
| EdAcc and LibriSpeech, both engines | 0 sentences changed; WER unchanged |
| Added words (a term where no word sounded like it) | 0 rows |
| Time of the pass | p50 0.09 ms, p95 0.27 ms; the first call of a new term list compiles it (about 11 ms) |
| Whistle `keywords`, `stt/dictation`, in process | WER 35.9 → 32.7 %, `entity` 15.8 → 21.2 % (29.3 % with the pass); p50 20 → 22 ms |
| Whistle `keywords`, EdAcc / LibriSpeech | 21 of 182 / 3 of 100 outputs changed; WER 28.2 → 28.2 % / 5.0 → 4.9 % |

**The speech LMs are slower than real time on this Mac** (real-time factor > 1.4 even without decoding), so a live
call with them stutters. They stay bench engines; Kokoro is the live voice. The Hinglish WER is too high for every
engine: the judge writes English words such as "office" in Devanagari.

## 7. Edge cases
| Case | What happens | Why |
|---|---|---|
| Audio shorter than 100 ms | empty text, no model call | nothing useful to hear |
| Two STT calls at once | one after the other | neither model is thread-safe |
| Parakeet and a speech LM at once | they take turns on the MPS lock; MLX runs freely | one torch MPS stream per process |
| A second speech LM is asked for | the first one is closed, then the second loads (4 to 10 s) | memory: one 2 GB model at a time |
| Cartesia is chosen and no key is set | 503 with the line to add to `.env`; the page shows the engine as "not set up" and disables it | fail early, with the fix |
| Cartesia is not reachable, or refuses the key | 502 with a plain reason; the voice worker retries a 5xx call after 2 s | `EngineError`; LiveKit's retry |
| Cartesia hears noise | it writes "Thank you." (12 of 12 noise clips, 2026-10-22): the VAD must stop noise before the STT | a Whisper habit, see section 6 |
| A speech LM does not stop | the token limit ends the piece; the log has a warning | about twice the expected length |
| The client closes during TTS | the generation stops at the next token | the stream sets a stop event |
| A speech LM is unloaded during a stream | the stream ends at the next token | `close()` sets the closed event |
| A model file is missing | a default engine: the service does not start; another engine: 503 that names `python -m orb_speech.models --all` | fail early, with the fix |
| `format` is not `written` or `raw` | 400 `format must be one of written, raw, not 'x'`, before the body is read | fail early |
| Text in quotes or backticks | passes through as it was | the speaker may mean the words as said |
| A number run that is not one number ("twenty twenty five", "two thirty", "one two three four") | stays as words, except a year after a month or a preposition | no wrong digits |
| "number one", "one of them", "a second", "a hundred", "GPT four" | stay words | ordinary talk; the cost is a few lines a dictation tool would write with digits |
| `at` between two words ("Meet me at noon", "Look at google.com") | stays, unless the name has a dot, an underscore or a digit, or follows a cue word, and a valid domain follows | `Look@google.com` would be damage |
| "dot com" with no name ("the dot com bubble"), "polka dot" | stays | the first label must not be a function word |
| "3 30" with no am or pm | stays | no marker, no time |
| "10 RM", "priaya.sharma at devslabs.ie" | `10 RM` stays; the email is joined as heard (`priaya.sharma@devslabs.ie`) | the formatter fixes the form, not the hearing |
| "May first", "March third" | `May 1st` with a capital M; lower case `may first` stays a verb | an ambiguous month needs a capital |
| `p.m.` at the end of a sentence | `PM.` (one dot) | the dot of `p.m.` is the sentence's too |
| Another language (Hindi, German, …) | passes through unchanged | English rules; Parakeet also writes other languages |
| An 8,000 character text of `dot dot dot …` | 29 ms, not a hang | the work caps (16 parts) |
| Hindi text for maya1 | 400: "maya1 speaks English only" | the model is English |
| A term that sounds like a rare English word: `oath` with OAuth, `terra firma` with Terraform, `get hub` with GitHub | it becomes the term | the word is not in the 12,530 ordinary words; a team adds only terms it uses |
| The words of a camelCase term said in other sense: "their user ID" with userId | it becomes `userId` | the words alone cannot tell; context is STT-4's job |
| Two ordinary words that only sound like a term: "post curves", "data dog" | they stay | precision first: an exact spelling is needed |

## 8. Expected results
- `python -m orb_speech.models --check --all` prints `ok` for every pinned file.
- `GET /v1/engines` lists `whistle, parakeet` and `kokoro, maya1, indic-speak`, with 54, 3 and 44 voices.
- `make test` passes 2,422 speech tests (most are fixture rows of the formatter and the vocabulary); `ORB_TEST_TTS_LM=1` adds 3 that load the speech LMs.
- `python -m orb_bench run tts` gives about the numbers of section 6.
- `POST /v1/stt` of a clip that says "pull request number four thousand eight hundred and twenty-one" gives `text`
  `... pull request #4821`, and `raw` has the spoken form. `?format=raw` gives both the same.
- The same raw text always gives the same `text`, so the voice worker's preflight transcript equals its final one.
