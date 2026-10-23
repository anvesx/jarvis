# LLD: orb-speech (local speech service)

Code: `src/orb_speech/` · Tests: `tests/` (54: 51 by default, 3 more with `ORB_TEST_TTS_LM=1`) · Status: built ·
Last update: 2026-10-22

**One small HTTP service gives the voice worker and the bench their local speech models.** Each request can name its
engine, so the bench switches models without a restart. No audio and no text leaves the machine.

| Kind | Engines | Default | Loads |
|---|---|---|---|
| Speech to text | `whistle` (Whistle 2.0.0, 17 MB), `parakeet` (Parakeet TDT 0.6B v3, 2.5 GB), `cartesia` (Cartesia Ink-Whisper, **a cloud service**) | `whistle` (the user's choice; decision 0005 has the accuracy) | Whistle at start; Parakeet on first use; Cartesia on first use, and only with `CARTESIA_API_KEY` (decision 0008) |
| Text to speech | `kokoro` (Kokoro-82M int8), `maya1` and `indic-speak` (3B speech LMs, 4-bit MLX) | `kokoro` | Kokoro at start; a speech LM on first use, one at a time |

## 1. Input
| Endpoint | Body | Limits | Answer |
|---|---|---|---|
| `POST /v1/stt?sample_rate=16000[&language=en][&engine=]` | PCM16 LE mono | ≤ 60 s; rate 8, 16, 24 or 48 kHz; a language of the engine (Whistle: en de fr es it nl pl; Parakeet: 25 European languages; Cartesia: any ISO-639 code, about 100) | `{"text", "language", "engine", "ms", "audio_ms"}`; 502 when the engine's own service fails; 503 when the engine cannot load |
| `POST /v1/tts` | `{"text", "engine"?, "voice"?, "speed"?, "style"?, "lang"?}` | text 1 to 2,000 characters; engine ≤ 32, voice ≤ 400, style ≤ 64, lang ≤ 16 characters; then the engine's own checks (below) | a stream of PCM16 mono; headers `X-Sample-Rate` (24000), `X-Engine` |
| `GET /v1/engines` | none | | per kind: the engines, the default, the loaded ones, `cloud` (engines that send the audio to another company) and `unavailable` (engine → the reason, for example a missing key); per TTS engine: `voices`, `styles`, `langs` (known without a load) |
| `GET /health` | none | | which models are loaded |

The engine checks of `/v1/tts`:
| Engine | `voice` | `style` | `lang` | `speed` |
|---|---|---|---|---|
| `kokoro` | one of its 54 voices; Hindi default `hf_alpha` | ignored | `en-us`, `en-gb`, `hi` | 0.5 to 2.0 |
| `maya1` | a description in words, ≤ 400 characters, without `"`, `<` or `>`; 3 presets | ignored (emotion tags such as `<laugh>` go in the text) | English only | none, or 1.0 |
| `indic-speak` | one of 44 names (case does not matter); default: the female voice of `lang` | one of 14 trained tags, or none | 22 Indian languages, `en`, `en-us`, `en-gb` | none, or 1.0 |

Settings: `ORB_STT_ENGINE` (`whistle`), `ORB_STT_LANGUAGE`, `ORB_TTS_ENGINE` (`kokoro`), `ORB_TTS_MODEL`,
`ORB_TTS_VOICE`, `ORB_SPEECH_PORT`, `ORB_MODELS_DIR`, `CARTESIA_API_KEY` and `ORB_CARTESIA_VERSION` (default
`2026-08-14`). The service reads only `CARTESIA_*` and `ORB_CARTESIA_*` keys from the repo `.env`. End-of-turn detection is not here: the voice worker runs
LiveKit's turn detector in its own process (decision 0006).

The callers are the voice worker, the agent service's bench proxy and the bench suites, on the same machine. The
service checks every limit before it touches a model.

## 2. Algorithm
- **Engines** (`engines.py`): a `Registry` per kind maps a name to a factory. `get(name)` builds the engine on first
  use, under a lock. A heavy engine (Parakeet; maya1 and indic-speak) counts against a limit of one per registry:
  loading another heavy engine first calls `close()` on the oldest one. A load error gives 503 with the reason.
- **One GPU user at a time** (`gpu.py`): PyTorch has one MPS command stream per process, and two threads must not use
  it at once. Every torch call on MPS (Parakeet, the SNAC and Vocos decoders) takes one reentrant lock. MLX has
  its own command queues and does not take it.
- **STT:** read the body in parts and stop at the limit → PCM16 to float32 → resample to 16 kHz (soxr) → the
  engine in a worker thread → `clean()`: stray quote marks at the ends and the fillers "uh", "um", "erm" removed.
  - **Whistle:** cactus-needle, on the CPU, with the language as an input.
  - **Parakeet:** the feature extractor (128 log-mel bins) → `ParakeetForTDT.generate` on MPS in float32. At load,
    0.2 s and 1 s of silence compile the GPU kernels. A keep-warm thread transcribes 0.2 s of silence after 1 s
    without a call, within 120 s of the last real call.
  - **Cartesia (cloud):** the samples go into a 16 kHz mono WAV, and one `POST https://api.cartesia.ai/stt` sends it
    as a multipart upload (`model=ink-whisper`, `language`) with the headers `Authorization: Bearer <key>` and
    `Cartesia-Version`. The answer's `text` goes through `clean()`. Less than 100 ms of audio makes no call. A failed
    call raises `EngineError` with a plain reason (HTTP 401 or 403: the key; 402: no credits; 429: too many requests;
    a network error: not reachable), and the route answers 502. A missing key makes the engine fail to load with
    the fix in the message (503), and `/v1/engines` shows it as `unavailable`.
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
| Hindi text for maya1 | 400: "maya1 speaks English only" | the model is English |

## 8. Expected results
- `python -m orb_speech.models --check --all` prints `ok` for every pinned file.
- `GET /v1/engines` lists `whistle, parakeet` and `kokoro, maya1, indic-speak`, with 54, 3 and 44 voices.
- `make test` passes 51 speech tests; `ORB_TEST_TTS_LM=1` adds 3 that load the speech LMs.
- `python -m orb_bench run tts` gives about the numbers of section 6.
