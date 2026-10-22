# LLD: orb-speech (local speech service)

Code: `src/orb_speech/` · Tests: `tests/` (14, of which 3 use the real models) · Status: built · Last update: 2026-10-22

**One small HTTP service gives the voice worker three local models:** Whistle (speech to text), Kokoro (text to
speech) and Smart Turn (has the person finished?). No audio and no text leaves the machine.

## 1. Input
| Endpoint | Body | Limits | Answer |
|---|---|---|---|
| `POST /v1/stt?sample_rate=16000[&language=en]` | PCM16 LE mono | ≤ 60 s; rate 8, 16, 24 or 48 kHz; language one of en de fr es it nl pl | `{"text", "language", "ms", "audio_ms"}` |
| `POST /v1/turn?sample_rate=16000` | PCM16 LE mono | ≤ 30 s (the model reads the last 8 s) | `{"probability", "complete", "threshold", "ms", "audio_ms"}` |
| `POST /v1/tts` | `{"text", "voice"?, "speed"?}` | text 1 to 2,000 characters; a known voice; speed 0.5 to 2.0 | a stream of PCM16 mono at 24 kHz; header `X-Sample-Rate` |
| `GET /health` | none | | which models are loaded |

The callers are the voice worker and the end-to-end test, on the same machine. The service trusts nothing in
the body: it checks every limit before it touches a model.

## 2. Algorithm
- **Start:** load Whistle (cactus-needle), Kokoro (kokoro-onnx, int8) and Smart Turn (ONNX Runtime) from
  `ORB_MODELS_DIR`, after a SHA-256 check of each file. About 3 s.
- **STT:** read the body in parts and stop at the limit → PCM16 to float32 → resample to 16 kHz (soxr) →
  Whistle in a worker thread, under a lock (Whistle is not thread-safe) → text, with stray quote marks at its ends
  removed (Whistle sometimes opens a transcript with `"`). The voice worker calls it for each piece of speech and
  once more for each whole turn.
- **Turn:** the same audio path → the last 8 s, padded with zeros at the start → 80 Whisper log-mel bins × 800
  frames → ONNX → a sigmoid probability → "complete" when ≥ 0.5.
- **TTS:** check the text, voice and speed → Kokoro `create_stream` yields audio per phoneme batch → float32 to
  PCM16 → sent in 100 ms parts. A closed connection stops the synthesis.

## 3. Maths
| Name | Value | Source |
|---|---|---|
| PCM16 → float | x / 32768 | standard |
| float → PCM16 | round(clip(x, −1, 1) × 32767) | standard; clipping stops wrap-around |
| Smart Turn window | 8 s × 16,000 Hz = 128,000 samples; zeros at the start when shorter | Smart Turn reference `inference.py` |
| Smart Turn features | 80 log-mel bins × 800 frames, `WhisperFeatureExtractor(chunk_length=8)` | reference code; input name `input_features` |
| Turn threshold | 0.5 (`ORB_TURN_THRESHOLD`) | the reference code's line |
| STT body limit | 60 s × 16,000 × 2 bytes = 1,920,000 bytes at 16 kHz | chosen: one turn is never that long |
| Shortest STT input | 100 ms; shorter gives empty text without a model call | chosen |
| TTS part size | 4,800 bytes = 2,400 samples = 100 ms at 24 kHz | chosen |
| Kokoro speed (int8) | real-time factor 0.093: 0.52 s for 5.6 s of speech | measured 2026-10-21, Apple M5 |
| Kokoro speed (fp32) | real-time factor 0.181 | measured; int8 is the default |

## 4. Test cases
| Case | Expected result | Test |
|---|---|---|
| PCM round trip | error below 1e-4 | `test_audio.py::test_pcm_round_trip_is_close` |
| Clipping; an odd byte count | clipped to ±0.99997; a clear error | `test_pcm_clips_and_rejects_odd_bytes` |
| Resampling | 48,000 → 16,000 samples; bad rates refused | `test_resample_lengths` |
| Turn window | pads at the start; keeps the last 8 s | `test_turn_window_pads_at_the_start_and_keeps_the_end` |
| Stray quote marks | removed at the ends; apostrophes kept | `test_stt_clean_removes_stray_quotes_only` |
| Health | `ok` is false when a model is missing | `test_server.py::test_health` |
| STT at 48 kHz in German | resampled to 16,000 samples; the language is passed on | `test_stt_converts_and_resamples` |
| Bad STT input | a clear 400 or 413 for each rule | `test_stt_rejects_bad_input_with_reasons` |
| Turn endpoint | complete / not complete from the model | `test_turn` |
| TTS stream | PCM at 24 kHz; bad voice 400; empty text and speed 5 rejected | `test_tts_streams_pcm_with_its_rate` |
| A model is not loaded | 503 on its endpoint | `test_missing_engines_give_503` |
| Real models: Kokoro, then Whistle | the same words back, in < 500 ms | `test_models.py::test_tts_then_stt_gives_the_same_words` |
| Real models: Smart Turn | a full question is complete; a sentence cut at "so the" is not | `test_smart_turn_tells_full_from_cut` |
| Model files | the SHA-256 of each file matches | `test_checksums_of_the_fetched_files` |

## 5. Security
- Binds 127.0.0.1 only. No authentication, so it must never be reachable from a network.
- **The log has sizes and timings only**, never words: `stt 4140 ms audio -> 52 chars in 25.5 ms`.
- **No network at run time:** `dev.sh` starts it with `HF_HUB_OFFLINE=1`, and `NEEDLE_TELEMETRY=0` stops the
  cactus-needle usage counts (anonymous counts, read in its source on 2026-10-21; no audio or text).
- **Model files are pinned** by SHA-256 in `models.py`. A changed file is refused at start.
- Body limits stop a client from filling memory; the voice name must be one that Kokoro knows.

## 6. Performance
Measured on an Apple M5, 2026-10-21, during the end-to-end voice test:
| Call | Measured |
|---|---|
| Whistle, utterances of 2.0 to 6.4 s | 15 to 26 ms |
| Smart Turn | 12 to 47 ms |
| Kokoro, first audio of a sentence of 38 to 112 characters | 277 to 622 ms (about 5.5 ms per character) |

Kokoro is the slowest step: it makes a whole phoneme batch before the first audio. **Next:** the TTS plugin can
send the first clause (from about 20 characters) instead of the whole first sentence; expected gain 0.2 to
0.4 s on the first audio.

## 7. Edge cases
| Case | What happens | Why |
|---|---|---|
| Audio shorter than 100 ms | empty text, no model call | nothing useful to hear |
| Two STT calls at once | they run one after the other | Whistle is not thread-safe |
| The client closes during TTS | synthesis stops after the current batch | `create_stream` stops when its consumer stops |
| A model file is missing | the service does not start; the message names `python -m orb_speech.models` | fail at start, not on the first call |
| Kokoro's final, falling tone | Smart Turn can call an unfinished fragment "complete" | synthetic speech; check with real speech |

## 8. Expected results
- `python -m orb_speech.models --check` prints `ok` for the four files.
- `GET /health` gives `"ok": true`.
- `make test` passes the 13 tests, the 3 model tests included.
