# 0003 · Local speech models: Whistle, Kokoro, Smart Turn

Date: 2026-10-21 · Status: accepted

## Decision
| Job | Model | Size | License | Measured on an Apple M5 |
|---|---|---|---|---|
| Speech to text | Whistle 2.0.0 (Cactus Compute) | 16.9 MB | Apache-2.0 | 12 to 35 ms for one sentence |
| Text to speech | Kokoro-82M v1.0, int8 ONNX | 114 MB | Apache-2.0 | 0.52 s for 5.6 s of speech (real-time factor 0.093) |
| End of turn | Smart Turn v3.2, CPU ONNX (Pipecat) | 8.7 MB | BSD-2-Clause | 12 to 47 ms |
| Voice activity | Silero VAD v5 (in the Go SDK) | 2 MB | MIT | less than 1 ms per 32 ms window |

All run on this machine. No audio and no transcript leaves it.

## Corrections and rejected options
- **Whistle is speech-to-text, not text-to-speech.** It was first named as the TTS. Its page calls it
  "an open speech recognition model". So Kokoro is the TTS.
- **The LiveKit turn detector is not used, because of its license.** The LiveKit Model License says:
  "You cannot use the LiveKit models on a standalone basis or with any other frameworks." orb runs on its own
  Go SDK, which is a different framework. Smart Turn has an open license and hears prosody, not only text.
- **Fish Audio OpenAudio S1-mini is not used in the product, because its weights are non-commercial.**
  It has 51 emotion markers such as `(curious)`, `(serious)` and `(confident)`, which would fit the personas.
  But the weights are CC-BY-NC-SA-4.0. It is allowed for internal demos and research only. For the product,
  use Fish Audio's paid API or a commercial license, or an open model with emotion control.
- **Candidates to check for an expressive voice:** Chatterbox (Resemble AI) and Orpheus TTS (Canopy Labs).
  Their licenses and speed on this Mac are not verified yet.

## Known limit
Kokoro reads every fragment with a final, falling tone. So synthetic speech is a weak test of turn detection:
"I think the main problem is that" scored 0.97 ("finished"). Tune the turn timing with real speech.

## Integrity
`orb_speech.models` pins the SHA-256 of every model file and refuses a file that does not match. The speech
service starts with `HF_HUB_OFFLINE=1` and `NEEDLE_TELEMETRY=0`, so it makes no network call once the files
are on disk.
