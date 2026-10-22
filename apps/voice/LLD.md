# LLD: apps/voice (the voice worker)

Code: `main.go`, `debugaudio.go` · Tests: `main_test.go` (3) and `scripts/e2e_voice.py` (end to end) · Status: built · Last update: 2026-10-22

**The voice worker is a LiveKit agent on lk-go-agent-sdk.** For each room it joins, it listens with Silero VAD,
cuts speech into pieces for Whistle, hears the end of the turn with Smart Turn, transcribes the whole turn in one
call, hands it to the orb agent service, and speaks the reply with Kokoro.

## 1. Input
- A job from the LiveKit server for each new room (`--dev` server on 127.0.0.1:7880).
- The room's audio, decoded and resampled by the SDK to 16 kHz mono. The browser sends it with echo
  cancellation, noise suppression and automatic gain on.
- Settings (environment or the repo `.env`):

| Variable | Default | Meaning |
|---|---|---|
| `ORB_AGENT_URL` / `ORB_SPEECH_URL` | http://127.0.0.1:8100 / :8200 | the agent and speech services |
| `ORB_STT_LANGUAGE` | en | Whistle language; empty detects it |
| `ORB_TTS_VOICE` | the service default (af_heart) | Kokoro voice |
| `ORB_VAD_THRESHOLD` / `ORB_VAD_NEG_THRESHOLD` | 0.5 / 0.35 | speech starts above the first line and ends below the second (hysteresis) |
| `ORB_VAD_MIN_SPEECH_MS` | 250 | voice needed to start speech |
| `ORB_VAD_MIN_SILENCE_MS` | 250 | silence that ends speech |
| `ORB_MIN_ENDPOINTING_MS` / `ORB_MAX_ENDPOINTING_MS` | 200 / 2500 | wait after silence when Smart Turn says "finished" / "not finished" |
| `ORB_MIN_INTERRUPTION_MS` | 400 | while the agent is busy, speech must last this long after the VAD start (about 550 ms of voice in all) to interrupt it |
| `ORB_LOG_TEXT` | 0 | 1 puts transcripts and replies in the log |
| `ORB_DEBUG_AUDIO_DIR` | not set | debug only: every transcribed piece and turn is saved as WAV + text |
| `ONNXRUNTIME_LIB_PATH` | /opt/homebrew/lib/libonnxruntime.dylib when it exists | for Silero |

## 2. Algorithm
1. Map the room to a session: `orb-<id>` gives `<id>`; any other room gets `room-<12 hex of its SHA-256>`.
2. `PUT /v1/sessions/<id>` creates the session when it is missing ("Open discussion").
3. Build the pipeline: one Silero model, two streams (one for the session's endpointing and barge-in, one for the
   STT adapter's pieces); `stt.NewStreamAdapter(speech.Recognizer)`; `agentllm.LLM`; `speech.TTS`;
   `speech.TurnDetector`.
4. Per turn (the SDK's ORB_CHANGES.md has the details): VAD silence → the piece's text (Whistle) → Smart Turn →
   commit after 200 ms or up to 2.5 s → **the whole turn transcribed again in one call** → the agent service →
   Kokoro, the first clause first, then sentence by sentence → the room.
5. While the agent thinks or speaks: speech interrupts it only after the guard (and while voice is still there).
   Shorter speech (an echo, a cough, "mm-hmm") is ignored and is not a turn.
6. When the person goes on before any audio of the reply played, the two parts become one turn, sent with
   `replace_last`; when the cutting speech had no words, the cut turn is answered again after 1 s.
7. Logs: `piece heard`, `user turn`, `speech ignored`, `agent decision`, `reply check`; a warning when the session
   dropped input frames (checked every 30 s).

## 3. Maths
**Why these values** (measured 2026-10-22 on an Apple M5; details in the SDK's ORB_CHANGES.md):
| Finding | Number |
|---|---|
| Whole turn in one Whistle call against joined pieces, at 10 dB SNR | 4.2 % against 12.9 % word error |
| Kokoro's "Okay." / "Sure." as voice, after LiveKit's Opus round trip | long enough (> 400 ms) to pass a 300 ms guard; 400 ms holds them |
| Silence that ends speech | 250 ms: the whole-turn transcript repairs a cut piece, so a short wait no longer costs accuracy |

**The latency of a short voice turn** (from the end of the person's speech to the first agent audio):
| Step | Time | Source |
|---|---|---|
| Silero: silence that ends the speech | 250 ms | setting |
| Whistle (piece) + Smart Turn + Whistle (whole turn) | about 0.1 to 0.17 s | measured |
| Jev reads the turn | 0.29 to 0.56 s | measured |
| Gemma: first token, then the first clause | 0.28 to 0.39 s, then 0.2 to 0.5 s | measured |
| Kokoro: first audio of the first piece | 0.21 to 0.62 s (about 5.5 ms per character) | measured |
| WebRTC and playout | about 0.2 to 0.3 s | estimated from the difference |
| **Measured end to end** | **1.25 to 1.29 s** (Rival turn), **2.62 s** (Examiner turn with a long first sentence) | `e2e_voice.py`, two runs |

## 4. Test cases
| Case | Expected result | Test |
|---|---|---|
| Room names | `orb-<id>` gives the ID; another name gives one stable, valid ID | `main_test.go::TestSessionFor` |
| `.env` | set variables win; quotes and `export` work | `TestLoadDotenvKeepsSetVariables` |
| Debug WAV | a valid 16 kHz PCM header and size | `TestWriteWAVHeader` |
| End to end: two spoken turns | the same words heard; Rival, then Examiner | `scripts/e2e_voice.py` |
| End to end: three clauses with 450 ms pauses in pink noise at 20 dB SNR | **one** message, ≤ 15 % word error | `e2e_voice.py` (2026-10-22: 1 message, 10 %; the only miss is "back end" for "backend") |
| End to end: "Sure." while the agent speaks | the agent keeps talking; no turn | `e2e_voice.py` |
| End to end: barge-in | the agent's audio stops within 1.5 s | `e2e_voice.py` (0.80 s and 0.92 s) |

Result on 2026-10-22: **PASS** in two runs in a row. No input frames were dropped.

## 5. Security
- Transcripts and replies stay out of the log unless `ORB_LOG_TEXT=1`.
- `ORB_DEBUG_AUDIO_DIR` saves real voices: it is off by default, the files are created with mode 0600, and
  `data/debug_audio/` is in `.gitignore`. Turn it on only with the speaker's consent.
- The worker talks to services on 127.0.0.1 only, and to the LiveKit server with the dev credentials in dev
  mode. A production server needs `LIVEKIT_URL`, `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET`.
- The room name only picks a session; it never builds a file path except through `sessionFor`, which allows
  `[a-z0-9-]` only.

## 6. Performance
See the budget above. **Next, with the expected gain:**
1. Start the Jev read on the final transcript, before the commit: up to 0.4 s.
2. Bias Whistle with the topic's words (its `keywords` option): fixes errors such as "database" heard as
   "day or base" (seen in a test run).
3. Tune the guard and the silence with real voices: run a call with `ORB_DEBUG_AUDIO_DIR=data/debug_audio`, then
   replay the saved turns offline.

## 7. Edge cases
| Case | What happens | Why |
|---|---|---|
| A pause inside a sentence, before the agent spoke | one merged turn, transcribed as a whole; the agent undoes the unheard one | `replace_last` |
| An echo, a cough or "mm-hmm" while the agent speaks | ignored; the agent keeps talking | the barge-in guard |
| A short real answer ("No") while the agent speaks | ignored too | the guard's price; the person can say it again |
| A false interruption of an unheard reply | the cut turn is answered again after 1 s | no dead end |
| A barge-in after the agent spoke | the audio stops; the agent service keeps the whole reply text | the agent does not know yet what was heard (roadmap) |
| Two people in one room | the worker hears only the first audio track | `RunRoomSession` subscribes to one track |
| The speech or agent service is down | errors in the log; the turn gets no reply | the worker does not hide failures |
| The LLM copies the persona note into its reply | the agent drops those lines before they reach the TTS | see orb-discussion LLD |

## 8. Expected results
`make dev`, then `make e2e` prints `PASS`. In the browser, Talk shows "The agent joined. Speak when you are
ready."; pauses inside a sentence do not cut it; a short "mm-hmm" does not stop the agent; speaking over the
agent stops it in under 1 s.
