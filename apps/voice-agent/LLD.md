# LLD: orb-voice-agent (the voice worker)

Code: `src/orb_voice/` · Tests: `tests/` (22) · Status: built · Last update: 2026-10-22

**One Python worker on LiveKit Agents 1.8.8 hears the person, decides when the turn ends, and speaks the agent's
reply.** Silero and LiveKit's turn detector are fixed parts. Speech to text, the reply and text to speech are orb's
services, so the bench can switch their models per session (decision 0006).

```
LiveKit room (browser mic, AEC on) ─▶ Silero VAD ─▶ LiveSTT (interim words, a preflight at each pause)
   ─▶ turn detector v1-mini (local) ─▶ OrbLLM: agent service /v1/sessions/{id}/turns (Jev + Ollama)
   ─▶ FirstClauseTokenizer ─▶ OrbTTS: speech service /v1/tts ─▶ room
events (states, transcripts, metrics) ─▶ room data channel "orb.bench" ─▶ bench page and turn suite
```

## 1. Input
| Input | From | Trust |
|---|---|---|
| A job for each new room | the LiveKit server (automatic dispatch, no agent name) | the dev server on 127.0.0.1 |
| The person's audio, 48 kHz | the room | untrusted: noise, echo and any words |
| The session config `{stt_engine, llm_model, tts_engine, tts_voice, tts_style, tts_lang}` | `PUT /v1/sessions/{id}` of the agent service, at join | checked by the agent service |
| Settings (table in section 3) | the environment; the repo `.env` fills in only unset `ORB_*` and `LIVEKIT_*` keys | the operator |

The room name gives the session: `orb-<id>` is the session `<id>` (3 to 48 of `a-z 0-9 -`); any other room gets
`room-` + the first 12 hex digits of its SHA-256.

## 2. Algorithm
1. **Start** (`setup`, once per process): read the settings and load Silero. The health server binds 127.0.0.1:8081.
   One idle process waits warm for the next room.
2. **Join** (`entry`): find the session, `PUT` it at the agent service (topic "Open discussion" if new), read its
   config, then ask the speech service which STT engines are cloud services (`GET /v1/engines`). Build the
   `AgentSession` with the session's engines. For Parakeet, one silent STT call opens the speech service's keep-warm
   window. For a cloud engine (Cartesia), the worker logs a warning, asks for interim transcripts only every
   `ORB_CLOUD_INTERIM_EVERY_S` (1.0 s), and sends no warm-up call (silence is billed).
3. **Hear** (`LiveSTT`, own VAD stream):
   - Speech start: a new piece; the frames from Silero's prefix (0.5 s) are kept.
   - While speaking: every 0.3 s of speech, the piece so far goes to `/v1/stt` as an interim transcript. A late
     interim of an old piece is dropped.
   - **Preflight:** after 0.15 s of silence in the piece, the piece so far is transcribed and sent as a preflight
     transcript. LiveKit starts the reply from it (preemptive generation, also its audio). When speech comes back,
     the preflight is dropped.
   - Speech end (Silero: 0.55 s of silence): the final transcript. When the preflight's pause ended the piece, the
     final is the preflight text (no second call), so LiveKit keeps the early reply.
4. **Decide the end of the turn** (LiveKit): the turn detector reads the audio of the turn. Likely finished: commit
   after `min_delay` (0.3 s) from the end of speech; unlikely: wait up to `max_delay` (2.5 s) for more speech.
5. **Reply** (`OrbLLM`): send only the person's words since the agent last spoke, `channel="voice"`, to
   `/v1/sessions/{id}/turns`, and stream the `token` events. `replace_last` is true when an earlier turn was sent
   and none of its audio played (`TurnTracker`): the agent service then undoes that turn and its Jev read. Errors are
   not retried (a retry would send the turn twice).
6. **Speak** (`FirstClauseTokenizer` + `OrbTTS`): the first piece ends at a sentence end after 20 characters, a
   comma, semicolon or colon after 30, or the last word boundary within 60. The rest goes sentence by sentence.
   Each piece is one streamed `/v1/tts` call at 24 kHz.
7. **Interruptions** (LiveKit, mode "vad"): after 0.5 s of the person's speech with 2 or more words in the
   transcript, the agent's audio pauses. With no words after 2 s, it resumes (false interruption). Otherwise the
   reply is cut and the new turn begins.
8. **Report:** every state change, transcript and metric goes to the room's data channel (topic `orb.bench`). The log
   has sizes and timings, and words only with `ORB_LOG_TEXT=1`.

## 3. Maths
| Setting | Value | Env | Source |
|---|---|---|---|
| Silero activation / deactivation | 0.5 / 0.35 | `ORB_VAD_THRESHOLD` | LiveKit plugin defaults |
| Silero min speech / min silence / prefix | 0.05 / 0.55 / 0.5 s | `ORB_VAD_MIN_SPEECH_S`, `_MIN_SILENCE_S`, `_PREFIX_S` | plugin defaults |
| Endpointing min / max delay | 0.3 / 2.5 s | `ORB_MIN_ENDPOINTING_S`, `ORB_MAX_ENDPOINTING_S` | LiveKit defaults with the audio turn detector |
| Interruption | 0.5 s and 2 words; resume after 2.0 s | `ORB_MIN_INTERRUPTION_S`, `_WORDS` | chosen: "Sure." is 1 word |
| Interim transcript | every 0.3 s of speech | `ORB_INTERIM_EVERY_S` | measured: 0.2 s did not stop barge-in sooner (0.95 s) |
| Interim transcript, cloud STT | every 1.0 s of speech | `ORB_CLOUD_INTERIM_EVERY_S` | each call pays for the whole piece again: for 5 s of speech, 17.5 s of billed audio instead of 47 s (orb-speech LLD) |
| Preflight | after 0.15 s of silence; 0 is off | `ORB_PREFLIGHT_S` | measured: p50 2.20 → 1.73 s with the TTS rules |
| Preemptive replies per user turn | at most 3; only for turns under 10 s | LiveKit defaults | each one costs a Jev read |
| First TTS piece | sentence ≥ 20, clause ≥ 30, word cut ≤ 60 characters | `ORB_FIRST_CLAUSE_CHARS`, `ORB_FIRST_PIECE_MAX_CHARS` | Kokoro: ~5.5 ms per character before its first audio |
| Room output queue | 200 ms | LiveKit | residual audio after a pause |
| AEC warm-up | interruptions off for the first 3 s of agent speech in a session | LiveKit | lets the browser's echo canceller adapt |

## 4. Test cases
| Case | Expected result | Test |
|---|---|---|
| Room names | `orb-<id>` maps to `<id>`; others to a stable `room-` hash | `test_session_for_rooms_of_the_web_page_and_others` |
| Repo `.env` | only `ORB_*` and `LIVEKIT_*` keys, never over a set value | `test_repo_env_copies_only_orb_and_livekit_keys` |
| Settings | defaults and overrides from the environment | `test_settings_defaults_and_overrides` |
| Cloud STT cadence | slower than the local cadence; an override | `test_a_cloud_stt_gets_a_slower_interim_cadence` |
| Which STT engines are cloud | the set and the default from the speech service; an older service, bad JSON and a service that is down all mean "local" | `test_the_worker_learns_which_stt_engines_are_cloud_services` |
| LiveKit credentials | the dev key or a short secret only with a server on this machine | `test_dev_credentials_only_on_this_machine` |
| STT request | PCM16, rate, language, engine; the service's error message on failure | `test_stt_sends_*`, `test_stt_errors_carry_the_service_message` |
| TTS request | engine, voice, style and language passed; 24 kHz only | `test_tts_sends_*`, `test_tts_refuses_another_sample_rate` |
| Turn text | the person's words since the agent spoke, joined | `test_turn_text_joins_the_words_since_the_agent_spoke` |
| LLM stream | tokens out, events forwarded, `replace_last` false, true, false | `test_llm_streams_tokens_forwards_events_and_undoes_unheard_turns` |
| LLM failure | an error, and exactly one HTTP call | `test_llm_failures_are_reported_and_never_retried` |
| Interim words | they grow while the person speaks; one final at the end | `test_live_stt_sends_growing_interim_words_then_one_final` |
| Preflight | sent at the pause; the final has the same text and needs no second STT call | `test_a_preflight_at_a_pause_becomes_the_final` |
| Preflight, then more speech | the final hears the whole piece | `test_a_preflight_is_dropped_when_the_person_goes_on` |
| Preflight off | no preflight event | `test_no_preflight_when_it_is_off` |
| First TTS piece | the clause, a short first sentence untouched, no cut in "12,500", a word cut within 60 | `test_first_clause_*`, `test_a_short_first_sentence_*`, `test_a_long_first_sentence_*` |
| The whole worker | `make e2e`: words, personas, a noisy turn, "Sure.", barge-in | `scripts/e2e_voice.py` |
| Turn behavior and latency | `python -m orb_bench run turn` | orb-bench LLD |

## 5. Security
- **Local only:** the LiveKit dev server, the agent service and the speech service are on 127.0.0.1, and so is the
  worker's health server (LiveKit's default is every interface).
- **No cloud models:** the turn detector is pinned to the local `v1-mini`; interruptions use mode "vad". The cloud
  turn model, the adaptive barge-in model and the noise cancellation (BVC) are billed LiveKit services and are off.
- **Secrets:** the worker reads only `ORB_*` and `LIVEKIT_*` keys from the repo `.env`, so other secrets in it never
  enter the process. The worker does not start with the dev key of `livekit-server --dev`, or a secret of fewer
  than 32 characters, unless `LIVEKIT_URL` is on this machine (the agent service has the same rule).
- **A cloud STT sends the audio of the call away** (decision 0008): only when the session chose it (the bar of the
  page), and the worker logs a warning at the start of the call. The key is in the speech service, not in this worker.
- **Words stay out of the log** unless `ORB_LOG_TEXT=1`. The data channel carries the transcripts to the room's own
  participants (the person and the bench page).
- **Debug audio** (`ORB_DEBUG_AUDIO_DIR`) saves real voices as WAV and text, mode 0600 in a 0700 folder. It is off
  by default and must be on only with the person's consent.
- **Licenses:** LiveKit Agents is Apache-2.0. The turn detector's LiveKit Model License allows it only inside LiveKit
  Agents.

## 6. Performance
Measured on an Apple M5 (16 GB), 2026-10-22, with Whistle, gemma4:e2b and Kokoro (turn suite run 20261022-205034):
| Measure | Value |
|---|---|
| End of the person's speech → first agent audio | p50 1.73 s, p90 1.93 s |
| Turn commit after the end of speech | p50 0.40 s |
| Barge-in: the 2nd word heard / the agent paused / its audio stopped | 0.95 / 0.98 / 0.97 to 1.06 s |
| `make e2e`, three runs | first audio 1.08 to 1.75 s; barge-in 0.79 to 1.03 s |

The budget of a turn, from the end of speech: preflight 0.2 s → Jev read about 0.5 s → Ollama first token 0.2 to
0.5 s → first piece of text 0.15 s → Kokoro 0.3 s → transport 0.1 s. The turn commit is no longer on this path.
Next steps, with their expected gain:
- Take the Jev read off the path: about 0.5 s. It needs a change in how the agent decides.
- Read the conversation into Ollama's cache while Jev reads: about 0.05 to 0.1 s.
- A shorter first piece (40 characters): about 0.1 s, with a less natural break.

## 7. Edge cases
| Case | What happens | Why |
|---|---|---|
| The person pauses mid-sentence for 1.2 s | no reply; the two parts become one message | the turn detector waits up to 2.5 s; the preflight reply is dropped or undone |
| "Sure." while the agent speaks | the agent goes on; the word joins the next turn | 1 word < 2; LiveKit keeps the transcript |
| Noise only | no turn (4 of 4 clips in the turn suite) | Silero does not start speech |
| The agent's own voice at a low level, no echo canceller | a false turn, and the agent stops | measured in the turn suite; the browser's AEC must remove it |
| The person speaks before any reply audio played | the unheard turn is undone (`replace_last`) | `TurnTracker` |
| The agent service fails | an error event, no retry, the session goes on | a retry would send the turn twice |
| The LLM writes no words | the agent stays silent for the turn; the log has a warning; the page shows "The LLM wrote no words for this turn" | a model can spend its tokens on hidden text; the voice limit is 1,024 tokens (orb-discussion LLD) |
| The LLM thinks silently first | the first word comes after 3 to 11 s on short lines such as "Can you hear me clearly?" (gemma4:e2b) | hidden tokens; choose another LLM in the bar (decision 0007) |
| A cloud STT is chosen | interim calls every 1.0 s; barge-in is slower (about 1.0 to 1.5 s) | the interruption needs 2 words in the interim text |
| A heavy TTS engine (maya1, indic-speak) | the reply stutters | slower than real time on this Mac (decision 0007) |
| A room that the web page did not make | a session of its own, `room-<hash>` | `session_for` |

## 8. Expected results
- `scripts/dev.sh` prints `ready voice` and the worker log shows `agent session ready` for each room.
- `make e2e` prints PASS: the same words, the expected personas, one message for the noisy turn, "Sure." ignored,
  and barge-in within 1.5 s.
- `python -m orb_bench run turn` shows about the numbers of section 6, no early replies and no noise turns.
