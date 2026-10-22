# LLD: orb changes to lk-go-agent-sdk

Upstream: github.com/beingPro007/lk-go-agent-sdk at commit `79619137effd4c1c2d8d28867f59db2f76624d74`
(2026-08-21), copied into orb on 2026-10-21 without its `.git`. Every change is backward compatible: the
interfaces gained optional parts only, the new options default to the old behavior, and all 26 upstream tests
still pass. Last update: 2026-10-22.

| File | Change |
|---|---|
| `voice/turn.go` | `AudioTurnDetector`; the turn audio buffer (60 s for transcription, the last 8 s for the detector) |
| `voice/session.go` | the endpointing decision asks the detector on VAD silence too; whole-turn transcription; the barge-in guard (`WithMinInterruptionDuration`); a cut, unheard turn joins the next turn, or is answered again when no words follow; `DroppedFrames()` |
| `voice/speech_handle.go` | `Played()`: did any audio of this speech reach the output |
| `voice/events.go` | `EventUserTurn` (the committed text), `EventUserSpeechIgnored` |
| `llm/llm.go` | `Message.Replaces`: this user message says the previous, unheard turn again |
| `stt/adapter.go` | `NewStreamAdapter` (a batch `Recognizer` plus a VAD gives a streaming STT); the adapter is also a `Recognizer` |
| `plugins/silero/{vad,detector}.go` | hysteresis: `WithNegativeThreshold` (default threshold − 0.15) |
| tests | 24 new: `voice/turn_test.go` (9), `voice/interrupt_test.go` (9), `stt/adapter_test.go` (4), `plugins/silero/detector_test.go` (+2) |

## Why (from a real call, 2026-10-22)
The speech service log of a live call showed fragments: pieces of 700 ms gave 3 characters, 1,720 ms gave 6.
Pieces of 200 to 540 ms while the agent spoke became text such as "Thank you." and cut the agent off. Smart
Turn, fed with noise and fragments, said "not finished" (0.06 to 0.49), so turns waited the full 2.5 s.
Four causes in the code:
1. The Silero detector had one threshold. A dip below 0.5 between words for 200 ms ended a piece.
2. Each piece was transcribed alone and the texts were joined, so the STT never heard a whole sentence.
3. Any start of speech interrupted the agent, including an echo or a cough.
4. When such speech cut a reply that nobody heard and brought no words, nothing ever answered the turn.

## 1. Input
Room audio frames at the STT rate (16 kHz); VAD events (start, end, and one inference result per 32 ms window);
STT events; turn-detector answers; the recognizer's text for a whole turn.

## 2. Algorithm
**VAD hysteresis (silero).** Speech starts after `minSpeech` of probability ≥ threshold (0.5) and ends after
`minSilence` below the lower line (0.35). A dip between 0.35 and 0.5 does not count as silence.

**Endpointing (session.go).** The session commits a user turn when the user is quiet with text in hand: a VAD
end of speech, a final transcript after the VAD heard silence, or an STT end of speech. With a turn detector,
the session arms the max delay as a safety net and asks the detector off the event loop; ≥ threshold waits the
min delay (or commits at once for an STT end of speech), lower holds until the max delay, an error counts as
"finished". A counter (`gen`) drops answers about an older state of the turn.

**Whole-turn transcription.** When the STT is also a `stt.Recognizer` (the stream adapter is), the session keeps
the turn's audio (up to 60 s) and, at commit, transcribes it in one call inside the generation goroutine. The
joined piece texts are the fallback when that call fails (an `EventError` reports it). `EventUserTurn` carries
the text that the LLM gets. The turn audio restarts (with 1 s of preroll) at a start of speech when the agent is
idle; while the agent is busy it stays, because the speech may continue the unheard turn, and it is cut to the
new speech only when that speech proves to be a new turn.

**Barge-in guard.** With `WithMinInterruptionDuration(d)` and a VAD: a start of speech while the agent thinks or
speaks arms a timer. When it fires and the user still has voice (a VAD window ≥ 0.5 in the last 100 ms), the
agent is interrupted. Otherwise the speech is ignored: no interruption, and its transcript becomes
`EventUserSpeechIgnored`, not a turn. Zero (the default) keeps the old behavior: any start of speech interrupts.

**An unheard reply.** `pumpAudio` marks a handle as played on its first frame. When speech interrupts a reply that
never played, the turn's text is carried: the next commit sends both parts as one user message with
`Replaces: true`, and the cut reply stays out of the chat. When that speech brings no words, the carried turn is
answered again after 1 s (also `Replaces: true`).

**Stream adapter (stt/adapter.go).** A VAD stream cuts the input into pieces with 600 ms of preroll; each piece
goes to the `Recognizer` in order; the stream emits StartOfSpeech, FinalTranscript (when not empty), EndOfSpeech.

## 3. Maths
| Name | Value | Source |
|---|---|---|
| VAD start / end lines | 0.5 / 0.35 (threshold − 0.15, at least 0.01) | the Silero authors' advice; LiveKit Agents uses the same rule |
| Whole turn against joined pieces (Whistle) | word error 5.1 % vs 6.8 % (clean), 3.9 % vs 5.6 % (20 dB SNR), **4.2 % vs 12.9 % (10 dB SNR)** | measured 2026-10-22: 4 turns of 3 clauses with 450 ms pauses, Kokoro voices, pink noise |
| Noise-only pieces into Whistle | 12 of 12 gave empty text (pink and white, 0.3 and 0.7 s, rms 0.003 to 0.03) | measured; "Thank you." in the call was real sound, not pure noise |
| Barge-in guard | timer from the VAD start (which needs 250 ms of voice) + voice in the last 100 ms | chosen: orb uses 400 ms, about 550 ms of voice in all, like LiveKit's 0.5 s |
| Turn audio | 60 s kept; the detector gets the last 8 s | Smart Turn reads 8 s; Whistle transcribes up to 30 s in one pass |
| Resume delay | 1 s after the end of speech with no words | chosen |
| Turn transcription timeout | 5 s, then the piece texts are used | chosen |

## 4. Test cases
| Case | Expected result | Test |
|---|---|---|
| VAD silence, detector "not finished" / "finished" | hold to the max delay / commit after the min delay | `TestVADEndpointingAsksTheTurnDetector`, `TestVADEndpointingConfidentDetector` |
| The user talks again before the detector answers | the old answer is dropped | `TestStaleDetectorAnswerIsDropped` |
| An audio detector | gets the turn's frames and text | `TestAudioTurnDetectorHearsTheTurn` |
| A failing detector | an error event; the turn still commits | `TestFailedDetectorDoesNotHoldTheTurn` |
| STT end of speech after a known answer | commits at once, no second call | `TestSTTEndOfSpeechReusesTheAnswer` |
| Turn audio buffer | keeps the newest audio; `keepLast` works | `TestTurnAudioKeepsTheLastSeconds` |
| The user goes on before any audio / after some audio | one merged turn / a new turn | `TestUnheardReplyMergesWithTheNextTurn`, `TestHeardReplyIsNotMerged` |
| 50 ms of sound while the agent speaks | not interrupted; its text is ignored, not a turn | `interrupt_test.go::TestShortSpeechDoesNotInterrupt` |
| Speech longer than the guard | interrupts; not before the guard | `TestLongSpeechInterrupts` |
| The guard fires in the VAD's hangover after a short word | no interruption | `TestHangoverDoesNotInterrupt` |
| A false interruption of an unheard reply | the turn is answered again, `Replaces: true` | `TestFalseInterruptionResumes` |
| A merged turn | `Replaces: true`; a first turn is not marked | `TestMergedTurnIsMarkedReplaces` |
| A continuation while the agent thinks, first part 1.5 s | transcribed from all 1.7 s of audio | `TestContinuationKeepsTheFirstPartsAudio` |
| A Recognizer STT | the LLM gets the whole-turn text; `EventUserTurn` | `TestWholeTurnIsTranscribed` |
| The whole-turn call fails | the piece texts; an error event | `TestTurnTranscriptionFallsBack` |
| A full input channel | dropped frames are counted | `TestDroppedFramesAreCounted` |
| Silero: a dip between the lines / a drop below the lower line | speech goes on / ends | `TestDetectorHysteresisKeepsSpeechThroughDips` |
| Silero: no lower line set | the old single-threshold behavior | `TestDetectorNoHysteresisByDefault` |
| Stream adapter | Start, Final, End; errors; `EndInput`; rate mismatch | `stt/adapter_test.go` (4) |
| All upstream tests | pass unchanged | `session_test.go`, `detector_test.go`, others |

All voice, STT and Silero tests passed 10 runs in a row with `-race` (2026-10-22).

## 5. Security
No new input from outside the process. The whole-turn audio stays in memory (at most 60 s of 16 kHz PCM,
1.9 MB) and goes only to the configured recognizer.

## 6. Performance
- The detector and the whole-turn transcription run off the event loop; timers bound every wait.
- The whole-turn call with Whistle: 20 to 60 ms for 2 to 9 s of audio (measured in the speech service log).
- A frame snapshot copies frame headers, not samples.
- End to end on 2026-10-22 (apps/voice/LLD.md): first agent audio 1.25 to 1.29 s after a short turn; barge-in
  stops the audio 0.80 to 0.92 s after the person starts.

## 7. Edge cases
| Case | What happens | Why |
|---|---|---|
| The transcript arrives before the VAD start | the detector may run twice for one silence | VAD and STT use separate channels |
| The detector or the recognizer never answers | the max delay commits / the 5 s timeout uses the piece texts | safety nets |
| A real one-word answer while the agent speaks ("No") | ignored when shorter than the guard | the guard's price; the person can say it again |
| An STT without a VAD | the guard does not apply; STT start of speech interrupts at once | the guard needs voice timing |
| A VAD that sends no inference events | counts as voiced, so the guard behaves as a plain timer | backward compatible |

## 8. Expected results
`make test` passes. In a call: pauses inside a sentence do not split it, an echo or a short "mm-hmm" does not
stop the agent, and real speech stops it in under 1 s.

## Upstream
All changes are general, not orb-specific: they can go to github.com/beingPro007/lk-go-agent-sdk as one pull
request with these tests.
