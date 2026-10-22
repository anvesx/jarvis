# LLD: orb-plugins (Go plugins for the voice pipeline)

Code: `agentllm/`, `speech/` · Tests: 12 · Status: built · Last update: 2026-10-22

**Four plugins connect lk-go-agent-sdk to orb's services:** the agent service acts as the LLM, and the speech
service gives speech to text, text to speech and the end-of-turn detector.

| Plugin | SDK interface | Calls |
|---|---|---|
| `agentllm.LLM` | `llm.LLM` | `PUT /v1/sessions/{id}`, `POST /v1/sessions/{id}/turns` on the agent service |
| `speech.Recognizer` | `stt.Recognizer` (wrap it with `stt.NewStreamAdapter`) | `POST /v1/stt` |
| `speech.TTS` | `tts.TTS` | `POST /v1/tts` |
| `speech.TurnDetector` | `voice.AudioTurnDetector` | `POST /v1/turn` |

## 1. Input
- `agentllm`: the SDK's chat context (only the newest user message is used) and a context that the SDK cancels
  on barge-in. The answer is NDJSON events.
- `speech`: 16-bit PCM frames from the SDK, text tokens from the LLM, and the JSON or PCM answers of the speech
  service.

## 2. Algorithm
**agentllm.Chat:** take the newest user text → decide `replace_last` → POST the turn → read NDJSON lines (a
buffer of up to 8 MB for a large `state` event) → `token` → a chat chunk; `error` → the stream error; other
events → the event handler (logs) → the stream closes at the end or on cancel.

`replace_last` is the SDK's own flag on the user message (`llm.Message.Replaces`): the session sets it when the
person went on talking before any audio of the reply played, or when a cut, unheard turn is answered again.
(Until 2026-10-22 the plugin guessed it from a text prefix; whole-turn transcription can change the first words,
so the guess broke.)

**speech.TTS:** tokens are added to a buffer → `cut` takes the first speakable piece → each piece is one POST,
in order → the PCM answer is read in 20 ms frames (480 samples at 24 kHz) → after each piece, an empty frame
with `Final` closes the segment. `Close` (barge-in) cancels the request.

`cut` rules: a piece ends at `.`, `!` or `?` before a space, or at a line break, when it has at least 20
characters. The first piece of a reply may also end at a comma, semicolon or colon before a space after 30 or
more characters, so Kokoro (about 5.5 ms per character) starts sooner. With no end in 220 characters, a piece
ends at the last comma, semicolon or colon (else at a space).

**speech.Recognizer and TurnDetector:** pack the frames as PCM16 (one sample rate, mono) → POST → JSON.

## 3. Maths
| Name | Value | Source |
|---|---|---|
| TTS frame | 480 samples = 20 ms at 24 kHz = 960 bytes | chosen: the output track takes any size; 20 ms keeps barge-in fast |
| Shortest piece | 20 characters | chosen: stops "Dr." and "3.5" from cutting a sentence |
| Longest piece with no sentence end | 220 characters | chosen: bounds the wait for the first audio |
| HTTP | response header timeout 20 s (speech) and 30 s (agent); no total timeout | a reply streams as long as it needs |
| Text-only turn call | returns 1 ("no opinion") | Smart Turn needs audio |

## 4. Test cases
| Case | Expected result | Test |
|---|---|---|
| A turn streams | the reply text; `read, decision, reply, check` reach the handler; the body has only the newest message and channel voice | `agentllm/TestChatStreamsTokensAndForwardsEvents` |
| Service errors | the `{"error"}` message reaches `Err()`; no user message fails | `TestChatReportsServiceErrors` |
| Barge-in | the stream closes after cancel | `TestChatStopsWhenCancelled` |
| Session setup | `PUT` with the topic; an unreachable service fails | `TestEnsureCreatesTheSession` |
| `replace_last` | follows the SDK's `Replaces` flag: false, then true | `TestReplaceLastFollowsTheSDKFlag` |
| Recognizer | sends PCM16 at 16 kHz with the language; reads text and length | `speech/TestRecognizerSendsPCMAndReadsText` |
| Recognizer errors | the service message; mixed rates refused | `TestRecognizerErrorsCarryTheServiceMessage` |
| Turn detector | sends the audio; text-only and empty calls return 1 | `TestTurnDetectorUsesTheAudio` |
| Sentence cutting | five cases (short, a number, `?`, a line break, a long text) | `TestCutSentences` |
| The first piece | may end at a clause after 30 characters; later pieces wait; "1,000" is not a clause end | `TestFirstPieceMayEndAtAClause` |
| TTS | two sentences, two requests, 20 ms frames, two `Final` markers | `TestTTSSynthesizesSentenceBySentence` |
| TTS barge-in | the stream stops after `Close` | `TestTTSStopsWhenClosed` |

## 5. Security
The plugins call only the URLs they are given (default 127.0.0.1). They carry no secret. Error messages repeat
the service's own message, which never holds a secret or a transcript.

## 6. Performance
One keep-alive HTTP connection per service (up to 8 idle per host). On one machine a call costs about 1 ms more
than an in-process call. Kokoro's first audio per piece (0.28 to 0.62 s) dominates; see orb-speech.

## 7. Edge cases
| Case | What happens | Why |
|---|---|---|
| A cut inside a number ("3.5") | no cut | a cut needs a space after the mark |
| "e.g. this" | may cut after "e.g." when 20 characters came before | a known, harmless break |
| The agent service restarts mid-call | the next turn fails with a clear error; `Ensure` recreates the session on the next job | sessions live in memory |
| The SDK marks a turn `Replaces` | the plugin sends `replace_last`; the server checks again that the old words match the start of the new text (60 %) | protects a real earlier turn |

## 8. Expected results
`make test` passes the 11 tests with `-race`. In a call, the worker log shows `agent decision` and
`reply check` lines for every turn.
