# 0008 · Cartesia as an optional cloud speech-to-text engine

Date: 2026-10-22 · Status: accepted, for now (the owner asked for it "for now") · Extends: 0007

## Decision
- **Cartesia Ink-Whisper** is one more choice in the speech-to-text list of the model bar, next to Whistle (the
  default) and Parakeet. It works through Cartesia's batch API (`POST https://api.cartesia.ai/stt`), so the bench's
  STT tab and suites can score it like the local engines.
- **It is a cloud service.** The audio of a call goes to Cartesia. That breaks the rule "local by default" (the only
  other external call of a turn is the Jev reader, which sends text). So the engine is opt-in at every step:
  - It is never the default, and it is unchecked in the bench's engine list.
  - It loads only when `CARTESIA_API_KEY` is set. Without a key the page shows it as "not set up" and disables it.
  - The page warns when it is chosen, and the voice worker logs a warning when a call starts with it.
  - The key lives in `.env` (not in git). The speech service loads only `CARTESIA_*` lines of that file.
- **Cost control:** Cartesia bills 1 credit for each 2 seconds of audio, silence included. The worker sends a growing
  piece again for each interim transcript, and each call pays for the whole piece. So for a cloud engine the worker
  asks every 1.0 s of speech (`ORB_CLOUD_INTERIM_EVERY_S`) and not every 0.3 s, and it skips the warm-up call. For a
  5 s sentence the billed audio drops from about 47 s (9.3 ×) to about 17.5 s (3.5 ×).

## Measured (2026-10-22, Apple M5; a small sample: the first 12 clips of each group; not enough to rank the engines)
| | Cartesia | Parakeet | Whistle |
|---|---|---|---|
| WER, LibriSpeech / synthetic / EdAcc | 0.5 % / 0.5 % / 12.5 % | 0.5 % / 0.9 % / 6.5 % | 3.2 % / 7.4 % / 12.5 % |
| Time for a call (p50) | 144 to 167 ms (HTTPS, kept-open connection) | 119 to 128 ms | 25 to 31 ms |
| Words from 12 noise-only clips | **12 of 12** ("Thank you.") | 3 of 12 | 0 of 12 |

In the browser test of the merged page, a call with Cartesia chosen in the bar worked end to end: the agent heard the
lines and answered them.

## Consequences
- **Noise:** Cartesia is Whisper-based and writes "Thank you." for noise. In a call, Silero stops pure noise before the
  STT. A noise burst that Silero takes for speech would become a turn with "Thank you.".
- **Latency:** a call costs 150 ms or more, against 25 ms for Whistle, and it depends on the network. The turn stays
  inside the budget of decision 0006 (the STT is not on the path after the preflight).
- **A better integration is possible:** Cartesia has a realtime WebSocket API (`wss://api.cartesia.ai/stt/websocket`,
  `ink-whisper`, 16 kHz PCM in chunks of 100 ms, transcript deltas with `is_final`). It bills each second of audio
  once, and it gives interim text at a low delay. It needs a streaming STT plugin in the worker, so it is not part of
  this change. It is on the roadmap.
- **Privacy:** before a real conversation with company data uses this engine, the owner must clear it with the
  company (the same rule as for the Jev reader in the architecture page). Test audio in the bench is synthetic or from
  public research sets.
- **Terms:** Cartesia's terms and data policy apply to the audio that is sent. They were not reviewed for this change.
