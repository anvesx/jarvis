# 0006 · The voice worker moves to LiveKit Agents (Python)

Date: 2026-10-22 · Status: accepted · Replaces: the Go voice part of 0002, the turn-detector reasons of 0003, and
Smart Turn

## Why
In the call of 2026-10-22 at 15:34, the Go worker failed in three ways:
- The VAD marked long stretches with no words as speech.
- Short words that the STT invented from noise ("Yeah.") became turns.
- The agent was cut almost every time it started to speak.

An audit of orb's Silero port found three rules that differed from the reference code. The LiveKit turn detector
hears more than Smart Turn (14 languages, Hindi included), but its license allows it only inside LiveKit Agents
(decision 0003). LiveKit Agents also brings tested parts: Silero, interruptions with a word count, false-interruption
resume, and preemptive generation.

## Decision
- **One Python worker on LiveKit Agents 1.8.8** (`apps/voice-agent`, package `orb_voice`, Apache-2.0). It replaces the
  Go worker (`apps/voice`), the copy of the Go SDK (`packages/go/lk-go-agent-sdk`) and the Go plugins
  (`packages/go/orb-plugins`). They are removed from the repo; git history keeps them (commit 2d0ef66).
- **Fixed parts:** LiveKit's Silero plugin with its defaults, and the local turn detector
  `inference.TurnDetector(version="v1-mini")`. In dev mode the default is the cloud model, so the version is pinned.
- **orb's parts as plugins:** STT and TTS call the speech service (any engine per session), and the LLM calls the
  agent service, as before (`channel="voice"`, `replace_last` undo).
- **Interruptions:** mode "vad" (the "adaptive" mode is a billed cloud model), 0.5 s of speech and 2 words, with
  false-interruption resume. So "Sure." and "Mm-hmm." never cut the agent.
- **Low latency, measured one change at a time with the turn suite:**
  - Preemptive generation also makes the audio early (`preemptive_tts`).
  - A preflight transcript after 0.15 s of silence starts the reply about 0.4 s before the turn ends. The final
    transcript reuses its text, so LiveKit keeps the early reply.
  - The first TTS piece ends at a comma after 30 characters, or at a word boundary within 60 characters.
- **Smart Turn is removed** from the speech service, with its pin and its endpoint.

## Measured (Apple M5, 16 GB; gemma4:e2b and Kokoro)
| | Go worker (Parakeet) | LiveKit worker, first version | LiveKit worker, now |
|---|---|---|---|
| End of speech → first agent audio, `make e2e` | 1.84 to 2.77 s | 1.85 to 3.56 s | **1.08 to 1.75 s** |
| Same, turn suite p50 / p90 (Whistle) | not run | 2.59 / 3.33 s | **1.73 / 1.93 s** |
| Barge-in: the agent's audio stops, `make e2e` | | 0.94 to 1.12 s | **0.79 to 1.03 s** |
| Early replies on 3 unfinished sentences (1.2 s pause) | | 0 | 0 |
| New turns from 4 noise clips | | 0 | 0 |
| "Sure." while the agent speaks | ignored | ignored | ignored |

`make e2e` passes 3 of 3 runs with the new worker.

Where the 1.7 s goes now: the preflight waits 0.2 s; the Jev read takes about 0.5 s; Ollama's first token 0.2 to
0.5 s; the first piece of text 0.15 s; Kokoro 0.3 s; transport 0.1 s. The turn commit (0.4 s) is no longer on the
path. A lower floor needs the Jev read off the path, which changes how the agent decides.

## Not used
- LiveKit's noise cancellation (BVC) and the adaptive interruption model: cloud services.
- The text turn detector `MultilingualModel`: deprecated.
- `max_retries` above 3 for preemptive replies: each early reply costs one Jev read.

## Known limits
- **Echo:** the agent's own voice at a low level, with no echo cancellation, becomes a turn and stops the agent
  (the turn suite's echo case). Browsers cancel echo; a live call must confirm it.
- **A backchannel is kept:** "Sure." during agent speech does not interrupt, but LiveKit puts it in front of the
  next turn ("Sure. Explain in three sentences …").
- **Licenses:** the LiveKit Model License allows the turn detector only inside LiveKit Agents. orb must stay on
  LiveKit Agents to use it.
