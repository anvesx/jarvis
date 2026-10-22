# 0002 · One agent interface, many channels; the brain in Python, voice in Go

Date: 2026-10-21 · Status: accepted

## Decision
**An agent is a brain behind one interface. A channel only moves turns in and events out.**

- `orb_core.agent` defines `Agent`, `Session` and `Event`. A session takes a user text and streams events:
  `read`, `decision`, `token`, `reply`, `check`, `state`, `error`.
- The agent service (`apps/agent`) hosts the agents behind one HTTP API. A turn is
  `POST /v1/sessions/{id}/turns`, and the answer is a stream of NDJSON events.
- Channels: the web chat (in the agent service), voice (the Go worker in `apps/voice`), later Slack.
  Every channel uses the same API, so an agent works on every channel without change.

**The brain stays in Python. The voice pipeline is Go, on `lk-go-agent-sdk`.**

## Why
- The brain needs the TypeSafe SDK, the debt rules and, next, document retrieval: all Python.
- The voice pipeline needs real-time audio, WebRTC and goroutines. `lk-go-agent-sdk` already does
  STT → LLM → TTS with VAD, barge-in and turn detection, and it is ours to change.
- An HTTP boundary costs about 1 ms on one machine. In return, the voice worker and the agent restart,
  scale and get tested separately.

## Consequences
- The voice worker treats the agent service as its LLM (`orb-plugins/agentllm`). It sends only the newest user
  text: the session lives in the agent service.
- When a person goes on talking before they heard a reply, the worker sends the whole sentence with
  `replace_last=true`, and the agent undoes the unheard turn (see apps/voice/LLD.md).
- The agent does not know yet how much of a reply the person heard after a barge-in. That is a next step
  (see the roadmap).
