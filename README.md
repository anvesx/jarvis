# orb

**orb is a platform for company agents.** Each agent is a brain behind one interface, and channels (a web chat,
voice, later Slack) carry turns to it. The goal is a company-level Doc Brain that answers questions from a
company's own documents; the [roadmap](docs/roadmap.md) builds it one feature per iteration.

**Iteration 0 (2026-10-21)** ships the foundation: the discussion agent, which does not just agree, in text and
in local voice.

```
browser ─ text ─▶ agent service ─▶ Jev reads · debt rules choose a persona · Gemma speaks
browser ─ voice ─▶ LiveKit ─▶ Go worker: Silero VAD · Whistle · Smart Turn ─▶ agent service ─▶ Kokoro ─▶ room
```

## Quick start (macOS, Apple Silicon)
```sh
make setup      # Homebrew libraries, the `orb` conda env, the speech models (once)
make test       # 136 tests, no network, no API cost
make dev        # every service; open http://127.0.0.1:8100
make e2e        # in a second terminal: a synthetic person talks to the voice agent
```
The Jev key comes from `JEV_API_KEY`, or from the file that `JEV_ENV_FILE` names (default `~/jev-test/.env`).
See `.env.example` for every setting.

## Layout
| Path | What it is | Design |
|---|---|---|
| `packages/python/orb-core` | settings, the Jev reader, the Ollama client, the agent interface, the page loader | [LLD](packages/python/orb-core/LLD.md) |
| `packages/python/orb-discussion` | the debt engine and the discussion agent | [LLD](packages/python/orb-discussion/LLD.md) |
| `packages/python/orb-speech` | Whistle, Kokoro, Smart Turn, and the speech service | [LLD](packages/python/orb-speech/LLD.md) |
| `packages/go/lk-go-agent-sdk` | the Go voice SDK, with orb's turn-detection changes | [changes](packages/go/lk-go-agent-sdk/ORB_CHANGES.md) |
| `packages/go/orb-plugins` | Go plugins: the agent as an LLM; speech STT, TTS and turn detector | [LLD](packages/go/orb-plugins/LLD.md) |
| `apps/agent` | the agent service and the web chat | [LLD](apps/agent/LLD.md) |
| `apps/voice` | the voice worker | [LLD](apps/voice/LLD.md) |
| `docs/` | [architecture](docs/architecture.md), [roadmap](docs/roadmap.md), [LLD template](docs/lld-template.md), [decisions](docs/decisions/) | |
| `data/` | the reference chat and its recorded Jev run (test data); `sessions/`, `logs/`, `bin/` are not in git | |
| `scripts/` | `dev.sh` starts everything; `e2e_voice.py` tests voice end to end | |

`packages/` are reusable parts: they never import an app. `apps/` wire them into products.

## Rules of the repo
- **Every component has an LLD** with eight sections: input, algorithm, maths, test cases, security,
  performance, edge cases, expected results ([template](docs/lld-template.md)).
- **Numbers are measured** and dated, or their source is named.
- **Local by default.** Every service binds 127.0.0.1; the only external call in a turn is the Jev reader.

## Measured (Apple M5, 2026-10-21)
| | |
|---|---|
| Jev reads a person message | 0.49 s median, 11 yes/no questions, about $0.00007 |
| Gemma first token | 0.34 s median |
| Person stops speaking → first agent audio | 1.40 s and 2.11 s |
| Whistle / Smart Turn / Kokoro first audio | 15 to 26 ms / 12 to 47 ms / 0.28 to 0.62 s |
