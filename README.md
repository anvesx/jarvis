# orb

**orb is a platform for company agents.** Each agent is a brain behind one interface, and channels (a web chat,
voice, later Slack) carry turns to it. The goal is a company-level Doc Brain that answers questions from a
company's own documents; the [roadmap](docs/roadmap.md) builds it one feature per iteration.

**Iteration 0 (2026-10-21)** ships the foundation: the discussion agent, which does not just agree, in text and
in local voice. **On 2026-10-22** the voice worker moved to LiveKit Agents, and a bench now measures every model.

```
browser ─ text ─▶ agent service ─▶ Jev reads · debt rules choose a persona · Gemma speaks
browser ─ voice ─▶ LiveKit ─▶ voice worker (LiveKit Agents): Silero · Whistle · LiveKit turn detector
                              ─▶ agent service ─▶ Kokoro ─▶ room
```

## Quick start (macOS, Apple Silicon)
```sh
make setup      # Homebrew (LiveKit, Ollama, ffmpeg), the `orb` conda env, the speech models (once)
make test       # 189 tests, no network, no API cost
make dev        # every service; open http://127.0.0.1:8100: the voice call (an orb) and the bench tools are tabs of one page
make e2e        # in a second terminal: a synthetic person talks to the voice agent
make check      # start every service, test them (unit, live and voice end to end), and stop them: about 4 minutes
make bench      # the git-ignored test audio for the bench (once)
```
The Jev key comes from `JEV_API_KEY`, or from the file that `JEV_ENV_FILE` names (default `~/jev-test/.env`).
See `.env.example` for every setting.

**Another device or a tunnel:** the agent service serves this computer. It refuses a request from a tunnel or another
device unless the request has `ORB_SHARE_KEY` (open `https://<host>/?key=<the key>` once). With `ORB_NGROK_DOMAIN`,
`make dev` runs ngrok for that address. The page and the voice signal use that one address; the audio goes straight to
this computer, so a device needs a network path to it. See [decision 0009](docs/decisions/0009-sharing.md).

## Layout
| Path | What it is | Design |
|---|---|---|
| `packages/python/orb-core` | settings, the Jev reader, the Ollama client, the agent interface, the page loader | [LLD](packages/python/orb-core/LLD.md) |
| `packages/python/orb-discussion` | the debt engine and the discussion agent | [LLD](packages/python/orb-discussion/LLD.md) |
| `packages/python/orb-speech` | the speech service: Whistle, Parakeet or Cartesia (cloud); Kokoro, maya1 or indic-speak | [LLD](packages/python/orb-speech/LLD.md) |
| `packages/python/orb-bench` | the test audio and the scored STT, TTS, LLM and turn suites | [LLD](packages/python/orb-bench/LLD.md) |
| `apps/agent` | the agent service and the one page: the voice call, the model bar and the bench tools | [LLD](apps/agent/LLD.md) |
| `apps/voice-agent` | the voice worker on LiveKit Agents | [LLD](apps/voice-agent/LLD.md) |
| `docs/` | [architecture](docs/architecture.md), [roadmap](docs/roadmap.md), [LLD template](docs/lld-template.md), [decisions](docs/decisions/) | |
| `data/` | the reference chat and its recorded Jev run (test data); `sessions/`, `logs/`, `testbench/` are not in git | |
| `scripts/` | `dev.sh` starts everything; `check_stack.sh` starts it, tests it and stops it; `e2e_voice.py` tests voice end to end | |

`packages/` are reusable parts: they never import an app. `apps/` wire them into products.

## Rules of the repo
- **Every component has an LLD** with eight sections: input, algorithm, maths, test cases, security,
  performance, edge cases, expected results ([template](docs/lld-template.md)).
- **Numbers are measured** and dated, or their source is named.
- **Local by default.** Every service binds 127.0.0.1, and a request from a tunnel needs a key; the only external call
  in a turn is the Jev reader.

## Measured (Apple M5, 16 GB)
| | |
|---|---|
| Jev reads a person message | 0.49 s median, 11 yes/no questions, about $0.00007 (2026-10-21) |
| Gemma first token | 0.34 s median (2026-10-21) |
| Person stops speaking → first agent audio | `make e2e` 1.08 to 1.75 s; turn suite p50 1.73 s, p90 1.93 s (2026-10-22, decision 0006) |
| Barge-in: the agent's audio stops | 0.79 to 1.03 s after the person starts (2026-10-22) |
| Speech-to-text word errors, real accented speech | Whistle 26.8 %, Parakeet 11.5 % (bench, decision 0007) |
| Kokoro / maya1 / indic-speak first audio | 0.14 to 0.99 s / 0.70 to 0.76 s / 0.78 to 0.88 s; only Kokoro is faster than real time (decision 0007) |

Model credits: Whistle by Cactus Compute; Parakeet TDT 0.6B v3 by NVIDIA (CC-BY-4.0); Kokoro-82M; maya1 by Maya
Research; indic-speak by Bodhan AI; LiveKit's turn detector and Silero VAD through LiveKit Agents. The orb of the call
page is LiveKit's Aura visualizer from `livekit/components-js` (Apache-2.0; license and notice in
`apps/agent/src/orb_agent/static/vendor/`).
