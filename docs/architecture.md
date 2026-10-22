# orb architecture

Last update: 2026-10-21 (iteration 0)

## The idea in one sentence
**orb is a platform for company agents: each agent is a brain behind one interface, and channels (web, voice,
later Slack) carry turns to it.**

## First principles
1. **A channel never decides.** It turns its medium into text turns and renders events. Text, voice and Slack
   must get the same answer for the same turn.
2. **An agent owns its session.** The chat, the ledger of debts and later the retrieved documents live in the
   agent, not in the channel.
3. **Every decision can be traced.** The discussion agent records each Jev question, its answer and the rule
   that it caused. A later agent records each retrieved source.
4. **Measured, not guessed.** A threshold or a latency in a document has a source.
5. **Local by default.** The LLM, speech and voice transport run on the machine. The only network call in a
   turn is the Jev reader (see Security).

## Layers
```
 CHANNELS            web chat (browser)        voice (browser → LiveKit → Go worker)        Slack (later)
                            │                              │
                            ▼                              ▼
 AGENT SERVICE      apps/agent · one HTTP API · sessions · NDJSON events · LiveKit tokens
                            │
                            ▼
 AGENTS             DiscussionAgent (orb-discussion)       DocAgent, OnboardingAgent, Router (later)
                            │
                            ▼
 COMPONENTS         orb-core: config · Jev reader · Ollama client · agent interface · page loader
                    orb-speech: Whistle STT · Kokoro TTS · Smart Turn          (Python, HTTP on :8200)
                    lk-go-agent-sdk + orb-plugins: VAD · turn logic · STT/TTS/LLM plugins   (Go)
```

## One text turn
```
browser ── POST /v1/sessions/{id}/turns {"text"} ──▶ agent service
   1. Jev reads the message: one request, about 11 yes/no questions          0.49 s median
   2. the debt rules choose the need and the persona                         < 1 ms
   3. Gemma writes the reply with the persona note, token by token           first word 0.34 s, all 0.95 s
   4. Jev reads the reply in the background, and the rules check it          0.35 s median
◀── NDJSON: read · decision · token … · reply · check · state
```

## One voice turn
```
browser mic (echo cancellation, noise suppression) ─▶ LiveKit (:7880) ─▶ Go worker (apps/voice)
   Silero VAD: speech starts above 0.5, ends after 250 ms below 0.35 (hysteresis)
   Whistle (speech service) transcribes each piece                           15 to 50 ms
   Smart Turn (speech service) hears whether the person finished             12 to 69 ms
   finished: wait 200 ms more · not finished: hold up to 2.5 s
   Whistle transcribes the WHOLE turn again in one call                      20 to 60 ms
   the turn goes to the agent service, exactly like a text turn             (steps 1 to 4 above)
   Kokoro speaks the first clause, then each sentence                        first audio 0.21 to 0.62 s
◀── agent audio in the room
Measured 2026-10-22: 1.25 to 1.29 s from the end of a short turn to the first agent audio.
```
While the agent thinks or speaks, speech must last about 550 ms (with voice still there) to interrupt it, so an
echo or an "mm-hmm" does not cut it; real speech stops it in under 1 s. When the person goes on talking before
any audio of the reply played, the two parts become one turn, transcribed as a whole, and the agent undoes the
unheard turn (`replace_last`).

## Where state lives
| State | Where | Lifetime |
|---|---|---|
| Session: chat, turns, ledger of debts, Jev trace | agent service memory | until the service stops; a JSON snapshot after every turn in `data/sessions/` |
| Voice chat context | Go worker | one room |
| Models | `~/.cache/orb/models`, `~/.cache/cactus-needle` | pinned by SHA-256 |
| Secrets | process environment, `.env`, the file named by `JEV_ENV_FILE` | never in logs or snapshots |

## Processes and ports (all on 127.0.0.1)
| Process | Port | Start |
|---|---|---|
| Ollama (Gemma `gemma4:e2b`) | 11434 | `ollama serve` |
| Agent service + web chat | 8100 | `python -m orb_agent.server` |
| Speech service | 8200 | `python -m orb_speech.server` |
| LiveKit server (dev) | 7880, 7881, UDP | `livekit-server --dev --bind 127.0.0.1 --node-ip 127.0.0.1` |
| Voice worker | 8081 (health) | `data/bin/orb-voice dev` |

`make dev` starts all of them and stops all of them with Ctrl+C.

## Security model (iteration 0)
- **Local only.** Every service binds 127.0.0.1. There are no user accounts yet, so no service may be
  reachable from a network.
- **Jev is the one external call.** Each Jev request sends the text of the new message and up to 12 earlier
  messages (600 characters each) to the TypeSafe API. For a company's internal documents (Doc Brain), this must
  be cleared with the company, or the reader must run on a local model.
- **Secrets** are read from the environment and `.env` files. `describe()` and `/health` report only "set" or
  "missing". The voice worker keeps transcripts out of its log unless `ORB_LOG_TEXT=1`.
- **User input is untrusted.** Limits: topic 2,000 characters, message 8,000, session ID 3 to 48 safe
  characters. A URL in a topic is fetched only when every redirect hop resolves to a public address.
- **Models are pinned** by SHA-256, and the speech service works offline.

## Repo layout
```
packages/python/orb-core        config, Jev reader, Ollama client, agent interface, page loader
packages/python/orb-discussion  the debt engine and the discussion agent
packages/python/orb-speech      Whistle, Kokoro, Smart Turn and the speech service
packages/go/lk-go-agent-sdk     the Go voice SDK (from github.com/beingPro007/lk-go-agent-sdk @ 7961913, with orb changes)
packages/go/orb-plugins         Go plugins: the agent as an LLM; speech STT, TTS and turn detector
apps/agent                      the agent service and the web chat
apps/voice                      the voice worker
docs/                           this file, the roadmap, the LLD template, decisions
data/                           the reference chat and its Jev run (test data); sessions, logs and binaries (not in git)
scripts/                        dev.sh (start everything), e2e_voice.py (end-to-end voice test)
```
