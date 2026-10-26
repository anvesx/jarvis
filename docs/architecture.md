# orb architecture

Last update: 2026-10-22 (iteration 0, voice on LiveKit Agents)

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
   turn is the Jev reader, unless a session chooses the Cartesia speech-to-text (see Security).
6. **Every model choice is measured.** orb-bench scores each STT, TTS and LLM on the same test audio and cases,
   and the bench page switches them for a live call.

## Layers
```
 CHANNELS            web chat (browser)        voice (browser → LiveKit → voice worker)     Slack (later)
                            │                              │
                            ▼                              ▼
 AGENT SERVICE      apps/agent · one HTTP API · sessions · NDJSON events · LiveKit tokens
                            │
                            ▼
 AGENTS             DiscussionAgent (orb-discussion)       DocAgent, OnboardingAgent, Router (later)
                            │
                            ▼
 COMPONENTS         orb-core: config · Jev reader · Ollama client · agent interface · page loader
                    orb-speech: Whistle, Parakeet or Cartesia (cloud) STT · Kokoro, maya1 or indic-speak TTS (:8200)
                    voice worker (apps/voice-agent): LiveKit Agents · Silero · turn detector · plugins
                    orb-bench: test audio · STT, TTS, LLM and turn suites · tabs of the one page (/)
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
browser mic (echo cancellation, noise suppression) ─▶ LiveKit (:7880) ─▶ voice worker (apps/voice-agent)
   Silero VAD (LiveKit plugin): speech starts above 0.5; a piece ends after 0.55 s of silence
   Whistle transcribes the piece every 0.3 s while the person speaks (interim words; decision 0012)
   Parakeet writes the preflight and the final: it hears each piece once
   after the final, off the turn path: the guarded LLM polish (qwen2.5:1.5b) rewrites the caption (decision 0013)
   0.15 s into a pause: a preflight transcript ─▶ the reply starts early (preemptive, with its audio)
   LiveKit's turn detector (v1-mini, local) hears whether the person finished:
      finished: commit 0.3 s after the speech · not finished: hold up to 2.5 s; the early reply is dropped
   the turn goes to the agent service, exactly like a text turn             (steps 1 to 4 above)
   Kokoro speaks a short first piece (a clause, or 60 characters), then each sentence
◀── agent audio in the room
Measured 2026-10-22 (Whistle, gemma4:e2b, Kokoro): first agent audio p50 1.73 s, p90 1.93 s after the person
stops (turn suite); 1.08 to 1.75 s in `make e2e`. Budget: preflight 0.2 s, Jev 0.5 s, Ollama first token 0.2 to
0.5 s, first piece 0.15 s, Kokoro 0.3 s, transport 0.1 s.
```
While the agent speaks, the person's speech pauses it after 0.5 s and 2 words; with no words in 2 s it resumes.
So "Sure." and "Mm-hmm." never cut it, and a real sentence stops it in about 1 s. When the person goes on talking
before any audio of the reply played, the two parts become one turn, and the agent undoes the unheard turn
(`replace_last`). Interruptions are off for the first 3 s of agent speech in a call, while the browser's echo
canceller adapts.

## Where state lives
| State | Where | Lifetime |
|---|---|---|
| Session: chat, turns, ledger of debts, Jev trace | agent service memory | until the service stops; a JSON snapshot after every turn in `data/sessions/` |
| Voice chat context | voice worker (LiveKit Agents) | one room |
| Models | `~/.cache/orb/models`, `~/.cache/cactus-needle` | pinned by SHA-256 |
| Secrets | process environment, `.env`, the file named by `JEV_ENV_FILE` | never in logs or snapshots |

## Processes and ports (the signal and HTTP ports are on 127.0.0.1)
| Process | Port | Start |
|---|---|---|
| Ollama (Gemma `gemma4:e2b`) | 11434 | `ollama serve` |
| Agent service + web chat | 8100 | `python -m orb_agent.server` |
| Speech service | 8200 | `python -m orb_speech.server` |
| LiveKit server (dev) | 7880, 7881, UDP 7882 | `livekit-server --dev --bind 127.0.0.1 --node-ip 127.0.0.1` |
| Voice worker | 8081 (health) | `python -m orb_voice dev --no-reload` |
| ngrok (optional) | none | `ngrok http 127.0.0.1:8100 --url https://$ORB_NGROK_DOMAIN`, run by `make dev` when the domain is set |

`make dev` starts all of them and stops all of them with Ctrl+C. `make check` starts them, tests them and stops them. The LiveKit dev server listens for media (TCP 7881,
UDP 7882) on every interface of the computer; a session needs its signal port (7880, loopback only) and a token.
With `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET` set, `make dev` runs the server with those keys; with `LIVEKIT_URL` on
another host, it starts no LiveKit server (decision 0009). The agent service forwards the signal (`/rtc`) to the local
server, so the page and the signal share one address; the audio does not pass there.

## Security model (iteration 0)
- **Local only, and a key for anything else.** Every service binds 127.0.0.1. There are no user accounts yet, so the
  agent service refuses every request that comes through a tunnel or from another device (a Host that is not a
  loopback name, or a proxy header) unless it has `ORB_SHARE_KEY`. With no key set, every such request is refused
  (decision 0009). Everyone with the key can do everything. The speech service is not reachable except through this
  service.
- **Jev is the one external call that is always on.** Each Jev request sends the text of the new message and up to
  12 earlier messages (600 characters each) to the TypeSafe API. For a company's internal documents (Doc Brain),
  this must be cleared with the company, or the reader must run on a local model.
- **Cartesia is an optional second external call, and it sends audio.** Only a session that chose the Cartesia STT in
  the model bar uses it (decision 0008). It needs `CARTESIA_API_KEY`, it is never the default, and the page and the
  worker warn when it is on. The same clearance rule applies before it hears company conversations.
- **Secrets** are read from the environment and `.env` files. `describe()` and `/health` report only "set" or
  "missing". The voice worker keeps transcripts out of its log unless `ORB_LOG_TEXT=1`.
- **User input is untrusted.** Limits: topic 2,000 characters, message 8,000, session ID 3 to 48 safe
  characters. A URL in a topic is fetched only when every redirect hop resolves to a public address.
- **Models are pinned** by SHA-256 (and Hugging Face commits), and the speech service works offline. No code from a
  model download runs; the two pickle files load with `weights_only=True`.
- **No LiveKit cloud models:** the turn detector runs locally (`v1-mini`); the adaptive barge-in model and the noise
  cancellation are billed cloud services and stay off.

## Repo layout
```
packages/python/orb-core        config, Jev reader, Ollama client, agent interface, page loader
packages/python/orb-discussion  the debt engine and the discussion agent
packages/python/orb-speech      the speech service: Whistle or Parakeet; Kokoro, maya1 or indic-speak
packages/python/orb-bench       the test audio and the scored suites
apps/agent                      the agent service, the web chat and the bench page
apps/voice-agent                the voice worker on LiveKit Agents
docs/                           this file, the roadmap, the LLD template, decisions
data/                           the reference chat and its Jev run (test data); sessions, logs and the bench data (not in git)
scripts/                        dev.sh (start everything), e2e_voice.py (end-to-end voice test), stt_bench.py (decision 0005)
```
