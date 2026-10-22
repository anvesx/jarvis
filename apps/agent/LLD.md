# LLD: apps/agent (the agent service and the web chat)

Code: `src/orb_agent/` · Tests: `tests/test_server.py` (7) · Status: built · Last update: 2026-10-21

**The agent service hosts every agent behind one HTTP API, serves the web chat, and signs LiveKit tokens for
voice.** Today it hosts one agent: `discussion`.

## 1. Input
| Route | Body | Limits | Answer |
|---|---|---|---|
| `POST /v1/sessions` | `{"topic", "agent"?}` | topic 1 to 2,000 characters | 201 and the state |
| `GET /v1/sessions` | none | | `[{id, topic, agent}]` |
| `GET /v1/sessions/{id}[?since=v]` | none | ID rule (orb-core) | the state, or `{"version": v, "unchanged": true}` |
| `PUT /v1/sessions/{id}` | `{"topic", "agent"?}` | | the state; creates the session when it is missing |
| `POST /v1/sessions/{id}/turns` | `{"text", "channel"?, "replace_last"?}` | text 1 to 8,000; channel web, voice, slack or test | NDJSON events |
| `POST /v1/voice/token` | `{"session_id"}` | the session must exist | `{"url", "room", "identity", "token"}` |
| `GET /health` | none | | agents, LLM, voice; no secret values |
| `GET /` | none | | the web chat |

## 2. Algorithm
- **Start:** load settings (`orb_core.config.bootstrap`), make the Jev reader and the Ollama client, build the
  session store with the discussion agent, read the voice settings.
- **New session:** the store creates it, then `prepare()` runs in the background: it loads a URL from the topic
  and warms Ollama with the system prompt.
- **Turn:** stream the session's events as NDJSON lines. After the stream ends (or the client leaves), wait for
  the background read of the reply and save a snapshot to `data/sessions/<id>.json`.
- **Voice token:** room `orb-<session id>`, a new identity `user-<8 hex>`, grants to join, publish and
  subscribe, valid 60 minutes.
- **Web chat** (`static/index.html`, no build step): start a session, send turns and render the NDJSON stream,
  show the engine's view (next move, six promises, debts, last Jev call, timing). **Talk** gets a token, joins the
  room with the LiveKit browser client 2.22.3 (in `static/vendor/`, Apache-2.0, works offline), and polls the
  state every 700 ms while the call runs. The page is dark by default; the Light / Dark chip in the header
  switches the theme and keeps the choice in the browser (`localStorage`, key `orb-theme`).

## 3. Maths
| Name | Value | Source |
|---|---|---|
| Dev credentials | `devkey` / `secret`, only when no key is set and the LiveKit URL is a loopback address | the public defaults of `livekit-server --dev` |
| Remote secret | 32 or more characters, else voice refuses to start | RFC 7518 §3.2: an HS256 key needs 256 bits |
| Token lifetime | 60 minutes | chosen: one call |
| State polling | 700 ms, with `since=version` to skip unchanged states | chosen: fast enough to follow a spoken turn |

## 4. Test cases
| Case | Expected result | Test |
|---|---|---|
| Page and health | the page loads; health lists the agent and voice; no secret in it | `test_page_and_health` |
| A full turn over HTTP | NDJSON events in order; the Rival; the check passes; `since` works; the snapshot is saved | `test_session_turn_and_save` |
| Bad input | 404 for a missing session; 400 for a bad ID, channel or agent; 422 for empty text | `test_errors_are_clear` |
| `PUT` | creates once; a second call keeps the first topic | `test_put_creates_once` |
| Voice token | signed for `orb-<id>` with room-join rights; 404 for a missing session | `test_voice_token_is_for_the_session_room` |
| Voice off | 503 with the variables to set | `test_voice_off_gives_503` |
| Voice settings | dev credentials only on loopback; no keys on a remote server turns voice off; a short secret is refused | `test_voice_config_rules` |

Also checked by hand on 2026-10-21 in headless Chrome with a fake microphone: start, a typed turn with the
sidebar filled, Talk (the agent joins the room), Hang up.

## 5. Security
- **No user accounts.** Anyone who reaches port 8100 can read every session. It binds 127.0.0.1 only, and it
  must stay so until iteration 6 adds accounts.
- **Tokens** are signed only for rooms of sessions that exist, with a short lifetime. The dev secret is public:
  it is accepted only for a loopback LiveKit URL.
- **XSS:** the page escapes every text from a person, the LLM or Jev before it renders it (`esc`).
- **The fonts** come from Google Fonts. Offline, the page uses system fonts.

## 6. Performance
The API adds well under 1 ms to a turn on one machine; the turn's time is the agent's (orb-discussion LLD). A
state is the whole session: 22.5 KB after 8 messages, about 3 KB per message (measured 2026-10-21).
`since=version` keeps polls small when nothing changed.

## 7. Edge cases
| Case | What happens | Why |
|---|---|---|
| The browser closes during a turn | the agent finishes the read; the snapshot is saved | cancel-safe read (orb-discussion) |
| Two channels send turns to one session | they run one after the other | the session lock |
| The service restarts | live sessions are lost; snapshots stay on disk | sessions live in memory (iteration 6: a store) |
| 200 live sessions | the least recently used one is dropped | the store limit |

## 8. Expected results
`make dev` and http://127.0.0.1:8100: the chips show the LLM and voice; a topic starts a session; each message
streams a reply and fills the sidebar; Talk joins the call.
