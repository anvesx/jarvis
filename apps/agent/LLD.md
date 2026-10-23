# LLD: apps/agent (the agent service and the web chat)

Code: `src/orb_agent/` · Tests: `tests/test_server.py` (19), `tests/test_access.py` (13), `tests/test_livekit_proxy.py` (6),
`tests/test_bench.py` (4) · Status: built · Last update: 2026-10-23

**The agent service hosts every agent behind one HTTP API, serves the page (the voice call and the bench tools), and
signs LiveKit tokens for voice. It serves this computer: a request from a tunnel or another device needs a key.**
Today it hosts one agent, `discussion`, with one instance for each LLM that a session asks for.

## 1. Input
| Route | Body | Limits | Answer |
|---|---|---|---|
| `POST /v1/sessions` | `{"topic", "agent"?, "config"?}` | topic 1 to 2,000 characters; config below | 201 and the state |
| `PUT /v1/sessions/{id}/config` | the config | the LLM cannot change after the start | the config |
| `GET /v1/models` | none | | `{"llm": {"default", "installed"}}` from Ollama's list |
| `GET /v1/sessions` | none | | `[{id, topic, agent}]` |
| `GET /v1/sessions/{id}[?since=v]` | none | ID rule (orb-core) | the state, or `{"version": v, "unchanged": true}` |
| `PUT /v1/sessions/{id}` | `{"topic", "agent"?}` | | the state; creates the session when it is missing |
| `POST /v1/sessions/{id}/turns` | `{"text", "channel"?, "replace_last"?}` | text 1 to 8,000; channel web, voice, slack or test | NDJSON events |
| `POST /v1/voice/token` | `{"session_id"}` | the session must exist | `{"url", "room", "identity", "token"}` |
| `GET /v1/voice/status` | none | | `{"url", "dev", "local", "proxy", "livekit", "worker"}`, or `null` when voice is off. `proxy`: the signal is forwarded (below). `livekit`, `worker`: whether the LiveKit server (its health page says OK) and the voice worker (a TCP connection) answer |
| `WS /rtc`, `WS /rtc/v1` (any query) | the frames of the LiveKit client | only when the LiveKit server is on this computer; behind the gate | the frames of the LiveKit server (the voice signal), forwarded. The handshake is refused (and `validate` says why) when the server does not answer |
| `GET /rtc/validate`, `GET /rtc/v1/validate` | none | these two paths only | LiveKit's status and text, or 502; any other `/rtc/...` path gives 404 |
| `GET /health` | none | | agents, LLM and the voice status; no secret values |
| `GET /` | none | | the page |
| `GET /bench` | none | | a redirect (307) to `/`: the bench tools are tabs of the page |
| `/v1/speech/engines`, `/v1/speech/stt`, `/v1/speech/tts` | as the speech service | | the speech service, proxied (same origin for the page) |
| `GET /v1/bench/corpus/{group}`, `GET /v1/bench/clip/{group}/{name}` | none | the group from a fixed list; the file inside its folder | a manifest; a WAV |
| `POST /v1/bench/stt` | `{"group", "id", "engine"?}` | lengths ≤ 32, 120, 32 | the text, the reference, WER and ms |
| `POST /v1/bench/runs`, `GET /v1/bench/runs/{id}` | `{"suite", "engines"?, "groups"?, "models"?, "limit"?, "judge"?}` | suite stt, tts, llm or turn; limit 1 to 1000; one run at a time (409) | 202 and the job; its progress |
| `GET /v1/bench/results/{suite}[/{run}]`, `GET /v1/bench/audio/{run}/{name}` | none | a known suite and run; the file inside its folder | the runs; a result; a TTS WAV |

**Every route needs the key when the request is remote** (the access gate, section 2).

The session config: `stt_engine`, `llm_model`, `tts_engine`, `tts_voice` (≤ 400 characters: a maya1 voice is a
description), `tts_style` and `tts_lang`. The service checks `llm_model` against Ollama's installed models. The
speech service checks the rest when the voice worker uses them.

## 2. Algorithm
- **Start:** load settings (`orb_core.config.bootstrap`), make the Jev reader and the Ollama client, build the
  session store with the discussion agent, read the voice settings and the share key.
- **New session:** the store creates it, then `prepare()` runs in the background: it loads a URL from the topic
  and warms Ollama with the system prompt. A config with another LLM gets the agent `discussion@<model>`, made
  once; the voice worker reads the config with `PUT /v1/sessions/{id}` when it joins the room.
- **Turn:** stream the session's events as NDJSON lines. After the stream ends (or the client leaves), wait for
  the background read of the reply and save a snapshot to `data/sessions/<id>.json`.
- **Voice token:** room `orb-<session id>`, a new identity `user-<8 hex>`, grants to join, publish and
  subscribe, valid 60 minutes.
- **Access gate** (`access.py`, an ASGI middleware before every route): a request is *remote* when its Host is not
  `127.0.0.1`, `localhost` or `[::1]`, or when it has a proxy header (`CF-Connecting-IP`, `CF-Ray`, `X-Forwarded-For`,
  `X-Forwarded-Host`, `X-Real-IP`, `True-Client-IP`, `Forwarded`). A local request passes. A remote request passes
  only with the key `ORB_SHARE_KEY`, in the cookie `orb_key`, the header `X-Orb-Key` or a Bearer token. A GET with
  `?key=<key>` sets the cookie (HttpOnly, SameSite=Lax, 30 days, Secure behind https) and redirects (303) to the same
  path without the key. Every other remote request gets 403: a page for a browser, JSON for a script. A websocket
  gets close code 1008. With no key set, every remote request gets 403. A key of fewer than 16 characters stops the
  service at start (decision 0009).
- **Voice status** (`/v1/voice/status` and the `voice` field of `/health`): the LiveKit URL that browsers get, whether it
  is a local address, whether the signal is forwarded, whether the LiveKit server answers (`GET /` says OK, 1.5 s limit; a
  tunnel with no server behind it answers with a page of its own, so a TCP connection would say yes too often) and whether
  the voice worker accepts a TCP connection (0.4 s limit); both at once. The page asks at load and at each press of Talk.
- **Signal forwarder** (`livekit_proxy.py`): when the LiveKit server (`LIVEKIT_URL`) is on this computer, `/rtc` and
  `/rtc/v1` are forwarded to it, frame by frame (text and binary, any size, the query unchanged), with the close code of
  the server passed on. The two `validate` pages are forwarded too. The target is fixed, the routes sit behind the gate,
  and LiveKit still checks the token. The clean-up of a connection is shielded from cancellation. The audio does not
  pass here: it goes straight between the device and the LiveKit server.
- **The page** (`static/index.html`, no build step, one file served from disk): a model bar on top (STT, LLM, TTS,
  voice, language, style; Silero and the LiveKit turn detector are shown as fixed), then five tabs.
  - **Talk** is the voice call. It has an orb (LiveKit's Aura visualizer, ported to the page: the shader and the state
    table are theirs, Apache-2.0, license and notice in `static/vendor/`), a status line, captions of what the STT
    heard and what the TTS speaks (LiveKit's `lk.transcription` text streams, each labelled with its model) and two
    buttons: **Talk / Hang up** and **New session**. It has no start screen and no text box. **Talk** makes a
    session by itself (topic "Open discussion") with the bar's models, gets a token, joins the room with the
    LiveKit browser client 2.22.3 (in `static/vendor/`, works offline) and polls the state of the session every
    700 ms while the call runs. The orb follows the `lk.agent.state` attribute of the agent and the volume of the
    audio tracks. A folded section ("Conversation, numbers and the engine's view") has the log, the next move, the
    six promises, the debts, the last Jev call and the timing of the turns.
  - **Before a call,** the page shows why voice cannot start (`voiceBlock`): the page is not secure (the browser
    blocks the microphone); voice is off on the server; the voice address is local, the signal is not forwarded and the
    page was opened from another device; the LiveKit server does not answer; the voice worker does not run. The first
    three can never work from that page, so Talk is disabled. The last two can change, so Talk stays on, and the check
    runs again at each press. The reason shows in the chip, under the orb and in the label of the orb.
  - **The signal address** (`signalUrl`): a page on another device cannot reach `ws://127.0.0.1:7880`, so it opens the
    signal at its own address (`wss://<page host>/rtc/v1`); the owner at home still connects straight to 127.0.0.1.
    If the audio then cannot connect ("pc connection" in the error), the page says that the audio could not reach this
    computer and that the device needs a network path to it.
  - **STT**, **TTS**, **LLM** and **Suites** are the bench tools (`bench.py`): check one engine on a corpus clip, a
    recording or a typed text, and run the orb-bench suites. A run goes in a background thread; the page polls its
    progress (orb-bench LLD).
  - A new session takes the bar's models as its config. A change of the bar during a session saves the speech models
    to that session (`PUT /v1/sessions/{id}/config`). A different LLM needs **New session**, because the LLM of a
    session is fixed at its start. The voice worker reads the config when you press Talk.
  - A cloud STT engine is marked "(cloud)", a warning shows when it is chosen, and an engine without a key is
    disabled ("not set up"). In the STT tab a cloud engine starts unchecked.
  - An empty reply (the LLM wrote no words) shows as a red line under the orb.
  - The page is dark by default. The Light / Dark chip switches the theme and keeps the choice in the browser
    (`localStorage`, key `orb-theme`).

## 3. Maths
| Name | Value | Source |
|---|---|---|
| Dev credentials | `devkey` / `secret`, only when no key is set and the LiveKit URL is a loopback address | the public defaults of `livekit-server --dev` |
| Remote secret | 32 or more characters, else voice refuses to start | RFC 7518 §3.2: an HS256 key needs 256 bits |
| Token lifetime | 60 minutes | chosen: one call |
| Share key | 16 or more characters; `openssl rand -hex 16` gives 128 bits | 16 hex characters are 64 bits: too few for a service on the internet |
| Key cookie | 30 days | chosen: a person opens the link again after a month |
| Status probe | 1.5 s for the LiveKit health page, 0.4 s for the worker connection, both at once | a local port answers in under 1 ms; a dead remote host must not hold the page |
| Refusal log | one line for each 10 s, with the count | a scanner can send thousands of requests |
| State polling | 700 ms on this computer, 3 s through a tunnel; `since=version` skips unchanged states | chosen: fast enough to follow a spoken turn; a free ngrok account limits requests |

## 4. Test cases
| Case | Expected result | Test |
|---|---|---|
| Page and health | the page loads; health lists the agent and voice; no secret in it | `test_page_and_health` |
| Gate: this computer | 127.0.0.1, localhost and [::1] pass, with and without a key | `test_this_computer_needs_no_key` |
| Gate: default | with no key, every remote request gets 403, and the handler does not run | `test_a_remote_request_is_refused_when_no_key_is_set` |
| Gate: proxy headers, rebinding | a local Host with a tunnel header is remote; another name that points here is remote | `test_a_proxy_header_makes_a_loopback_host_remote`, `test_another_host_name_is_remote_also_when_it_points_here` |
| Gate: the key | the header, a Bearer token and the cookie pass; a wrong, short or long key gets 403 | `test_the_key_lets_a_remote_request_in` |
| Gate: the link | `?key=` sets the cookie and redirects without the key; a wrong key and a POST get 403; the redirect never leaves the site | `test_the_link_*`, `test_the_redirect_never_leaves_the_site` |
| Gate: key rules | under 16 characters stops the service; a blank key means no sharing; the real app is gated | `test_a_short_key_is_refused_*`, `test_the_real_service_is_private_to_this_computer` |
| Gate: log | no key and no query in the log; one line for each period, with the count | `test_a_refusal_is_logged_*`, `test_the_log_gets_one_line_*` |
| Gate: stream, websocket | a stream passes with the key; a websocket closes with 1008 without it | `test_a_stream_passes_*`, `test_a_websocket_*` |
| Voice status | what runs, for a local and for a remote server, with and without a probe; the listening check; the LiveKit health page (OK only: a tunnel page, a 502, a silent server and a closed port all say no) | `test_the_voice_status_*`, `test_host_and_port_*`, `test_livekit_answers_*` |
| Signal forwarder | text and binary frames pass both ways, the query byte for byte, large frames; the close code of the server reaches the client; a server that is down refuses the handshake and `validate` says 502; `validate` gives LiveKit's text and no other path is forwarded; the gate refuses a remote client without the key; no route for a server that is not on this computer | `tests/test_livekit_proxy.py` |
| Voice messages | the page has the checks and the texts | `test_the_page_says_why_voice_cannot_start_on_another_device` |
| Talk tab | the orb, the captions, two buttons, no start screen, the license files | `test_the_talk_tab_*`, `test_the_captions_*` |
| A full turn over HTTP | NDJSON events in order; the Rival; the check passes; `since` works; the snapshot is saved | `test_session_turn_and_save` |
| Bad input | 404 for a missing session; 400 for a bad ID, channel or agent; 422 for empty text | `test_errors_are_clear` |
| `PUT` | creates once; a second call keeps the first topic | `test_put_creates_once` |
| Voice token | signed for `orb-<id>` with room-join rights; 404 for a missing session | `test_voice_token_is_for_the_session_room` |
| Voice off | 503 with the variables to set | `test_voice_off_gives_503` |
| Voice settings | dev credentials only on loopback; no keys on a remote server turns voice off; a short secret is refused | `test_voice_config_rules` |
| Session config | a session with another LLM uses it; the worker reads the config; one agent per model | `test_session_config_picks_an_llm_and_the_worker_reads_it` |
| Bad config | an unknown model gives 400 with the installed list; the LLM cannot change later | `test_session_config_refuses_unknown_models_and_llm_changes` |
| One page | `/bench` redirects to `/`; the five tabs and the six model fields are in the page; every id that the script looks up exists in the page | `test_the_bench_tools_are_tabs_of_the_chat_page` |
| Bench paths | `..` and unknown names never leave their folders | `test_inside_refuses_paths_that_leave_the_folder`, `test_corpus_and_clips_*`, `test_results_*` |
| Bench runs | 400 for an unknown suite, 409 while a run goes, 422 for a limit over 1000 | `test_one_run_at_a_time_and_unknown_suites_are_refused` |

Also checked by hand in headless Chrome with a fake microphone (2026-10-22): a call with the orb and the captions,
for each STT engine. And six browser scenarios as another device, with a host name that is not local: no key, the key
link on plain http, on a secure origin and at phone width, the worker off, and the owner at home (decision 0009).

## 5. Security
- **No user accounts, so the service serves this computer.** It binds 127.0.0.1. A tunnel or a proxy still puts it
  on a network, so the access gate refuses every remote request unless it has `ORB_SHARE_KEY` (decision 0009).
  Everyone with the key can do everything, and iteration 6 adds accounts. The gate trusts the Host header and the
  proxy headers, so a proxy that hides both looks local. A Host that is not a loopback name counts as remote: this
  also stops DNS rebinding.
- **The signal forwarder** is not an open proxy: its target is fixed by `LIVEKIT_URL`, it exists only for a server on
  this computer, it forwards only `/rtc`, `/rtc/v1` and the two `validate` paths (not LiveKit's admin API), it sits behind
  the gate, and LiveKit checks the token. The query (with the token) is never logged.
- **The key** is in the environment or `.env` only. It is in no log line, no health answer and no response. The cookie
  is HttpOnly, and a key in a link works for a GET page load only.
- **Tokens** are signed only for rooms of sessions that exist, with a short lifetime. The dev secret is public:
  it is accepted only for a loopback LiveKit URL.
- **XSS:** the page escapes every text from a person, the LLM or Jev before it renders it (`esc`).
- **The fonts** come from Google Fonts. Offline, the page uses system fonts.
- **The bench API** reads only files inside the bench's folders (`inside()` resolves the path first) and only from
  fixed lists of groups and suites. It starts one suite at a time: a suite uses the same models as a live call, and
  the llm and turn suites spend Jev reads.

## 6. Performance
The API adds well under 1 ms to a turn on one machine; the turn's time is the agent's (orb-discussion LLD). The
gate reads the Host header and a few header names for each request, and compares a key for a remote one. The voice
status makes two TCP connections at once (0.4 s at most). A
state is the whole session: 22.5 KB after 8 messages, about 3 KB per message (measured 2026-10-21).
`since=version` keeps polls small when nothing changed.

## 7. Edge cases
| Case | What happens | Why |
|---|---|---|
| The browser closes during a turn | the agent finishes the read; the snapshot is saved | cancel-safe read (orb-discussion) |
| Two channels send turns to one session | they run one after the other | the session lock |
| The service restarts | live sessions are lost; snapshots stay on disk | sessions live in memory (iteration 6: a store) |
| 200 live sessions | the least recently used one is dropped | the store limit |
| A tunnel and no `ORB_SHARE_KEY` | every remote request gets 403; the browser shows a page that says the orb is private | default-deny |
| A phone opens the tunnel link with the key | the page loads; Talk says that the voice server is on the owner's computer, until `LIVEKIT_PUBLIC_URL` names a server that the phone can reach | decision 0009 |
| The voice worker starts after the page loaded | the check at the press of Talk finds it | the status is asked again |
| The key link is opened again | it works again, and the cookie is renewed | the key is stateless |
| A device on another network | the page and the signal work; the audio does not connect, and the page says that the audio could not reach this computer | the audio goes straight to the addresses of this computer (decision 0009) |
| ngrok's first-visit notice | the person presses Visit Site once; the key in the link survives; the notice shows this computer's IP address | free ngrok account |
| The LiveKit server is down while a tunnel is up | the status says the LiveKit server does not answer; the signal handshake is refused | the status asks the server, not the tunnel |

## 8. Expected results
`make dev` and http://127.0.0.1:8100: the chips show the LLM and voice; **Talk** starts a session and joins the call;
the orb follows the call and the captions show both sides. From a tunnel (`ORB_NGROK_DOMAIN`), the link with `?key=<ORB_SHARE_KEY>` opens the
page once and then works without the key, and Talk reaches the voice server through the same address; with no key, the page
says that the orb is private.
