# TDD: Gemini 3.8 Live as a speech-to-speech mode, and a black-and-white call page

Status: draft · Last update: 2026-10-26 · Scope: a second voice mode beside orb's own pipeline, and a light UI refresh

**A person should be able to pick "Gemini Live" before a call and talk to one Google model that hears and speaks,
with the same captions and call page as today.** orb's own pipeline (Parakeet → orb agent → Kokoro) stays the
default and keeps working unchanged. The call page also moves to the black-and-white look of the smartrouter
dashboard, with its padding and layout bugs fixed.

## 1. What Gemini 3.8 Live is

Google released it on 2026-09-15 (blog: "Build real-time voice applications with Gemini 3.8 Live"). Facts checked
on 2026-10-26 against Google's docs and the `livekit-plugins-google` 1.8.9 wheel:

| Fact | Value | Source |
|---|---|---|
| Model ids (Gemini API) | `gemini-3.8-live` (stable), `gemini-3.8-live-extended-thinking` | ai.google.dev models page, plugin `LiveAPIModels` |
| In / out | audio (+ video, text) in, **audio out only** (native audio: no TEXT response) | Live API capabilities |
| Audio | in PCM16 16 kHz, out PCM16 24 kHz, over one WebSocket | Live API docs |
| Captions | `input_audio_transcription` and `output_audio_transcription`; the plugin turns them into LiveKit transcripts | Live API capabilities, plugin |
| Turns | server VAD (`automatic_activity_detection`), barge-in (`interrupted`), proactive audio always on | Live API, 3.8 model page |
| Length | one connection ≈ 10 min (GoAway before it ends); a session 15 min audio-only, unlimited with `context_window_compression` (sliding window); `session_resumption` handles live 2 h | session management |
| Thinking | 3.8 Live: fixed, no `thinking_level`; extended thinking: low / medium / high | Live API thinking |
| Tools | async (`NON_BLOCKING`) by default; Google Search; no code execution | Live API tools |
| Languages | 97+, chosen by the model (no language code; steer in the instructions) | blog, capabilities |
| Voices | the Gemini TTS set (Puck, Kore, Charon, Aoede, …; 30 names in the plugin's `Voice`) | plugin `api_proto.py` |
| Price (paid tier) | audio in $3 / 1M tokens ≈ $0.005 / min, audio out $12 / 1M ≈ $0.018 / min | pricing page |
| Data | paid tier: not used to train; logged for abuse monitoring. Free tier: used to improve Google products | Gemini API terms |
| LiveKit | `livekit-plugins-google` 1.8.9 (2026-10-24), `google.realtime.RealtimeModel`, needs `livekit-agents>=1.8.9`; key from `GOOGLE_API_KEY` | PyPI, plugin metadata |

Gemini 4 Argon is not this: it takes audio in but writes only text, and has no Live API support.

## 2. What orb has today

```
browser ──LiveKit──▶ voice worker (AgentSession)
                       stt = LiveSTT(Whistle interim, Parakeet final) ── speech service :8200
                       llm = OrbLLM ── agent service :8100 (Jev reader, personas, debt board, gemma4:e2b)
                       tts = OrbTTS (Kokoro) ── speech service
                     ──"orb.bench" data events──▶ call page (captions, turns table, events)
```

- The worker builds one `AgentSession` per room from the session config (`stt_engine`, `tts_*`) that the agent
  service keeps (`PUT /v1/sessions/{id}`, `SessionConfig`).
- The agent service, not the LLM alone, writes each reply: it reads the person (Jev), picks a persona and a debt,
  and adds a per-turn note. **A speech-to-speech model skips all of that.** It gets one system prompt at the start.
- `load_repo_env` copies only `ORB_` and `LIVEKIT_` keys from `.env` into the worker, so `GOOGLE_API_KEY` does
  not reach it today.
- Installed: `livekit-agents` 1.8.8, no Google plugin.
- Cloud speech is opt-in, never the default, and the page warns that audio leaves the machine (decision 0008).

## 3. Design

### 3.1 Two voice modes, chosen per session

| Mode | `voice_mode` | Hears | Thinks | Speaks |
|---|---|---|---|---|
| orb (default) | `orb` | Whistle + Parakeet / Qwen3 | orb agent (personas, debts) | Kokoro / others |
| Gemini Live | `gemini-live` | Gemini | Gemini, with orb's system prompt | Gemini voice |

The mode is fixed for a call, like the LLM is today (changing it applies to the next call). The session config
gains `voice_mode` (default `orb`), `gemini_model` (default `gemini-3.8-live`) and `gemini_voice` (default `Puck`).

**What Gemini Live loses, on purpose, in this first version:** the Jev read, persona switching, the debt board and
promise checks, replace_last undo, and the agent service's history of the call. The page says so in the mode's
note. Bringing the debt engine back as Gemini tools (async function calls) is a later step (section 7).

### 3.2 Voice worker

```python
if cfg.voice_mode == "gemini-live":
    session = AgentSession(llm=google.realtime.RealtimeModel(
        model=cfg.gemini_model, voice=cfg.gemini_voice, instructions=instructions,
        input_audio_transcription=AudioTranscriptionConfig(), output_audio_transcription=AudioTranscriptionConfig(),
        context_window_compression=ContextWindowCompressionConfig(sliding_window=SlidingWindow()),
        session_resumption=SessionResumptionConfig()), vad=vad)
else:
    ...today's pipeline, unchanged...
```

- `instructions`: from the agent service (`GET /v1/sessions/{id}/instructions`, new): orb's system prompt for the
  session's topic and source, plus a voice rule ("speak in short turns, at most 3 sentences; no lists or markdown").
- No STT load, no polish, no `LiveSTT`, no `OrbTTS`, no turn detector: Gemini detects turns on the server.
  The local Silero VAD stays for the page's "Hearing you…" state only if LiveKit needs it; else it is dropped.
- The same handlers run: `user_input_transcribed` → `transcript` events and the "You" caption;
  `conversation_item_added` and `agent_state_changed` as today. `agent_*` events of the agent service are silent.
- The `session` event says `stt gemini · llm gemini-3.8-live · tts gemini (Puck)`.
- A failed connection (no key, bad key, quota) becomes an `error` event with a clear message, as for the STT load:
  "Gemini Live did not start: the API key was refused. Check GOOGLE_API_KEY in .env." / "…quota or rate limit…" /
  "…no network…".
- The worker logs a warning at the start of every Gemini call: the audio of this call goes to Google.

### 3.3 Agent service

- `SessionConfig`: `voice_mode`, `gemini_model`, `gemini_voice`, checked against lists (the plugin's models and
  voices), stored with the session like the other fields.
- `GET /v1/models` adds `gemini: {available, models, default_model, voices, default_voice}` (available when
  `GOOGLE_API_KEY` is set; the value is never sent). A `gemini-live` config without the key gets 400.
- A session that starts in `gemini-live` never warms the LLM (`prepare(warm=False)`): warming loads gemma4:e2b
  (7.9 GB) for a call that does not use it. A session that started in `orb` mode and switches later has already warmed
  it; New session avoids that.
- `GET /v1/sessions/{id}/instructions` returns the text of 3.2.

### 3.4 Keys and dependencies

- `apps/voice-agent/pyproject.toml`: `livekit-agents[silero,google]~=1.8.9` (from 1.8.8; one minor step).
- `load_repo_env` (worker and agent service) also copies `GOOGLE_API_KEY`, by its exact name. No other key is added.
- `.env.example`: `GOOGLE_API_KEY=` with a comment (Gemini API, paid tier keeps the audio out of training).
- `scripts/check_stack.sh`: says whether the key is set (never prints it).

### 3.5 Call page

- The model bar gets a **Mode** field first: "orb pipeline" / "Gemini Live (cloud)". With Gemini Live:
  - STT, LLM, TTS, Language and Style fields hide; **Voice** lists the Gemini voices; a **Model** field offers
    3.8 Live and 3.8 Live Extended Thinking.
  - The cloud note (same style as Cartesia's): "Gemini Live sends the audio of this call to Google. Personas and the
    debt board are off in this mode."
  - The option is disabled with the reason when the key is missing.
- Captions, the orb, Talk / Hang up and the events log work as today.

### 3.6 Black-and-white UI (smartrouter look), kept small

Tokens from `~/Developer/smartrouter/dashboard/src/styles.css`:

| Token | Dark | Light |
|---|---|---|
| bg / surface / border | `#000` / `#0c0c0c` / `#1f1f1f` | `#fff` / `#fafafa` / `#e5e5e5` |
| ink / muted | `#fafafa` / `#8b8b8b` | `#0a0a0a` / `#6b6b6b` |
| primary button | white on black text | black on white text |
| accent (state only) | green `#22c55e` for live / ok, red `#ef4444` for errors | `#15803d` / `#dc2626` |

- Font: Inter (one family; JetBrains Mono stays for numbers and labels). Radius 10 px on fields, buttons, cards.
- The orb keeps its LiveKit Aura canvas but in white/grey instead of cyan.
- Fixes found while reading the CSS: the model bar wraps badly between 640 and 900 px (6 fields in auto-fit
  columns); captions jump when the label appears; the details grid has no gap; tool cards and tables have
  uneven padding; header chips wrap under the title on phones. Each is checked at 390, 768 and 1280 px.
- Not in scope: new layouts, new components, the bench tabs' logic.

## 4. Rules

- orb pipeline stays the default; with `voice_mode` unset nothing changes (tests prove it).
- No key, no Gemini: the mode is disabled with a reason, never a silent failure.
- The key is never logged, sent to the page, or written to a session file.
- Each ticket is one commit; tests use fakes (no Google call in tests).

## 5. Measurement (owner, one live call per mode)

| Measure | How | Target |
|---|---|---|
| First audio after you stop | the turns table's Total column | Gemini ≤ 1.0 s (orb today: about 1.5–3 s on this Mac) |
| Barge-in | speak over a long reply | Gemini stops within about 0.5 s |
| Captions | both lines fill in | both show words for every turn |
| Hinglish | 3 mixed sentences | Gemini answers in the same mix |
| Long call | stay 12 min (past one connection) | the call goes on (GoAway + resumption) |

## 6. Tickets

| Ticket | What | Depends on | Done when |
|---|---|---|---|
| GL-0 | Dependencies and key: livekit 1.8.9 + Google plugin, `GOOGLE_API_KEY` passthrough, `.env.example`, `check_stack.sh` | — | worker and agent tests pass on 1.8.9; key set → `check_stack` says so |
| GL-1 | Agent service: `voice_mode`, `gemini_model`, `gemini_voice` in the session config; `gemini` in voice status; `/instructions` | GL-0 | `test_server.py` covers defaults, bad values, status without a key, instructions text |
| GL-2 | Voice worker: Gemini Live branch, captions and bench events, error messages, cloud warning | GL-1 | `test_realtime.py` (fake RealtimeModel builder): options passed, no STT load, errors mapped; pipeline tests unchanged |
| GL-3 | Call page: Mode field, Gemini voice/model, cloud note, disabled reason | GL-1 | page works in both modes; no console errors |
| GL-4 | Black-and-white UI and the padding / layout fixes | — | screenshots at 390 / 768 / 1280 px, dark and light |
| GL-5 | Owner's live test (section 5) and a decision record 0015 | GL-2, GL-3 | numbers in this file |

## 7. Later (not in this work)

- The debt engine as Gemini tools: the agent service's `decision` and `check` as async function calls, so Gemini
  Live keeps personas and promises.
- Writing the Gemini call's transcript into the agent session's history, so a later orb-pipeline call knows it.
- Gemini 3.5 Transcribe (`gemini-3.5-transcribe-live`) as a cloud STT engine for the orb pipeline (85+ languages,
  code-switching, 1,000-term vocabulary): a candidate for Hinglish beside Qwen3.

## 8. Risks

- **Plugin bugs (LiveKit issues, last 2 months):** reconnects drop the reply in progress (#7867), barge-in with
  manual activity detection (#8003, we keep server VAD), a 5 s `generate_reply` timeout (#7481). First-version
  mitigation: no tools, server VAD, no `generate_reply` greeting.
- **livekit-agents 1.8.8 → 1.8.9** may change the pipeline: the full worker test suite runs before GL-1.
- **Cost and privacy:** about $0.023 per minute of talk; a free-tier key lets Google train on the audio. The page
  and the log say so; a real company call needs the owner's clearance first (as decision 0008).
- **Rate limits** are per project and not published; a refused session gets the quota message.
