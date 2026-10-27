# 0015 · Gemini 3.8 Live as an optional speech-to-speech mode beside the orb pipeline

Date: 2026-10-27 · Status: accepted, live test pending (the owner tests on localhost) · Design: docs/design/gemini-live.md

## Why
The owner wants to try Google's Gemini 3.8 Live (released 2026-09-15): one model that hears, thinks and speaks over
one WebSocket, with 97+ languages, barge-in and its own turn detection. The orb pipeline on this 16 GB Mac runs three
local model families (Whistle/Parakeet or Qwen3, gemma4:e2b, Kokoro), which swap and slow each other down.

## Decision
- **A second voice mode, chosen per session:** `voice_mode` is `orb` (the default, unchanged) or `gemini-live`. The
  call page has a Mode field first in its model bar; Gemini Live shows its model (`gemini-3.8-live`, or the extended
  thinking one) and its voice (30 Gemini voices, default Puck).
- **In Gemini Live, the voice worker runs `AgentSession(llm=google.realtime.RealtimeModel(...))` and nothing else**
  (livekit-plugins-google 1.8.9). Gemini detects the turns on its server; its two transcripts become the captions. The
  speech service, the orb agent's turns and Ollama are not used, so none of their models load (only the worker's small
  Silero VAD, as in every worker process).
- **Gemini gets orb's system prompt** for the session's topic and source (without the per-turn note rule) and short
  spoken-turn rules, from `GET /v1/sessions/{id}/instructions`.
- **No warm-up of the local LLM for a Gemini session** (`prepare(warm=False)`): warming loads gemma4:e2b, 7.9 GB.
- **`scripts/dev.sh --gemini`** starts only LiveKit, the agent service and the voice worker. The agent service then
  offers only Gemini Live (`ORB_GEMINI_ONLY`): it refuses an orb voice session and never warms the LLM.
- **It is a cloud service, opt-in like Cartesia (0008):** never the default; disabled with the reason without
  `GOOGLE_API_KEY`; the page says the audio goes to Google; the worker logs it at every call. The key is read by its
  exact name and never logged or sent to the page.
- **Long calls:** sliding-window context compression and session resumption, so a call can pass one connection
  (about 10 minutes) and the 15-minute audio session limit.

## What Gemini Live does not have (first version)
Jev's read of each message, persona switching, the debt board and promise checks, replace_last undo, and the agent
service's history of the call. The page says so. They can come back as Gemini tool calls (the agent's `decision` and
`check` as async functions): design section 7.

## Rejected
- **Gemini behind orb's agent (half cascade: Gemini text out, orb's TTS).** Native-audio Live models answer only in
  audio; text comes as a transcript.
- **Gemini 4 Argon.** It takes audio in but writes only text, and has no Live API support.
- **Replacing the orb pipeline.** It is local and free per minute, and it keeps the debt engine.

## Consequences
- About $0.023 per minute of talk (audio in $0.005, out $0.018); a free-tier key lets Google use the audio to improve
  its products. A real company call needs the owner's clearance first, as for Cartesia.
- LiveKit Agents moved from 1.8.8 to 1.8.9 (the plugin needs it); the pipeline's tests pass on it.
- Known plugin issues: a reconnect can drop the reply in progress (#7867); a 5 s `generate_reply` timeout (#7481, not
  used here). The live test checks a call longer than 10 minutes.

## Live test (owner)
To fill in after the test: first-audio time, barge-in, captions, Hinglish, a 12-minute call.
