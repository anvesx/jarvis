# Roadmap: from the discussion agent to Doc Brain

Last update: 2026-10-22

**The goal:** a company-level "Doc Brain" that answers questions from a company's own documents, for example a
CA firm that gets client questions about GST claims. Must-have features: a multi-agent interface, an onboarding
agent, Claude-style artifacts and Slack.

**The method:** one feature per iteration. Each iteration starts from first principles, ships an LLD for every
new component (input, algorithm, maths, tests, security, performance, edge cases, expected results), and ends
with a measurable exit check. A later iteration builds on the parts of the earlier ones.

---

## Iteration 0 · Foundation (done, 2026-10-21)
- The monorepo, with reusable packages and apps (decision 0001).
- One agent interface for every channel (decision 0002).
- The discussion agent on the debt engine (decision 0004).
- Local voice: Silero VAD, Whistle, Smart Turn, Kokoro, on the Go SDK (decision 0003).
- Speech to text changed to Parakeet TDT 0.6B v3 on 2026-10-22: 57 % fewer word errors than Whistle on real
  accented speech (decision 0005).
- **Exit check, met:** 136 tests pass. The end-to-end voice test passes: right transcript, right personas,
  first agent audio 1.40 s and 2.11 s after the person stops, barge-in stops the audio.

## Iteration 0.5 · Voice on LiveKit Agents, and the bench (done, 2026-10-22)
- The voice worker is Python on LiveKit Agents, with LiveKit's Silero plugin and its local turn detector; the Go
  worker, the Go SDK copy and Smart Turn are removed (decision 0006).
- Each session can choose its STT, LLM and TTS. Whistle is the default STT; Kokoro stays the voice; maya1 and
  indic-speak are bench engines (decision 0007).
- orb-bench: git-ignored test audio (594 STT clips, noise, echo) and four scored suites. The bench tools and the chat
  are tabs of one page, with one model bar that chooses the models of a chat (STT, LLM, TTS, voice).
- Cartesia as an optional cloud STT choice (decision 0008).
- Lower latency: a preflight transcript starts the reply early; the first TTS piece is short.
- **Exit check, met:** 154 tests pass. `make e2e` passes 3 of 3: first agent audio 1.08 to 1.75 s, barge-in 0.79 to
  1.03 s. Turn suite: p50 1.73 s, p90 1.93 s; no early replies on unfinished sentences; no turns from noise.

## Iteration 0.6 · The call page, and sharing (2026-10-22; voice for other devices is open)
- The first tab of the page is the call: an orb (LiveKit's Aura visualizer), live captions of the STT and the TTS, and two
  buttons (Talk, New session). A session starts by itself when you press Talk.
- The agent service serves this computer. A request from a tunnel or another device needs `ORB_SHARE_KEY`, and with no
  key it is refused (decision 0009). The page says why voice cannot start from a device.
- One address (the owner's ngrok domain) carries the page and the voice signal: the service forwards `/rtc` to the
  LiveKit server that `dev.sh` runs here with its own keys.
- **Exit check, met:** 189 tests pass. A call through the ngrok domain (headless Chrome, a fake microphone) connected,
  the agent heard and answered. Through the real tunnel, every request without the key got 403.
- **Open:** the audio of a device on another network. ngrok's free plan cannot carry it (a TCP endpoint needs a card, and
  there is no UDP). Options: a card on the ngrok account plus a relay, LiveKit Cloud, a small public host, or a VPN.

## Iteration 1 · Knowledge: answers from documents
**First principle:** an answer is only as good as its source, so every claim must point to a passage, and
"the documents do not say" is a valid answer.
- `orb-docs`: ingest PDF, DOCX and web pages; split them into passages; index them (BM25 and a local
  embedding model); search.
- `DocAgent`: retrieve, answer with citations, and refuse when no passage supports the answer.
- Verify with Jev: for each claim of the answer, "Does passage P support claim C?" (a yes/no question). Drop or
  flag every claim with no support.
- **Exit check:** a test set of 50 GST questions with known answers from sample documents. Measure: correct
  answers, wrong answers, correct refusals, and citation accuracy.

## Iteration 2 · Artifacts
**First principle:** some answers are documents, not chat: a checklist, a table of deadlines, a draft letter.
- A new event type `artifact` with a typed body (document, table, checklist) and versions.
- A side panel in the web chat; the voice channel says "I put it in your panel".
- **Exit check:** an agent produces, updates and versions an artifact in one session, and the history shows
  every version.

## Iteration 3 · Multi-agent interface
**First principle:** one session, many specialists, and a person always knows who answers.
- A router agent that picks the agent for each turn (Jev answers "which agent should take this turn?").
- Handoff with a reason; the UI shows the active agent.
- **Exit check:** on a mixed test set, the router picks the right agent for at least 95 % of turns.

## Iteration 4 · Onboarding agent
**First principle:** a company must be able to set up its own Doc Brain without an engineer.
- An agent that leads the setup: connect sources, upload documents, set roles, try test questions.
- Its output is a setup artifact (iteration 2).
- **Exit check:** a new person sets up a sample firm and gets a cited answer in under 15 minutes.

## Iteration 5 · Slack
**First principle:** Slack is one more channel; the agent does not change.
- A Slack adapter (Socket Mode): a thread is a session; mentions become turns; replies stream.
- Permissions: who can ask which agent.
- **Exit check:** the same question gets the same answer in the web chat and in Slack.

## Iteration 6 · Hardening
- User accounts, tenants and isolation between companies; an audit log; deployment; monitoring.
- A decision on the reader for confidential documents: the TypeSafe API, or a local model.

---

## Open items from iteration 0
- **What the person heard.** After a barge-in, the agent service keeps the whole reply. The worker should report
  which sentences played, and the agent should keep only those.
- **Faster voice turns.** Done on 2026-10-22: a preflight transcript starts the Jev read and the reply 0.2 s after
  the person stops, and Kokoro starts on a short first piece. The floor is now about 1.6 s. Next: a decision
  that does not wait for the Jev read (about 0.5 s), which changes how the persona is chosen.
- **Echo in real calls.** With no echo canceller, the agent's own voice becomes a turn (turn suite). Check a real
  browser call on laptop speakers.
- **Tune with real voices.** The VAD, the barge-in guard and the turn timing were fixed on 2026-10-22 from one real
  call and checked with synthetic speech. Next: record a few real calls with `ORB_DEBUG_AUDIO_DIR` (with consent)
  and replay the saved turns offline to tune the values.
- **Bias the STT with the topic.** Topic words such as "database" are sometimes heard wrong ("day or base" with
  Whistle). Whistle has a `keywords` option that the speech service does not use yet; Parakeet has none.
- **Words from noise.** Whistle writes "Thank you." for hum and brown noise (11 of 24 noise clips), and Cartesia for
  all of the first 12; Silero stopped every noise clip in the turn suite. Watch for it in real calls.
- **Silent thinking of gemma4:e2b.** On short lines such as "Can you hear me clearly?" it writes 200 to 700 hidden
  tokens first, so the first word comes after 3 to 11 s (browser test, 2026-10-22). Options: another default LLM
  (the suite has the numbers), or a way to switch the thinking off for real.
- **Cartesia over the realtime WebSocket.** The batch engine pays for the whole piece at each interim call. The
  WebSocket bills each second once and gives interim text at a low delay (decision 0008).
- **An expressive voice.** Map each persona to a tone. maya1 (emotion tags) and indic-speak (style tags) can do it
  but are slower than real time on this Mac (decision 0007); Fish Audio's open model is non-commercial.
- **A non-thinking Qwen.** Ollama's `qwen3:4b` is the thinking build; `qwen3:4b-instruct` would need a 2.5 GB pull.
