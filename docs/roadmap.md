# Roadmap: from the discussion agent to Doc Brain

Last update: 2026-10-21

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
- **Exit check, met:** 136 tests pass. The end-to-end voice test passes: right transcript, right personas,
  first agent audio 1.40 s and 2.11 s after the person stops, barge-in stops the audio.

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
- **Faster voice turns.** Two changes would save about 0.4 to 0.8 s: start Kokoro on the first clause instead
  of the first sentence, and start the Jev read on the final transcript, before the turn commits.
- **Tune with real voices.** The VAD, the barge-in guard and the turn timing were fixed on 2026-10-22 from one real
  call and checked with synthetic speech. Next: record a few real calls with `ORB_DEBUG_AUDIO_DIR` (with consent)
  and replay the saved turns offline to tune the values.
- **Bias the STT with the topic.** Whistle can favor given words (`keywords`). Passing the topic's words would fix
  errors such as "database" heard as "day or base".
- **An expressive voice.** Map each persona to a tone. Fish Audio's open model has the markers but a
  non-commercial license (decision 0003).
