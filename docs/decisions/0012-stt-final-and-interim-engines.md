# 0012 · Speech to text: Parakeet hears each piece once; Whistle writes the interim words

Date: 2026-10-24 · Status: accepted · Ticket: STT-1 (docs/design/stt-accuracy.md) · Replaces the engine choice of 0007

## Why
`LiveSTT` makes interim words by sending the whole piece of speech so far to the engine again every 0.3 s. With
Whistle (17 MB, about 25 ms a call) that cost little. Parakeet writes far better text, but the same loop with Parakeet
transcribes a 10 s sentence about 33 times. Each call reads the audio again from the start. The owner asked what
real products do, and for the design that keeps the final text — the text the agent acts on — best.

Interim words are not only captions. LiveKit counts them for `min_interruption_words`: a barge-in needs 2 words, so
"Sure." never cuts the agent. A future UI without captions still needs them.

## What real systems do (research/streaming-stt.md, primary sources)
- Deepgram, Google, Apple, OpenAI and AssemblyAI all send provisional partials and then a final. Every one of them
  says the partials can change. Google adds a `stability` score; Apple calls them "volatile". The app acts only on
  the final.
- No vendor documents whether its partials come from carried encoder state or from re-running. Cache-aware
  streaming models (NeMo FastConformer, Nemotron) do process each frame once.
- No streaming version of Parakeet TDT 0.6B v3 exists. Its model card offers only a buffered script.
- Wispr Flow shows no captions: it transcribes once, after the key is released.
- A fast pass for partials with a stronger pass for the final is a known pattern (Google's two-pass RNN-T + LAS).
  Its known cost is caption text that changes at the end.

## Measured (Apple M4, 16 GB, 2026-10-24)
44 clips: 20 dictation sentences (2–9 s) and 24 EdAcc and LibriSpeech pieces (8–25 s, mean 10.2 s). Every prefix
on a 0.3 s grid was timed, then replayed with the worker's own scheduling (one interim in flight; the GPU call of a
dropped interim still runs, so the final waits for it). Scripts: `bench/scripts/live_*.py` in the research folder.

Long pieces (8–25 s):

| Design | WER | Final, p50 / max | GPU work per piece | GPU busy while speaking | Interim lag |
|---|---|---|---|---|---|
| Parakeet once at the end, no interims | 3.8 % | 236 / 348 ms | 244 ms | 2.4 % | — |
| Parakeet interims every 0.3 s (the loop with Parakeet) | 3.8 % | 342 / 574 ms | 4,831 ms (20×) | 47 % | 0.13 s |
| Parakeet interims every 1.0 s | 3.8 % | 236 / 537 ms | 1,387 ms (5.7×) | 14 % | 0.14 s |
| **Whistle interims every 0.3 s, Parakeet final (chosen)** | **3.8 %** | **236 / 348 ms** | **244 ms GPU + 957 ms CPU** | **2.4 %** | **0.03 s** |
| parakeet-mlx `transcribe_stream`, 1.0 s chunks | 3.6 % | 650 / 1,041 ms | 4,459 ms | 44 % | 0.43 s |
| parakeet-mlx `transcribe_stream`, 0.5 s chunks | 7.6 % | 755 / 1,309 ms | 8,517 ms | 84 % | 0.44 s |
| parakeet-mlx, right context 8–32 frames (cheap) | 48–54 % | 343–417 ms | 6.3–7.0 s | 62–69 % | 0.31–0.34 s |

Short dictation (2–9 s): Parakeet alone 130 ms p50; the Parakeet loop at 0.3 s does 11× the work (1,492 ms GPU) and
keeps the GPU 34 % busy; Whistle interims leave the final at 130 ms.

- **Whistle and Parakeet do not slow each other.** Parakeet on 10 long clips took 245 ms p50 alone and 223 ms with
  Whistle interims running beside it (Parakeet holds the GPU lock `gpu.MPS`; Whistle does not take it).
- **Interims and the final disagree on 9–15 % of words.** Whistle's text differs from Parakeet's final by 8.7 % of
  words on long pieces and 14.5 % on dictation. That is the price of the cheap interims, and it is what "the caption
  changes at the end" means.
- **parakeet-mlx streaming is not streaming here.** Each chunk encodes the right-context window again. With the
  default 256-frame window (20 s), every chunk of a short piece re-encodes all of it, so it costs more than the
  loop. Small windows are cheap but wreck the words (48–74 % WER).
- MLX bf16 batch Parakeet: 157 ms (short) and 305 ms (long) p50, slower than the torch float32 engine (130 and
  236 ms) with the same WER, so the engine stays (decision 0005).

## Decision
- **The final text comes from Parakeet, which hears each piece once.** `ORB_STT_ENGINE` defaults to `parakeet`. The
  preflight and the final both come from it, so they stay equal and LiveKit keeps the early reply.
- **The interim words come from Whistle.** `LiveSTT` takes a second engine, `interim=`. The worker setting is
  `ORB_VOICE_STT_INTERIM_ENGINE` (default `whistle`).
- **An interim never reaches the agent.** The UI should show interim text as provisional and replace it with the
  final, as Apple and Deepgram advise.
- **A cloud engine as the final engine no longer pays for interims.** The interims stay local unless the interim
  engine is a cloud engine; then `ORB_CLOUD_INTERIM_EVERY_S` applies.

## Rejected
- **Parakeet for interims, at any rate.** It does 5.7–20× the GPU work of the final, and in the worst case it delays
  the final by up to 230 ms. The interim words get no better than Parakeet's own final.
- **parakeet-mlx `transcribe_stream`.** It was measured above: slower, more work, or wrong words.
- **One cache-aware streaming model for both** (Nemotron 3.5 streaming, Parakeet EOU). This is closest to the cloud
  products, but no streaming Parakeet v3 exists. NVIDIA's own cards show the streaming models' word errors above
  Parakeet v3's, and they need a new runtime (FluidAudio CoreML or an MLX port). STT-5 measures them on orb's corpus.
- **LocalAgreement (whisper-streaming).** It re-runs a growing buffer and confirms words after 2 agreeing passes. The
  paper reports 3.3 s latency, and its authors now call it outdated.

## Consequences
- The speech service loads Parakeet at start (2.5 GB in float32); Whistle loads on the first interim (17 MB).
- **Memory on a 16 GB Mac.** With the agent LLM gemma4:e2b loaded (7.9 GB), Parakeet float32 pushed the machine
  into swap and its calls took 21–41 s, so `make e2e` failed (2026-10-24). Whistle passed. On such a machine, close
  other large apps or set `ORB_STT_ENGINE=whistle`; a smaller agent LLM is a separate decision.
- Captions can change at the end of a sentence (9–15 % of words). The final is always Parakeet's text.
- Revisit when a streaming model matches Parakeet v3's WER on orb's corpus (STT-5), or when captions go away. The
  interims stay either way, because the barge-in word count needs them.
