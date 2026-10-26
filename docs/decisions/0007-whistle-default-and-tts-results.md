# 0007 · Whistle is the default STT; Kokoro stays the voice; the bench results

Date: 2026-10-22 · Status: accepted; the STT default is changed by 0012 (Parakeet final, Whistle interims) · Changes: the default engine of 0005; adds the TTS and LLM measurements

## Decision
- **Whistle is the default speech-to-text engine** again (the owner's choice: it is about 5 times faster).
  Parakeet stays an engine that a session or the bench can choose, so the choice stays measurable.
- **Kokoro stays the live voice.** maya1 and indic-speak are bench engines: they are slower than real time here.
- **Chat LLMs on the bench:** gemma4:e2b (the default), gemma3:4b, granite4:3b and llama3.2:3b. Ollama's
  `qwen3:4b` is the thinking build (2507), and `think: false` does not stop it, so it fails every voice case.
- **Speech to text also has a cloud choice,** Cartesia (decision 0008).

## STT (orb-bench stt run 20261022-194658, every clip through the call path)
| Engine | LibriSpeech | Synthetic (clean / 20 / 10 / 5 dB) | EdAcc South Asian / US | Text from 24 noise clips | p50 |
|---|---|---|---|---|---|
| Whistle | 5.2 % | 6.0 / 7.3 / 13.8 / 28.5 % | 26.8 / 33.4 % | 45.8 % ("Thank you.") | 17 to 36 ms |
| Parakeet | 2.4 % | 3.0 / 4.2 / 4.8 / 7.7 % | 11.5 / 14.8 % | 29.2 % ("Mm-hmm.") | 89 to 173 ms |

Silero stops the noise first: the turn suite had 0 new turns from 4 noise clips with Whistle.

## TTS (orb-bench tts run 20261022-201558; judge: Parakeet for English, Whisper large-v3-turbo for Hindi)
| Engine | First audio | Real-time factor | Word errors of the judge |
|---|---|---|---|
| Kokoro, English (af_heart) | 0.14 to 0.99 s | 0.09 to 0.12 | 0 % on 4 texts |
| Kokoro, Hindi (hf_alpha) | 0.33 to 0.47 s | 0.10 to 0.12 | 9.1 % Hindi, 25 % Hinglish |
| maya1 (4-bit MLX) | 0.70 to 0.76 s | 1.53 to 1.89 | 0 to 7.7 % |
| indic-speak, Kavya (4-bit MLX) | 0.78 to 0.88 s | 1.42 to 1.64 | 0 % Hindi, 0 to 14 % English, 37.5 % Hinglish |

- **Why the speech LMs cannot be live here:** each token reads the 1.86 GB of weights. The M5 makes 57 to 59 tokens
  per second, and real time needs 82, so the real-time factor is at least 1.41 even with no audio decoding.
- The Hinglish numbers are too high for every engine: the judge writes English words such as "office" in Devanagari.
- **Not measured:** drama-3 does not fit in 16 GB; Fish S1-mini made speech at 4.4 times real time and its license
  is non-commercial (decision 0003).

## LLM (orb-bench llm run 20261022-200105; five voice cases each, Jev in the loop)
| Model | Cases passed | First token (agent, with Jev) |
|---|---|---|
| gemma4:e2b | 5 of 5 | 0.71 to 0.78 s |
| gemma3:4b | 5 of 5 | 0.95 to 1.43 s |
| granite4:3b | 5 of 5 | 0.71 to 0.97 s |
| llama3.2:3b | 4 of 5 (one reply began with the line "assistant"; the reply filter now drops it) | 0.64 to 0.83 s |
| qwen3:4b (thinking build) | 0 of 5 (its thinking became the reply: 3,000 to 10,000 characters, up to 124 s) | 0.89 to 2.48 s |

A voice reply now has two limits, so a model that ignores the length rule cannot talk for minutes: 1,024 tokens for
Ollama (it counts the hidden tokens) and 700 visible characters (the person hears the start of the reply).

**A mistake that the limits had, and its fix (same day):** the first limit was 200 tokens, and in a live call the agent
stayed silent. gemma4:e2b writes 0 to 370 hidden tokens before its first word on short or confusing lines such as "Can
you hear me clearly?", also with `think` set to false, and Ollama's limit counts them. In the call, 2 of 3 turns used
all 200 tokens before the first word. The LLM suite had only clean lines, so it did not catch this. It now has three
meta lines of that call, each run 3 times, and it counts silent replies. The hidden thinking still costs time: the first
word of the first line came after 3 to 11 s in the browser test (the table above shows the first token with Jev, for
clean lines).

## Consequences
- More word errors on accented speech with the default (26.8 % against 11.5 %). The bench shows both engines side
  by side; one setting changes the default (`ORB_STT_ENGINE=parakeet`).
- The speech LMs need `mlx`, `mlx-lm` and `snac` (the `speech-lm` extra of orb-speech) and 4.3 GB of weights
  (`python -m orb_speech.models --all`). Only one of them is loaded at a time.
