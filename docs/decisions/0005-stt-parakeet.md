# 0005 · Speech to text: Parakeet TDT 0.6B v3 replaces Whistle

Date: 2026-10-22 · Status: changed by 0007 (Whistle is the default again; Parakeet stays an engine) · Changes: the speech-to-text row of decision 0003

## Why
After the voice fixes of 2026-10-22 (VAD hysteresis, whole-turn transcription, barge-in guard), real calls still
gave bad transcripts. The audio path was checked first:
- The STT gets audio from before the VAD start: 600 ms for each piece, 1 s for each turn. The first word is not cut.
- LiveKit's media-sdk decodes Opus and resamples 48 kHz → 16 kHz with soxr ("LQ"), which has an anti-alias filter.
- **Measured:** Whistle makes 25.6 % word errors on the original clips and 26.3 % on the same clips after the
  call path (Opus 32 kbps, soxr LQ). The codec and the resampler add 0.7 points; they are not the cause.

The cause is the model. Whistle is a 16.9 MB file, smaller than Whisper "tiny" (39 M parameters). It is accurate
on clean synthetic speech (4.2 % at 10 dB SNR, decision 0003), but not on real conversational speech.

## Measurement
`scripts/stt_bench.py`: 182 clips of real conversational English from the EdAcc test split. 142 clips come from four
South Asian speakers (two Indian English, two with Sinhalese as their first language) and 40 from US English speakers
as a control. Every clip went through the call path. Apple M5, 16 GB; each clip is timed alone, after one warm-up.

| Engine | South Asian WER | US WER | Median time per clip | p90 | Size | License |
|---|---|---|---|---|---|---|
| Whistle 2.0.0 (before) | 26.3 % | 34.6 % | 20 ms | 40 ms | 17 MB | Apache-2.0 |
| **Parakeet TDT 0.6B v3** | **11.2 %** | **15.0 %** | **109 ms** | **191 ms** | 2.5 GB | CC-BY-4.0 |
| Parakeet CTC 0.6B | 14.4 % | 18.5 % | 63 ms | 110 ms | 2.4 GB | CC-BY-4.0 |
| Whisper large-v3-turbo (fp16) | 14.9 % | 20.9 % | 341 ms | 391 ms | 1.6 GB | MIT |

Parakeet TDT v3 makes 57 % fewer errors than Whistle on the South Asian speakers and 57 % fewer on the US control.
It also writes punctuation and capital letters; the CTC model does not. Whisper is three times slower on this
Mac, through `transformers` on the GPU.

EdAcc is spontaneous conversation between two people, so all the numbers are high. Speech to an agent is probably
clearer, so live error rates are probably lower. That is not measured yet.

## Decision
- The speech service uses **NVIDIA Parakeet TDT 0.6B v3** through `transformers` (`ParakeetForTDT`), in float32,
  on the Apple GPU (MPS), else on the CPU. `ORB_STT_ENGINE=whistle` goes back to Whistle.
- **float32, not float16:** float16 gives the same words, but in one run 33 of 182 calls took over 250 ms, up to
  964 ms, while new GPU kernels compiled. float32: only clips longer than 15 s took over 250 ms.
- `orb_speech.models` pins the Hugging Face commit (`541d1f99…`) and the SHA-256 of all six files. The service
  refuses a changed file and starts with `HF_HUB_OFFLINE=1`.

## Consequences
- **Latency:** each STT call costs about 70 to 110 ms more than Whistle for a turn of 2 to 8 s. One whole-turn call
  is on the path to the first agent audio.
- **Memory and disk:** about 2.5 GB more of each, in the Hugging Face cache.
- **The language** is not an input: the model finds it among 25 European languages. `ORB_STT_LANGUAGE` is still
  checked and returned.
- **New dependencies:** `torch`, `transformers>=5.19` and `librosa` (for the mel filters of the feature extractor).
- **CC-BY-4.0:** the model must be credited to NVIDIA where orb names its models (README, this record).
- **No keyword bias:** Whistle had a `keywords` option for topic words; this path has none. The roadmap item
  "bias the STT with the topic" needs another way.
- **Fillers:** Parakeet writes "uh" and "um". `stt.clean()` removes them before the text reaches the agent.
