# LLD: orb-bench (the test bench)

Code: `src/orb_bench/` · Tests: `tests/` (10) · Status: built · Last update: 2026-10-22

**orb-bench proves each model choice with a measurement.** It builds git-ignored test audio, runs four scored suites
against the running services, and saves every result. The bench page (`/bench` of the agent service) runs the same
suites and shows the results, and it switches the STT, LLM and TTS models for a live call.

| Suite | What it measures | Needs |
|---|---|---|
| `stt` | word error rate per engine, corpus and part; words invented from noise; time per call | the speech service |
| `tts` | time to first audio, real-time factor, word error rate of a judge STT on the speech | the speech service |
| `llm` | voice replies per model: a reply exists, short, no leaked note, the expected persona; first-token time | the agent service, Jev, Ollama |
| `turn` | a synthetic person in a real LiveKit room: first agent audio, turn commit, early replies, noise, backchannel, echo, barge-in | every service of `scripts/dev.sh` |

## 1. Input
- **The corpus** (`python -m orb_bench corpus`, `data/testbench/corpus/`, git-ignored, 94 MB). Every audio clip went
  through the call path (Opus 32 kbps, then soxr LQ to 16 kHz), as in decision 0005.
  | Group | Clips | What |
  |---|---|---|
  | `stt/synthetic` | 288 | 12 sentences with orb's topic words × 6 Kokoro voices (2 Hindi voices reading English) × clean and pink noise at 20, 10 and 5 dB SNR |
  | `stt/edacc` | 182 | the EdAcc clips of decision 0005: 4 South Asian speakers (142) and US English (40); seed 7 |
  | `stt/librispeech` | 100 | LibriSpeech test-clean, 5 to 40 words; seed 11 |
  | `noise` | 24 | white, pink, brown and hum noise at 3 levels and 2 lengths, no speech |
  | `echo` | 6 | the agent's voice (Kokoro af_heart) at low levels |
  | `llm`, `tts` | 8 cases (3 of them run 3 times), 6 texts | voice turns with expected personas, and 3 short lines about the call itself from a real call ("Can you hear me clearly?"); English, Hindi and Hinglish texts |
- **The services** (`ORB_AGENT_URL`, `ORB_SPEECH_URL`; 127.0.0.1 by default). Their answers are measured, not trusted.
- **The bench API** (`apps/agent/src/orb_agent/bench.py`) takes a suite name, engines, groups, models and a clip
  limit (1 to 1000), each checked against a fixed list.

## 2. Algorithm
- **stt:** each clip of each group → `POST /v1/stt?engine=` → normalize the reference and the text → word edits. A noise
  clip with any text counts as invented. The summary has one row per engine, corpus and part, and one total per
  engine and corpus.
- **tts:** for each config (engine, voice, languages): one warm-up request (not measured: it loads the engine), then
  each text of its languages → stream `/v1/tts` → time to the first bytes and to the end → save the WAV → the judge
  hears it (Parakeet through the speech service for English, Whisper large-v3-turbo on the GPU for Hindi) → WER.
- **llm:** for each model: one warm-up turn (Ollama loads the model), then each case in a new session with
  `config.llm_model` → stream the voice turns → checks on the last reply. A case with `repeat` runs that many times,
  because a model does not answer alike twice. The summary of a model has the cases, the passes, the silent replies
  and the first-word time (median and slowest).
- **turn** (`room.py`): the person joins the room of a new session, publishes a microphone and listens to the agent's
  track (loud = RMS above 0.01 in a 20 ms frame). It also records the worker's events from the room's data channel
  (`orb.bench`). Each case sends Kokoro speech (voice am_michael) in real time, 20 ms per frame:
  - finished: 4 sentences; the time from the end of the clip to the first loud agent frame, and to the turn commit.
  - unfinished: 3 sentences cut mid-phrase, 1.2 s of silence, then the rest. Agent audio in the pause is an early reply.
  - noise: 4 clips of 2.5 s. A new person message or agent audio is a false turn.
  - backchannel, echo and barge-in, each while the agent speaks a long answer: "Sure." must not stop it or become a
    message; the agent's own voice at level 0.02 is reported; for a full sentence, the time until the agent's
    audio stops, the second word in the transcript and the agent's pause.
- Every run is saved as `data/testbench/results/<suite>/<run>.json` (the run is the time it ended). `python -m
  orb_bench show <suite> [run]` prints it; the bench page lists and shows the runs.

## 3. Maths
| Name | Value | Source |
|---|---|---|
| Word error rate | (substitutions + deletions + insertions) / reference words, summed over the clips of a cell | standard |
| English normalizer | Whisper's English normalizer (case, punctuation, fillers, numbers) | `transformers`, as decision 0005 |
| Other languages | lower case; Unicode punctuation (P) and symbols (S) removed; vowel signs (M) kept | chosen: a danda must not count as a word |
| Noise for a target SNR | gain = rms(speech) / (rms(noise) × 10^(SNR / 20)) | standard |
| Percentile | linear between the two nearest ranks | standard |
| Real-time factor | time to make the audio / its length | standard |
| Loud agent frame | RMS above 0.01 in 20 ms | chosen: above room noise, below speech |
| Reply checks | at most 3 sentences and 400 characters; no "Persona:", "Why:", "[" or a first line that is only a chat role | the voice length rule |

## 4. Test cases
| Case | Expected result | Test |
|---|---|---|
| Normalizers | English drops fillers, punctuation and case; Hindi keeps vowel signs and drops the danda | `test_english_normalizer_*`, `test_hindi_normalizer_*` |
| Edits, WER, percentiles | exact values on small inputs | `test_edits_wer_and_percentile` |
| Noise and mixing | unit RMS; the mix hits the SNR | `test_noise_is_unit_rms_and_mix_hits_the_snr` |
| Padding | silence at both ends | `test_pad_adds_silence_at_both_ends` |
| stt suite on fake HTTP | scored speech; invented words counted on noise | `test_stt_suite_scores_speech_and_counts_invented_words` |
| llm suite on fake HTTP | the checks and the persona; a leaked note fails | `test_llm_suite_checks_replies_and_personas` |
| tts suite on fake HTTP | length, judge WER, a saved WAV; one warm-up per config, no row for it | `test_tts_suite_measures_and_judges` |
| turn summary | each failure kind counted | `test_turn_summary_counts_each_failure_kind` |
| llm suite, repeated cases | a case with `repeat` runs that many times (`mic_check#1` …); a silent reply fails `has_reply` and is counted; the summary has the first-word p50 and the slowest | `test_llm_suite_repeats_a_case_and_counts_silent_replies` |

The suites themselves run against the real services (`python -m orb_bench run …`); their numbers are in the LLDs of
the components and in decisions 0006 and 0007.

## 5. Security
- **Test audio stays out of git** (`data/testbench/` is in `.gitignore`): EdAcc and LibriSpeech are CC-BY research
  sets, and the corpus is regenerated, not shared.
- **Downloads:** EdAcc and LibriSpeech come from the Hugging Face hub (public datasets, parquet files read with
  pyarrow). The Whisper judge comes from the Hugging Face cache.
- **A cloud STT engine** (Cartesia) is in the page's engine list but unchecked, and it is not in the default of the
  command line. A run with it sends the corpus audio (synthetic speech or public research sets, never a person's
  voice) to the cloud, and it costs credits (1 for each 2 s of audio, silence included: the noise clips too).
- **Cost:** the llm and turn suites call Jev, about $0.00007 per read (README). A turn run has about 16 turns, and
  each turn has a read of the person's message and a read of the reply, so about $0.0025 per run.
- **No secrets in results:** a result has texts, transcripts and timings, never a key.
- **The bench API** is on the agent service, on 127.0.0.1 only. Clip and result paths are checked to stay inside
  their folders; one suite runs at a time.

## 6. Performance
| Run (Apple M5, 16 GB, 2026-10-22) | Time |
|---|---|
| `tts` (4 configs, 16 texts, both judges) | 1 min 39 s (run 20261022-201558) |
| `llm` (5 models × 5 cases, a model load each) | about 10 min (run 20261022-200105) |
| `turn` | 3 min 47 s (run 20261022-202907) |
| `stt` (2 engines, 594 clips), `corpus` | not timed; Whistle needs 17 to 36 ms and Parakeet 89 to 173 ms per clip |

The llm, tts and turn suites use the same models as a live call, so the bench page refuses a second run while one
is going, and a live call during a run changes its numbers.

## 7. Edge cases
| Case | What happens | Why |
|---|---|---|
| A corpus group is missing | the suite stops with "Run: python -m orb_bench corpus" | the corpus is not in git |
| An LLM writes no words for a turn | the row fails `has_reply`, and the summary counts it as silent | hidden tokens used up the limit (decision 0007); the first suite had only clean lines and missed it |
| An engine does not load | its rows have the error; the other engines go on | 503 from the speech service |
| Kokoro speech ends with silence | `first_audio_s` counts from the end of the clip, silence included | the numbers favor the agent by that silence, in every run alike |
| Hinglish texts | the judge writes English words in Devanagari, so the WER is too high | a judge limit, noted with the results |
| A model reloads during the llm suite | not measured: one warm-up turn per model | Ollama keeps one model in memory |
| The worker's events arrive late | the times come from the worker's own clock, on the same machine | the person and the worker share the clock |

## 8. Expected results
- `python -m orb_bench corpus` prints the clip count of each group (288, 182, 100, 24, 6).
- `python -m orb_bench run <suite>` prints a table and saves a result file; `/bench` shows the same runs.
- `make test` passes the 10 tests.
