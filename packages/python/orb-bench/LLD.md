# LLD: orb-bench (the test bench)

Code: `src/orb_bench/` · Tests: `tests/` (23) · Status: built · Last update: 2026-10-23

**orb-bench proves each model choice with a measurement.** It builds git-ignored test audio, runs four scored suites
against the running services, re-scores saved STT runs offline, and saves every result. The bench page (`/bench` of the agent service) runs the same
suites and shows the results, and it switches the STT, LLM and TTS models for a live call.

| Suite | What it measures | Needs |
|---|---|---|
| `stt` | word error rate per engine, corpus and part; for written-form groups also fwer, entity and punct_f1; words invented from noise; time per call | the speech service |
| `format` | the same scores for a saved `stt` run after named text stages, plus added words and the time of each stage | nothing (offline) |
| `tts` | time to first audio, real-time factor, word error rate of a judge STT on the speech | the speech service |
| `llm` | voice replies per model: a reply exists, short, no leaked note, the expected persona; first-token time | the agent service, Jev, Ollama |
| `turn` | a synthetic person in a real LiveKit room: first agent audio, turn commit, early replies, noise, backchannel, echo, barge-in | every service of `scripts/dev.sh` |

## 1. Input
- **The corpus** (`python -m orb_bench corpus`, `data/testbench/corpus/`, git-ignored, 94 MB before `stt/dictation`).
  **`ORB_TESTBENCH_DIR`** moves the whole test bench (corpus and results); it is how the worktrees of one repo share
  one corpus (`orb-data/testbench`, 167 MB with `stt/dictation`, measured 2026-10-23). Every audio clip went
  through the call path (Opus 32 kbps, then soxr LQ to 16 kHz), as in decision 0005.
  | Group | Clips | What |
  |---|---|---|
  | `stt/synthetic` | 288 | 12 sentences with orb's topic words × 6 Kokoro voices (2 Hindi voices reading English) × clean and pink noise at 20, 10 and 5 dB SNR |
  | `stt/dictation` | 480 | 40 written-form sentences × the same 6 voices × clean and pink noise at 10 dB. Each has `spoken` (what Kokoro reads), `ref` (the written form a dictation tool should output), `keys` (entities that must be exact) and `cls` (its error class). 14 classes: number, time, date, money, percent, ordinal, email, url, identifier, command, domain, correction, list, plain (control). The sentences are in `dictation.py` |
  | `stt/edacc` | 182 | the EdAcc clips of decision 0005: 4 South Asian speakers (142) and US English (40); seed 7 |
  | `stt/librispeech` | 100 | LibriSpeech test-clean, 5 to 40 words; seed 11 |
  | `noise` | 24 | white, pink, brown and hum noise at 3 levels and 2 lengths, no speech |
  | `echo` | 6 | the agent's voice (Kokoro af_heart) at low levels |
  | `llm`, `tts` | 8 cases (3 of them run 3 times), 6 texts | voice turns with expected personas, and 3 short lines about the call itself from a real call ("Can you hear me clearly?"); English, Hindi and Hinglish texts |
- **The services** (`ORB_AGENT_URL`, `ORB_SPEECH_URL`; 127.0.0.1 by default). Their answers are measured, not trusted.
- **A saved `stt` run** (the input of `format`): its rows hold `ref`, `keys`, the engine's `raw` text and the scores.
- **The bench API** (`apps/agent/src/orb_agent/bench.py`) takes a suite name, engines, groups, models, stage names
  (`format` only) and a clip limit (1 to 1000), each checked against a fixed list. It runs only stages that are
  registered in code; the API cannot import a module.

## 2. Algorithm
- **stt:** each clip of each group → `POST /v1/stt?engine=` → normalize the reference and the text → word edits. A clip
  of a written-form group (`written` in its manifest: `stt/synthetic`, `stt/dictation`) is also scored without the
  normalizer: fwer, entity and punct_f1. A noise clip with any text counts as invented. The summary has one row per
  engine, corpus and part, and one total (`all`) per engine and corpus. A dictation clip counts in its variant
  (`clean`, `snr10`) and in its class (`cls:<class>`). Every row keeps the engine's `raw` text.
- **format** (`suites/format.py`): load the `stt` run (the newest, or `--source`) → for each stage combination (`raw`,
  `written`, `vocab+written`: names joined by `+`, applied left to right) → apply it to the `raw` text of every row
  and time each stage → score the row again → `added_words(raw, output, ref)` → the same summary as `stt`, with one
  row per combination, the added-word count and the p50 and p95 of the stage time. It never calls the speech service
  and never imports a speech model.
  - **Stage registry.** A stage is a function `str -> str`. `raw` (the identity) is registered. `register(name, fn)`
    adds one; a string `"package.module:function"` is imported on first use. On the command line, `--define
    name=package.module:function` does the same.
  - **STT-2** registers `written` as `orb_speech.textfmt:written`. **STT-3** registers `vocab` the same way. Neither
    needs a change of the suite. A stage that needs a vocabulary holds it inside the function; it must not read
    `keys`.
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
| fwer | word edits / reference tokens, on tokens that keep case. A token is a word, a flag (`-b`, `--watch`) or one punctuation mark. Joiners inside a word (`. , : ' / @ _ -`) keep emails, times (`3:30`), money (`12,450.30`) and paths in one token. No normalizer | chosen: the normalized WER cannot see written form |
| entity | keys found / keys. A key is found when it is in the text with the same case and is not inside a longer word (`4821` is not in `14821`, `-b` is not in `--build`) | chosen: exact written form |
| punct_f1 | F1 over the marks `. , ? ! ; :` that stand alone. The words are aligned first (edit distance), then a mark counts when the same mark follows the same reference word. Micro-averaged over the clips of a cell | chosen: a wrong word elsewhere must not move a mark |
| added words | content words of the output that are in neither the stage input nor the reference. A content word is a lower-case letter run, with camelCase and dotted names split, without function words (a list of 58 in `metrics.py`) and without tokens that start with a digit | chosen: digits and symbols are allowed to appear, as TDD 4.1 says |
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
| Tokens | marks split off, joiners kept, flags whole | `test_written_tokens_split_marks_but_keep_entities_whole` |
| fwer | sees case, punctuation and number format that WER ignores | `test_fwer_sees_case_punctuation_and_number_format_that_wer_ignores` |
| entity | exact, case-sensitive, not inside longer words | `test_entity_counts_exact_keys_not_parts_of_longer_words` |
| punct_f1 | marks follow aligned words; wrong words keep right marks | `test_punct_f1_aligns_words_first_so_a_wrong_word_does_not_move_a_mark` |
| added words | an invented word is found; digits, symbols and reference words are not | `test_added_words_are_content_words_in_neither_the_input_nor_the_reference` |
| Row scores | written metrics only for written-form rows that have a reference | `test_row_scores_gives_written_metrics_only_to_written_rows_with_a_reference` |
| Dictation sentences | 40 sentences, 14 classes, each key is in its own ref | `test_dictation_sentences_cover_every_class_with_exact_keys` |
| stt summary, written form | fwer, entity, punct_f1 for dictation and its classes; none for EdAcc | `test_stt_suite_adds_written_metrics_only_for_written_form_groups` |
| stt summary, perfect text | WER 0, fwer 0, entity 100 %, punct_f1 100 % | `test_stt_suite_scores_a_perfect_written_transcript` |
| format with `raw` | the same numbers as the stt run, no service call | `test_format_suite_with_raw_reproduces_the_stt_run_without_a_service` |
| format stages | stages run in order, are timed, and an added word is counted | `test_format_suite_runs_registered_stages_in_order_and_scores_them` |
| format registry | an unknown stage names the known ones; `"module:function"` loads lazily | `test_format_suite_names_the_unknown_stage_and_loads_a_dotted_path_lazily` |
| format without a run | stops and says to run `stt` | `test_format_suite_needs_an_stt_run` |
| Bench API, format suite | the API runs `format` with the stage names it is given, lists its results, and lists `stt/dictation` | `apps/agent/tests/test_bench.py::test_the_format_suite_runs_offline_and_the_dictation_group_is_listed` |

The suites themselves run against the real services (`python -m orb_bench run …`); their numbers are in the LLDs of
the components and in decisions 0006 and 0007.

## 5. Security
- **Test audio stays out of git** (`data/testbench/` is in `.gitignore`): EdAcc and LibriSpeech are CC-BY research
  sets, and the corpus is regenerated, not shared.
- **Stages are code.** `--define name=package.module:function` imports a module of the operator's choice, so it runs
  with the operator's rights. The bench API cannot do this: it runs only stages that code registered.
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
| `stt` (2 engines, 1074 clips each, 2148 calls; Apple M4, 16 GB, 2026-10-23) | 5 min 25 s (run 20261023-122740) |
| `format` with `raw` on that run (2148 rows) | 1.7 s, no service (run 20261023-123334) |
| `corpus`: `stt/dictation` (240 syntheses, 480 call-path encodings; Apple M4, 2026-10-23) | about 4 min 20 s (from file times: the manifest of `stt/synthetic` at 12:12:26, of `stt/dictation` at 12:16:44) |
| time per call, same `stt` run | Whistle p50 45 to 74 ms, Parakeet p50 155 to 204 ms (decision 0005: 17 to 36 and 89 to 173 ms; the machine was shared, so these are higher) |

**The bottleneck of `format`** is reading the run file. A formatter that takes 5 ms a row adds 11 s for the 2148 rows.

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
| A key appears in the wrong place | the entity still counts it | the check is exact text, not position; a stage that moves text is caught by fwer |
| "p.m." in the output | its dot counts as a mark after a word in punct_f1 | the tokenizer cannot tell an abbreviation from a sentence end; it lowers the score of engines that write "p.m." |
| A stored run has no `keys`, `raw` or `written` (made before 2026-10-23) | `format` uses `hyp` for `raw` and scores no written-form metric | old rows do not have them; run `stt` again |
| A stage raises | the `format` run stops with the error | a hidden failure would score as raw text |
| The `stt` run used a service that formats text (STT-2) | `hyp` is the formatted text, `raw` the service's `raw` field | the format suite starts from `raw` so a stage is never applied twice |
| The worker's events arrive late | the times come from the worker's own clock, on the same machine | the person and the worker share the clock |

## 8. Expected results
- `python -m orb_bench corpus` prints the clip count of each group (288 synthetic, 480 dictation, 182, 100, 24, 6).
- `ORB_TESTBENCH_DIR=<dir> python -m orb_bench run stt` reads and writes `<dir>`.
- `python -m orb_bench run <suite>` prints a table and saves a result file; `/bench` shows the same runs.
- `python -m orb_bench run format --stages raw` prints the numbers of the source `stt` run again, with 0 added words.
- **Baseline (Apple M4, 2026-10-23, run 20261023-122740).** Normalized WER, Whistle / Parakeet: EdAcc South Asian
  26.8 / 11.5 %, LibriSpeech 5.2 / 2.4 %, synthetic 14.3 / 5.0 %, dictation 36.1 / 19.8 %. On dictation, Parakeet has
  fwer 30.2 %, entity 29.0 % and punct_f1 75.5 %. Invented text on noise: Whistle 45.8 %, Parakeet 29.2 %. The full
  table is in `docs/design/stt-accuracy.md` section 5. A later ticket compares its run with these.
- `make test` passes the 23 tests.
