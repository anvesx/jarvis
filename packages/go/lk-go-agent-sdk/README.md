# lk-go-agent-sdk

A Go SDK for building realtime voice agents on [LiveKit](https://livekit.io), mirroring the architecture of [livekit/agents](https://github.com/livekit/agents) (Python).

> **Status:** pre-release, but end-to-end working. A full conversation — user speech → Silero VAD → Deepgram → Gemini (with a tool call) → Cartesia → audio published back into the room — has been run against a live LiveKit server and the real vendor APIs.

```go
func entrypoint(job *agents.JobContext) error {
    stt, _ := deepgram.New()
    model, _ := gemini.New(job.Context())
    tts, _ := cartesia.New()
    vad, _ := silero.New()
    defer vad.Close()

    session := voice.NewAgentSession(
        voice.WithSTT(stt),
        voice.WithLLM(model),
        voice.WithTTS(tts),
        voice.WithVAD(vad),
    )

    return voice.RunRoomSession(job, session, &voice.Agent{
        Instructions: "You are a helpful voice assistant. Keep replies short.",
    })
}

func main() {
    cli.Run(agents.WorkerOptions{Entrypoint: entrypoint})
}
```

```sh
cd examples/voiceagent && go run . dev
```

## Install

```sh
go get github.com/beingPro007/lk-go-agent-sdk
```

The core (`agents`, `cli`, `stt`, `tts`, `llm`, `vad`, `audio`) is pure Go. The audio track adapters (`audio/track`) and anything that depends on them require CGO — see [System requirements](#system-requirements).

## Architecture

| Package | Mirrors (Python) | Purpose |
|---|---|---|
| `agents` (root) | `worker.py`, `job.py` | Worker protocol client, job lifecycle |
| `cli/` | `cli/` | `dev` / `start` commands, health endpoint, drain |
| `stt/`, `tts/`, `llm/`, `vad/` | same | Vendor-neutral streaming interfaces |
| `audio/` | `utils/audio` | `Frame` (pure Go) |
| `audio/track/` | `io.py` room I/O | Opus ↔ PCM track adapters (CGO) |
| `voice/` | `voice/` | `Agent`, `AgentSession`, `SpeechHandle`, room bridge |
| `plugins/*` | `livekit-plugins-*` | Vendor implementations (separate Go modules) |

Every stream in the SDK — STT, TTS, LLM, VAD — exposes the same shape, which is the Go translation of Python's async generators:

```go
for ev := range stream.Chan() {
    // ...
}
if err := stream.Err(); err != nil {
    // ...
}
```

## The voice pipeline

`AgentSession` wires STT → LLM → TTS and owns turn taking.

- **Turn taking.** A turn ends on the STT's end-of-speech signal (Deepgram endpointing). When a VAD is configured it also ends the turn `WithMinEndpointingDelay` (500ms) after the user falls silent with a final transcript in hand — without this a turn hangs forever if the user's audio stops, e.g. they mute mid-call. Supply a `TurnDetector` to override the decision; when it reports the user is not finished, the turn is held open until `WithMaxEndpointingDelay` expires.
- **Barge-in.** VAD start-of-speech cancels the in-flight generation and clears the audio already queued on the output. Without a VAD, the STT's start-of-speech is used instead.
- **Tool calling.** `llm.Tool` carries a `Handler`; the session runs the handler, feeds the result back to the model, and keeps looping until the model answers with text (bounded by `WithMaxToolSteps`, default 5).
- **Preemptive generation.** With `WithPreemptiveGeneration(true)`, inference starts on the final transcript rather than at the turn boundary. Audio stays gated until the turn commits, and the run is discarded if the user keeps talking.
- **Speech handles.** Every utterance is a `*SpeechHandle` — `ID()`, `Wait()`, `Interrupt()`, `Interrupted()`, `Text()`. Reach the live one via `session.CurrentSpeech()`.
- **Imperative speech.** `session.Say(ctx, "welcome aboard")` speaks text directly, bypassing the LLM, and returns its handle.
- **Events.** `session.Events()` yields transcripts, response deltas, state changes, tool calls and errors.
- **Sample rates.** Room input is captured at the STT's own sample rate; a configured VAD must accept that rate (Silero: 16 kHz), and the published track uses the TTS's rate. Override either with `voice.WithInputSampleRate` / `voice.WithOutputSampleRate`.

### Pipeline nodes

Each stage is a function you can replace on the `Agent`, which is how you add
custom processing without forking the session:

```go
agent := &voice.Agent{
    LLMNode: func(ctx context.Context, chatCtx *llm.ChatContext, tools []llm.Tool) (<-chan llm.ChatChunk, error) {
        // e.g. rewrite the prompt, then delegate to any LLM
    },
}
```

`STTNode`, `LLMNode` and `TTSNode` all take the stage's input channel and return its output channel. Leave them nil to use the configured plugin.

## Plugins

Each plugin is its own Go module so the core carries no vendor dependencies.

| Plugin | Kind | Env | Notes |
|---|---|---|---|
| `plugins/deepgram` | STT | `DEEPGRAM_API_KEY` | Streaming websocket, `nova-3`, endpointing + interim results |
| `plugins/gemini` | LLM | `GEMINI_API_KEY` | Official `google.golang.org/genai` SDK, streaming, function calling. Defaults to `gemini-3.6-flash` with minimal thinking — raise it with `WithThinkingLevel` if you want reasoning over latency |
| `plugins/cartesia` | TTS | `CARTESIA_API_KEY`, `CARTESIA_VOICE` | Streaming websocket, `sonic-2`, PCM out |
| `plugins/silero` | VAD | `ONNXRUNTIME_LIB_PATH` | Silero v5 ONNX, model embedded, CGO |

```go
import "github.com/beingPro007/lk-go-agent-sdk/plugins/deepgram"
```

## System requirements

- Go 1.26+
- For the audio path (`audio/track`, `voice`, any voice agent): `libopus`, `libopusfile` and `libsoxr`.
  ```sh
  sudo apt install libopus-dev libopusfile-dev libsoxr-dev
  ```
- For `plugins/silero`: an ONNX Runtime shared library **1.22.0 or newer**, pointed at by `ONNXRUNTIME_LIB_PATH` or `silero.WithLibraryPath(...)`. Grab one from the [ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases).

## Development

Run a local LiveKit server:

```sh
docker run --rm -p 7880:7880 -p 7881:7881 -p 7882:7882/udp \
  livekit/livekit-server --dev --bind 0.0.0.0
```

Dev credentials: URL `ws://localhost:7880`, key `devkey`, secret `secret` — the `dev` command falls back to these when `LIVEKIT_URL` is unset.

```sh
go build ./...                  # core module
go test ./...                   # core module
(cd plugins/deepgram && go test ./...)
```

Plugin modules and `examples/voiceagent` are separate modules and are not covered by the root `./...`; build them from their own directory.

## Design notes

- **Goroutines, not subprocesses.** Jobs run in goroutines with panic recovery instead of Python's `ipc/` subprocess pool.
- **Channels, not async generators.** `Chan()` + `Err()` + `Close()` is the faithful translation of `async for` with exception propagation.
- **CGO is contained.** It enters only through `audio/track` (Opus/soxr via `livekit/media-sdk`) and `plugins/silero` (ONNX Runtime). The worker, job, CLI and plugin protocol layers stay pure Go.

## Not yet implemented

- An end-of-utterance turn-detector model. The `voice.TurnDetector` interface and its endpointing behaviour exist; no plugin implements it yet.
- Realtime speech-to-speech models (the interfaces are kept model-agnostic for it).
- Multi-agent handoff, and text/transcription room I/O.
