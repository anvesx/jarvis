# Examples

Each example is a worker. Run it with `dev` for local defaults and text logs, or `start` for production JSON logs.

| Example | What it shows |
|---|---|
| `minimal/` | The smallest worker: accept a job, join the room, wait for it to end |
| `echo/` | The audio path: subscribe to a track, decode to PCM, re-encode and publish it back |
| `voiceagent/` | A full voice agent: Deepgram STT, Gemini LLM, Cartesia TTS, Silero VAD, plus a tool the model can call |

```sh
go run ./examples/minimal dev
go run ./examples/echo dev
```

`voiceagent/` is its own Go module (it depends on the plugin modules), so run it from its directory:

```sh
cd examples/voiceagent && go run . dev
```

It reads credentials from the environment, falling back to a `.env` file in the
example directory or at the repository root:

```
LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
DEEPGRAM_API_KEY, GEMINI_API_KEY, CARTESIA_API_KEY, CARTESIA_VOICE
ONNXRUNTIME_LIB_PATH
```
