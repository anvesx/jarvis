# apps/agent

The agent service: one HTTP API for every channel, the web chat, and LiveKit tokens for voice. The design is
in [LLD.md](LLD.md).

```sh
JEV_ENV_FILE=~/jev-test/.env python -m orb_agent.server     # http://127.0.0.1:8100
```
It needs Ollama with `gemma4:e2b`. For voice, also run LiveKit, the speech service and `apps/voice`, or run
everything with `make dev`.
