# apps/voice

The voice worker: a LiveKit agent on [lk-go-agent-sdk](../../packages/go/lk-go-agent-sdk) that talks with the
orb agent service. The design and the latency budget are in [LLD.md](LLD.md).

```sh
livekit-server --dev --bind 127.0.0.1 --node-ip 127.0.0.1      # LiveKit on :7880
python -m orb_speech.server                                     # speech on :8200
python -m orb_agent.server                                      # agent on :8100
cd apps/voice && CGO_ENABLED=1 go run . dev                      # this worker
```
Or start everything with `make dev`, then run `make e2e`.
