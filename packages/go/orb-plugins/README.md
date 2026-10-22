# orb-plugins

Go plugins that connect [lk-go-agent-sdk](../lk-go-agent-sdk) to orb: the agent service as the LLM, and the
local speech service as STT, TTS and the turn detector. The design is in [LLD.md](LLD.md).

```go
svc := speech.NewClient("http://127.0.0.1:8200")
recognizer, _ := stt.NewStreamAdapter(speech.NewRecognizer(svc, "en"), vad)
session := voice.NewAgentSession(
    voice.WithSTT(recognizer),
    voice.WithLLM(agentllm.New("http://127.0.0.1:8100", sessionID, "Open discussion")),
    voice.WithTTS(speech.NewTTS(svc, "af_heart")),
    voice.WithVAD(vad),
    voice.WithTurnDetector(speech.NewTurnDetector(svc), 0.5),
)
```
The full worker is in [apps/voice](../../../apps/voice/main.go).
