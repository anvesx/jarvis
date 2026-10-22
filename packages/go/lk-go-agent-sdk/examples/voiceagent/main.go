package main

import (
	"bufio"
	"context"
	"log/slog"
	"os"
	"strings"
	"time"

	agents "github.com/beingPro007/lk-go-agent-sdk"
	"github.com/beingPro007/lk-go-agent-sdk/cli"
	"github.com/beingPro007/lk-go-agent-sdk/llm"
	"github.com/beingPro007/lk-go-agent-sdk/plugins/cartesia"
	"github.com/beingPro007/lk-go-agent-sdk/plugins/deepgram"
	"github.com/beingPro007/lk-go-agent-sdk/plugins/gemini"
	"github.com/beingPro007/lk-go-agent-sdk/plugins/silero"
	"github.com/beingPro007/lk-go-agent-sdk/voice"
)

func main() {
	loadDotenv(".env", "../../.env")
	cli.Run(agents.WorkerOptions{
		Entrypoint: entrypoint,
	})
}

func loadDotenv(paths ...string) {
	for _, p := range paths {
		f, err := os.Open(p)
		if err != nil {
			continue
		}
		sc := bufio.NewScanner(f)
		for sc.Scan() {
			line := strings.TrimSpace(sc.Text())
			if line == "" || strings.HasPrefix(line, "#") {
				continue
			}
			k, v, ok := strings.Cut(line, "=")
			if !ok {
				continue
			}
			k, v = strings.TrimSpace(k), strings.TrimSpace(v)
			if v == "" {
				continue
			}
			if _, set := os.LookupEnv(k); !set {
				os.Setenv(k, v)
			}
		}
		f.Close()
		return
	}
}

func entrypoint(job *agents.JobContext) error {
	stt, err := deepgram.New()
	if err != nil {
		return err
	}
	model, err := gemini.New(job.Context())
	if err != nil {
		return err
	}
	tts, err := cartesia.New()
	if err != nil {
		return err
	}
	vad, err := silero.New()
	if err != nil {
		return err
	}
	defer vad.Close()

	session := voice.NewAgentSession(
		voice.WithSTT(stt),
		voice.WithLLM(model),
		voice.WithTTS(tts),
		voice.WithVAD(vad),
	)

	go func() {
		for ev := range session.Events() {
			switch ev.Type {
			case voice.EventUserTranscript:
				slog.Info("user", "text", ev.Transcript, "final", ev.IsFinal)
			case voice.EventAgentResponse:
				slog.Info("agent", "delta", ev.Text)
			case voice.EventToolCall:
				slog.Info("tool call", "name", ev.ToolCall.Name, "args", ev.Text)
			case voice.EventToolResult:
				slog.Info("tool result", "name", ev.ToolCall.Name, "result", ev.Text)
			case voice.EventError:
				slog.Error("session error", "error", ev.Err)
			default:
				slog.Info(ev.Type.String(), "state", ev.State.String())
			}
		}
	}()

	agent := &voice.Agent{
		Instructions: "You are a helpful voice assistant. Keep your responses short, natural, and conversational.",
		Tools: []llm.Tool{{
			Name:        "get_current_time",
			Description: "Get the current time of day.",
			Parameters: map[string]any{
				"type":       "object",
				"properties": map[string]any{},
			},
			Handler: func(ctx context.Context, args string) (string, error) {
				return time.Now().Format("3:04 PM"), nil
			},
		}},
	}
	return voice.RunRoomSession(job, session, agent)
}
