// The orb voice worker: a LiveKit agent built on lk-go-agent-sdk.
//
//	cd apps/voice && go run . dev
//
// For a room named orb-<session id> it joins, listens and talks:
//
//	room audio → Silero VAD → Whistle STT → orb agent (Jev + debt rules + Gemma) → Kokoro TTS → room audio
//
// Smart Turn hears the end of each turn and decides when the user has
// finished. STT, TTS and Smart Turn run in the speech service
// (packages/python/orb-speech); the agent runs in the agent service
// (apps/agent). Both must run on this machine. See apps/voice/README.md.
package main

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"log/slog"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	agents "github.com/beingPro007/lk-go-agent-sdk"
	"github.com/beingPro007/lk-go-agent-sdk/cli"
	"github.com/beingPro007/lk-go-agent-sdk/plugins/silero"
	"github.com/beingPro007/lk-go-agent-sdk/stt"
	"github.com/beingPro007/lk-go-agent-sdk/voice"
	"github.com/beingPro007/orb/packages/go/orb-plugins/agentllm"
	"github.com/beingPro007/orb/packages/go/orb-plugins/speech"
)

const roomPrefix = "orb-"

var sessionID = regexp.MustCompile(`^[a-z0-9][a-z0-9-]{2,47}$`)

func main() {
	loadDotenv(repoEnv())
	if os.Getenv("ONNXRUNTIME_LIB_PATH") == "" {
		if _, err := os.Stat("/opt/homebrew/lib/libonnxruntime.dylib"); err == nil {
			os.Setenv("ONNXRUNTIME_LIB_PATH", "/opt/homebrew/lib/libonnxruntime.dylib")
		}
	}
	cli.Run(agents.WorkerOptions{Entrypoint: entrypoint})
}

// sessionFor maps a room to an agent session. Rooms that the web page makes
// are orb-<session id>; any other room gets a stable session of its own.
func sessionFor(room string) string {
	if id, ok := strings.CutPrefix(room, roomPrefix); ok && sessionID.MatchString(id) {
		return id
	}
	sum := sha256.Sum256([]byte(room))
	return "room-" + hex.EncodeToString(sum[:6])
}

func entrypoint(job *agents.JobContext) error {
	t0 := time.Now()
	room := job.Job().GetRoom().GetName()
	sid := sessionFor(room)
	log := slog.With("room", room, "session", sid)
	log.Info("job starts")
	logText := os.Getenv("ORB_LOG_TEXT") == "1" // transcripts stay out of the log unless asked for

	speechSvc := speech.NewClient(env("ORB_SPEECH_URL", "http://127.0.0.1:8200"))
	model := agentllm.New(env("ORB_AGENT_URL", "http://127.0.0.1:8100"), sid, "Open discussion",
		agentllm.WithEventHandler(func(e agentllm.Event) { logAgentEvent(log, e, logText) }))
	if err := model.Ensure(job.Context()); err != nil {
		return err
	}
	log.Info("agent session ready", "ms", time.Since(t0).Milliseconds())

	// Speech starts after 250 ms above 0.5 and ends after 250 ms below 0.35 (hysteresis), so a dip between
	// words does not cut a sentence. See apps/voice/LLD.md for the measurements.
	vadModel, err := silero.New(
		silero.WithThreshold(float32(num("ORB_VAD_THRESHOLD", 0.5))),
		silero.WithNegativeThreshold(float32(num("ORB_VAD_NEG_THRESHOLD", 0.35))),
		silero.WithMinSpeechDuration(ms("ORB_VAD_MIN_SPEECH_MS", 250)),
		silero.WithMinSilenceDuration(ms("ORB_VAD_MIN_SILENCE_MS", 250)),
	)
	if err != nil {
		return err
	}
	defer vadModel.Close()

	var rec stt.Recognizer = speech.NewRecognizer(speechSvc, env("ORB_STT_LANGUAGE", "en"))
	if dir := os.Getenv("ORB_DEBUG_AUDIO_DIR"); dir != "" {
		rec = newDumpRecognizer(rec, filepath.Join(dir, sid), log)
		log.Warn("debug audio is on: every transcribed piece and turn is saved", "dir", filepath.Join(dir, sid))
	}
	recognizer, err := stt.NewStreamAdapter(rec, vadModel,
		stt.WithErrorHandler(func(err error) { log.Error("speech-to-text failed", "error", err) }))
	if err != nil {
		return err
	}

	session := voice.NewAgentSession(
		voice.WithSTT(recognizer), // the adapter also transcribes each whole turn at commit
		voice.WithLLM(model),
		voice.WithTTS(speech.NewTTS(speechSvc, os.Getenv("ORB_TTS_VOICE"))),
		voice.WithVAD(vadModel),
		voice.WithTurnDetector(speech.NewTurnDetector(speechSvc), 0.5),
		voice.WithMinEndpointingDelay(ms("ORB_MIN_ENDPOINTING_MS", 200)),
		voice.WithMaxEndpointingDelay(ms("ORB_MAX_ENDPOINTING_MS", 2500)),
		// counted from the VAD's start of speech, which already needs 250 ms of voice, and the voice must still be
		// there (within 100 ms): about 550 ms of voice in all, as in LiveKit Agents
		voice.WithMinInterruptionDuration(ms("ORB_MIN_INTERRUPTION_MS", 400)),
	)
	defer func() { log.Info("voice session ends", "dropped_frames", session.DroppedFrames()) }()
	go reportDrops(job, session, log)

	go func() {
		for ev := range session.Events() {
			switch ev.Type {
			case voice.EventUserTranscript:
				if ev.IsFinal {
					logSpeech(log, "piece heard", ev.Transcript, logText)
				}
			case voice.EventUserTurn:
				logSpeech(log, "user turn", ev.Transcript, logText)
			case voice.EventUserSpeechIgnored:
				logSpeech(log, "speech ignored (too short to interrupt the agent)", ev.Transcript, logText)
			case voice.EventStateChanged:
				log.Info("state", "now", ev.State.String())
			case voice.EventError:
				log.Error("session error", "error", ev.Err)
			case voice.EventUserStartedSpeaking, voice.EventUserStoppedSpeaking, voice.EventAgentStartedSpeaking, voice.EventAgentStoppedSpeaking:
				log.Debug(ev.Type.String())
			}
		}
	}()

	log.Info("voice session starts", "setup_ms", time.Since(t0).Milliseconds())
	return voice.RunRoomSession(job, session, &voice.Agent{}) // the agent service owns the prompt
}

func logSpeech(log *slog.Logger, msg, text string, logText bool) {
	if logText {
		log.Info(msg, "text", text)
	} else {
		log.Info(msg, "chars", len(text))
	}
}

// reportDrops logs when the session dropped input frames: gaps in the audio
// that the STT hears.
func reportDrops(job *agents.JobContext, s *voice.AgentSession, log *slog.Logger) {
	t := time.NewTicker(30 * time.Second)
	defer t.Stop()
	var last uint64
	for {
		select {
		case <-job.Done():
			return
		case <-t.C:
			if n := s.DroppedFrames(); n > last {
				log.Warn("input frames dropped", "total", n, "new", n-last)
				last = n
			}
		}
	}
}

func logAgentEvent(log *slog.Logger, e agentllm.Event, logText bool) {
	switch e.Type {
	case "decision":
		var d struct {
			Persona string `json:"persona"`
			Debt    string `json:"debt"`
			Kind    string `json:"kind"`
			Stage   string `json:"stage"`
		}
		if json_(e.Data, &d) {
			log.Info("agent decision", "persona", d.Persona, "debt", d.Debt, "kind", d.Kind, "stage", d.Stage)
		}
	case "check":
		var c struct {
			Text string `json:"text"`
			Done *bool  `json:"done"`
		}
		if json_(e.Data, &c) {
			log.Info("reply check", "done", c.Done, "text", c.Text)
		}
	case "reply":
		var r struct {
			Text   string             `json:"text"`
			Timing map[string]float64 `json:"timing"`
		}
		if json_(e.Data, &r) {
			if logText {
				log.Info("agent said", "text", r.Text, "timing", r.Timing)
			} else {
				log.Info("agent said", "chars", len(r.Text), "timing", r.Timing)
			}
		}
	}
}

func env(name, fallback string) string {
	if v := strings.TrimSpace(os.Getenv(name)); v != "" {
		return v
	}
	return fallback
}

func num(name string, fallback float64) float64 {
	if v, err := strconv.ParseFloat(os.Getenv(name), 64); err == nil && v > 0 {
		return v
	}
	return fallback
}

func ms(name string, fallback int) time.Duration {
	if n, err := strconv.Atoi(os.Getenv(name)); err == nil && n > 0 {
		return time.Duration(n) * time.Millisecond
	}
	return time.Duration(fallback) * time.Millisecond
}

// repoEnv is the .env of the orb repo: the nearest parent folder with go.work.
func repoEnv() string {
	dir, err := os.Getwd()
	if err != nil {
		return ""
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.work")); err == nil {
			return filepath.Join(dir, ".env")
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return ""
		}
		dir = parent
	}
}

// loadDotenv sets the variables of a KEY=VALUE file that are not set yet.
func loadDotenv(path string) {
	if path == "" {
		return
	}
	f, err := os.Open(path)
	if err != nil {
		return
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		k, v, ok := strings.Cut(strings.TrimPrefix(line, "export "), "=")
		k, v = strings.TrimSpace(k), strings.Trim(strings.TrimSpace(v), `"'`)
		if !ok || k == "" || v == "" {
			continue
		}
		if _, set := os.LookupEnv(k); !set {
			os.Setenv(k, v)
		}
	}
}
