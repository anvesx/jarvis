package voice

import (
	"context"
	"sync"
	"time"

	"github.com/beingPro007/lk-go-agent-sdk/audio"
	"github.com/beingPro007/lk-go-agent-sdk/llm"
)

// TurnDetector predicts whether the user has finished their turn, from the
// chat so far. chatCtx ends with the user's text of the current turn.
// It returns the probability that the turn is complete.
//
// The session asks the detector whenever the user goes quiet with text in
// hand: on a VAD end of speech, on a final transcript after the VAD heard
// silence, and on an STT end of speech. A probability at or above the
// threshold commits the turn after the min endpointing delay (or at once for
// an STT end of speech); a lower one holds it open until the max delay.
type TurnDetector interface {
	PredictEndOfTurn(ctx context.Context, chatCtx *llm.ChatContext) (float64, error)
}

// AudioTurnDetector is a TurnDetector that also reads the audio of the user's
// turn, for models that hear prosody (for example Smart Turn). When the
// configured detector implements it, the session calls PredictEndOfTurnAudio.
// turn holds the user's audio since the turn began, oldest first, at the
// input sample rate, and never more than the last 8 seconds.
type AudioTurnDetector interface {
	TurnDetector
	PredictEndOfTurnAudio(ctx context.Context, chatCtx *llm.ChatContext, turn []audio.Frame) (float64, error)
}

const (
	turnAudioMax     = 60 * time.Second // the whole turn, for transcription in one call
	turnDetectorMax  = 8 * time.Second  // what an AudioTurnDetector gets: Smart Turn reads the last 8 s
	turnAudioPreroll = time.Second      // kept when a new turn starts: the VAD fires after some speech
)

// turnAudio keeps the most recent input audio, so an AudioTurnDetector can
// hear the whole turn. The frame pump fills it and the endpointing decision
// reads it, so it is safe for concurrent use.
type turnAudio struct {
	mu     sync.Mutex
	frames []audio.Frame
	dur    time.Duration
	max    time.Duration
}

func newTurnAudio(max time.Duration) *turnAudio { return &turnAudio{max: max} }

func (t *turnAudio) push(f audio.Frame) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.frames = append(t.frames, f)
	t.dur += f.Duration()
	for t.dur > t.max && len(t.frames) > 1 {
		t.dur -= t.frames[0].Duration()
		t.frames = t.frames[1:]
	}
}

// keepLast drops the oldest frames until at most d of audio is left.
func (t *turnAudio) keepLast(d time.Duration) {
	t.mu.Lock()
	defer t.mu.Unlock()
	for len(t.frames) > 0 && t.dur > d {
		t.dur -= t.frames[0].Duration()
		t.frames = t.frames[1:]
	}
}

func (t *turnAudio) snapshot() []audio.Frame {
	t.mu.Lock()
	defer t.mu.Unlock()
	out := make([]audio.Frame, len(t.frames))
	copy(out, t.frames)
	return out
}

func (t *turnAudio) duration() time.Duration {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.dur
}

// predict asks the turn detector about the current turn, with its audio when
// the detector can hear.
func (s *AgentSession) predict(ctx context.Context, userText string, turn []audio.Frame) (float64, error) {
	local := s.snapshot()
	local.Add(llm.RoleUser, userText)
	if d, ok := s.turnDetector.(AudioTurnDetector); ok {
		return d.PredictEndOfTurnAudio(ctx, local, lastAudio(turn, turnDetectorMax))
	}
	return s.turnDetector.PredictEndOfTurn(ctx, local)
}

type turnResult struct {
	gen int
	p   float64
	err error
}

// lastAudio is the newest d of frames, oldest first.
func lastAudio(frames []audio.Frame, d time.Duration) []audio.Frame {
	var total time.Duration
	for i := len(frames) - 1; i >= 0; i-- {
		total += frames[i].Duration()
		if total > d {
			return frames[i+1:]
		}
	}
	return frames
}
