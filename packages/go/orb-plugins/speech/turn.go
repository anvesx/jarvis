package speech

import (
	"context"
	"fmt"

	"github.com/beingPro007/lk-go-agent-sdk/audio"
	"github.com/beingPro007/lk-go-agent-sdk/llm"
)

// TurnDetector is a voice.AudioTurnDetector backed by Smart Turn v3.2 in the
// speech service. It hears the user's audio, not the text.
type TurnDetector struct {
	c *Client
}

// NewTurnDetector returns a turn detector.
func NewTurnDetector(c *Client) *TurnDetector { return &TurnDetector{c: c} }

// PredictEndOfTurn is the text-only call. Smart Turn needs audio, so without
// it the detector has no opinion and returns 1: the session keeps its usual
// timing.
func (d *TurnDetector) PredictEndOfTurn(context.Context, *llm.ChatContext) (float64, error) {
	return 1, nil
}

// PredictEndOfTurnAudio returns the probability that the user has finished,
// from the audio of the turn.
func (d *TurnDetector) PredictEndOfTurnAudio(ctx context.Context, _ *llm.ChatContext, turn []audio.Frame) (float64, error) {
	body, rate, err := pcm16(turn)
	if err != nil {
		return 0, err
	}
	if len(body) == 0 {
		return 1, nil
	}
	var out struct {
		Probability float64 `json:"probability"`
	}
	if err := d.c.postJSON(ctx, fmt.Sprintf("/v1/turn?sample_rate=%d", rate), "application/octet-stream", body, &out); err != nil {
		return 0, err
	}
	return out.Probability, nil
}
