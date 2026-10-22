package speech

import (
	"context"
	"fmt"
	"net/url"
	"time"

	"github.com/beingPro007/lk-go-agent-sdk/audio"
	"github.com/beingPro007/lk-go-agent-sdk/stt"
)

// Recognizer is an stt.Recognizer backed by Whistle in the speech service.
// It transcribes one finished piece of speech per call. Wrap it with
// stt.NewStreamAdapter and a VAD to get a streaming STT.
type Recognizer struct {
	c        *Client
	language string
}

// NewRecognizer returns a recognizer. language is "" to detect it, or one of
// en, de, fr, es, it, nl, pl.
func NewRecognizer(c *Client, language string) *Recognizer {
	return &Recognizer{c: c, language: language}
}

// SampleRate is the rate that Whistle reads.
func (r *Recognizer) SampleRate() int { return 16000 }

// Recognize sends the speech and returns its text.
func (r *Recognizer) Recognize(ctx context.Context, speech []audio.Frame) (stt.SpeechData, error) {
	body, rate, err := pcm16(speech)
	if err != nil || len(body) == 0 {
		return stt.SpeechData{}, err
	}
	q := url.Values{"sample_rate": {fmt.Sprint(rate)}}
	if r.language != "" {
		q.Set("language", r.language)
	}
	var out struct {
		Text     string  `json:"text"`
		Language string  `json:"language"`
		AudioMS  float64 `json:"audio_ms"`
	}
	if err := r.c.postJSON(ctx, "/v1/stt?"+q.Encode(), "application/octet-stream", body, &out); err != nil {
		return stt.SpeechData{}, err
	}
	return stt.SpeechData{Text: out.Text, Language: out.Language, Confidence: 1,
		EndTime: time.Duration(out.AudioMS) * time.Millisecond}, nil
}
