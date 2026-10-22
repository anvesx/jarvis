package tts

import (
	"context"

	"github.com/beingPro007/lk-go-agent-sdk/audio"
)

type SynthesizedAudio struct {
	Frame     audio.Frame
	Text      string
	SegmentID string
	Final     bool
}

type TTS interface {
	SampleRate() int
	NumChannels() int
	Stream(ctx context.Context) SynthesizeStream
}

type SynthesizeStream interface {
	PushText(token string)
	Flush()
	EndInput()
	Chan() <-chan SynthesizedAudio
	Err() error
	Close() error
}
