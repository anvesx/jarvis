package vad

import (
	"context"
	"time"

	"github.com/beingPro007/lk-go-agent-sdk/audio"
)

type EventType int

const (
	StartOfSpeech EventType = iota
	InferenceDone
	EndOfSpeech
)

type Event struct {
	Type            EventType
	Speaking        bool
	Probability     float64
	SpeechDuration  time.Duration
	SilenceDuration time.Duration
	Frames          []audio.Frame
}

type VAD interface {
	SampleRate() int
	Stream(ctx context.Context) Stream
}

type Stream interface {
	PushFrame(frame audio.Frame)
	Flush()
	EndInput()
	Chan() <-chan Event
	Err() error
	Close() error
}
