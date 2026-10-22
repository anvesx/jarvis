package stt

import (
	"context"
	"time"

	"github.com/beingPro007/lk-go-agent-sdk/audio"
)

type SpeechEventType int

const (
	StartOfSpeech SpeechEventType = iota
	InterimTranscript
	FinalTranscript
	EndOfSpeech
)

type SpeechData struct {
	Language   string
	Text       string
	StartTime  time.Duration
	EndTime    time.Duration
	Confidence float64
}

type SpeechEvent struct {
	Type         SpeechEventType
	Alternatives []SpeechData
}

type STT interface {
	SampleRate() int
	NumChannels() int
	Stream(ctx context.Context) SpeechStream
}

type SpeechStream interface {
	PushFrame(frame audio.Frame)
	Flush()
	EndInput()
	Chan() <-chan SpeechEvent
	Err() error
	Close() error
}
