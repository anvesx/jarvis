package voice

import "github.com/beingPro007/lk-go-agent-sdk/audio"

type AudioInput interface {
	Frames() <-chan audio.Frame
}

type AudioOutput interface {
	CaptureFrame(audio.Frame) error
	ClearBuffer()
	WaitForPlayout()
}
