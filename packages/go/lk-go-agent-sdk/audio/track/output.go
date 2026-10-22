package track

import (
	"github.com/beingPro007/lk-go-agent-sdk/audio"
	"github.com/livekit/media-sdk"
	"github.com/livekit/protocol/logger"
	lkmedia "github.com/livekit/server-sdk-go/v2/pkg/media"
)

type Output struct {
	track *lkmedia.PCMLocalTrack
}

func NewOutput(sampleRate, channels int) (*Output, error) {
	t, err := lkmedia.NewPCMLocalTrack(sampleRate, channels, logger.GetLogger())
	if err != nil {
		return nil, err
	}
	return &Output{track: t}, nil
}

func (out *Output) Track() *lkmedia.PCMLocalTrack {
	return out.track
}

func (out *Output) CaptureFrame(f audio.Frame) error {
	return out.track.WriteSample(media.PCM16Sample(f.Data))
}

func (out *Output) WaitForPlayout() {
	out.track.WaitForPlayout()
}

func (out *Output) ClearBuffer() {
	out.track.ClearQueue()
}

func (out *Output) Close() error {
	out.track.ClearQueue()
	return out.track.Close()
}
