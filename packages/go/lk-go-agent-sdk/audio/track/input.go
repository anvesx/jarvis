package track

import (
	"sync"

	"github.com/beingPro007/lk-go-agent-sdk/audio"
	"github.com/livekit/media-sdk"
	lkmedia "github.com/livekit/server-sdk-go/v2/pkg/media"
	"github.com/pion/webrtc/v4"
)

type Input struct {
	sampleRate int
	channels   int
	frames     chan audio.Frame

	mu     sync.RWMutex
	closed bool
	track  *lkmedia.PCMRemoteTrack
}

func NewInput(sampleRate, channels, buffer int) *Input {
	if buffer <= 0 {
		buffer = 100
	}
	return &Input{
		sampleRate: sampleRate,
		channels:   channels,
		frames:     make(chan audio.Frame, buffer),
	}
}

func NewRemoteTrackInput(remote *webrtc.TrackRemote, sampleRate, channels int) (*Input, error) {
	in := NewInput(sampleRate, channels, 0)
	pcm, err := lkmedia.NewPCMRemoteTrack(remote, in,
		lkmedia.WithTargetSampleRate(sampleRate),
		lkmedia.WithTargetChannels(channels),
	)
	if err != nil {
		return nil, err
	}
	in.mu.Lock()
	in.track = pcm
	in.mu.Unlock()
	return in, nil
}

func (in *Input) WriteSample(sample media.PCM16Sample) error {
	in.mu.RLock()
	defer in.mu.RUnlock()
	if in.closed {
		return nil
	}
	data := make([]int16, len(sample))
	copy(data, sample)
	select {
	case in.frames <- audio.Frame{Data: data, SampleRate: in.sampleRate, Channels: in.channels}:
	default:
	}
	return nil
}

func (in *Input) Frames() <-chan audio.Frame {
	return in.frames
}

func (in *Input) Close() error {
	in.mu.Lock()
	defer in.mu.Unlock()
	if in.closed {
		return nil
	}
	in.closed = true
	if in.track != nil {
		in.track.Close()
	}
	close(in.frames)
	return nil
}
