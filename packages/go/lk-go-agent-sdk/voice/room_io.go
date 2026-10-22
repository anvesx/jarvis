package voice

import (
	"fmt"
	"sync"

	agents "github.com/beingPro007/lk-go-agent-sdk"
	"github.com/beingPro007/lk-go-agent-sdk/audio/track"
	lksdk "github.com/livekit/server-sdk-go/v2"
	"github.com/pion/webrtc/v4"
)

type RoomOption func(*roomConfig)

type roomConfig struct {
	inputSampleRate  int
	inputChannels    int
	outputSampleRate int
	outputChannels   int
	trackName        string
}

func WithInputSampleRate(r int) RoomOption  { return func(c *roomConfig) { c.inputSampleRate = r } }
func WithInputChannels(n int) RoomOption    { return func(c *roomConfig) { c.inputChannels = n } }
func WithOutputSampleRate(r int) RoomOption { return func(c *roomConfig) { c.outputSampleRate = r } }
func WithOutputChannels(n int) RoomOption   { return func(c *roomConfig) { c.outputChannels = n } }
func WithTrackName(n string) RoomOption     { return func(c *roomConfig) { c.trackName = n } }

func RunRoomSession(job *agents.JobContext, s *AgentSession, agent *Agent, opts ...RoomOption) error {
	cfg := &roomConfig{trackName: "agent"}
	for _, o := range opts {
		o(cfg)
	}

	inRate := cfg.inputSampleRate
	inCh := cfg.inputChannels
	if inRate == 0 && s.stt != nil {
		inRate = s.stt.SampleRate()
		inCh = s.stt.NumChannels()
	}
	if inRate == 0 {
		inRate = 16000
	}
	if inCh == 0 {
		inCh = 1
	}

	outRate := cfg.outputSampleRate
	outCh := cfg.outputChannels
	if outRate == 0 && s.tts != nil {
		outRate = s.tts.SampleRate()
		outCh = s.tts.NumChannels()
	}
	if outRate == 0 {
		outRate = 24000
	}
	if outCh == 0 {
		outCh = 1
	}

	if s.vad != nil && s.vad.SampleRate() != inRate {
		return fmt.Errorf("voice: VAD expects %d Hz but the input runs at %d Hz", s.vad.SampleRate(), inRate)
	}

	inputCh := make(chan *track.Input, 1)
	var once sync.Once
	cb := lksdk.NewRoomCallback()
	cb.OnTrackSubscribed = func(remote *webrtc.TrackRemote, pub *lksdk.RemoteTrackPublication, rp *lksdk.RemoteParticipant) {
		if remote.Kind() != webrtc.RTPCodecTypeAudio {
			return
		}
		in, err := track.NewRemoteTrackInput(remote, inRate, inCh)
		if err != nil {
			return
		}
		once.Do(func() { inputCh <- in })
	}

	room, err := job.Connect(cb)
	if err != nil {
		return err
	}

	out, err := track.NewOutput(outRate, outCh)
	if err != nil {
		return err
	}
	defer out.Close()
	if _, err := room.LocalParticipant.PublishTrack(out.Track(), &lksdk.TrackPublicationOptions{Name: cfg.trackName}); err != nil {
		return err
	}

	select {
	case in := <-inputCh:
		defer in.Close()
		return s.Start(job.Context(), agent, in, out)
	case <-job.Done():
		return nil
	}
}
