package main

import (
	"log/slog"

	agents "github.com/beingPro007/lk-go-agent-sdk"
	"github.com/beingPro007/lk-go-agent-sdk/audio/track"
	"github.com/beingPro007/lk-go-agent-sdk/cli"
	lksdk "github.com/livekit/server-sdk-go/v2"
	"github.com/pion/webrtc/v4"
)

const (
	sampleRate = 48000
	channels   = 1
)

func main() {
	cli.Run(agents.WorkerOptions{
		Entrypoint: echo,
	})
}

func echo(job *agents.JobContext) error {
	out, err := track.NewOutput(sampleRate, channels)
	if err != nil {
		return err
	}
	defer out.Close()

	cb := lksdk.NewRoomCallback()
	cb.OnTrackSubscribed = func(remote *webrtc.TrackRemote, pub *lksdk.RemoteTrackPublication, rp *lksdk.RemoteParticipant) {
		if remote.Kind() != webrtc.RTPCodecTypeAudio {
			return
		}
		in, err := track.NewRemoteTrackInput(remote, sampleRate, channels)
		if err != nil {
			slog.Error("failed to open input", "error", err)
			return
		}
		slog.Info("echoing audio track", "from", rp.Identity(), "track", pub.SID())
		go func() {
			defer in.Close()
			for f := range in.Frames() {
				if err := out.CaptureFrame(f); err != nil {
					slog.Error("capture frame failed", "error", err)
					return
				}
			}
		}()
	}

	room, err := job.Connect(cb)
	if err != nil {
		return err
	}
	if _, err := room.LocalParticipant.PublishTrack(out.Track(), &lksdk.TrackPublicationOptions{
		Name: "echo",
	}); err != nil {
		return err
	}
	slog.Info("pcm echo agent ready", "room", room.Name())

	<-job.Done()
	return nil
}
