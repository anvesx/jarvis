package track

import (
	"testing"

	"github.com/beingPro007/lk-go-agent-sdk/audio"
)

// A fresh Output exposes a publishable track and accepts a captured frame and a
// clean Close without error — the minimal happy path before publishing to a room.
func TestNewOutput(t *testing.T) {
	out, err := NewOutput(24000, 1)
	if err != nil {
		t.Fatalf("NewOutput: %v", err)
	}
	if out.Track() == nil {
		t.Fatal("nil track")
	}
	if err := out.CaptureFrame(audio.NewFrame([]int16{1, 2, 3}, 24000, 1)); err != nil {
		t.Fatalf("CaptureFrame: %v", err)
	}
	if err := out.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
}
