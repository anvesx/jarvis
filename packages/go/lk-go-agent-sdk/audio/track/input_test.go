package track

import (
	"testing"

	"github.com/livekit/media-sdk"
)

// A pushed PCM16 sample should surface on Frames() as an audio.Frame carrying
// the configured sample rate/channels and the same sample data.
func TestInputWriteSampleToFrame(t *testing.T) {
	in := NewInput(16000, 1, 4)
	in.WriteSample(media.PCM16Sample{1, 2, 3})
	f := <-in.Frames()
	if f.SampleRate != 16000 || f.Channels != 1 {
		t.Fatalf("meta wrong: %+v", f)
	}
	if len(f.Data) != 3 || f.Data[0] != 1 || f.Data[2] != 3 {
		t.Fatalf("data wrong: %v", f.Data)
	}
}

// When the frame buffer is full, WriteSample must drop rather than block or
// error, so a slow consumer never stalls the RTP decode path.
func TestInputDropsWhenFull(t *testing.T) {
	in := NewInput(16000, 1, 1)
	for i := 0; i < 10; i++ {
		if err := in.WriteSample(media.PCM16Sample{int16(i)}); err != nil {
			t.Fatalf("WriteSample err: %v", err)
		}
	}
}

// Close drains buffered frames (range ends), and both double-Close and
// WriteSample-after-Close must be safe no-ops (no send-on-closed panic).
func TestInputCloseEndsRange(t *testing.T) {
	in := NewInput(16000, 1, 4)
	in.WriteSample(media.PCM16Sample{5})
	in.Close()
	count := 0
	for range in.Frames() {
		count++
	}
	if count != 1 {
		t.Fatalf("expected 1 buffered frame, got %d", count)
	}
	if err := in.Close(); err != nil {
		t.Fatalf("double close: %v", err)
	}
	if err := in.WriteSample(media.PCM16Sample{9}); err != nil {
		t.Fatalf("write after close: %v", err)
	}
}
