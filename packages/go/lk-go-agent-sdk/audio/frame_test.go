package audio

import (
	"testing"
	"time"
)

// A stereo frame's per-channel sample count is total samples divided by channels.
func TestSamplesPerChannel(t *testing.T) {
	f := NewFrame(make([]int16, 960), 48000, 2)
	if got := f.SamplesPerChannel(); got != 480 {
		t.Fatalf("SamplesPerChannel = %d, want 480", got)
	}
}

// 480 samples per channel at 48kHz should report as exactly 10ms of audio.
func TestDuration(t *testing.T) {
	f := NewFrame(make([]int16, 480), 48000, 1)
	if got := f.Duration(); got != 10*time.Millisecond {
		t.Fatalf("Duration = %v, want 10ms", got)
	}
}

// Bytes -> FromBytes must be lossless, including the int16 min/max edge values.
func TestBytesRoundTrip(t *testing.T) {
	orig := NewFrame([]int16{0, 1, -1, 32767, -32768, 1234}, 16000, 1)
	got := FromBytes(orig.Bytes(), 16000, 1)
	if len(got.Data) != len(orig.Data) {
		t.Fatalf("len = %d, want %d", len(got.Data), len(orig.Data))
	}
	for i := range orig.Data {
		if got.Data[i] != orig.Data[i] {
			t.Fatalf("sample %d = %d, want %d", i, got.Data[i], orig.Data[i])
		}
	}
}

// Clone must copy the backing array so mutating the copy never touches the original.
func TestCloneIsDeep(t *testing.T) {
	orig := NewFrame([]int16{1, 2, 3}, 16000, 1)
	c := orig.Clone()
	c.Data[0] = 99
	if orig.Data[0] != 1 {
		t.Fatalf("Clone shares backing array: orig mutated to %d", orig.Data[0])
	}
}
