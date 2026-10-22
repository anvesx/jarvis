package audio

import (
	"encoding/binary"
	"time"
)

type Frame struct {
	Data       []int16
	SampleRate int
	Channels   int
}

func NewFrame(data []int16, sampleRate, channels int) Frame {
	return Frame{Data: data, SampleRate: sampleRate, Channels: channels}
}

func (f Frame) SamplesPerChannel() int {
	if f.Channels <= 0 {
		return 0
	}
	return len(f.Data) / f.Channels
}

func (f Frame) Duration() time.Duration {
	if f.SampleRate <= 0 {
		return 0
	}
	return time.Duration(f.SamplesPerChannel()) * time.Second / time.Duration(f.SampleRate)
}

func (f Frame) Bytes() []byte {
	out := make([]byte, len(f.Data)*2)
	for i, s := range f.Data {
		binary.LittleEndian.PutUint16(out[i*2:], uint16(s))
	}
	return out
}

func FromBytes(data []byte, sampleRate, channels int) Frame {
	if len(data)%2 != 0 {
		panic("audio: PCM16 byte slice must have even length")
	}
	samples := make([]int16, len(data)/2)
	for i := range samples {
		samples[i] = int16(binary.LittleEndian.Uint16(data[i*2:]))
	}
	return Frame{Data: samples, SampleRate: sampleRate, Channels: channels}
}

func (f Frame) Clone() Frame {
	data := make([]int16, len(f.Data))
	copy(data, f.Data)
	return Frame{Data: data, SampleRate: f.SampleRate, Channels: f.Channels}
}
