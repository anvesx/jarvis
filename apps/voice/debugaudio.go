package main

import (
	"context"
	"encoding/binary"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sync/atomic"
	"time"

	"github.com/beingPro007/lk-go-agent-sdk/audio"
	"github.com/beingPro007/lk-go-agent-sdk/stt"
)

// dumpRecognizer saves the audio of every recognition call as a WAV file,
// with its text next to it, so turn and VAD settings can be tuned with real
// voices. It is for debugging only (ORB_DEBUG_AUDIO_DIR): the files hold the
// user's voice.
type dumpRecognizer struct {
	r   stt.Recognizer
	dir string
	log *slog.Logger
	n   atomic.Int64
}

func newDumpRecognizer(r stt.Recognizer, dir string, log *slog.Logger) *dumpRecognizer {
	return &dumpRecognizer{r: r, dir: dir, log: log}
}

func (d *dumpRecognizer) SampleRate() int { return d.r.SampleRate() }

func (d *dumpRecognizer) Recognize(ctx context.Context, speech []audio.Frame) (stt.SpeechData, error) {
	data, err := d.r.Recognize(ctx, speech)
	var dur time.Duration
	for _, f := range speech {
		dur += f.Duration()
	}
	name := fmt.Sprintf("%s-%03d-%dms", time.Now().Format("150405.000"), d.n.Add(1), dur.Milliseconds())
	text := data.Text
	if err != nil {
		text = "ERROR: " + err.Error()
	}
	if werr := writeArtifacts(d.dir, name, speech, text); werr != nil {
		d.log.Warn("could not save the debug audio", "error", werr)
	}
	return data, err
}

func writeArtifacts(dir, name string, speech []audio.Frame, text string) error {
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, name+".txt"), []byte(text+"\n"), 0o600); err != nil {
		return err
	}
	return writeWAV(filepath.Join(dir, name+".wav"), speech)
}

// writeWAV writes mono 16-bit PCM frames as a WAV file.
func writeWAV(path string, frames []audio.Frame) error {
	rate, samples := 16000, 0
	if len(frames) > 0 {
		rate = frames[0].SampleRate
	}
	for _, f := range frames {
		samples += len(f.Data)
	}
	buf := make([]byte, 0, 44+samples*2)
	le := binary.LittleEndian
	buf = append(buf, "RIFF"...)
	buf = le.AppendUint32(buf, uint32(36+samples*2))
	buf = append(buf, "WAVEfmt "...)
	buf = le.AppendUint32(buf, 16)
	buf = le.AppendUint16(buf, 1) // PCM
	buf = le.AppendUint16(buf, 1) // mono
	buf = le.AppendUint32(buf, uint32(rate))
	buf = le.AppendUint32(buf, uint32(rate*2))
	buf = le.AppendUint16(buf, 2)
	buf = le.AppendUint16(buf, 16)
	buf = append(buf, "data"...)
	buf = le.AppendUint32(buf, uint32(samples*2))
	for _, f := range frames {
		for _, s := range f.Data {
			buf = le.AppendUint16(buf, uint16(s))
		}
	}
	return os.WriteFile(path, buf, 0o600)
}
