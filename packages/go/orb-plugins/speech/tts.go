package speech

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"sync"
	"unicode"

	"github.com/beingPro007/lk-go-agent-sdk/audio"
	"github.com/beingPro007/lk-go-agent-sdk/tts"
)

const (
	ttsRate        = 24000
	frameSamples   = 480 // 20 ms at 24 kHz
	minSentence    = 20  // a shorter piece waits for more text, so "Dr." or "e.g." do not cut a sentence
	maxPiece       = 220 // a longer piece with no sentence end is cut at a comma or a space, to keep the first audio fast
	firstClause    = 30  // the first piece of a reply may end at a comma after this many characters: the first audio comes sooner
	sentenceBuffer = 32
)

// TTS is a tts.TTS backed by Kokoro in the speech service. It cuts the text
// into sentences as the LLM streams it, and synthesizes one sentence at a
// time, in order, so the first audio starts after the first sentence.
type TTS struct {
	c     *Client
	voice string
	speed float64
}

// NewTTS returns a TTS with a Kokoro voice ("" uses the service default).
func NewTTS(c *Client, voice string) *TTS { return &TTS{c: c, voice: voice} }

// WithSpeed sets the speaking speed, from 0.5 to 2.0. Zero keeps the default.
func (t *TTS) WithSpeed(speed float64) *TTS { t.speed = speed; return t }

func (t *TTS) SampleRate() int  { return ttsRate }
func (t *TTS) NumChannels() int { return 1 }

// Stream starts one utterance.
func (t *TTS) Stream(ctx context.Context) tts.SynthesizeStream {
	ctx, cancel := context.WithCancel(ctx)
	s := &ttsStream{t: t, ctx: ctx, cancel: cancel, sentences: make(chan string, sentenceBuffer),
		out: make(chan tts.SynthesizedAudio, 64)}
	go s.run()
	return s
}

type ttsStream struct {
	t      *TTS
	ctx    context.Context
	cancel context.CancelFunc

	mu        sync.Mutex
	buf       strings.Builder
	ended     bool
	sent      int // pieces sent so far
	sentences chan string
	out       chan tts.SynthesizedAudio
	err       error
}

func (s *ttsStream) PushText(token string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.ended {
		return
	}
	s.buf.WriteString(token)
	for {
		piece, rest, ok := cut(s.buf.String(), s.sent == 0)
		if !ok {
			return
		}
		s.buf.Reset()
		s.buf.WriteString(rest)
		s.send(piece)
	}
}

func (s *ttsStream) Flush() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.ended {
		return
	}
	s.flushLocked()
}

func (s *ttsStream) EndInput() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.ended {
		return
	}
	s.flushLocked()
	s.ended = true
	close(s.sentences)
}

func (s *ttsStream) flushLocked() {
	if rest := strings.TrimSpace(s.buf.String()); rest != "" {
		s.send(rest)
	}
	s.buf.Reset()
}

func (s *ttsStream) send(piece string) {
	s.sent++
	select {
	case s.sentences <- piece:
	case <-s.ctx.Done():
	}
}

func (s *ttsStream) Chan() <-chan tts.SynthesizedAudio { return s.out }

func (s *ttsStream) Err() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.err
}

func (s *ttsStream) Close() error {
	s.cancel()
	return nil
}

func (s *ttsStream) run() {
	defer close(s.out)
	segment := 0
	for {
		select {
		case <-s.ctx.Done():
			return
		case piece, ok := <-s.sentences:
			if !ok {
				return
			}
			segment++
			if err := s.speak(piece, fmt.Sprintf("s%d", segment)); err != nil {
				if s.ctx.Err() != nil {
					return
				}
				s.mu.Lock()
				s.err = err
				s.mu.Unlock()
			}
		}
	}
}

// speak synthesizes one piece and emits it as 20 ms frames, then an empty
// frame marked Final that ends the segment.
func (s *ttsStream) speak(text, segment string) error {
	body, _ := json.Marshal(map[string]any{"text": text, "voice": s.t.voice, "speed": nullable(s.t.speed)})
	resp, err := s.t.c.post(s.ctx, "/v1/tts", "application/json", body)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	buf := make([]byte, frameSamples*2)
	for {
		n, err := io.ReadFull(resp.Body, buf)
		if n >= 2 {
			samples := make([]int16, n/2)
			for i := range samples {
				samples[i] = int16(binary.LittleEndian.Uint16(buf[i*2:]))
			}
			select {
			case s.out <- tts.SynthesizedAudio{Frame: audio.NewFrame(samples, ttsRate, 1), Text: text, SegmentID: segment}:
			case <-s.ctx.Done():
				return s.ctx.Err()
			}
		}
		if err == io.EOF || err == io.ErrUnexpectedEOF {
			break
		}
		if err != nil {
			return fmt.Errorf("speech: read the audio: %w", err)
		}
	}
	select {
	case s.out <- tts.SynthesizedAudio{Text: text, SegmentID: segment, Final: true}:
	case <-s.ctx.Done():
	}
	return nil
}

func nullable(v float64) any {
	if v == 0 {
		return nil
	}
	return v
}

// cut takes the first speakable piece from text: a sentence that ends with
// . ! ? or a line break and is at least minSentence long, or, when the text
// is longer than maxPiece with no sentence end, the part up to the last comma
// or space. For the first piece of a reply (first), a clause that ends with a
// comma, semicolon or colon after firstClause characters is enough: Kokoro
// needs about 5.5 ms per character, so a short first piece starts the voice
// sooner. ok is false when the text needs more tokens first.
func cut(text string, first bool) (piece, rest string, ok bool) {
	runes := []rune(text)
	for i, r := range runes {
		end := r == '\n' || ((r == '.' || r == '!' || r == '?') && i+1 < len(runes) && unicode.IsSpace(runes[i+1]))
		if end && len(strings.TrimSpace(string(runes[:i+1]))) >= minSentence {
			return strings.TrimSpace(string(runes[:i+1])), strings.TrimLeftFunc(string(runes[i+1:]), unicode.IsSpace), true
		}
	}
	if first {
		for i, r := range runes {
			clause := (r == ',' || r == ';' || r == ':') && i+1 < len(runes) && unicode.IsSpace(runes[i+1])
			if clause && len(strings.TrimSpace(string(runes[:i+1]))) >= firstClause {
				return strings.TrimSpace(string(runes[:i+1])), strings.TrimLeftFunc(string(runes[i+1:]), unicode.IsSpace), true
			}
		}
	}
	if len(runes) > maxPiece {
		head := string(runes[:maxPiece])
		at := strings.LastIndexAny(head, ",;:")
		if at < minSentence {
			at = strings.LastIndexByte(head, ' ')
		}
		if at >= minSentence {
			return strings.TrimSpace(head[:at+1]), strings.TrimLeftFunc(string(runes)[at+1:], unicode.IsSpace), true
		}
	}
	return "", text, false
}
