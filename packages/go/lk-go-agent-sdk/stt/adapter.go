package stt

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/beingPro007/lk-go-agent-sdk/audio"
	"github.com/beingPro007/lk-go-agent-sdk/vad"
)

// Recognizer transcribes one finished piece of speech. Batch speech-to-text
// models (Whistle, Whisper and similar) implement it. speech holds frames at
// SampleRate, oldest first.
type Recognizer interface {
	SampleRate() int
	Recognize(ctx context.Context, speech []audio.Frame) (SpeechData, error)
}

// AdapterOption configures NewStreamAdapter.
type AdapterOption func(*adapter)

// WithPreroll sets how much audio before the VAD's start of speech is added
// to each piece. The VAD fires only after some speech (Silero: 250 ms by
// default), so the preroll must be longer than that. Default 600 ms.
func WithPreroll(d time.Duration) AdapterOption { return func(a *adapter) { a.preroll = d } }

// WithErrorHandler gets every recognition error as it happens. The stream
// also returns the last error from Err when it ends.
func WithErrorHandler(fn func(error)) AdapterOption { return func(a *adapter) { a.onErr = fn } }

type adapter struct {
	r       Recognizer
	v       vad.VAD
	preroll time.Duration
	onErr   func(error)
}

// NewStreamAdapter makes a streaming STT from a Recognizer and a VAD. The VAD
// cuts the input into pieces of speech. Each piece is recognized when the VAD
// reports its end, in order, and the stream emits StartOfSpeech, then
// FinalTranscript (when the text is not empty), then EndOfSpeech.
func NewStreamAdapter(r Recognizer, v vad.VAD, opts ...AdapterOption) (STT, error) {
	if r.SampleRate() != v.SampleRate() {
		return nil, fmt.Errorf("stt: the recognizer runs at %d Hz but the VAD at %d Hz", r.SampleRate(), v.SampleRate())
	}
	a := &adapter{r: r, v: v, preroll: 600 * time.Millisecond}
	for _, o := range opts {
		o(a)
	}
	return a, nil
}

func (a *adapter) SampleRate() int  { return a.r.SampleRate() }
func (a *adapter) NumChannels() int { return 1 }

// Recognize transcribes any stretch of speech with the adapter's recognizer.
// A voice session uses it to transcribe a whole user turn in one call at
// commit: one call with all the words is more accurate than the joined texts
// of the pieces (measured with Whistle: 4.2 % against 12.9 % word error at
// 10 dB SNR).
func (a *adapter) Recognize(ctx context.Context, speech []audio.Frame) (SpeechData, error) {
	return a.r.Recognize(ctx, speech)
}

func (a *adapter) Stream(ctx context.Context) SpeechStream {
	ctx, cancel := context.WithCancel(ctx)
	s := &adapterStream{
		a:      a,
		ctx:    ctx,
		cancel: cancel,
		vs:     a.v.Stream(ctx),
		events: make(chan SpeechEvent, 32),
		pieces: make(chan []audio.Frame, 8),
	}
	go s.readVAD()
	go s.recognize()
	return s
}

type adapterStream struct {
	a      *adapter
	ctx    context.Context
	cancel context.CancelFunc
	vs     vad.Stream
	events chan SpeechEvent
	pieces chan []audio.Frame

	mu        sync.Mutex
	recent    []audio.Frame // the last preroll of audio, while nobody speaks
	recentDur time.Duration
	speech    []audio.Frame // the piece of speech so far
	speaking  bool
	err       error
}

func (s *adapterStream) PushFrame(f audio.Frame) {
	s.mu.Lock()
	if s.speaking {
		s.speech = append(s.speech, f)
	} else {
		s.recent = append(s.recent, f)
		s.recentDur += f.Duration()
		for len(s.recent) > 1 && s.recentDur-s.recent[0].Duration() >= s.a.preroll {
			s.recentDur -= s.recent[0].Duration()
			s.recent = s.recent[1:]
		}
	}
	s.mu.Unlock()
	s.vs.PushFrame(f)
}

func (s *adapterStream) Flush()    { s.vs.Flush() }
func (s *adapterStream) EndInput() { s.vs.EndInput() }

func (s *adapterStream) Chan() <-chan SpeechEvent { return s.events }

func (s *adapterStream) Err() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.err != nil {
		return s.err
	}
	return s.vs.Err()
}

func (s *adapterStream) Close() error {
	s.cancel()
	return s.vs.Close()
}

func (s *adapterStream) readVAD() {
	defer close(s.pieces)
	for ev := range s.vs.Chan() {
		switch ev.Type {
		case vad.StartOfSpeech:
			s.mu.Lock()
			s.speaking = true
			s.speech = append([]audio.Frame(nil), s.recent...)
			s.recent, s.recentDur = nil, 0
			s.mu.Unlock()
			s.emit(SpeechEvent{Type: StartOfSpeech})
		case vad.EndOfSpeech:
			s.mu.Lock()
			piece := s.speech
			s.speech, s.speaking = nil, false
			s.mu.Unlock()
			if len(piece) == 0 {
				continue
			}
			select {
			case s.pieces <- piece:
			case <-s.ctx.Done():
				return
			}
		}
	}
}

func (s *adapterStream) recognize() {
	defer close(s.events)
	for piece := range s.pieces {
		data, err := s.a.r.Recognize(s.ctx, piece)
		if err != nil {
			if s.ctx.Err() != nil {
				return
			}
			s.mu.Lock()
			s.err = err
			s.mu.Unlock()
			if s.a.onErr != nil {
				s.a.onErr(err)
			}
		} else if strings.TrimSpace(data.Text) != "" {
			s.emit(SpeechEvent{Type: FinalTranscript, Alternatives: []SpeechData{data}})
		}
		s.emit(SpeechEvent{Type: EndOfSpeech})
	}
}

func (s *adapterStream) emit(ev SpeechEvent) {
	select {
	case s.events <- ev:
	case <-s.ctx.Done():
	}
}
