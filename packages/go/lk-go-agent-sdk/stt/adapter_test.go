package stt

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/beingPro007/lk-go-agent-sdk/audio"
	"github.com/beingPro007/lk-go-agent-sdk/vad"
)

// fakeVAD lets a test decide when speech starts and ends. EndInput ends an
// open piece of speech and closes the stream, like the Silero plugin.
type fakeVAD struct {
	rate int
	mu   sync.Mutex
	last *fakeVADStream
}

func (f *fakeVAD) SampleRate() int { return f.rate }
func (f *fakeVAD) Stream(context.Context) vad.Stream {
	s := &fakeVADStream{events: make(chan vad.Event, 16)}
	f.mu.Lock()
	f.last = s
	f.mu.Unlock()
	return s
}
func (f *fakeVAD) stream() *fakeVADStream {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.last
}

type fakeVADStream struct {
	events   chan vad.Event
	mu       sync.Mutex
	speaking bool
	once     sync.Once
	frames   int
}

func (s *fakeVADStream) PushFrame(audio.Frame) { s.mu.Lock(); s.frames++; s.mu.Unlock() }
func (s *fakeVADStream) Flush()                {}
func (s *fakeVADStream) EndInput() {
	s.once.Do(func() {
		s.mu.Lock()
		if s.speaking {
			s.events <- vad.Event{Type: vad.EndOfSpeech}
		}
		s.mu.Unlock()
		close(s.events)
	})
}
func (s *fakeVADStream) Chan() <-chan vad.Event { return s.events }
func (s *fakeVADStream) Err() error             { return nil }
func (s *fakeVADStream) Close() error           { return nil }
func (s *fakeVADStream) start() {
	s.mu.Lock()
	s.speaking = true
	s.mu.Unlock()
	s.events <- vad.Event{Type: vad.StartOfSpeech}
}
func (s *fakeVADStream) stop() {
	s.mu.Lock()
	s.speaking = false
	s.mu.Unlock()
	s.events <- vad.Event{Type: vad.EndOfSpeech}
}

// fakeRecognizer returns scripted texts and records the samples it got.
type fakeRecognizer struct {
	mu      sync.Mutex
	texts   []string
	err     error
	samples []int
}

func (r *fakeRecognizer) SampleRate() int { return 16000 }
func (r *fakeRecognizer) Recognize(ctx context.Context, speech []audio.Frame) (SpeechData, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	n := 0
	for _, f := range speech {
		n += len(f.Data)
	}
	r.samples = append(r.samples, n)
	if r.err != nil {
		return SpeechData{}, r.err
	}
	text := ""
	if len(r.texts) > 0 {
		text, r.texts = r.texts[0], r.texts[1:]
	}
	return SpeechData{Text: text, Language: "en"}, nil
}

func frame() audio.Frame { return audio.NewFrame(make([]int16, 320), 16000, 1) } // 20 ms

func next(t *testing.T, ch <-chan SpeechEvent) SpeechEvent {
	t.Helper()
	select {
	case ev, ok := <-ch:
		if !ok {
			t.Fatal("the stream closed early")
		}
		return ev
	case <-time.After(2 * time.Second):
		t.Fatal("no event in time")
	}
	return SpeechEvent{}
}

func waitFrames(t *testing.T, s *fakeVADStream, n int) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		s.mu.Lock()
		got := s.frames
		s.mu.Unlock()
		if got >= n {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("the VAD did not get %d frames", n)
}

func TestAdapterEmitsStartFinalEnd(t *testing.T) {
	v, r := &fakeVAD{rate: 16000}, &fakeRecognizer{texts: []string{"hello there"}}
	a, err := NewStreamAdapter(r, v, WithPreroll(100*time.Millisecond))
	if err != nil {
		t.Fatal(err)
	}
	st := a.Stream(context.Background())
	defer st.Close()

	for i := 0; i < 10; i++ { // 200 ms of silence: only the last 100 ms stay as preroll
		st.PushFrame(frame())
	}
	vs := v.stream()
	waitFrames(t, vs, 10)
	vs.start()
	if ev := next(t, st.Chan()); ev.Type != StartOfSpeech {
		t.Fatalf("first event %v, want StartOfSpeech", ev.Type)
	}
	for i := 0; i < 15; i++ { // 300 ms of speech
		st.PushFrame(frame())
	}
	waitFrames(t, vs, 25)
	vs.stop()
	ev := next(t, st.Chan())
	if ev.Type != FinalTranscript || ev.Alternatives[0].Text != "hello there" {
		t.Fatalf("got %+v, want the final transcript", ev)
	}
	if ev := next(t, st.Chan()); ev.Type != EndOfSpeech {
		t.Fatalf("got %v, want EndOfSpeech", ev.Type)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if want := (5 + 15) * 320; r.samples[0] != want { // 100 ms preroll + 300 ms speech
		t.Fatalf("the recognizer got %d samples, want %d", r.samples[0], want)
	}
}

func TestAdapterSkipsEmptyTextAndReportsErrors(t *testing.T) {
	v := &fakeVAD{rate: 16000}
	var handled []error
	var hmu sync.Mutex
	r := &fakeRecognizer{texts: []string{"  "}}
	a, _ := NewStreamAdapter(r, v, WithErrorHandler(func(err error) { hmu.Lock(); handled = append(handled, err); hmu.Unlock() }))
	st := a.Stream(context.Background())
	defer st.Close()
	vs := v.stream()

	vs.start()
	next(t, st.Chan())
	st.PushFrame(frame())
	waitFrames(t, vs, 1)
	vs.stop()
	if ev := next(t, st.Chan()); ev.Type != EndOfSpeech {
		t.Fatalf("an empty text gave %v, want only EndOfSpeech", ev.Type)
	}

	r.mu.Lock()
	r.err = errors.New("speech service is down")
	r.mu.Unlock()
	vs.start()
	next(t, st.Chan())
	st.PushFrame(frame())
	waitFrames(t, vs, 2)
	vs.stop()
	if ev := next(t, st.Chan()); ev.Type != EndOfSpeech {
		t.Fatalf("a failed piece gave %v, want EndOfSpeech", ev.Type)
	}
	if st.Err() == nil {
		t.Fatal("Err() does not report the recognition error")
	}
	hmu.Lock()
	defer hmu.Unlock()
	if len(handled) != 1 {
		t.Fatalf("the error handler ran %d times, want 1", len(handled))
	}
}

func TestAdapterEndInputFinishesTheOpenPiece(t *testing.T) {
	v, r := &fakeVAD{rate: 16000}, &fakeRecognizer{texts: []string{"last words"}}
	a, _ := NewStreamAdapter(r, v)
	st := a.Stream(context.Background())
	defer st.Close()
	vs := v.stream()

	vs.start()
	next(t, st.Chan())
	st.PushFrame(frame())
	waitFrames(t, vs, 1)
	st.EndInput()
	if ev := next(t, st.Chan()); ev.Type != FinalTranscript || ev.Alternatives[0].Text != "last words" {
		t.Fatalf("got %+v, want the last words", ev)
	}
	next(t, st.Chan()) // EndOfSpeech
	select {
	case _, ok := <-st.Chan():
		if ok {
			t.Fatal("the stream did not close after EndInput")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("the stream did not close after EndInput")
	}
}

func TestAdapterRejectsMismatchedRates(t *testing.T) {
	if _, err := NewStreamAdapter(&fakeRecognizer{}, &fakeVAD{rate: 8000}); err == nil {
		t.Fatal("a recognizer at 16 kHz and a VAD at 8 kHz must not be combined")
	}
}
