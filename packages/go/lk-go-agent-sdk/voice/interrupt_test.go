package voice

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/beingPro007/lk-go-agent-sdk/audio"
	"github.com/beingPro007/lk-go-agent-sdk/llm"
	"github.com/beingPro007/lk-go-agent-sdk/stt"
)

// recognizingSTT is a fake STT that can also transcribe a whole turn, like
// stt.NewStreamAdapter.
type recognizingSTT struct {
	*fakeSTT
	text string
	err  error

	mu      sync.Mutex
	calls   int
	samples int
}

func (r *recognizingSTT) Recognize(_ context.Context, speech []audio.Frame) (stt.SpeechData, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.calls++
	r.samples = 0
	for _, f := range speech {
		r.samples += len(f.Data)
	}
	return stt.SpeechData{Text: r.text}, r.err
}

func (r *recognizingSTT) seen() (int, int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.calls, r.samples
}

func lastUser(c *llm.ChatContext) llm.Message {
	for i := len(c.Messages) - 1; i >= 0; i-- {
		if c.Messages[i].Role == llm.RoleUser {
			return c.Messages[i]
		}
	}
	return llm.Message{}
}

// TestShortSpeechDoesNotInterrupt: an echo or an "mm-hmm" while the agent
// speaks does not cut it, and its transcript is not a turn.
func TestShortSpeechDoesNotInterrupt(t *testing.T) {
	speech, voiceDetect := newFakeSTT(), newFakeVAD()
	model := newFakeLLM(textTurn("a long answer"), textTurn("unused"))
	model.hold = make(chan struct{})
	out := &fakeOutput{}
	s := NewAgentSession(WithSTT(speech), WithLLM(model), WithTTS(&fakeTTS{}), WithVAD(voiceDetect),
		WithMinEndpointingDelay(20*time.Millisecond), WithMinInterruptionDuration(300*time.Millisecond))
	events := collectEvents(s)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	errc := startSession(t, ctx, s, &Agent{}, newFakeInput(), out)

	voiceDetect.startSpeech()
	time.Sleep(10 * time.Millisecond)
	speech.final("tell me a story")
	time.Sleep(10 * time.Millisecond)
	voiceDetect.stopSpeech()
	waitFor(t, func() bool { return out.frames() > 0 }, "the agent to speak")
	handle := s.CurrentSpeech()

	voiceDetect.startSpeech() // 50 ms of sound: shorter than the 300 ms minimum
	time.Sleep(50 * time.Millisecond)
	voiceDetect.stopSpeech()
	speech.final("mm hmm")
	waitFor(t, func() bool { return hasEvent(events(), EventUserSpeechIgnored) }, "the short speech to be ignored")
	time.Sleep(400 * time.Millisecond)
	if handle.Interrupted() {
		t.Fatal("50 ms of sound interrupted the agent")
	}
	if model.callCount() != 1 {
		t.Fatalf("the ignored speech became a turn (%d LLM calls)", model.callCount())
	}
	close(model.hold)
	speech.finish()
	waitStart(t, errc)
}

// TestLongSpeechInterrupts: speech that lasts the minimum cuts the agent.
func TestLongSpeechInterrupts(t *testing.T) {
	speech, voiceDetect := newFakeSTT(), newFakeVAD()
	model := newFakeLLM(textTurn("a long answer"), textTurn("ok"))
	model.hold = make(chan struct{})
	out := &fakeOutput{}
	s := NewAgentSession(WithSTT(speech), WithLLM(model), WithTTS(&fakeTTS{}), WithVAD(voiceDetect),
		WithMinEndpointingDelay(20*time.Millisecond), WithMinInterruptionDuration(100*time.Millisecond))
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	errc := startSession(t, ctx, s, &Agent{}, newFakeInput(), out)

	voiceDetect.startSpeech()
	time.Sleep(10 * time.Millisecond)
	speech.final("tell me a story")
	time.Sleep(10 * time.Millisecond)
	voiceDetect.stopSpeech()
	waitFor(t, func() bool { return out.frames() > 0 }, "the agent to speak")
	handle := s.CurrentSpeech()

	voiceDetect.startSpeech() // the user keeps talking
	time.Sleep(50 * time.Millisecond)
	if handle.Interrupted() {
		t.Fatal("interrupted before the minimum duration")
	}
	waitFor(t, handle.Interrupted, "the long speech to interrupt the agent")
	close(model.hold)
	speech.finish()
	waitStart(t, errc)
}

// TestFalseInterruptionResumes: speech cut a reply that nobody heard, and no
// words came; the cut turn is answered again, marked as a replacement.
func TestFalseInterruptionResumes(t *testing.T) {
	speech, voiceDetect := newFakeSTT(), newFakeVAD()
	model := newFakeLLM(textTurn("never heard"), textTurn("the answer"))
	model.holdFirst = make(chan struct{})
	defer close(model.holdFirst)
	out := &fakeOutput{}
	s := NewAgentSession(WithSTT(speech), WithLLM(model), WithTTS(&fakeTTS{}), WithVAD(voiceDetect),
		WithMinEndpointingDelay(20*time.Millisecond), WithMinInterruptionDuration(50*time.Millisecond))
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	errc := startSession(t, ctx, s, &Agent{}, newFakeInput(), out)

	voiceDetect.startSpeech()
	time.Sleep(10 * time.Millisecond)
	speech.final("what is sharding")
	time.Sleep(10 * time.Millisecond)
	voiceDetect.stopSpeech()
	waitFor(t, func() bool { return model.callCount() == 1 }, "the first turn to commit")

	voiceDetect.startSpeech() // 120 ms of noise: long enough to cut, but no words
	time.Sleep(120 * time.Millisecond)
	voiceDetect.stopSpeech()
	waitFor(t, func() bool { return model.callCount() == 2 }, "the cut turn to be answered again")
	waitFor(t, func() bool { return out.frames() > 0 }, "the answer to play")
	if m := lastUser(model.context(1)); m.Content != "what is sharding" || !m.Replaces {
		t.Fatalf("resumed turn %+v, want the same text marked Replaces", m)
	}
	speech.finish()
	waitStart(t, errc)
}

// TestMergedTurnIsMarkedReplaces: a turn that joins an unheard turn carries
// the Replaces flag for the LLM.
func TestMergedTurnIsMarkedReplaces(t *testing.T) {
	speech, voiceDetect := newFakeSTT(), newFakeVAD()
	model := newFakeLLM(textTurn("never heard"), textTurn("ok"))
	model.holdFirst = make(chan struct{})
	defer close(model.holdFirst)
	s := NewAgentSession(WithSTT(speech), WithLLM(model), WithTTS(&fakeTTS{}), WithVAD(voiceDetect),
		WithMinEndpointingDelay(20*time.Millisecond))
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	errc := startSession(t, ctx, s, &Agent{}, newFakeInput(), &fakeOutput{})

	voiceDetect.startSpeech()
	time.Sleep(10 * time.Millisecond)
	speech.final("we should store passwords")
	time.Sleep(10 * time.Millisecond)
	voiceDetect.stopSpeech()
	waitFor(t, func() bool { return model.callCount() == 1 }, "the first part to commit")
	voiceDetect.startSpeech()
	time.Sleep(10 * time.Millisecond)
	speech.final("in a json file")
	time.Sleep(10 * time.Millisecond)
	voiceDetect.stopSpeech()
	waitFor(t, func() bool { return model.callCount() == 2 }, "the merged turn")
	if m := lastUser(model.context(1)); !m.Replaces || m.Content != "we should store passwords in a json file" {
		t.Fatalf("merged turn %+v", m)
	}
	if m := lastUser(model.context(0)); m.Replaces {
		t.Fatal("a first turn must not be marked Replaces")
	}
	speech.finish()
	waitStart(t, errc)
}

// TestWholeTurnIsTranscribed: with an STT that can recognize a whole stretch
// of audio, the turn's text comes from one call over the turn's audio.
func TestWholeTurnIsTranscribed(t *testing.T) {
	rec := &recognizingSTT{fakeSTT: newFakeSTT(), text: "we should store the passwords in a vault"}
	model := newFakeLLM(textTurn("ok"))
	in := newFakeInput()
	s := NewAgentSession(WithSTT(rec), WithLLM(model), WithTTS(&fakeTTS{}))
	events := collectEvents(s)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	errc := startSession(t, ctx, s, &Agent{}, in, &fakeOutput{})

	for i := 0; i < 5; i++ {
		in.frames <- audio.NewFrame(make([]int16, 320), 16000, 1)
	}
	time.Sleep(30 * time.Millisecond)
	rec.final("we should store the")
	rec.final("password in a vault")
	rec.endOfSpeech()
	waitFor(t, func() bool { return model.callCount() == 1 }, "the turn")
	if got := lastUser(model.context(0)).Content; got != "we should store the passwords in a vault" {
		t.Fatalf("the LLM got %q, want the whole-turn transcript", got)
	}
	if calls, samples := rec.seen(); calls != 1 || samples != 5*320 {
		t.Fatalf("recognizer calls=%d samples=%d", calls, samples)
	}
	waitFor(t, func() bool { return hasEvent(events(), EventUserTurn) }, "the user-turn event")
	rec.finish()
	waitStart(t, errc)
}

// TestTurnTranscriptionFallsBack: when the whole-turn call fails, the text of
// the pieces is used, and the error is reported.
func TestTurnTranscriptionFallsBack(t *testing.T) {
	rec := &recognizingSTT{fakeSTT: newFakeSTT(), err: errors.New("speech service is down")}
	model := newFakeLLM(textTurn("ok"))
	in := newFakeInput()
	s := NewAgentSession(WithSTT(rec), WithLLM(model), WithTTS(&fakeTTS{}))
	events := collectEvents(s)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	errc := startSession(t, ctx, s, &Agent{}, in, &fakeOutput{})

	in.frames <- audio.NewFrame(make([]int16, 320), 16000, 1)
	time.Sleep(20 * time.Millisecond)
	rec.final("hello there")
	rec.endOfSpeech()
	waitFor(t, func() bool { return model.callCount() == 1 }, "the turn")
	if got := lastUser(model.context(0)).Content; got != "hello there" {
		t.Fatalf("the LLM got %q, want the text of the pieces", got)
	}
	waitFor(t, func() bool { return hasEvent(events(), EventError) }, "the error event")
	rec.finish()
	waitStart(t, errc)
}

func TestDroppedFramesAreCounted(t *testing.T) {
	s := NewAgentSession()
	ch := make(chan audio.Frame, 1)
	for i := 0; i < 3; i++ {
		s.pushFrame(ch, audio.NewFrame(make([]int16, 320), 16000, 1))
	}
	if s.DroppedFrames() != 2 {
		t.Fatalf("dropped %d, want 2", s.DroppedFrames())
	}
}

// TestHangoverDoesNotInterrupt: a short word leaves the VAD in its silence
// hangover when the interruption timer fires; no voice, so no interruption.
func TestHangoverDoesNotInterrupt(t *testing.T) {
	speech, voiceDetect := newFakeSTT(), newFakeVAD()
	model := newFakeLLM(textTurn("a long answer"), textTurn("unused"))
	model.hold = make(chan struct{})
	out := &fakeOutput{}
	s := NewAgentSession(WithSTT(speech), WithLLM(model), WithTTS(&fakeTTS{}), WithVAD(voiceDetect),
		WithMinEndpointingDelay(20*time.Millisecond), WithMinInterruptionDuration(300*time.Millisecond))
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	errc := startSession(t, ctx, s, &Agent{}, newFakeInput(), out)

	voiceDetect.startSpeech()
	time.Sleep(10 * time.Millisecond)
	speech.final("tell me a story")
	time.Sleep(10 * time.Millisecond)
	voiceDetect.stopSpeech()
	waitFor(t, func() bool { return out.frames() > 0 }, "the agent to speak")
	handle := s.CurrentSpeech()

	voiceDetect.infer(0.9) // a short "yeah"
	voiceDetect.startSpeech()
	voiceDetect.infer(0.1) // the word ended: only the VAD's silence hangover is left
	time.Sleep(500 * time.Millisecond)
	if handle.Interrupted() {
		t.Fatal("the silence hangover after a short word interrupted the agent")
	}
	voiceDetect.stopSpeech()
	close(model.hold)
	speech.finish()
	waitStart(t, errc)
}

// TestContinuationKeepsTheFirstPartsAudio: the user goes on talking while the
// agent thinks; the merged turn is transcribed from all of its audio, the
// first part included, even when that part is longer than the preroll.
func TestContinuationKeepsTheFirstPartsAudio(t *testing.T) {
	rec := &recognizingSTT{fakeSTT: newFakeSTT(), text: "the whole merged turn"}
	voiceDetect := newFakeVAD()
	model := newFakeLLM(textTurn("never heard"), textTurn("ok"))
	model.holdFirst = make(chan struct{})
	defer close(model.holdFirst)
	in := newFakeInput()
	s := NewAgentSession(WithSTT(rec), WithLLM(model), WithTTS(&fakeTTS{}), WithVAD(voiceDetect),
		WithMinEndpointingDelay(20*time.Millisecond), WithMinInterruptionDuration(50*time.Millisecond))
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	errc := startSession(t, ctx, s, &Agent{}, in, &fakeOutput{})
	frame := func() audio.Frame { return audio.NewFrame(make([]int16, 320), 16000, 1) } // 20 ms

	voiceDetect.startSpeech()
	for i := 0; i < 75; i++ { // 1.5 s: longer than the 1 s preroll
		in.frames <- frame()
	}
	time.Sleep(30 * time.Millisecond)
	rec.final("i think we should use go")
	time.Sleep(10 * time.Millisecond)
	voiceDetect.stopSpeech()
	waitFor(t, func() bool { return model.callCount() == 1 }, "the first part to commit")

	voiceDetect.startSpeech() // the user goes on while the agent thinks
	for i := 0; i < 10; i++ {
		in.frames <- frame()
	}
	time.Sleep(100 * time.Millisecond) // past the 50 ms minimum: this cuts the unheard reply
	rec.final("because the team knows it")
	time.Sleep(10 * time.Millisecond)
	voiceDetect.stopSpeech()
	waitFor(t, func() bool { return model.callCount() == 2 }, "the merged turn")
	if _, samples := rec.seen(); samples < 85*320 {
		t.Fatalf("the merged turn was transcribed from %d samples; want all %d of both parts", samples, 85*320)
	}
	if m := lastUser(model.context(1)); !m.Replaces || m.Content != "the whole merged turn" {
		t.Fatalf("merged turn %+v", m)
	}
	rec.finish()
	waitStart(t, errc)
}
