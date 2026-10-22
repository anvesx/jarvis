package voice

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/beingPro007/lk-go-agent-sdk/audio"
	"github.com/beingPro007/lk-go-agent-sdk/llm"
)

// scriptedDetector answers with a fixed probability after a delay and records
// what it was asked.
type scriptedDetector struct {
	mu      sync.Mutex
	p       float64
	delay   time.Duration
	err     error
	calls   int
	text    string
	samples int
	heard   bool
}

func (d *scriptedDetector) answer(ctx context.Context, c *llm.ChatContext, turn []audio.Frame, heard bool) (float64, error) {
	select {
	case <-time.After(d.delay):
	case <-ctx.Done():
		return 0, ctx.Err()
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	d.calls++
	d.text = c.Messages[len(c.Messages)-1].Content
	d.heard = heard
	d.samples = 0
	for _, f := range turn {
		d.samples += len(f.Data)
	}
	return d.p, d.err
}

func (d *scriptedDetector) PredictEndOfTurn(ctx context.Context, c *llm.ChatContext) (float64, error) {
	return d.answer(ctx, c, nil, false)
}

func (d *scriptedDetector) callCount() int {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.calls
}

// audioDetector is the same detector, but it can hear the turn.
type audioDetector struct{ *scriptedDetector }

func (d audioDetector) PredictEndOfTurnAudio(ctx context.Context, c *llm.ChatContext, turn []audio.Frame) (float64, error) {
	return d.answer(ctx, c, turn, true)
}

func vadSession(t *testing.T, det TurnDetector, min, max time.Duration, in *fakeInput) (*AgentSession, *fakeSTT, *fakeVAD, *fakeLLM, context.CancelFunc, chan error) {
	t.Helper()
	speech, voiceDetect := newFakeSTT(), newFakeVAD()
	model := newFakeLLM(textTurn("ok"), textTurn("ok"))
	s := NewAgentSession(
		WithSTT(speech), WithLLM(model), WithTTS(&fakeTTS{}), WithVAD(voiceDetect),
		WithTurnDetector(det, 0.5), WithMinEndpointingDelay(min), WithMaxEndpointingDelay(max),
	)
	ctx, cancel := context.WithCancel(context.Background())
	errc := startSession(t, ctx, s, &Agent{}, in, &fakeOutput{})
	return s, speech, voiceDetect, model, cancel, errc
}

// TestVADEndpointingAsksTheTurnDetector: silence plus text asks the detector,
// and "not finished" holds the turn until the max delay.
func TestVADEndpointingAsksTheTurnDetector(t *testing.T) {
	det := &scriptedDetector{p: 0.1}
	_, speech, voiceDetect, model, cancel, errc := vadSession(t, det, 20*time.Millisecond, 300*time.Millisecond, newFakeInput())
	defer cancel()

	voiceDetect.startSpeech()
	speech.final("i was thinking that")
	voiceDetect.stopSpeech()
	// The VAD and the STT use separate channels, so the transcript can arrive
	// before the start of speech; then the detector is asked twice.
	waitFor(t, func() bool { return det.callCount() >= 1 }, "the detector to be asked")
	time.Sleep(120 * time.Millisecond)
	if model.callCount() != 0 {
		t.Fatal("the turn committed although the detector said the user is not finished")
	}
	waitFor(t, func() bool { return model.callCount() == 1 }, "the max delay to commit the turn")

	speech.finish()
	waitStart(t, errc)
}

// TestVADEndpointingConfidentDetector: "finished" commits after the min delay.
func TestVADEndpointingConfidentDetector(t *testing.T) {
	det := &scriptedDetector{p: 0.9}
	_, speech, voiceDetect, model, cancel, errc := vadSession(t, det, 30*time.Millisecond, 10*time.Second, newFakeInput())
	defer cancel()

	voiceDetect.startSpeech()
	speech.final("that is all")
	voiceDetect.stopSpeech()
	start := time.Now()
	waitFor(t, func() bool { return model.callCount() == 1 }, "the turn to commit after the min delay")
	if time.Since(start) > 2*time.Second {
		t.Fatal("a confident detector waited for the max delay")
	}

	speech.finish()
	waitStart(t, errc)
}

// TestStaleDetectorAnswerIsDropped: the user talks again before the detector
// answers, so its answer about the shorter turn must not commit anything.
func TestStaleDetectorAnswerIsDropped(t *testing.T) {
	det := &scriptedDetector{p: 0.9, delay: 150 * time.Millisecond}
	_, speech, voiceDetect, model, cancel, errc := vadSession(t, det, 20*time.Millisecond, 10*time.Second, newFakeInput())
	defer cancel()

	voiceDetect.startSpeech()
	speech.final("i was")
	voiceDetect.stopSpeech()
	time.Sleep(30 * time.Millisecond)
	voiceDetect.startSpeech() // the user goes on
	time.Sleep(250 * time.Millisecond)
	if model.callCount() != 0 {
		t.Fatal("a stale detector answer committed the turn")
	}
	speech.final("going to say this")
	voiceDetect.stopSpeech()
	waitFor(t, func() bool { return model.callCount() == 1 }, "the full turn to commit")
	if got := model.context(0).Messages[0].Content; got != "i was going to say this" {
		t.Fatalf("committed %q, want the whole turn", got)
	}

	speech.finish()
	waitStart(t, errc)
}

// TestAudioTurnDetectorHearsTheTurn: an AudioTurnDetector gets the turn's
// audio and the turn's text.
func TestAudioTurnDetectorHearsTheTurn(t *testing.T) {
	det := audioDetector{&scriptedDetector{p: 0.9}}
	in := newFakeInput()
	_, speech, voiceDetect, model, cancel, errc := vadSession(t, det, 20*time.Millisecond, 10*time.Second, in)
	defer cancel()

	voiceDetect.startSpeech()
	for i := 0; i < 5; i++ {
		in.frames <- audio.NewFrame(make([]int16, 320), 16000, 1) // 5 x 20 ms
	}
	time.Sleep(50 * time.Millisecond)
	speech.final("are you there")
	voiceDetect.stopSpeech()
	waitFor(t, func() bool { return model.callCount() == 1 }, "the turn to commit")

	det.mu.Lock()
	heard, samples, text := det.heard, det.samples, det.text
	det.mu.Unlock()
	if !heard || samples != 5*320 || strings.TrimSpace(text) != "are you there" {
		t.Fatalf("detector heard=%v samples=%d text=%q", heard, samples, text)
	}

	speech.finish()
	waitStart(t, errc)
}

// TestFailedDetectorDoesNotHoldTheTurn: an error reports itself and the turn
// commits after the min delay.
func TestFailedDetectorDoesNotHoldTheTurn(t *testing.T) {
	det := &scriptedDetector{err: errors.New("speech service is down")}
	s, speech, voiceDetect, model, cancel, errc := vadSession(t, det, 30*time.Millisecond, 10*time.Second, newFakeInput())
	defer cancel()
	events := collectEvents(s)

	voiceDetect.startSpeech()
	speech.final("hello")
	voiceDetect.stopSpeech()
	waitFor(t, func() bool { return model.callCount() == 1 }, "the turn to commit despite the error")
	waitFor(t, func() bool { return hasEvent(events(), EventError) }, "the error to be reported")

	speech.finish()
	waitStart(t, errc)
}

// TestSTTEndOfSpeechReusesTheAnswer: an STT end of speech after a confident
// answer commits at once, without a second detector call.
func TestSTTEndOfSpeechReusesTheAnswer(t *testing.T) {
	det := &scriptedDetector{p: 0.9}
	_, speech, voiceDetect, model, cancel, errc := vadSession(t, det, 5*time.Second, 10*time.Second, newFakeInput())
	defer cancel()

	voiceDetect.startSpeech()
	speech.final("done now")
	voiceDetect.stopSpeech()
	waitFor(t, func() bool { return det.callCount() >= 1 }, "the detector to answer")
	time.Sleep(30 * time.Millisecond) // let the answer reach the session
	before := det.callCount()
	speech.endOfSpeech()
	waitFor(t, func() bool { return model.callCount() == 1 }, "the STT end of speech to commit at once")
	if det.callCount() != before {
		t.Fatalf("the STT end of speech asked the detector again (%d calls, want %d)", det.callCount(), before)
	}

	speech.finish()
	waitStart(t, errc)
}

func TestTurnAudioKeepsTheLastSeconds(t *testing.T) {
	ta := newTurnAudio(100 * time.Millisecond)
	for i := 0; i < 8; i++ {
		ta.push(audio.NewFrame(make([]int16, 320), 16000, 1)) // 20 ms each
	}
	if got := ta.duration(); got != 100*time.Millisecond {
		t.Fatalf("duration %v, want 100ms", got)
	}
	ta.keepLast(40 * time.Millisecond)
	if got := len(ta.snapshot()); got != 2 {
		t.Fatalf("%d frames after keepLast(40ms), want 2", got)
	}
	ta.keepLast(0)
	if len(ta.snapshot()) != 0 || ta.duration() != 0 {
		t.Fatal("keepLast(0) did not empty the buffer")
	}
}

// TestUnheardReplyMergesWithTheNextTurn: the user goes on talking before any
// audio of the reply played, so both parts become one turn, and the unheard
// reply stays out of the chat.
func TestUnheardReplyMergesWithTheNextTurn(t *testing.T) {
	speech, voiceDetect := newFakeSTT(), newFakeVAD()
	model := newFakeLLM(textTurn("never heard"), textTurn("ok"))
	model.holdFirst = make(chan struct{})
	defer close(model.holdFirst)
	out := &fakeOutput{}
	s := NewAgentSession(WithSTT(speech), WithLLM(model), WithTTS(&fakeTTS{}), WithVAD(voiceDetect),
		WithMinEndpointingDelay(30*time.Millisecond))
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	errc := startSession(t, ctx, s, &Agent{}, newFakeInput(), out)

	voiceDetect.startSpeech()
	time.Sleep(10 * time.Millisecond)
	speech.final("we should store passwords")
	time.Sleep(10 * time.Millisecond)
	voiceDetect.stopSpeech()
	waitFor(t, func() bool { return model.callCount() == 1 }, "the first part to commit")

	voiceDetect.startSpeech() // the user goes on before the agent says anything
	time.Sleep(10 * time.Millisecond)
	speech.final("in a json file")
	time.Sleep(10 * time.Millisecond)
	voiceDetect.stopSpeech()
	waitFor(t, func() bool { return model.callCount() == 2 }, "the merged turn to commit")
	waitFor(t, func() bool { return out.frames() > 0 }, "the merged reply to play")

	got := model.context(1).Messages
	if last := got[len(got)-1]; last.Content != "we should store passwords in a json file" {
		t.Fatalf("the merged turn is %q", last.Content)
	}
	speech.finish()
	waitStart(t, errc)
	for _, m := range s.ChatContext().Messages {
		if m.Content == "we should store passwords" || m.Content == "never heard" {
			t.Fatalf("the unheard first part is in the chat: %+v", s.ChatContext().Messages)
		}
	}
}

// TestHeardReplyIsNotMerged: once the user heard some of the reply, the next
// speech is a new turn.
func TestHeardReplyIsNotMerged(t *testing.T) {
	speech, voiceDetect := newFakeSTT(), newFakeVAD()
	model := newFakeLLM(textTurn("a long answer"), textTurn("ok"))
	model.hold = make(chan struct{})
	out := &fakeOutput{}
	s := NewAgentSession(WithSTT(speech), WithLLM(model), WithTTS(&fakeTTS{}), WithVAD(voiceDetect),
		WithMinEndpointingDelay(30*time.Millisecond))
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	errc := startSession(t, ctx, s, &Agent{}, newFakeInput(), out)

	voiceDetect.startSpeech()
	time.Sleep(10 * time.Millisecond)
	speech.final("tell me a story")
	time.Sleep(10 * time.Millisecond)
	voiceDetect.stopSpeech()
	waitFor(t, func() bool { return out.frames() > 0 }, "the reply to play")

	voiceDetect.startSpeech() // barge-in after the user heard some audio
	time.Sleep(10 * time.Millisecond)
	speech.final("stop please")
	time.Sleep(10 * time.Millisecond)
	voiceDetect.stopSpeech()
	waitFor(t, func() bool { return model.callCount() == 2 }, "the new turn to commit")
	close(model.hold)
	got := model.context(1).Messages
	if last := got[len(got)-1]; last.Content != "stop please" {
		t.Fatalf("the new turn is %q, want only the new words", last.Content)
	}
	speech.finish()
	waitStart(t, errc)
}
