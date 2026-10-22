package voice

import (
	"context"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/beingPro007/lk-go-agent-sdk/audio"
	"github.com/beingPro007/lk-go-agent-sdk/llm"
	"github.com/beingPro007/lk-go-agent-sdk/tts"
)

func waitFor(t *testing.T, cond func() bool, what string) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(2 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
}

// startSession runs Start in the background and waits until the session has
// taken ownership of the audio output, i.e. it is ready to receive events.
func startSession(t *testing.T, ctx context.Context, s *AgentSession, agent *Agent, in AudioInput, out AudioOutput) chan error {
	t.Helper()
	errc := make(chan error, 1)
	go func() { errc <- s.Start(ctx, agent, in, out) }()
	waitFor(t, func() bool {
		s.mu.Lock()
		defer s.mu.Unlock()
		return s.out != nil
	}, "session to start")
	return errc
}

func collectEvents(s *AgentSession) func() []Event {
	var mu sync.Mutex
	var got []Event
	go func() {
		for ev := range s.Events() {
			mu.Lock()
			got = append(got, ev)
			mu.Unlock()
		}
	}()
	return func() []Event {
		mu.Lock()
		defer mu.Unlock()
		out := make([]Event, len(got))
		copy(out, got)
		return out
	}
}

func hasEvent(events []Event, t EventType) bool {
	for _, ev := range events {
		if ev.Type == t {
			return true
		}
	}
	return false
}

func waitStart(t *testing.T, errc chan error) {
	t.Helper()
	select {
	case err := <-errc:
		if err != nil && err != context.Canceled {
			t.Fatalf("Start: %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("session did not finish in time")
	}
}

// TestFullTurn drives one complete conversational turn through the real
// orchestrator with fake plugins: a transcript plus an end-of-turn should
// produce an LLM reply, synthesize it, and land the audio on the output.
func TestFullTurn(t *testing.T) {
	speech := newFakeSTT()
	model := newFakeLLM(textTurn("hi there"))
	out := &fakeOutput{}

	s := NewAgentSession(WithSTT(speech), WithLLM(model), WithTTS(&fakeTTS{}))
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	errc := startSession(t, ctx, s, &Agent{Instructions: "be nice"}, newFakeInput(), out)

	speech.final("hello world")
	speech.endOfSpeech()
	speech.finish()
	waitStart(t, errc)

	if out.frames() == 0 {
		t.Fatal("no audio frames reached the output")
	}

	// The chat context should record the system prompt, the user turn and the reply.
	msgs := s.ChatContext().Messages
	if len(msgs) != 3 {
		t.Fatalf("chat context has %d messages, want 3: %+v", len(msgs), msgs)
	}
	if msgs[1].Role != llm.RoleUser || msgs[1].Content != "hello world" {
		t.Fatalf("user message wrong: %+v", msgs[1])
	}
	if msgs[2].Role != llm.RoleAssistant || msgs[2].Content != "hi there" {
		t.Fatalf("assistant message wrong: %+v", msgs[2])
	}
}

// TestStartRequiresPlugins verifies Start refuses to run without the core plugins.
func TestStartRequiresPlugins(t *testing.T) {
	s := NewAgentSession(WithSTT(newFakeSTT()))
	err := s.Start(context.Background(), &Agent{}, newFakeInput(), &fakeOutput{})
	if err == nil {
		t.Fatal("expected an error when LLM and TTS are missing")
	}
}

// TestBargeIn checks the interruption path: while the agent
// is speaking, a VAD start-of-speech must cancel the in-flight generation, clear
// the audio already queued on the output, and mark the speech handle interrupted.
func TestBargeIn(t *testing.T) {
	speech := newFakeSTT()
	voiceDetect := newFakeVAD()
	model := newFakeLLM(textTurn("a long winded answer"))
	model.hold = make(chan struct{})
	out := &fakeOutput{}

	s := NewAgentSession(WithSTT(speech), WithLLM(model), WithTTS(&fakeTTS{}), WithVAD(voiceDetect))
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	errc := startSession(t, ctx, s, &Agent{}, newFakeInput(), out)

	speech.final("tell me a story")
	speech.endOfSpeech()

	// Wait until the agent is actually speaking before barging in.
	waitFor(t, func() bool { return out.frames() > 0 }, "agent to start speaking")
	handle := s.CurrentSpeech()
	if handle == nil {
		t.Fatal("no speech handle while the agent is speaking")
	}

	voiceDetect.startSpeech()

	waitFor(t, func() bool { return handle.Interrupted() }, "speech handle to be interrupted")
	waitFor(t, out.wasCleared, "output buffer to be cleared")

	// The interrupted generation must still finish and release the pipeline.
	select {
	case <-handle.Done():
	case <-time.After(3 * time.Second):
		t.Fatal("interrupted generation never finished")
	}

	close(model.hold)
	speech.finish()
	waitStart(t, errc)

	// What the agent managed to say before being cut off is kept in history.
	msgs := s.ChatContext().Messages
	last := msgs[len(msgs)-1]
	if last.Role != llm.RoleAssistant || last.Content == "" {
		t.Fatalf("truncated reply was not recorded: %+v", msgs)
	}
}

// TestToolCallLoop verifies the function-calling loop: the model asks for a
// tool, the session runs the handler, feeds the result back, and the second
// model turn becomes the spoken answer.
func TestToolCallLoop(t *testing.T) {
	speech := newFakeSTT()
	model := newFakeLLM(
		toolTurn(llm.ToolCall{ID: "call_1", Name: "get_time", Arguments: `{"tz":"UTC"}`}),
		textTurn("it is noon"),
	)
	out := &fakeOutput{}

	var gotArgs string
	agent := &Agent{
		Instructions: "be useful",
		Tools: []llm.Tool{{
			Name:        "get_time",
			Description: "current time",
			Handler: func(ctx context.Context, args string) (string, error) {
				gotArgs = args
				return "12:00", nil
			},
		}},
	}

	s := NewAgentSession(WithSTT(speech), WithLLM(model), WithTTS(&fakeTTS{}))
	events := collectEvents(s)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	errc := startSession(t, ctx, s, agent, newFakeInput(), out)

	speech.final("what time is it")
	speech.endOfSpeech()
	speech.finish()
	waitStart(t, errc)

	if gotArgs != `{"tz":"UTC"}` {
		t.Fatalf("handler got arguments %q", gotArgs)
	}
	if model.callCount() != 2 {
		t.Fatalf("LLM was called %d times, want 2 (tool round + answer)", model.callCount())
	}

	// The second model call must see the assistant tool call and its result.
	second := model.context(1)
	if second == nil || len(second.Messages) < 4 {
		t.Fatalf("second LLM call saw %+v", second)
	}
	toolMsg := second.Messages[len(second.Messages)-1]
	if toolMsg.Role != llm.RoleTool || toolMsg.Content != "12:00" || toolMsg.Name != "get_time" {
		t.Fatalf("tool result message wrong: %+v", toolMsg)
	}
	if toolMsg.ToolCallID != "call_1" {
		t.Fatalf("tool result lost the call id: %+v", toolMsg)
	}

	// The final answer is what gets spoken and stored.
	msgs := s.ChatContext().Messages
	last := msgs[len(msgs)-1]
	if last.Role != llm.RoleAssistant || last.Content != "it is noon" {
		t.Fatalf("final assistant message wrong: %+v", msgs)
	}
	got := events()
	if !hasEvent(got, EventToolCall) || !hasEvent(got, EventToolResult) {
		t.Fatalf("tool events were not emitted: %+v", got)
	}
}

// TestUnknownTool checks a tool call with no matching handler
// does not stall the turn: the error goes back to the model as the result.
func TestUnknownTool(t *testing.T) {
	speech := newFakeSTT()
	model := newFakeLLM(
		toolTurn(llm.ToolCall{ID: "call_1", Name: "missing", Arguments: "{}"}),
		textTurn("sorry"),
	)

	s := NewAgentSession(WithSTT(speech), WithLLM(model), WithTTS(&fakeTTS{}))
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	errc := startSession(t, ctx, s, &Agent{}, newFakeInput(), &fakeOutput{})

	speech.final("do the thing")
	speech.endOfSpeech()
	speech.finish()
	waitStart(t, errc)

	second := model.context(1)
	if second == nil {
		t.Fatal("model was not called a second time")
	}
	last := second.Messages[len(second.Messages)-1]
	if last.Role != llm.RoleTool || !strings.Contains(last.Content, "unknown tool") {
		t.Fatalf("unknown tool was not reported back: %+v", last)
	}
}

// TestNodeOverrides verifies an Agent can replace the LLM and TTS pipeline
// nodes, bypassing the configured plugins entirely.
func TestNodeOverrides(t *testing.T) {
	speech := newFakeSTT()
	model := newFakeLLM(textTurn("from the plugin"))
	out := &fakeOutput{}

	agent := &Agent{
		LLMNode: func(ctx context.Context, chatCtx *llm.ChatContext, tools []llm.Tool) (<-chan llm.ChatChunk, error) {
			ch := make(chan llm.ChatChunk, 1)
			ch <- llm.ChatChunk{Delta: llm.ChoiceDelta{Role: llm.RoleAssistant, Content: "from the node"}}
			close(ch)
			return ch, nil
		},
		TTSNode: func(ctx context.Context, text <-chan string) (<-chan tts.SynthesizedAudio, error) {
			ch := make(chan tts.SynthesizedAudio, 8)
			go func() {
				defer close(ch)
				for range text {
					ch <- tts.SynthesizedAudio{Frame: audio.NewFrame([]int16{7, 7}, 24000, 1)}
				}
			}()
			return ch, nil
		},
	}

	s := NewAgentSession(WithSTT(speech), WithLLM(model), WithTTS(&fakeTTS{}))
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	errc := startSession(t, ctx, s, agent, newFakeInput(), out)

	speech.final("hello")
	speech.endOfSpeech()
	speech.finish()
	waitStart(t, errc)

	if model.callCount() != 0 {
		t.Fatalf("the LLM plugin was used despite the node override")
	}
	if out.frames() == 0 {
		t.Fatal("the TTS node override produced no audio")
	}
	msgs := s.ChatContext().Messages
	if msgs[len(msgs)-1].Content != "from the node" {
		t.Fatalf("reply did not come from the node: %+v", msgs)
	}
}

// TestPreemptiveGeneration checks that inference starts on a final transcript
// but the audio stays gated until the turn is actually committed, and that the
// committed turn reuses the same generation instead of running a second one.
func TestPreemptiveGeneration(t *testing.T) {
	speech := newFakeSTT()
	model := newFakeLLM(textTurn("sure"))
	out := &fakeOutput{}

	s := NewAgentSession(
		WithSTT(speech), WithLLM(model), WithTTS(&fakeTTS{}),
		WithPreemptiveGeneration(true),
	)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	errc := startSession(t, ctx, s, &Agent{}, newFakeInput(), out)

	speech.final("can you help")

	// Inference runs ahead of the turn boundary...
	waitFor(t, func() bool { return model.callCount() == 1 }, "preemptive inference to start")
	// ...but nothing may be played before the turn is committed.
	if out.frames() != 0 {
		t.Fatal("preemptive audio was played before end of turn")
	}

	speech.endOfSpeech()
	waitFor(t, func() bool { return out.frames() > 0 }, "committed audio to play")

	speech.finish()
	waitStart(t, errc)

	if model.callCount() != 1 {
		t.Fatalf("LLM ran %d times, want 1 (the preemptive run should be reused)", model.callCount())
	}
	msgs := s.ChatContext().Messages
	if msgs[len(msgs)-1].Content != "sure" {
		t.Fatalf("reply was not committed: %+v", msgs)
	}
}

// TestPreemptiveGenerationDiscarded checks that a preemptive run is thrown away
// when the user keeps talking, and the committed turn covers the full sentence.
func TestPreemptiveGenerationDiscarded(t *testing.T) {
	speech := newFakeSTT()
	model := newFakeLLM(textTurn("first"), textTurn("second"))
	out := &fakeOutput{}

	s := NewAgentSession(
		WithSTT(speech), WithLLM(model), WithTTS(&fakeTTS{}),
		WithPreemptiveGeneration(true),
	)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	errc := startSession(t, ctx, s, &Agent{}, newFakeInput(), out)

	speech.final("can you help")
	waitFor(t, func() bool { return model.callCount() == 1 }, "first preemptive run")
	speech.final("me with this")
	waitFor(t, func() bool { return model.callCount() == 2 }, "second preemptive run")

	speech.endOfSpeech()
	waitFor(t, func() bool { return out.frames() > 0 }, "committed audio to play")
	speech.finish()
	waitStart(t, errc)

	// Only the final, whole-sentence turn is kept in history.
	msgs := s.ChatContext().Messages
	if len(msgs) != 2 {
		t.Fatalf("discarded generation leaked into history: %+v", msgs)
	}
	if msgs[0].Role != llm.RoleUser || msgs[0].Content != "can you help me with this" {
		t.Fatalf("user turn wrong: %+v", msgs[0])
	}
	if msgs[1].Content != "second" {
		t.Fatalf("wrong reply committed: %+v", msgs[1])
	}
}

// TestTurnDetectorDefersTurn verifies a turn detector that says "not done yet"
// holds the turn open until the endpointing delay expires.
func TestTurnDetectorDefersTurn(t *testing.T) {
	speech := newFakeSTT()
	model := newFakeLLM(textTurn("go on"))

	s := NewAgentSession(
		WithSTT(speech), WithLLM(model), WithTTS(&fakeTTS{}),
		WithTurnDetector(&fakeTurnDetector{score: 0.1}, 0.5),
		WithMaxEndpointingDelay(200*time.Millisecond),
	)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	errc := startSession(t, ctx, s, &Agent{}, newFakeInput(), &fakeOutput{})

	speech.final("i was thinking")
	speech.endOfSpeech()

	// The detector said the user is not finished, so nothing runs yet.
	time.Sleep(60 * time.Millisecond)
	if model.callCount() != 0 {
		t.Fatal("generation started despite the turn detector deferring")
	}

	// Once the endpointing delay expires the turn commits anyway.
	waitFor(t, func() bool { return model.callCount() == 1 }, "deferred turn to commit")

	speech.finish()
	waitStart(t, errc)
}

// TestTurnDetectorCommitsImmediately verifies a confident detector lets the
// turn through without waiting for the endpointing delay.
func TestTurnDetectorCommitsImmediately(t *testing.T) {
	speech := newFakeSTT()
	model := newFakeLLM(textTurn("ok"))

	s := NewAgentSession(
		WithSTT(speech), WithLLM(model), WithTTS(&fakeTTS{}),
		WithTurnDetector(&fakeTurnDetector{score: 0.9}, 0.5),
		WithMaxEndpointingDelay(10*time.Second),
	)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	errc := startSession(t, ctx, s, &Agent{}, newFakeInput(), &fakeOutput{})

	speech.final("that is all")
	speech.endOfSpeech()
	waitFor(t, func() bool { return model.callCount() == 1 }, "turn to commit immediately")

	speech.finish()
	waitStart(t, errc)
}

// TestSay checks the imperative speech path: text goes straight to TTS, the
// returned handle completes, and the utterance is recorded in history.
func TestSay(t *testing.T) {
	speech := newFakeSTT()
	out := &fakeOutput{}

	s := NewAgentSession(WithSTT(speech), WithLLM(newFakeLLM()), WithTTS(&fakeTTS{}))
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	errc := startSession(t, ctx, s, &Agent{}, newFakeInput(), out)

	handle, err := s.Say(ctx, "welcome aboard")
	if err != nil {
		t.Fatalf("Say: %v", err)
	}
	handle.Wait()

	if out.frames() == 0 {
		t.Fatal("Say produced no audio")
	}
	msgs := s.ChatContext().Messages
	if len(msgs) != 1 || msgs[0].Role != llm.RoleAssistant || msgs[0].Content != "welcome aboard" {
		t.Fatalf("Say did not record the utterance: %+v", msgs)
	}

	speech.finish()
	waitStart(t, errc)
}

// TestVADEndpointingCommitsTurn covers the case where the STT never reports
// end-of-speech — which happens whenever the user's audio goes quiet, e.g. they
// mute mid-call. With a VAD present, a final transcript plus VAD silence must
// still close the turn instead of hanging forever.
func TestVADEndpointingCommitsTurn(t *testing.T) {
	speech := newFakeSTT()
	voiceDetect := newFakeVAD()
	model := newFakeLLM(textTurn("go ahead"))
	out := &fakeOutput{}

	s := NewAgentSession(
		WithSTT(speech), WithLLM(model), WithTTS(&fakeTTS{}), WithVAD(voiceDetect),
		WithMinEndpointingDelay(50*time.Millisecond),
	)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	errc := startSession(t, ctx, s, &Agent{}, newFakeInput(), out)

	// The user speaks and stops; the VAD notices the silence.
	voiceDetect.startSpeech()
	voiceDetect.stopSpeech()
	// The transcript lands afterwards, and no stt.EndOfSpeech ever arrives.
	speech.final("are you there")

	waitFor(t, func() bool { return model.callCount() == 1 }, "VAD endpointing to close the turn")
	waitFor(t, func() bool { return out.frames() > 0 }, "the agent to answer")

	speech.finish()
	waitStart(t, errc)

	msgs := s.ChatContext().Messages
	if len(msgs) != 2 || msgs[0].Content != "are you there" {
		t.Fatalf("turn was not committed correctly: %+v", msgs)
	}
}

// TestVADEndpointingWaitsWhileSpeaking verifies the endpointing timer does not
// fire while the VAD still hears the user, so a mid-sentence pause does not cut
// them off.
func TestVADEndpointingWaitsWhileSpeaking(t *testing.T) {
	speech := newFakeSTT()
	voiceDetect := newFakeVAD()
	model := newFakeLLM(textTurn("ok"))

	s := NewAgentSession(
		WithSTT(speech), WithLLM(model), WithTTS(&fakeTTS{}), WithVAD(voiceDetect),
		WithMinEndpointingDelay(50*time.Millisecond),
	)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	errc := startSession(t, ctx, s, &Agent{}, newFakeInput(), &fakeOutput{})

	voiceDetect.startSpeech()
	speech.final("i was going to say")
	time.Sleep(200 * time.Millisecond)
	if model.callCount() != 0 {
		t.Fatal("turn committed while the user was still speaking")
	}

	// Once they stop, the turn closes.
	voiceDetect.stopSpeech()
	waitFor(t, func() bool { return model.callCount() == 1 }, "turn to close after the user stops")

	speech.finish()
	waitStart(t, errc)
}
