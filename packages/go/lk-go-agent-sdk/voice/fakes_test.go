package voice

import (
	"context"
	"sync"

	"github.com/beingPro007/lk-go-agent-sdk/audio"
	"github.com/beingPro007/lk-go-agent-sdk/llm"
	"github.com/beingPro007/lk-go-agent-sdk/stt"
	"github.com/beingPro007/lk-go-agent-sdk/tts"
	"github.com/beingPro007/lk-go-agent-sdk/vad"
)

// fakeSTT hands out streams over a channel the test owns, so a test can feed
// speech events at exactly the moment it wants them.
type fakeSTT struct{ events chan stt.SpeechEvent }

func newFakeSTT() *fakeSTT { return &fakeSTT{events: make(chan stt.SpeechEvent, 16)} }

func (f *fakeSTT) SampleRate() int  { return 16000 }
func (f *fakeSTT) NumChannels() int { return 1 }
func (f *fakeSTT) Stream(context.Context) stt.SpeechStream {
	return &fakeSTTStream{events: f.events}
}

func (f *fakeSTT) final(text string) {
	f.events <- stt.SpeechEvent{Type: stt.FinalTranscript, Alternatives: []stt.SpeechData{{Text: text}}}
}
func (f *fakeSTT) endOfSpeech() { f.events <- stt.SpeechEvent{Type: stt.EndOfSpeech} }
func (f *fakeSTT) finish()      { close(f.events) }

// The stream never closes the shared channel; only the test does.
type fakeSTTStream struct{ events chan stt.SpeechEvent }

func (s *fakeSTTStream) PushFrame(audio.Frame)        {}
func (s *fakeSTTStream) Flush()                       {}
func (s *fakeSTTStream) EndInput()                    {}
func (s *fakeSTTStream) Chan() <-chan stt.SpeechEvent { return s.events }
func (s *fakeSTTStream) Err() error                   { return nil }
func (s *fakeSTTStream) Close() error                 { return nil }

// fakeLLM replays one scripted list of chunks per Chat call and records the
// chat context it was given, so tests can assert what the model actually saw.
// When hold is set the stream stays open after its chunks until released,
// which keeps a generation in flight for interruption tests.
type fakeLLM struct {
	turns [][]llm.ChatChunk
	hold  chan struct{}
	// holdFirst, when set, keeps the first Chat call from sending anything
	// until it is closed, so a test can interrupt a reply before any audio.
	holdFirst chan struct{}

	mu      sync.Mutex
	calls   int
	seen    []*llm.ChatContext
	tools   []llm.Tool
	started chan struct{}
	once    sync.Once
}

func newFakeLLM(turns ...[]llm.ChatChunk) *fakeLLM {
	return &fakeLLM{turns: turns, started: make(chan struct{})}
}

func textTurn(s string) []llm.ChatChunk {
	return []llm.ChatChunk{{Delta: llm.ChoiceDelta{Role: llm.RoleAssistant, Content: s}}}
}

func toolTurn(calls ...llm.ToolCall) []llm.ChatChunk {
	return []llm.ChatChunk{{Delta: llm.ChoiceDelta{Role: llm.RoleAssistant, ToolCalls: calls}}}
}

func (f *fakeLLM) Chat(ctx context.Context, chatCtx *llm.ChatContext, tools []llm.Tool) llm.LLMStream {
	f.mu.Lock()
	i := f.calls
	f.calls++
	f.seen = append(f.seen, chatCtx.Clone())
	f.tools = tools
	var chunks []llm.ChatChunk
	if i < len(f.turns) {
		chunks = f.turns[i]
	}
	f.mu.Unlock()

	ch := make(chan llm.ChatChunk, len(chunks)+1)
	go func() {
		defer close(ch)
		if i == 0 && f.holdFirst != nil {
			select {
			case <-f.holdFirst:
			case <-ctx.Done():
				return
			}
		}
		for _, c := range chunks {
			select {
			case ch <- c:
			case <-ctx.Done():
				return
			}
		}
		f.once.Do(func() { close(f.started) })
		if f.hold != nil {
			select {
			case <-f.hold:
			case <-ctx.Done():
			}
		}
	}()
	return &fakeLLMStream{chunks: ch}
}

func (f *fakeLLM) callCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.calls
}

func (f *fakeLLM) context(i int) *llm.ChatContext {
	f.mu.Lock()
	defer f.mu.Unlock()
	if i >= len(f.seen) {
		return nil
	}
	return f.seen[i]
}

type fakeLLMStream struct{ chunks chan llm.ChatChunk }

func (s *fakeLLMStream) Chan() <-chan llm.ChatChunk { return s.chunks }
func (s *fakeLLMStream) Err() error                 { return nil }
func (s *fakeLLMStream) Close() error               { return nil }

// fakeTTS turns each pushed text token into one audio frame.
type fakeTTS struct{}

func (f *fakeTTS) SampleRate() int  { return 24000 }
func (f *fakeTTS) NumChannels() int { return 1 }
func (f *fakeTTS) Stream(context.Context) tts.SynthesizeStream {
	return &fakeTTSStream{audio: make(chan tts.SynthesizedAudio, 32)}
}

type fakeTTSStream struct {
	audio chan tts.SynthesizedAudio
	once  sync.Once
}

func (s *fakeTTSStream) PushText(string) {
	s.audio <- tts.SynthesizedAudio{Frame: audio.NewFrame([]int16{1, 2, 3, 4}, 24000, 1)}
}
func (s *fakeTTSStream) Flush()                            {}
func (s *fakeTTSStream) EndInput()                         { s.once.Do(func() { close(s.audio) }) }
func (s *fakeTTSStream) Chan() <-chan tts.SynthesizedAudio { return s.audio }
func (s *fakeTTSStream) Err() error                        { return nil }
func (s *fakeTTSStream) Close() error                      { s.once.Do(func() { close(s.audio) }); return nil }

// fakeVAD lets a test fire speech events on demand, which is how barge-in is
// triggered in the pipeline.
type fakeVAD struct{ events chan vad.Event }

func newFakeVAD() *fakeVAD { return &fakeVAD{events: make(chan vad.Event, 8)} }

func (f *fakeVAD) SampleRate() int                   { return 16000 }
func (f *fakeVAD) Stream(context.Context) vad.Stream { return &fakeVADStream{events: f.events} }
func (f *fakeVAD) startSpeech() {
	f.events <- vad.Event{Type: vad.StartOfSpeech, Speaking: true}
}
func (f *fakeVAD) stopSpeech() { f.events <- vad.Event{Type: vad.EndOfSpeech} }

// infer sends one inference result, the way Silero reports every window.
func (f *fakeVAD) infer(p float64) {
	f.events <- vad.Event{Type: vad.InferenceDone, Probability: p, Speaking: true}
}

type fakeVADStream struct{ events chan vad.Event }

func (s *fakeVADStream) PushFrame(audio.Frame) {}
func (s *fakeVADStream) Flush()                {}
func (s *fakeVADStream) EndInput()             {}
func (s *fakeVADStream) Chan() <-chan vad.Event {
	return s.events
}
func (s *fakeVADStream) Err() error   { return nil }
func (s *fakeVADStream) Close() error { return nil }

// fakeTurnDetector reports a fixed end-of-turn probability.
type fakeTurnDetector struct{ score float64 }

func (d *fakeTurnDetector) PredictEndOfTurn(context.Context, *llm.ChatContext) (float64, error) {
	return d.score, nil
}

// --- audio input / output doubles ---

type fakeInput struct{ frames chan audio.Frame }

func newFakeInput() *fakeInput { return &fakeInput{frames: make(chan audio.Frame, 8)} }

func (f *fakeInput) Frames() <-chan audio.Frame { return f.frames }

type fakeOutput struct {
	mu       sync.Mutex
	captured int
	cleared  bool
}

func (o *fakeOutput) CaptureFrame(audio.Frame) error {
	o.mu.Lock()
	o.captured++
	o.mu.Unlock()
	return nil
}
func (o *fakeOutput) ClearBuffer()    { o.mu.Lock(); o.cleared = true; o.mu.Unlock() }
func (o *fakeOutput) WaitForPlayout() {}

func (o *fakeOutput) frames() int {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.captured
}

func (o *fakeOutput) wasCleared() bool {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.cleared
}
