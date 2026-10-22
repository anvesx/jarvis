package cartesia

import (
	"encoding/base64"
	"encoding/json"
	"testing"

	"github.com/beingPro007/lk-go-agent-sdk/audio"
	"github.com/beingPro007/lk-go-agent-sdk/tts"
)

// newTestStream builds a socket-less stream so we can exercise handle() and
// message() without a live Cartesia connection.
func newTestStream() *synthStream {
	return &synthStream{
		tts:   &TTS{sampleRate: 24000, channels: 1},
		audio: make(chan tts.SynthesizedAudio, 8),
		done:  make(chan struct{}),
	}
}

// A base64 "chunk" message must decode back into a PCM audio.Frame with the
// configured sample rate — the reverse of Deepgram's encode path.
func TestHandleChunkDecodesPCM(t *testing.T) {
	st := newTestStream()
	src := audio.NewFrame([]int16{1, 2, 3}, 24000, 1)
	b64 := base64.StdEncoding.EncodeToString(src.Bytes())
	st.handle([]byte(`{"type":"chunk","data":"` + b64 + `","context_id":"x"}`))
	got := <-st.audio
	if len(got.Frame.Data) != 3 || got.Frame.Data[0] != 1 || got.Frame.Data[2] != 3 {
		t.Fatalf("decoded wrong: %v", got.Frame.Data)
	}
	if got.Frame.SampleRate != 24000 {
		t.Fatalf("sample rate = %d", got.Frame.SampleRate)
	}
}

// A "done" message marks the end of a synthesized segment via Final=true.
func TestHandleDoneMarksFinal(t *testing.T) {
	st := newTestStream()
	st.handle([]byte(`{"type":"done","context_id":"x"}`))
	got := <-st.audio
	if !got.Final {
		t.Fatal("expected Final=true on done")
	}
}

// Outgoing messages carry the streaming continue flag, transcript, and context id.
func TestMessageContinueFlag(t *testing.T) {
	tt := &TTS{model: "sonic-2", voice: "v", language: "en", sampleRate: 24000}
	var m map[string]any
	if err := json.Unmarshal(tt.message("ctx-1", "hello", true), &m); err != nil {
		t.Fatal(err)
	}
	if m["continue"] != true {
		t.Fatalf("continue = %v, want true", m["continue"])
	}
	if m["transcript"] != "hello" || m["context_id"] != "ctx-1" {
		t.Fatalf("message wrong: %+v", m)
	}
}

// The context id stays fixed within an utterance and rotates after a flush, so
// each utterance streams under its own Cartesia context.
func TestContextRotation(t *testing.T) {
	st := newTestStream()
	first := st.currentCtx()
	if st.currentCtx() != first {
		t.Fatal("ctx should be stable within an utterance")
	}
	st.mu.Lock()
	st.ctxID = ""
	st.mu.Unlock()
	if second := st.currentCtx(); second == first {
		t.Fatalf("ctx should rotate after flush: %q == %q", second, first)
	}
}
