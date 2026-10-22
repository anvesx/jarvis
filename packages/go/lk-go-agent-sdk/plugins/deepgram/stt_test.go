package deepgram

import (
	"testing"

	"github.com/beingPro007/lk-go-agent-sdk/stt"
)

// newTestStream builds a stream with no socket so we can exercise handle()
// (the JSON->SpeechEvent parser) in isolation, without a live Deepgram connection.
func newTestStream() *speechStream {
	return &speechStream{
		events: make(chan stt.SpeechEvent, 8),
		done:   make(chan struct{}),
	}
}

// A Deepgram Results message with is_final=true maps to a FinalTranscript event
// carrying the transcript and confidence.
func TestHandleFinalTranscript(t *testing.T) {
	st := newTestStream()
	st.handle([]byte(`{"type":"Results","is_final":true,"channel":{"alternatives":[{"transcript":"hello world","confidence":0.98}]}}`))
	ev := <-st.events
	if ev.Type != stt.FinalTranscript {
		t.Fatalf("type = %v, want FinalTranscript", ev.Type)
	}
	if len(ev.Alternatives) != 1 || ev.Alternatives[0].Text != "hello world" {
		t.Fatalf("alternatives wrong: %+v", ev.Alternatives)
	}
}

// A non-final Results message maps to an InterimTranscript event.
func TestHandleInterim(t *testing.T) {
	st := newTestStream()
	st.handle([]byte(`{"type":"Results","is_final":false,"channel":{"alternatives":[{"transcript":"hel"}]}}`))
	ev := <-st.events
	if ev.Type != stt.InterimTranscript {
		t.Fatalf("type = %v, want InterimTranscript", ev.Type)
	}
}

// Deepgram's vad_events SpeechStarted maps to a StartOfSpeech event.
func TestHandleSpeechStarted(t *testing.T) {
	st := newTestStream()
	st.handle([]byte(`{"type":"SpeechStarted"}`))
	ev := <-st.events
	if ev.Type != stt.StartOfSpeech {
		t.Fatalf("type = %v, want StartOfSpeech", ev.Type)
	}
}

// speech_final=true is an endpoint: it emits the FinalTranscript, then a separate
// EndOfSpeech event so the pipeline knows the turn ended.
func TestHandleSpeechFinalEmitsEndOfSpeech(t *testing.T) {
	st := newTestStream()
	st.handle([]byte(`{"type":"Results","is_final":true,"speech_final":true,"channel":{"alternatives":[{"transcript":"done"}]}}`))
	if ev := <-st.events; ev.Type != stt.FinalTranscript {
		t.Fatalf("first event = %v, want FinalTranscript", ev.Type)
	}
	if ev := <-st.events; ev.Type != stt.EndOfSpeech {
		t.Fatalf("second event = %v, want EndOfSpeech", ev.Type)
	}
}

// Empty transcripts (Deepgram sends these between utterances) must not emit events.
func TestHandleEmptyTranscriptIgnored(t *testing.T) {
	st := newTestStream()
	st.handle([]byte(`{"type":"Results","is_final":true,"channel":{"alternatives":[{"transcript":""}]}}`))
	select {
	case ev := <-st.events:
		t.Fatalf("expected no event, got %+v", ev)
	default:
	}
}
