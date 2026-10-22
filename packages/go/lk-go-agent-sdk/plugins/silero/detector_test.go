package silero

import (
	"testing"

	"github.com/beingPro007/lk-go-agent-sdk/vad"
)

// newDetector uses 16kHz with a 250ms min-speech (4000 samples) and 100ms
// min-silence (1600 samples) hangover — the plugin defaults.
func newDetector() *detector {
	return &detector{
		threshold:         0.5,
		sampleRate:        16000,
		minSpeechSamples:  4000,
		minSilenceSamples: 1600,
	}
}

func countType(evs []vad.Event, typ vad.EventType) int {
	n := 0
	for _, e := range evs {
		if e.Type == typ {
			n++
		}
	}
	return n
}

// Sustained high probability crosses min-speech and fires exactly one
// StartOfSpeech; sustained low probability afterwards crosses min-silence and
// fires exactly one EndOfSpeech.
func TestDetectorSpeechThenSilence(t *testing.T) {
	d := newDetector()
	var all []vad.Event

	// 10 windows of 512 samples = 5120 > 4000 min-speech -> should trigger.
	for i := 0; i < 10; i++ {
		all = append(all, d.step(0.9, 512)...)
	}
	if countType(all, vad.StartOfSpeech) != 1 {
		t.Fatalf("StartOfSpeech count = %d, want 1", countType(all, vad.StartOfSpeech))
	}
	if !d.speaking {
		t.Fatal("expected speaking after sustained high probability")
	}

	// 6 silent windows = 3072 > 1600 min-silence -> should end the utterance.
	before := len(all)
	for i := 0; i < 6; i++ {
		all = append(all, d.step(0.1, 512)...)
	}
	end := all[before:]
	if countType(end, vad.EndOfSpeech) != 1 {
		t.Fatalf("EndOfSpeech count = %d, want 1", countType(end, vad.EndOfSpeech))
	}
	if d.speaking {
		t.Fatal("expected not speaking after sustained silence")
	}
}

// Every window emits exactly one InferenceDone event regardless of speech state.
func TestDetectorInferenceEveryStep(t *testing.T) {
	d := newDetector()
	evs := d.step(0.3, 512)
	if countType(evs, vad.InferenceDone) != 1 {
		t.Fatalf("expected one InferenceDone per step, got %d", countType(evs, vad.InferenceDone))
	}
}

// A single high window (below the min-speech threshold) followed by silence must
// not trigger StartOfSpeech — debouncing brief blips.
func TestDetectorShortSpeechNoTrigger(t *testing.T) {
	d := newDetector()
	var all []vad.Event
	all = append(all, d.step(0.9, 512)...)
	all = append(all, d.step(0.1, 512)...)
	if countType(all, vad.StartOfSpeech) != 0 {
		t.Fatal("brief speech under minSpeech should not trigger StartOfSpeech")
	}
}

// reset() mid-utterance (e.g. Flush/EndInput) must close the turn with an
// EndOfSpeech and clear the speaking state.
func TestDetectorResetWhileSpeaking(t *testing.T) {
	d := newDetector()
	for i := 0; i < 10; i++ {
		d.step(0.9, 512)
	}
	evs := d.reset()
	if countType(evs, vad.EndOfSpeech) != 1 {
		t.Fatal("reset while speaking should emit EndOfSpeech")
	}
	if d.speaking {
		t.Fatal("reset should clear speaking")
	}
}

// With hysteresis, a dip between the two lines does not end the speech; a
// drop below the lower line does.
func TestDetectorHysteresisKeepsSpeechThroughDips(t *testing.T) {
	d := newDetector()
	d.negThreshold = 0.35
	for i := 0; i < 10; i++ {
		d.step(0.9, 512)
	}
	var mid []vad.Event
	for i := 0; i < 20; i++ { // 640 ms at 0.4: below 0.5, above 0.35
		mid = append(mid, d.step(0.4, 512)...)
	}
	if countType(mid, vad.EndOfSpeech) != 0 || !d.speaking {
		t.Fatal("a dip above the lower line ended the speech")
	}
	var end []vad.Event
	for i := 0; i < 6; i++ {
		end = append(end, d.step(0.2, 512)...)
	}
	if countType(end, vad.EndOfSpeech) != 1 {
		t.Fatal("a drop below the lower line did not end the speech")
	}
}

// Without a lower line, the old single-threshold behavior stays.
func TestDetectorNoHysteresisByDefault(t *testing.T) {
	d := newDetector()
	for i := 0; i < 10; i++ {
		d.step(0.9, 512)
	}
	var evs []vad.Event
	for i := 0; i < 6; i++ {
		evs = append(evs, d.step(0.4, 512)...)
	}
	if countType(evs, vad.EndOfSpeech) != 1 {
		t.Fatal("with no negative threshold, 0.4 must count as silence")
	}
}
