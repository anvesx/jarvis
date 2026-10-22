package silero

import (
	"time"

	"github.com/beingPro007/lk-go-agent-sdk/vad"
)

type detector struct {
	threshold         float32
	negThreshold      float32 // speech ends only below this line; 0 means the same as threshold
	sampleRate        int
	minSpeechSamples  int
	minSilenceSamples int

	speaking         bool
	speechSamples    int
	silenceSamples   int
	utteranceSamples int
}

func (d *detector) samples(n int) time.Duration {
	return time.Duration(n) * time.Second / time.Duration(d.sampleRate)
}

// lower is the line that ends speech. With hysteresis (negThreshold below
// threshold), a dip between words does not count as silence.
func (d *detector) lower() float32 {
	if d.negThreshold <= 0 || d.negThreshold > d.threshold {
		return d.threshold
	}
	return d.negThreshold
}

func (d *detector) step(prob float32, n int) []vad.Event {
	evs := []vad.Event{{Type: vad.InferenceDone, Probability: float64(prob), Speaking: d.speaking}}

	if !d.speaking {
		if prob < d.threshold {
			d.speechSamples = 0
			return evs
		}
		d.silenceSamples = 0
		d.speechSamples += n
		if d.speechSamples >= d.minSpeechSamples {
			d.speaking = true
			d.utteranceSamples = d.speechSamples
			evs = append(evs, vad.Event{Type: vad.StartOfSpeech, Probability: float64(prob), Speaking: true})
		}
		return evs
	}

	d.utteranceSamples += n
	if prob >= d.lower() {
		d.silenceSamples = 0
		return evs
	}
	d.silenceSamples += n
	if d.silenceSamples >= d.minSilenceSamples {
		evs = append(evs, vad.Event{
			Type:            vad.EndOfSpeech,
			Probability:     float64(prob),
			Speaking:        false,
			SpeechDuration:  d.samples(d.utteranceSamples),
			SilenceDuration: d.samples(d.silenceSamples),
		})
		d.speaking = false
		d.speechSamples = 0
		d.utteranceSamples = 0
		d.silenceSamples = 0
	}
	return evs
}

func (d *detector) reset() []vad.Event {
	var evs []vad.Event
	if d.speaking {
		evs = append(evs, vad.Event{
			Type:           vad.EndOfSpeech,
			Speaking:       false,
			SpeechDuration: d.samples(d.utteranceSamples),
		})
	}
	d.speaking = false
	d.speechSamples = 0
	d.utteranceSamples = 0
	d.silenceSamples = 0
	return evs
}
