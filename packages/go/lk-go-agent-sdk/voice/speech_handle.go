package voice

import (
	"context"
	"strconv"
	"sync"
	"sync/atomic"
)

var speechSeq atomic.Uint64

type SpeechHandle struct {
	id     string
	cancel context.CancelFunc

	done      chan struct{}
	committed chan struct{}

	doneOnce   sync.Once
	commitOnce sync.Once

	mu          sync.Mutex
	interrupted bool
	text        string

	played atomic.Bool // some audio of this speech reached the output
}

// Played reports whether any audio of this speech reached the output, that is
// whether the user can have heard some of it.
func (h *SpeechHandle) Played() bool { return h.played.Load() }

func newSpeechHandle(cancel context.CancelFunc, committed bool) *SpeechHandle {
	h := &SpeechHandle{
		id:        "speech_" + strconv.FormatUint(speechSeq.Add(1), 10),
		cancel:    cancel,
		done:      make(chan struct{}),
		committed: make(chan struct{}),
	}
	if committed {
		h.Commit()
	}
	return h
}

func (h *SpeechHandle) ID() string { return h.id }

func (h *SpeechHandle) Interrupt() {
	h.mu.Lock()
	h.interrupted = true
	h.mu.Unlock()
	h.cancel()
}

func (h *SpeechHandle) Interrupted() bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.interrupted
}

func (h *SpeechHandle) Commit() { h.commitOnce.Do(func() { close(h.committed) }) }

func (h *SpeechHandle) Committed() bool {
	select {
	case <-h.committed:
		return true
	default:
		return false
	}
}

func (h *SpeechHandle) Done() <-chan struct{} { return h.done }

func (h *SpeechHandle) Wait() { <-h.done }

func (h *SpeechHandle) Text() string {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.text
}

func (h *SpeechHandle) setText(s string) {
	h.mu.Lock()
	h.text = s
	h.mu.Unlock()
}

func (h *SpeechHandle) finish() { h.doneOnce.Do(func() { close(h.done) }) }
