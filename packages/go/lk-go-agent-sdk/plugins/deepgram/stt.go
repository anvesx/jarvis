package deepgram

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"sync"
	"time"

	"github.com/beingPro007/lk-go-agent-sdk/audio"
	"github.com/beingPro007/lk-go-agent-sdk/stt"
	"github.com/gorilla/websocket"
)

const (
	baseURL           = "wss://api.deepgram.com/v1/listen"
	defaultModel      = "nova-3"
	defaultLanguage   = "en"
	defaultSampleRate = 16000
	defaultChannels   = 1
	keepAliveInterval = 8 * time.Second
)

type STT struct {
	apiKey     string
	model      string
	language   string
	sampleRate int
	channels   int
}

type Option func(*STT)

func WithAPIKey(k string) Option   { return func(s *STT) { s.apiKey = k } }
func WithModel(m string) Option    { return func(s *STT) { s.model = m } }
func WithLanguage(l string) Option { return func(s *STT) { s.language = l } }
func WithSampleRate(r int) Option  { return func(s *STT) { s.sampleRate = r } }
func WithChannels(c int) Option    { return func(s *STT) { s.channels = c } }

func New(opts ...Option) (*STT, error) {
	s := &STT{
		apiKey:     os.Getenv("DEEPGRAM_API_KEY"),
		model:      defaultModel,
		language:   defaultLanguage,
		sampleRate: defaultSampleRate,
		channels:   defaultChannels,
	}
	for _, o := range opts {
		o(s)
	}
	if s.apiKey == "" {
		return nil, errors.New("deepgram: API key required (set DEEPGRAM_API_KEY or WithAPIKey)")
	}
	return s, nil
}

func (s *STT) SampleRate() int { return s.sampleRate }

func (s *STT) NumChannels() int { return s.channels }

func (s *STT) Stream(ctx context.Context) stt.SpeechStream {
	st := &speechStream{
		events: make(chan stt.SpeechEvent, 32),
		done:   make(chan struct{}),
	}

	header := http.Header{"Authorization": {"Token " + s.apiKey}}
	conn, _, err := websocket.DefaultDialer.DialContext(ctx, s.wsURL(), header)
	if err != nil {
		st.setErr(err)
		close(st.events)
		return st
	}
	st.conn = conn

	go st.readLoop(ctx)
	go st.keepAlive(ctx)
	return st
}

func (s *STT) wsURL() string {
	q := url.Values{}
	q.Set("model", s.model)
	q.Set("language", s.language)
	q.Set("encoding", "linear16")
	q.Set("sample_rate", strconv.Itoa(s.sampleRate))
	q.Set("channels", strconv.Itoa(s.channels))
	q.Set("interim_results", "true")
	q.Set("smart_format", "true")
	q.Set("vad_events", "true")
	q.Set("endpointing", "300")
	q.Set("utterance_end_ms", "1000")
	return baseURL + "?" + q.Encode()
}

type speechStream struct {
	conn   *websocket.Conn
	events chan stt.SpeechEvent
	done   chan struct{}

	mu     sync.Mutex
	sendMu sync.Mutex
	err    error
	closed bool
}

func (st *speechStream) PushFrame(f audio.Frame) {
	st.send(websocket.BinaryMessage, f.Bytes())
}

func (st *speechStream) Flush() {
	st.send(websocket.TextMessage, []byte(`{"type":"Finalize"}`))
}

func (st *speechStream) EndInput() {
	st.send(websocket.TextMessage, []byte(`{"type":"CloseStream"}`))
}

func (st *speechStream) Chan() <-chan stt.SpeechEvent { return st.events }

func (st *speechStream) Err() error {
	st.mu.Lock()
	defer st.mu.Unlock()
	return st.err
}

func (st *speechStream) Close() error {
	st.mu.Lock()
	if st.closed {
		st.mu.Unlock()
		return nil
	}
	st.closed = true
	conn := st.conn
	st.mu.Unlock()
	close(st.done)
	if conn != nil {
		return conn.Close()
	}
	return nil
}

func (st *speechStream) readLoop(ctx context.Context) {
	defer close(st.events)
	go func() {
		select {
		case <-ctx.Done():
			st.Close()
		case <-st.done:
		}
	}()
	for {
		_, data, err := st.conn.ReadMessage()
		if err != nil {
			if ctx.Err() == nil && !st.isClosed() {
				st.setErr(err)
			}
			return
		}
		st.handle(data)
	}
}

func (st *speechStream) keepAlive(ctx context.Context) {
	t := time.NewTicker(keepAliveInterval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-st.done:
			return
		case <-t.C:
			st.send(websocket.TextMessage, []byte(`{"type":"KeepAlive"}`))
		}
	}
}

type dgResponse struct {
	Type        string `json:"type"`
	IsFinal     bool   `json:"is_final"`
	SpeechFinal bool   `json:"speech_final"`
	Channel     struct {
		Alternatives []struct {
			Transcript string  `json:"transcript"`
			Confidence float64 `json:"confidence"`
		} `json:"alternatives"`
	} `json:"channel"`
}

func (st *speechStream) handle(data []byte) {
	var r dgResponse
	if err := json.Unmarshal(data, &r); err != nil {
		return
	}
	switch r.Type {
	case "SpeechStarted":
		st.emit(stt.SpeechEvent{Type: stt.StartOfSpeech})
	case "UtteranceEnd":
		st.emit(stt.SpeechEvent{Type: stt.EndOfSpeech})
	case "Results":
		if len(r.Channel.Alternatives) == 0 {
			return
		}
		alt := r.Channel.Alternatives[0]
		if alt.Transcript == "" {
			return
		}
		typ := stt.InterimTranscript
		if r.IsFinal {
			typ = stt.FinalTranscript
		}
		st.emit(stt.SpeechEvent{
			Type:         typ,
			Alternatives: []stt.SpeechData{{Text: alt.Transcript, Confidence: alt.Confidence}},
		})
		if r.SpeechFinal {
			st.emit(stt.SpeechEvent{Type: stt.EndOfSpeech})
		}
	}
}

func (st *speechStream) emit(ev stt.SpeechEvent) {
	select {
	case st.events <- ev:
	case <-st.done:
	}
}

func (st *speechStream) send(mt int, data []byte) {
	st.mu.Lock()
	conn := st.conn
	closed := st.closed
	st.mu.Unlock()
	if conn == nil || closed {
		return
	}
	st.sendMu.Lock()
	defer st.sendMu.Unlock()
	_ = conn.WriteMessage(mt, data)
}

func (st *speechStream) setErr(err error) {
	st.mu.Lock()
	if st.err == nil {
		st.err = err
	}
	st.mu.Unlock()
}

func (st *speechStream) isClosed() bool {
	st.mu.Lock()
	defer st.mu.Unlock()
	return st.closed
}
