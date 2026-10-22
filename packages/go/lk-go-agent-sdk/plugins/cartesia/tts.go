package cartesia

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"os"
	"sync"

	"github.com/beingPro007/lk-go-agent-sdk/audio"
	"github.com/beingPro007/lk-go-agent-sdk/tts"
	"github.com/gorilla/websocket"
)

const (
	baseURL           = "wss://api.cartesia.ai/tts/websocket"
	defaultModel      = "sonic-2"
	defaultVoice      = "a0e99841-438c-4a64-b679-ae501e7d6091"
	defaultVersion    = "2024-11-13"
	defaultLanguage   = "en"
	defaultSampleRate = 24000
	defaultChannels   = 1
)

type TTS struct {
	apiKey     string
	model      string
	voice      string
	language   string
	version    string
	sampleRate int
	channels   int
}

type Option func(*TTS)

func WithAPIKey(k string) Option   { return func(t *TTS) { t.apiKey = k } }
func WithModel(m string) Option    { return func(t *TTS) { t.model = m } }
func WithVoice(v string) Option    { return func(t *TTS) { t.voice = v } }
func WithLanguage(l string) Option { return func(t *TTS) { t.language = l } }
func WithSampleRate(r int) Option  { return func(t *TTS) { t.sampleRate = r } }

func New(opts ...Option) (*TTS, error) {
	voice := os.Getenv("CARTESIA_VOICE")
	if voice == "" {
		voice = defaultVoice
	}
	t := &TTS{
		apiKey:     os.Getenv("CARTESIA_API_KEY"),
		model:      defaultModel,
		voice:      voice,
		language:   defaultLanguage,
		version:    defaultVersion,
		sampleRate: defaultSampleRate,
		channels:   defaultChannels,
	}
	for _, o := range opts {
		o(t)
	}
	if t.apiKey == "" {
		return nil, errors.New("cartesia: API key required (set CARTESIA_API_KEY or WithAPIKey)")
	}
	return t, nil
}

func (t *TTS) SampleRate() int  { return t.sampleRate }
func (t *TTS) NumChannels() int { return t.channels }

func (t *TTS) Stream(ctx context.Context) tts.SynthesizeStream {
	st := &synthStream{
		tts:   t,
		audio: make(chan tts.SynthesizedAudio, 32),
		done:  make(chan struct{}),
	}
	conn, _, err := websocket.DefaultDialer.DialContext(ctx, t.wsURL(), nil)
	if err != nil {
		st.setErr(err)
		close(st.audio)
		return st
	}
	st.conn = conn
	go st.readLoop(ctx)
	return st
}

func (t *TTS) wsURL() string {
	q := url.Values{}
	q.Set("api_key", t.apiKey)
	q.Set("cartesia_version", t.version)
	return baseURL + "?" + q.Encode()
}

type outMsg struct {
	ModelID      string       `json:"model_id"`
	Transcript   string       `json:"transcript"`
	Voice        voiceSpec    `json:"voice"`
	Language     string       `json:"language,omitempty"`
	ContextID    string       `json:"context_id"`
	OutputFormat outputFormat `json:"output_format"`
	Continue     bool         `json:"continue"`
}

type voiceSpec struct {
	Mode string `json:"mode"`
	ID   string `json:"id"`
}

type outputFormat struct {
	Container  string `json:"container"`
	Encoding   string `json:"encoding"`
	SampleRate int    `json:"sample_rate"`
}

func (t *TTS) message(ctxID, transcript string, cont bool) []byte {
	b, _ := json.Marshal(outMsg{
		ModelID:      t.model,
		Transcript:   transcript,
		Voice:        voiceSpec{Mode: "id", ID: t.voice},
		Language:     t.language,
		ContextID:    ctxID,
		OutputFormat: outputFormat{Container: "raw", Encoding: "pcm_s16le", SampleRate: t.sampleRate},
		Continue:     cont,
	})
	return b
}

type synthStream struct {
	tts   *TTS
	conn  *websocket.Conn
	audio chan tts.SynthesizedAudio
	done  chan struct{}

	mu     sync.Mutex
	sendMu sync.Mutex
	err    error
	closed bool
	ctxID  string
	ctxN   int
}

func (st *synthStream) PushText(token string) {
	if token == "" {
		return
	}
	st.send(st.tts.message(st.currentCtx(), token, true))
}

func (st *synthStream) Flush() {
	st.send(st.tts.message(st.currentCtx(), "", false))
	st.mu.Lock()
	st.ctxID = ""
	st.mu.Unlock()
}

func (st *synthStream) EndInput() {
	st.Flush()
}

func (st *synthStream) Chan() <-chan tts.SynthesizedAudio { return st.audio }

func (st *synthStream) Err() error {
	st.mu.Lock()
	defer st.mu.Unlock()
	return st.err
}

func (st *synthStream) Close() error {
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

func (st *synthStream) currentCtx() string {
	st.mu.Lock()
	defer st.mu.Unlock()
	if st.ctxID == "" {
		st.ctxN++
		st.ctxID = fmt.Sprintf("ctx-%d", st.ctxN)
	}
	return st.ctxID
}

func (st *synthStream) readLoop(ctx context.Context) {
	defer close(st.audio)
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

type inMsg struct {
	Type      string `json:"type"`
	Data      string `json:"data"`
	ContextID string `json:"context_id"`
	Error     string `json:"error"`
}

func (st *synthStream) handle(data []byte) {
	var m inMsg
	if err := json.Unmarshal(data, &m); err != nil {
		return
	}
	switch m.Type {
	case "chunk":
		pcm, err := base64.StdEncoding.DecodeString(m.Data)
		if err != nil || len(pcm) == 0 {
			return
		}
		st.emit(tts.SynthesizedAudio{
			Frame:     audio.FromBytes(pcm, st.tts.sampleRate, st.tts.channels),
			SegmentID: m.ContextID,
		})
	case "done":
		st.emit(tts.SynthesizedAudio{SegmentID: m.ContextID, Final: true})
	case "error":
		st.setErr(errors.New("cartesia: " + m.Error))
	}
}

func (st *synthStream) emit(a tts.SynthesizedAudio) {
	select {
	case st.audio <- a:
	case <-st.done:
	}
}

func (st *synthStream) send(msg []byte) {
	st.mu.Lock()
	conn := st.conn
	closed := st.closed
	st.mu.Unlock()
	if conn == nil || closed {
		return
	}
	st.sendMu.Lock()
	defer st.sendMu.Unlock()
	_ = conn.WriteMessage(websocket.TextMessage, msg)
}

func (st *synthStream) setErr(err error) {
	st.mu.Lock()
	if st.err == nil {
		st.err = err
	}
	st.mu.Unlock()
}

func (st *synthStream) isClosed() bool {
	st.mu.Lock()
	defer st.mu.Unlock()
	return st.closed
}
