// Package agentllm is an llm.LLM that hands each user turn to the orb agent
// service and streams the reply text back.
//
// The agent service owns the session: the chat, the debts and the persona.
// So this plugin sends only the newest user message, with channel "voice",
// and it ignores the rest of the chat context and any tools.
package agentllm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/beingPro007/lk-go-agent-sdk/llm"
)

// Event is one non-text event of a turn: read, decision, reply, check, state.
type Event struct {
	Type string
	Data json.RawMessage
}

// Option configures an LLM.
type Option func(*LLM)

// WithHTTPClient replaces the default HTTP client (no overall timeout,
// because a reply streams for as long as it needs).
func WithHTTPClient(c *http.Client) Option { return func(l *LLM) { l.http = c } }

// WithEventHandler gets every event that is not reply text, for logs.
func WithEventHandler(fn func(Event)) Option { return func(l *LLM) { l.onEvent = fn } }

// LLM talks to one session of the orb agent service.
type LLM struct {
	baseURL   string
	sessionID string
	topic     string
	http      *http.Client
	onEvent   func(Event)
}

// New returns an LLM for the session sessionID. topic is used only when the
// session does not exist yet (see Ensure).
func New(baseURL, sessionID, topic string, opts ...Option) *LLM {
	l := &LLM{baseURL: strings.TrimRight(baseURL, "/"), sessionID: sessionID, topic: topic,
		http: &http.Client{Transport: &http.Transport{ResponseHeaderTimeout: 30 * time.Second}}}
	for _, o := range opts {
		o(l)
	}
	return l
}

// SessionID is the agent session that this LLM talks to.
func (l *LLM) SessionID() string { return l.sessionID }

// Ensure creates the session when it is missing, and checks that the service answers.
func (l *LLM) Ensure(ctx context.Context) error {
	body, _ := json.Marshal(map[string]string{"topic": l.topic})
	req, err := http.NewRequestWithContext(ctx, http.MethodPut, l.baseURL+"/v1/sessions/"+url.PathEscape(l.sessionID), bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := l.http.Do(req)
	if err != nil {
		return fmt.Errorf("agentllm: the agent service does not answer at %s: %w", l.baseURL, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("agentllm: create session %s: %s", l.sessionID, serviceError(resp))
	}
	return nil
}

// Chat sends the newest user message of chatCtx as one turn. When the SDK
// marks it as Replaces (the person went on talking before they heard any
// reply), the agent service undoes the unheard turn first (replace_last).
func (l *LLM) Chat(ctx context.Context, chatCtx *llm.ChatContext, tools []llm.Tool) llm.LLMStream {
	s := &stream{chunks: make(chan llm.ChatChunk, 64)}
	msg := lastUser(chatCtx)
	text := strings.TrimSpace(msg.Content)
	if text == "" {
		s.fail(errors.New("agentllm: the chat context has no user message"))
		close(s.chunks)
		return s
	}
	ctx, s.cancel = context.WithCancel(ctx)
	go s.run(ctx, l, text, msg.Replaces)
	return s
}

func lastUser(c *llm.ChatContext) llm.Message {
	if c == nil {
		return llm.Message{}
	}
	for i := len(c.Messages) - 1; i >= 0; i-- {
		if c.Messages[i].Role == llm.RoleUser {
			return c.Messages[i]
		}
	}
	return llm.Message{}
}

type stream struct {
	chunks chan llm.ChatChunk
	cancel context.CancelFunc
	mu     sync.Mutex
	err    error
}

func (s *stream) Chan() <-chan llm.ChatChunk { return s.chunks }

func (s *stream) Err() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.err
}

func (s *stream) Close() error {
	if s.cancel != nil {
		s.cancel()
	}
	return nil
}

func (s *stream) fail(err error) {
	s.mu.Lock()
	if s.err == nil {
		s.err = err
	}
	s.mu.Unlock()
}

func (s *stream) run(ctx context.Context, l *LLM, text string, replace bool) {
	defer close(s.chunks)
	body, _ := json.Marshal(map[string]any{"text": text, "channel": "voice", "replace_last": replace})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		l.baseURL+"/v1/sessions/"+url.PathEscape(l.sessionID)+"/turns", bytes.NewReader(body))
	if err != nil {
		s.fail(err)
		return
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := l.http.Do(req)
	if err != nil {
		if ctx.Err() == nil {
			s.fail(fmt.Errorf("agentllm: %w", err))
		}
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		s.fail(fmt.Errorf("agentllm: turn: %s", serviceError(resp)))
		return
	}
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 64*1024), 8*1024*1024) // a state event can be large
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var ev struct {
			Type string          `json:"type"`
			Data json.RawMessage `json:"data"`
		}
		if err := json.Unmarshal(line, &ev); err != nil {
			s.fail(fmt.Errorf("agentllm: bad event line: %w", err))
			return
		}
		switch ev.Type {
		case "token":
			var piece string
			if err := json.Unmarshal(ev.Data, &piece); err != nil || piece == "" {
				continue
			}
			select {
			case s.chunks <- llm.ChatChunk{Delta: llm.ChoiceDelta{Role: llm.RoleAssistant, Content: piece}}:
			case <-ctx.Done():
				return
			}
		case "error":
			var msg string
			_ = json.Unmarshal(ev.Data, &msg)
			s.fail(fmt.Errorf("agentllm: %s", msg))
		default:
			if l.onEvent != nil {
				l.onEvent(Event{Type: ev.Type, Data: ev.Data})
			}
		}
	}
	if err := sc.Err(); err != nil && ctx.Err() == nil {
		s.fail(fmt.Errorf("agentllm: read the reply: %w", err))
	}
}

// serviceError is the {"error": ...} message of a failed response, or its status.
func serviceError(resp *http.Response) string {
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
	var e struct {
		Error string `json:"error"`
	}
	if json.Unmarshal(raw, &e) == nil && e.Error != "" {
		return fmt.Sprintf("%s (HTTP %d)", e.Error, resp.StatusCode)
	}
	return resp.Status
}
