package gemini

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"sync"

	"github.com/beingPro007/lk-go-agent-sdk/llm"
	"google.golang.org/genai"
)

const (
	defaultModel    = "gemini-3.6-flash"
	defaultThinking = "minimal"
)

type LLM struct {
	client   *genai.Client
	model    string
	thinking genai.ThinkingLevel

	sigMu sync.Mutex
	sigs  map[string][]byte
}

type config struct {
	apiKey   string
	model    string
	thinking string
}

type Option func(*config)

func WithAPIKey(k string) Option { return func(c *config) { c.apiKey = k } }
func WithModel(m string) Option  { return func(c *config) { c.model = m } }

func WithThinkingLevel(l string) Option { return func(c *config) { c.thinking = l } }

func New(ctx context.Context, opts ...Option) (*LLM, error) {
	c := &config{apiKey: os.Getenv("GEMINI_API_KEY"), model: defaultModel, thinking: defaultThinking}
	for _, o := range opts {
		o(c)
	}
	if c.apiKey == "" {
		return nil, errors.New("gemini: API key required (set GEMINI_API_KEY or WithAPIKey)")
	}
	client, err := genai.NewClient(ctx, &genai.ClientConfig{APIKey: c.apiKey})
	if err != nil {
		return nil, err
	}
	return &LLM{client: client, model: c.model, thinking: thinkingLevel(c.thinking), sigs: map[string][]byte{}}, nil
}

func (l *LLM) Chat(ctx context.Context, chatCtx *llm.ChatContext, tools []llm.Tool) llm.LLMStream {
	ctx, cancel := context.WithCancel(ctx)
	st := &chatStream{
		chunks: make(chan llm.ChatChunk, 32),
		cancel: cancel,
	}
	contents, sys := toContents(chatCtx, l.signatures(chatCtx))
	cfg := &genai.GenerateContentConfig{SystemInstruction: sys}
	if l.thinking != genai.ThinkingLevelUnspecified {
		cfg.ThinkingConfig = &genai.ThinkingConfig{ThinkingLevel: l.thinking}
	}
	if len(tools) > 0 {
		cfg.Tools = toTools(tools)
	}
	go st.run(ctx, l, contents, cfg)
	return st
}

type chatStream struct {
	chunks chan llm.ChatChunk
	cancel context.CancelFunc

	mu     sync.Mutex
	err    error
	closed bool
}

func (st *chatStream) run(ctx context.Context, l *LLM, contents []*genai.Content, cfg *genai.GenerateContentConfig) {
	defer close(st.chunks)
	for resp, err := range l.client.Models.GenerateContentStream(ctx, l.model, contents, cfg) {
		if err != nil {
			if ctx.Err() == nil {
				st.setErr(err)
			}
			return
		}
		l.recordSignatures(resp)
		chunk, ok := respToChunk(resp)
		if !ok {
			continue
		}
		select {
		case st.chunks <- chunk:
		case <-ctx.Done():
			return
		}
	}
}

func (st *chatStream) Chan() <-chan llm.ChatChunk { return st.chunks }

func (st *chatStream) Err() error {
	st.mu.Lock()
	defer st.mu.Unlock()
	return st.err
}

func (st *chatStream) Close() error {
	st.mu.Lock()
	if st.closed {
		st.mu.Unlock()
		return nil
	}
	st.closed = true
	st.mu.Unlock()
	st.cancel()
	return nil
}

func (st *chatStream) setErr(err error) {
	st.mu.Lock()
	if st.err == nil {
		st.err = err
	}
	st.mu.Unlock()
}

func thinkingLevel(s string) genai.ThinkingLevel {
	switch strings.ToLower(s) {
	case "minimal":
		return genai.ThinkingLevelMinimal
	case "low":
		return genai.ThinkingLevelLow
	case "medium":
		return genai.ThinkingLevelMedium
	case "high":
		return genai.ThinkingLevelHigh
	}
	return genai.ThinkingLevelUnspecified
}

func (l *LLM) recordSignatures(resp *genai.GenerateContentResponse) {
	if resp == nil || len(resp.Candidates) == 0 || resp.Candidates[0].Content == nil {
		return
	}
	l.sigMu.Lock()
	defer l.sigMu.Unlock()
	for _, p := range resp.Candidates[0].Content.Parts {
		if p == nil || p.FunctionCall == nil || len(p.ThoughtSignature) == 0 {
			continue
		}
		id := p.FunctionCall.ID
		if id == "" {
			id = p.FunctionCall.Name
		}
		l.sigs[id] = p.ThoughtSignature
	}
}

func (l *LLM) signatures(chatCtx *llm.ChatContext) map[string][]byte {
	l.sigMu.Lock()
	defer l.sigMu.Unlock()
	if chatCtx == nil || len(l.sigs) == 0 {
		return nil
	}
	live := map[string][]byte{}
	for _, m := range chatCtx.Messages {
		for _, tc := range m.ToolCalls {
			if sig, ok := l.sigs[tc.ID]; ok {
				live[tc.ID] = sig
			}
		}
	}
	l.sigs = live
	return live
}

func respToChunk(resp *genai.GenerateContentResponse) (llm.ChatChunk, bool) {
	if resp == nil || len(resp.Candidates) == 0 || resp.Candidates[0].Content == nil {
		return llm.ChatChunk{}, false
	}
	delta := llm.ChoiceDelta{Role: llm.RoleAssistant}
	got := false
	for _, p := range resp.Candidates[0].Content.Parts {
		if p == nil {
			continue
		}
		if p.Text != "" {
			delta.Content += p.Text
			got = true
		}
		if p.FunctionCall != nil {
			args := ""
			if b, err := json.Marshal(p.FunctionCall.Args); err == nil {
				args = string(b)
			}
			id := p.FunctionCall.ID
			if id == "" {
				id = p.FunctionCall.Name
			}
			delta.ToolCalls = append(delta.ToolCalls, llm.ToolCall{
				ID:        id,
				Name:      p.FunctionCall.Name,
				Arguments: args,
			})
			got = true
		}
	}
	if !got {
		return llm.ChatChunk{}, false
	}
	return llm.ChatChunk{Delta: delta}, true
}

func toContents(chatCtx *llm.ChatContext, sigs map[string][]byte) ([]*genai.Content, *genai.Content) {
	var contents []*genai.Content
	var sys *genai.Content
	if chatCtx == nil {
		return contents, sys
	}
	for _, m := range chatCtx.Messages {
		switch m.Role {
		case llm.RoleSystem:
			sys = &genai.Content{Parts: []*genai.Part{{Text: m.Content}}}
		case llm.RoleTool:
			name := m.Name
			if name == "" {
				name = m.ToolCallID
			}
			contents = append(contents, &genai.Content{
				Role: "user",
				Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
					ID:       m.ToolCallID,
					Name:     name,
					Response: toolResponse(m.Content),
				}}},
			})
		default:
			role := "user"
			if m.Role == llm.RoleAssistant {
				role = "model"
			}
			var parts []*genai.Part
			if m.Content != "" {
				parts = append(parts, &genai.Part{Text: m.Content})
			}
			for _, tc := range m.ToolCalls {
				var args map[string]any
				_ = json.Unmarshal([]byte(tc.Arguments), &args)
				parts = append(parts, &genai.Part{
					FunctionCall: &genai.FunctionCall{
						ID:   tc.ID,
						Name: tc.Name,
						Args: args,
					},
					ThoughtSignature: sigs[tc.ID],
				})
			}
			if len(parts) > 0 {
				contents = append(contents, &genai.Content{Role: role, Parts: parts})
			}
		}
	}
	return contents, sys
}

func toolResponse(content string) map[string]any {
	var obj map[string]any
	if err := json.Unmarshal([]byte(content), &obj); err == nil && obj != nil {
		return obj
	}
	return map[string]any{"output": content}
}

func toTools(tools []llm.Tool) []*genai.Tool {
	decls := make([]*genai.FunctionDeclaration, 0, len(tools))
	for _, t := range tools {
		decls = append(decls, &genai.FunctionDeclaration{
			Name:        t.Name,
			Description: t.Description,
			Parameters:  toSchema(t.Parameters),
		})
	}
	return []*genai.Tool{{FunctionDeclarations: decls}}
}

func toSchema(m map[string]any) *genai.Schema {
	if m == nil {
		return nil
	}
	s := &genai.Schema{}
	if t, ok := m["type"].(string); ok {
		s.Type = jsonType(t)
	}
	if d, ok := m["description"].(string); ok {
		s.Description = d
	}
	if props, ok := m["properties"].(map[string]any); ok {
		s.Properties = map[string]*genai.Schema{}
		for k, v := range props {
			if vm, ok := v.(map[string]any); ok {
				s.Properties[k] = toSchema(vm)
			}
		}
	}
	if req, ok := m["required"].([]any); ok {
		for _, r := range req {
			if rs, ok := r.(string); ok {
				s.Required = append(s.Required, rs)
			}
		}
	}
	if items, ok := m["items"].(map[string]any); ok {
		s.Items = toSchema(items)
	}
	if enum, ok := m["enum"].([]any); ok {
		for _, e := range enum {
			if es, ok := e.(string); ok {
				s.Enum = append(s.Enum, es)
			}
		}
	}
	return s
}

func jsonType(t string) genai.Type {
	switch t {
	case "string":
		return genai.TypeString
	case "number":
		return genai.TypeNumber
	case "integer":
		return genai.TypeInteger
	case "boolean":
		return genai.TypeBoolean
	case "array":
		return genai.TypeArray
	case "object":
		return genai.TypeObject
	default:
		return genai.TypeUnspecified
	}
}
