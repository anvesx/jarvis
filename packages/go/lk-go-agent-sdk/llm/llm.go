package llm

import "context"

type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

type ToolCall struct {
	ID        string
	Name      string
	Arguments string
}

type Message struct {
	Role       Role
	Content    string
	Name       string
	ToolCalls  []ToolCall
	ToolCallID string
	// Replaces marks a user message that says the previous user turn again,
	// with more words: the user went on talking before any audio of the
	// reply played. An LLM that keeps its own history can drop the old turn.
	Replaces bool
}

type ChatContext struct {
	Messages []Message
}

func (c *ChatContext) Add(role Role, content string) {
	c.Messages = append(c.Messages, Message{Role: role, Content: content})
}

func (c *ChatContext) Clone() *ChatContext {
	msgs := make([]Message, len(c.Messages))
	copy(msgs, c.Messages)
	return &ChatContext{Messages: msgs}
}

func (c *ChatContext) AddToolCalls(calls []ToolCall) {
	c.Messages = append(c.Messages, Message{Role: RoleAssistant, ToolCalls: calls})
}

func (c *ChatContext) AddToolResult(call ToolCall, result string) {
	c.Messages = append(c.Messages, Message{
		Role:       RoleTool,
		Content:    result,
		Name:       call.Name,
		ToolCallID: call.ID,
	})
}

type ToolHandler func(ctx context.Context, arguments string) (string, error)

type Tool struct {
	Name        string
	Description string
	Parameters  map[string]any
	Handler     ToolHandler
}

type ChoiceDelta struct {
	Role      Role
	Content   string
	ToolCalls []ToolCall
}

type ChatChunk struct {
	ID    string
	Delta ChoiceDelta
}

type LLM interface {
	Chat(ctx context.Context, chatCtx *ChatContext, tools []Tool) LLMStream
}

type LLMStream interface {
	Chan() <-chan ChatChunk
	Err() error
	Close() error
}
