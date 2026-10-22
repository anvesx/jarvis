package voice

import "github.com/beingPro007/lk-go-agent-sdk/llm"

type AgentState int

const (
	StateListening AgentState = iota
	StateThinking
	StateSpeaking
)

func (s AgentState) String() string {
	switch s {
	case StateListening:
		return "listening"
	case StateThinking:
		return "thinking"
	case StateSpeaking:
		return "speaking"
	}
	return "unknown"
}

type EventType int

const (
	EventUserStartedSpeaking EventType = iota
	EventUserStoppedSpeaking
	EventUserTranscript
	EventAgentResponse
	EventAgentStartedSpeaking
	EventAgentStoppedSpeaking
	EventStateChanged
	EventToolCall
	EventToolResult
	EventError
	// EventUserTurn carries the text of a committed user turn, after the
	// whole turn was transcribed in one piece when the STT can do that.
	EventUserTurn
	// EventUserSpeechIgnored carries the text of speech that was too short to
	// interrupt the agent (an echo, a cough, "mm-hmm"). It is not a turn.
	EventUserSpeechIgnored
)

func (t EventType) String() string {
	switch t {
	case EventUserStartedSpeaking:
		return "user_started_speaking"
	case EventUserStoppedSpeaking:
		return "user_stopped_speaking"
	case EventUserTranscript:
		return "user_transcript"
	case EventAgentResponse:
		return "agent_response"
	case EventAgentStartedSpeaking:
		return "agent_started_speaking"
	case EventAgentStoppedSpeaking:
		return "agent_stopped_speaking"
	case EventStateChanged:
		return "state_changed"
	case EventToolCall:
		return "tool_call"
	case EventToolResult:
		return "tool_result"
	case EventError:
		return "error"
	case EventUserTurn:
		return "user_turn"
	case EventUserSpeechIgnored:
		return "user_speech_ignored"
	}
	return "unknown"
}

type Event struct {
	Type       EventType
	Transcript string
	IsFinal    bool
	Text       string
	State      AgentState
	SpeechID   string
	ToolCall   llm.ToolCall
	Err        error
}
