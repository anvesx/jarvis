package gemini

import (
	"testing"

	"github.com/beingPro007/lk-go-agent-sdk/llm"
	"google.golang.org/genai"
)

// A text part in a Gemini response becomes a ChatChunk with that text as the delta.
func TestRespToChunkText(t *testing.T) {
	resp := &genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{{
			Content: &genai.Content{Parts: []*genai.Part{{Text: "hello"}}},
		}},
	}
	chunk, ok := respToChunk(resp)
	if !ok {
		t.Fatal("expected chunk")
	}
	if chunk.Delta.Content != "hello" {
		t.Fatalf("content = %q, want hello", chunk.Delta.Content)
	}
}

// A Gemini functionCall part becomes a ToolCall with the args JSON-encoded into Arguments.
func TestRespToChunkFunctionCall(t *testing.T) {
	resp := &genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{{
			Content: &genai.Content{Parts: []*genai.Part{{
				FunctionCall: &genai.FunctionCall{Name: "get_weather", Args: map[string]any{"city": "paris"}},
			}}},
		}},
	}
	chunk, ok := respToChunk(resp)
	if !ok {
		t.Fatal("expected chunk")
	}
	if len(chunk.Delta.ToolCalls) != 1 {
		t.Fatalf("tool calls = %d, want 1", len(chunk.Delta.ToolCalls))
	}
	tc := chunk.Delta.ToolCalls[0]
	if tc.Name != "get_weather" || tc.Arguments != `{"city":"paris"}` {
		t.Fatalf("tool call wrong: %+v", tc)
	}
}

// An empty/candidate-less response yields no chunk (ok=false), so we skip it in the stream.
func TestRespToChunkEmpty(t *testing.T) {
	if _, ok := respToChunk(&genai.GenerateContentResponse{}); ok {
		t.Fatal("expected no chunk for empty response")
	}
}

// Our roles translate to Gemini's model: system -> SystemInstruction (not a message),
// user -> "user", assistant -> "model".
func TestToContentsRolesAndSystem(t *testing.T) {
	cc := &llm.ChatContext{}
	cc.Add(llm.RoleSystem, "be brief")
	cc.Add(llm.RoleUser, "hi")
	cc.Add(llm.RoleAssistant, "hello")
	contents, sys := toContents(cc, nil)
	if sys == nil || sys.Parts[0].Text != "be brief" {
		t.Fatalf("system instruction wrong: %+v", sys)
	}
	if len(contents) != 2 {
		t.Fatalf("contents = %d, want 2", len(contents))
	}
	if contents[0].Role != "user" || contents[1].Role != "model" {
		t.Fatalf("roles wrong: %q %q", contents[0].Role, contents[1].Role)
	}
}

// A JSON-schema tool parameters map converts recursively into a *genai.Schema
// (object type, typed properties, required list preserved).
func TestToSchema(t *testing.T) {
	m := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"city": map[string]any{"type": "string", "description": "city name"},
		},
		"required": []any{"city"},
	}
	s := toSchema(m)
	if s.Type != genai.TypeObject {
		t.Fatalf("type = %v, want OBJECT", s.Type)
	}
	if s.Properties["city"].Type != genai.TypeString {
		t.Fatalf("city type wrong: %v", s.Properties["city"].Type)
	}
	if len(s.Required) != 1 || s.Required[0] != "city" {
		t.Fatalf("required wrong: %v", s.Required)
	}
}

// A tool result must go back to Gemini as a functionResponse named after the
// function itself (Gemini matches on name), carrying the call id, while the
// assistant's own tool call is replayed as a functionCall part.
func TestToContentsToolRoundTrip(t *testing.T) {
	chatCtx := &llm.ChatContext{Messages: []llm.Message{
		{Role: llm.RoleUser, Content: "what time is it"},
		{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{
			{ID: "call_1", Name: "get_time", Arguments: `{"tz":"UTC"}`},
		}},
		{Role: llm.RoleTool, Name: "get_time", ToolCallID: "call_1", Content: "12:00"},
	}}

	contents, _ := toContents(chatCtx, nil)
	if len(contents) != 3 {
		t.Fatalf("got %d contents, want 3", len(contents))
	}

	call := contents[1].Parts[0].FunctionCall
	if contents[1].Role != "model" || call == nil || call.Name != "get_time" {
		t.Fatalf("assistant tool call wrong: %+v", contents[1])
	}
	if call.Args["tz"] != "UTC" {
		t.Fatalf("tool call arguments were not decoded: %+v", call.Args)
	}

	resp := contents[2].Parts[0].FunctionResponse
	if resp == nil {
		t.Fatalf("tool result was not a functionResponse: %+v", contents[2])
	}
	if resp.Name != "get_time" {
		t.Fatalf("functionResponse name = %q, want the function name", resp.Name)
	}
	if resp.ID != "call_1" {
		t.Fatalf("functionResponse id = %q, want call_1", resp.ID)
	}
	if resp.Response["output"] != "12:00" {
		t.Fatalf("tool output wrong: %+v", resp.Response)
	}
}

// Callers that only set ToolCallID (no Name) still get a usable functionResponse.
func TestToContentsToolResultFallsBackToCallID(t *testing.T) {
	chatCtx := &llm.ChatContext{Messages: []llm.Message{
		{Role: llm.RoleTool, ToolCallID: "get_time", Content: "12:00"},
	}}
	contents, _ := toContents(chatCtx, nil)
	if contents[0].Parts[0].FunctionResponse.Name != "get_time" {
		t.Fatalf("name fallback failed: %+v", contents[0].Parts[0].FunctionResponse)
	}
}

// A JSON tool result is handed to Gemini as a structured object rather than
// being buried in a string.
func TestToolResponseKeepsJSON(t *testing.T) {
	got := toolResponse(`{"temp":21,"unit":"C"}`)
	if got["unit"] != "C" {
		t.Fatalf("JSON result was not passed through: %+v", got)
	}
	if plain := toolResponse("just text"); plain["output"] != "just text" {
		t.Fatalf("plain result wrong: %+v", plain)
	}
}

// A Gemini functionCall with no id (the usual case) is given the function name
// as its id, so the round trip back to the model still matches up.
func TestRespToChunkFunctionCallIDDefaultsToName(t *testing.T) {
	resp := &genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{{
			Content: &genai.Content{Parts: []*genai.Part{{
				FunctionCall: &genai.FunctionCall{Name: "get_time", Args: map[string]any{}},
			}}},
		}},
	}
	chunk, ok := respToChunk(resp)
	if !ok || len(chunk.Delta.ToolCalls) != 1 {
		t.Fatalf("expected one tool call, got %+v", chunk)
	}
	if chunk.Delta.ToolCalls[0].ID != "get_time" {
		t.Fatalf("id = %q, want the function name", chunk.Delta.ToolCalls[0].ID)
	}
}

// Voice agents want the model answering fast, so thinking defaults to minimal;
// unknown values fall back to the model's own default.
func TestThinkingLevel(t *testing.T) {
	if thinkingLevel(defaultThinking) != genai.ThinkingLevelMinimal {
		t.Fatalf("default thinking level is not minimal")
	}
	if thinkingLevel("HIGH") != genai.ThinkingLevelHigh {
		t.Fatalf("level parsing is case sensitive")
	}
	if thinkingLevel("nonsense") != genai.ThinkingLevelUnspecified {
		t.Fatalf("unknown level should defer to the model default")
	}
}
