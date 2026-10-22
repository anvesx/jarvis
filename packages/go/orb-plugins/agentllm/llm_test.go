package agentllm

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/beingPro007/lk-go-agent-sdk/llm"
)

func chat(user string) *llm.ChatContext {
	c := &llm.ChatContext{}
	c.Add(llm.RoleSystem, "ignored by the agent service")
	c.Add(llm.RoleUser, "an older message")
	c.Add(llm.RoleAssistant, "an older reply")
	c.Add(llm.RoleUser, user)
	return c
}

func TestChatStreamsTokensAndForwardsEvents(t *testing.T) {
	var got map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1/sessions/room-1/turns" {
			t.Errorf("unexpected %s %s", r.Method, r.URL.Path)
		}
		json.NewDecoder(r.Body).Decode(&got)
		w.Header().Set("Content-Type", "application/x-ndjson")
		for _, line := range []string{
			`{"type":"read","data":{"id":"M1"}}`,
			`{"type":"decision","data":{"persona":"Examiner"}}`,
			`{"type":"token","data":"Why"}`, `{"type":"token","data":" do"}`, `{"type":"token","data":" you think so?"}`,
			`{"type":"reply","data":{"text":"Why do you think so?"}}`,
			`{"type":"check","data":{"done":true}}`,
		} {
			fmt.Fprintln(w, line)
		}
	}))
	defer srv.Close()

	var mu sync.Mutex
	var types []string
	l := New(srv.URL, "room-1", "T", WithEventHandler(func(e Event) { mu.Lock(); types = append(types, e.Type); mu.Unlock() }))
	st := l.Chat(context.Background(), chat("No, I don't agree."), nil)
	var text strings.Builder
	for c := range st.Chan() {
		text.WriteString(c.Delta.Content)
	}
	if err := st.Err(); err != nil {
		t.Fatal(err)
	}
	if text.String() != "Why do you think so?" {
		t.Fatalf("reply %q", text.String())
	}
	if got["text"] != "No, I don't agree." || got["channel"] != "voice" || got["replace_last"] != false {
		t.Fatalf("request body %v: want only the newest user message, channel voice", got)
	}
	mu.Lock()
	defer mu.Unlock()
	if strings.Join(types, ",") != "read,decision,reply,check" {
		t.Fatalf("events %v", types)
	}
}

func TestChatReportsServiceErrors(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/turns") && strings.Contains(r.URL.Path, "missing") {
			w.WriteHeader(http.StatusNotFound)
			io.WriteString(w, `{"error":"There is no session missing-1."}`)
			return
		}
		fmt.Fprintln(w, `{"type":"read","data":{}}`)
		fmt.Fprintln(w, `{"type":"error","data":"The LLM failed: model not found"}`)
	}))
	defer srv.Close()

	st := New(srv.URL, "missing-1", "T").Chat(context.Background(), chat("hi"), nil)
	for range st.Chan() {
	}
	if err := st.Err(); err == nil || !strings.Contains(err.Error(), "There is no session missing-1.") {
		t.Fatalf("err %v", err)
	}
	st = New(srv.URL, "room-2", "T").Chat(context.Background(), chat("hi"), nil)
	for range st.Chan() {
	}
	if err := st.Err(); err == nil || !strings.Contains(err.Error(), "model not found") {
		t.Fatalf("err %v", err)
	}
	st = New(srv.URL, "room-3", "T").Chat(context.Background(), &llm.ChatContext{}, nil)
	for range st.Chan() {
	}
	if st.Err() == nil {
		t.Fatal("a chat with no user message must fail")
	}
}

func TestChatStopsWhenCancelled(t *testing.T) {
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, `{"type":"token","data":"Hello"}`)
		w.(http.Flusher).Flush()
		select {
		case <-release:
		case <-r.Context().Done():
		}
	}))
	defer srv.Close()
	defer close(release)

	ctx, cancel := context.WithCancel(context.Background())
	st := New(srv.URL, "room-4", "T").Chat(ctx, chat("tell me a story"), nil)
	if c := <-st.Chan(); c.Delta.Content != "Hello" {
		t.Fatalf("first chunk %q", c.Delta.Content)
	}
	cancel() // the user barges in
	select {
	case _, ok := <-st.Chan():
		for ok {
			_, ok = <-st.Chan()
		}
	case <-time.After(2 * time.Second):
		t.Fatal("the stream did not close after cancel")
	}
}

func TestEnsureCreatesTheSession(t *testing.T) {
	var method, path string
	var body map[string]string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		method, path = r.Method, r.URL.Path
		json.NewDecoder(r.Body).Decode(&body)
		io.WriteString(w, `{"id":"room-5"}`)
	}))
	defer srv.Close()
	if err := New(srv.URL, "room-5", "Voice topic").Ensure(context.Background()); err != nil {
		t.Fatal(err)
	}
	if method != http.MethodPut || path != "/v1/sessions/room-5" || body["topic"] != "Voice topic" {
		t.Fatalf("%s %s %v", method, path, body)
	}
	if err := New("http://127.0.0.1:1", "room-6", "T").Ensure(context.Background()); err == nil {
		t.Fatal("Ensure must fail when the service does not answer")
	}
}

func TestReplaceLastFollowsTheSDKFlag(t *testing.T) {
	var mu sync.Mutex
	var bodies []map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var b map[string]any
		json.NewDecoder(r.Body).Decode(&b)
		mu.Lock()
		bodies = append(bodies, b)
		mu.Unlock()
		fmt.Fprintln(w, `{"type":"token","data":"ok"}`)
	}))
	defer srv.Close()
	l := New(srv.URL, "room-7", "T")
	for _, replaces := range []bool{false, true} {
		c := &llm.ChatContext{Messages: []llm.Message{{Role: llm.RoleUser, Content: "We should store passwords in a plain file.", Replaces: replaces}}}
		st := l.Chat(context.Background(), c, nil)
		for range st.Chan() {
		}
	}
	mu.Lock()
	defer mu.Unlock()
	if bodies[0]["replace_last"] != false || bodies[1]["replace_last"] != true {
		t.Fatalf("replace_last = %v, %v; want false, true", bodies[0]["replace_last"], bodies[1]["replace_last"])
	}
}
