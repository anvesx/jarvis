package speech

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/beingPro007/lk-go-agent-sdk/audio"
)

func frames(n, rate int) []audio.Frame {
	out := make([]audio.Frame, n)
	for i := range out {
		out[i] = audio.NewFrame(make([]int16, rate/50), rate, 1) // 20 ms
	}
	return out
}

func TestRecognizerSendsPCMAndReadsText(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		if r.URL.Path != "/v1/stt" || r.URL.Query().Get("sample_rate") != "16000" || r.URL.Query().Get("language") != "en" {
			t.Errorf("unexpected request %s", r.URL)
		}
		if len(body) != 10*320*2 {
			t.Errorf("body has %d bytes, want %d", len(body), 10*320*2)
		}
		io.WriteString(w, `{"text":"hello there","language":"en","ms":12.5,"audio_ms":200}`)
	}))
	defer srv.Close()
	got, err := NewRecognizer(NewClient(srv.URL), "en").Recognize(context.Background(), frames(10, 16000))
	if err != nil {
		t.Fatal(err)
	}
	if got.Text != "hello there" || got.Language != "en" || got.EndTime != 200*time.Millisecond {
		t.Fatalf("%+v", got)
	}
}

func TestRecognizerErrorsCarryTheServiceMessage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusRequestEntityTooLarge)
		io.WriteString(w, `{"error":"The audio is longer than 60 s."}`)
	}))
	defer srv.Close()
	_, err := NewRecognizer(NewClient(srv.URL), "").Recognize(context.Background(), frames(1, 16000))
	if err == nil || !strings.Contains(err.Error(), "The audio is longer than 60 s.") {
		t.Fatalf("err %v", err)
	}
	mixed := append(frames(1, 16000), frames(1, 48000)...)
	if _, err := NewRecognizer(NewClient(srv.URL), "").Recognize(context.Background(), mixed); err == nil {
		t.Fatal("frames at two sample rates must be refused")
	}
}

func TestTurnDetectorUsesTheAudio(t *testing.T) {
	var size int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		size = len(body)
		io.WriteString(w, `{"probability":0.83,"complete":true}`)
	}))
	defer srv.Close()
	d := NewTurnDetector(NewClient(srv.URL))
	p, err := d.PredictEndOfTurnAudio(context.Background(), nil, frames(50, 16000))
	if err != nil || p != 0.83 || size != 50*320*2 {
		t.Fatalf("p=%v err=%v size=%d", p, err, size)
	}
	if p, _ := d.PredictEndOfTurn(context.Background(), nil); p != 1 {
		t.Fatalf("text-only call gave %v, want 1 (no opinion)", p)
	}
	if p, _ := d.PredictEndOfTurnAudio(context.Background(), nil, nil); p != 1 {
		t.Fatalf("no audio gave %v, want 1", p)
	}
}

func TestCutSentences(t *testing.T) {
	cases := []struct {
		in, piece, rest string
		ok              bool
	}{
		{"Hello there, my friend. How are", "Hello there, my friend.", "How are", true},
		{"Hi. Short.", "", "Hi. Short.", false},                                       // too short: wait for more
		{"It costs 3.5 dollars and more", "", "It costs 3.5 dollars and more", false}, // a dot inside a number
		{"Is this really what you want? Yes", "Is this really what you want?", "Yes", true},
		{"A line that ends here\nnext", "A line that ends here", "next", true},
	}
	for _, c := range cases {
		piece, rest, ok := cut(c.in, false)
		if piece != c.piece || rest != c.rest || ok != c.ok {
			t.Errorf("cut(%q) = %q, %q, %v; want %q, %q, %v", c.in, piece, rest, ok, c.piece, c.rest, c.ok)
		}
	}
	long := strings.Repeat("word ", 30) + "and then, more words without an end " + strings.Repeat("x ", 20)
	piece, _, ok := cut(long, false)
	if !ok || len(piece) > maxPiece || len(piece) < minSentence {
		t.Fatalf("a long text was not cut well: %q", piece)
	}
}

func TestFirstPieceMayEndAtAClause(t *testing.T) {
	text := "Storing passwords in a plain file is risky, because anyone who copies the file"
	if p, rest, ok := cut(text, true); !ok || p != "Storing passwords in a plain file is risky," || rest != "because anyone who copies the file" {
		t.Fatalf("first piece %q %q %v", p, rest, ok)
	}
	if _, _, ok := cut(text, false); ok {
		t.Fatal("a later piece must wait for the end of its sentence")
	}
	if _, _, ok := cut("Well, yes", true); ok {
		t.Fatal("a clause shorter than 30 characters must wait")
	}
	if _, _, ok := cut("It costs 1,000 dollars and maybe more than that", true); ok {
		t.Fatal("a comma inside a number is not a clause end")
	}
}

func TestTTSSynthesizesSentenceBySentence(t *testing.T) {
	var mu sync.Mutex
	var texts []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Text  string   `json:"text"`
			Voice string   `json:"voice"`
			Speed *float64 `json:"speed"`
		}
		json.NewDecoder(r.Body).Decode(&req)
		mu.Lock()
		texts = append(texts, req.Text)
		mu.Unlock()
		if req.Voice != "af_heart" || req.Speed != nil {
			t.Errorf("voice %q speed %v", req.Voice, req.Speed)
		}
		w.Header().Set("X-Sample-Rate", "24000")
		pcm := make([]byte, 0, 1200*2)
		for i := 0; i < 1200; i++ { // 50 ms
			pcm = binary.LittleEndian.AppendUint16(pcm, uint16(int16(i)))
		}
		w.Write(pcm)
	}))
	defer srv.Close()

	st := NewTTS(NewClient(srv.URL), "af_heart").Stream(context.Background())
	for _, tok := range []string{"Hashing is ", "safer than a plain file. ", "Do you agree", " with that?"} {
		st.PushText(tok)
	}
	st.EndInput()
	var samples, finals int
	segments := map[string]bool{}
	for a := range st.Chan() {
		samples += len(a.Frame.Data)
		segments[a.SegmentID] = true
		if a.Final {
			finals++
		}
		if len(a.Frame.Data) > frameSamples {
			t.Fatalf("a frame has %d samples, want at most %d", len(a.Frame.Data), frameSamples)
		}
	}
	if err := st.Err(); err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	defer mu.Unlock()
	if strings.Join(texts, "|") != "Hashing is safer than a plain file.|Do you agree with that?" {
		t.Fatalf("sentences %q", texts)
	}
	if samples != 2*1200 || finals != 2 || len(segments) != 2 {
		t.Fatalf("samples=%d finals=%d segments=%d", samples, finals, len(segments))
	}
}

func TestTTSStopsWhenClosed(t *testing.T) {
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write(make([]byte, 960))
		w.(http.Flusher).Flush()
		select {
		case <-release:
		case <-r.Context().Done():
		}
	}))
	defer srv.Close()
	defer close(release)

	st := NewTTS(NewClient(srv.URL), "").Stream(context.Background())
	st.PushText("This sentence is long enough to be spoken. ")
	<-st.Chan()
	st.Close() // barge-in
	done := make(chan struct{})
	go func() {
		for range st.Chan() {
		}
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("the TTS stream did not stop after Close")
	}
}
