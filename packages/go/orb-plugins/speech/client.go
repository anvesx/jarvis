// Package speech connects the voice pipeline to the orb speech service
// (packages/python/orb-speech): Whistle speech-to-text, Kokoro text-to-speech
// and Smart Turn end-of-turn detection, all on this machine.
//
//	c := speech.NewClient("http://127.0.0.1:8200")
//	recognizer := speech.NewRecognizer(c, "")    // an stt.Recognizer: wrap it with stt.NewStreamAdapter
//	voice := speech.NewTTS(c, "af_heart")         // a tts.TTS
//	turns := speech.NewTurnDetector(c)            // a voice.AudioTurnDetector
package speech

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/beingPro007/lk-go-agent-sdk/audio"
)

// Client calls the speech service.
type Client struct {
	BaseURL string
	HTTP    *http.Client
}

// NewClient returns a client for the service at baseURL.
func NewClient(baseURL string) *Client {
	return &Client{BaseURL: strings.TrimRight(baseURL, "/"),
		HTTP: &http.Client{Transport: &http.Transport{ResponseHeaderTimeout: 20 * time.Second, MaxIdleConnsPerHost: 8}}}
}

func (c *Client) post(ctx context.Context, path, contentType string, body []byte) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+path, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", contentType)
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return nil, fmt.Errorf("speech: the speech service does not answer at %s: %w", c.BaseURL, err)
	}
	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		var e struct {
			Error string `json:"error"`
		}
		if json.Unmarshal(raw, &e) == nil && e.Error != "" {
			return nil, fmt.Errorf("speech: %s %s: %s (HTTP %d)", http.MethodPost, path, e.Error, resp.StatusCode)
		}
		return nil, fmt.Errorf("speech: %s %s: %s", http.MethodPost, path, resp.Status)
	}
	return resp, nil
}

func (c *Client) postJSON(ctx context.Context, path, contentType string, body []byte, out any) error {
	resp, err := c.post(ctx, path, contentType, body)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	return json.NewDecoder(resp.Body).Decode(out)
}

// pcm16 packs frames into little-endian 16-bit PCM. The frames must share one
// sample rate, which it returns.
func pcm16(frames []audio.Frame) ([]byte, int, error) {
	if len(frames) == 0 {
		return nil, 0, nil
	}
	rate := frames[0].SampleRate
	n := 0
	for _, f := range frames {
		if f.SampleRate != rate {
			return nil, 0, fmt.Errorf("speech: frames at %d Hz and %d Hz in one request", rate, f.SampleRate)
		}
		if f.Channels > 1 {
			return nil, 0, fmt.Errorf("speech: only mono audio is supported, not %d channels", f.Channels)
		}
		n += len(f.Data)
	}
	out := make([]byte, 0, n*2)
	for _, f := range frames {
		for _, s := range f.Data {
			out = binary.LittleEndian.AppendUint16(out, uint16(s))
		}
	}
	return out, rate, nil
}
