package cli

import (
	"bytes"
	"log/slog"
	"strings"
	"testing"

	lklogger "github.com/livekit/protocol/logger"
)

// TestBridgeLiveKitLogs verifies that log lines coming out of the LiveKit SDK
// and pion are re-emitted through our slog handler, with their structured
// fields intact, instead of the SDK's own zap logger.
func TestBridgeLiveKitLogs(t *testing.T) {
	var buf bytes.Buffer
	bridgeLiveKitLogs(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug})))

	lklogger.Infow("hello from livekit", "room", "test-room")
	lklogger.Debugw("noisy pion internals", "sdp", "v=0...")

	got := buf.String()
	// Debug-level SDK chatter is dropped even when our own logger is verbose.
	if strings.Contains(got, "noisy pion internals") {
		t.Fatalf("debug chatter from the SDK was not filtered: %q", got)
	}
	if !strings.Contains(got, "hello from livekit") {
		t.Fatalf("livekit log did not reach slog: %q", got)
	}
	if !strings.Contains(got, "test-room") {
		t.Fatalf("structured fields were dropped: %q", got)
	}
}
