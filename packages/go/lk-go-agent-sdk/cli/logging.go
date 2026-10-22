package cli

import (
	"context"
	"log/slog"

	"github.com/go-logr/logr"
	lklogger "github.com/livekit/protocol/logger"
	lksdk "github.com/livekit/server-sdk-go/v2"
)

type minLevelHandler struct {
	slog.Handler
	min slog.Level
}

func (h minLevelHandler) Enabled(ctx context.Context, l slog.Level) bool {
	return l >= h.min && h.Handler.Enabled(ctx, l)
}

func (h minLevelHandler) WithAttrs(attrs []slog.Attr) slog.Handler {
	return minLevelHandler{Handler: h.Handler.WithAttrs(attrs), min: h.min}
}

func (h minLevelHandler) WithGroup(name string) slog.Handler {
	return minLevelHandler{Handler: h.Handler.WithGroup(name), min: h.min}
}

func bridgeLiveKitLogs(l *slog.Logger) {
	h := minLevelHandler{Handler: l.Handler(), min: slog.LevelInfo}
	bridged := lklogger.LogRLogger(logr.FromSlogHandler(h))
	lklogger.SetLogger(bridged, "livekit")
	lksdk.SetLogger(bridged)
}
