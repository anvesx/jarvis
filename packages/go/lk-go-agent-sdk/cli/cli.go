package cli

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	agents "github.com/beingPro007/lk-go-agent-sdk"
)

func Run(opts agents.WorkerOptions) {
	os.Exit(run(opts))
}

func run(opts agents.WorkerOptions) int {
	if len(os.Args) < 2 {
		usage()
		return 2
	}
	cmd := os.Args[1]
	if cmd != "dev" && cmd != "start" {
		usage()
		return 2
	}
	dev := cmd == "dev"

	fs := flag.NewFlagSet(cmd, flag.ExitOnError)
	urlFlag := fs.String("url", "", "LiveKit server URL (defaults to $LIVEKIT_URL)")
	keyFlag := fs.String("api-key", "", "LiveKit API key (defaults to $LIVEKIT_API_KEY)")
	secretFlag := fs.String("api-secret", "", "LiveKit API secret (defaults to $LIVEKIT_API_SECRET)")
	healthPort := fs.Int("health-port", 8081, "port for the HTTP health endpoint (0 disables)")
	drainTimeout := fs.Duration("drain-timeout", 60*time.Second, "how long to wait for active jobs on shutdown")
	fs.Parse(os.Args[2:])

	logger := newLogger(dev)
	slog.SetDefault(logger)
	bridgeLiveKitLogs(logger)
	if opts.Logger == nil {
		opts.Logger = logger
	}

	if *urlFlag != "" {
		opts.URL = *urlFlag
	}
	if *keyFlag != "" {
		opts.APIKey = *keyFlag
	}
	if *secretFlag != "" {
		opts.APISecret = *secretFlag
	}
	if dev && opts.URL == "" && os.Getenv("LIVEKIT_URL") == "" {
		opts.URL = "ws://localhost:7880"
		if opts.APIKey == "" && os.Getenv("LIVEKIT_API_KEY") == "" {
			opts.APIKey = "devkey"
			opts.APISecret = "secret"
		}
		logger.Warn("no LIVEKIT_URL set, using local dev defaults", "url", opts.URL)
	}

	worker, err := agents.NewWorker(opts)
	if err != nil {
		logger.Error("invalid worker options", "error", err)
		return 1
	}

	healthSrv := startHealthServer(worker, *healthPort, logger)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)

	runErr := make(chan error, 1)
	go func() { runErr <- worker.Run(ctx) }()

	select {
	case err := <-runErr:
		if err != nil {
			logger.Error("worker exited", "error", err)
			return 1
		}
		return 0
	case sig := <-sigCh:
		logger.Info("signal received, draining", "signal", sig.String(), "timeout", drainTimeout.String())
		go func() {
			<-sigCh
			logger.Warn("second signal received, forcing exit")
			os.Exit(1)
		}()

		drainCtx, cancelDrain := context.WithTimeout(context.Background(), *drainTimeout)
		defer cancelDrain()
		if err := worker.Drain(drainCtx); err != nil {
			logger.Warn("drain timed out, remaining jobs cancelled", "error", err)
		}
		cancel()
		<-runErr
		if healthSrv != nil {
			healthSrv.Shutdown(context.Background())
		}
		logger.Info("shutdown complete")
		return 0
	}
}

func newLogger(dev bool) *slog.Logger {
	if dev {
		return slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelDebug}))
	}
	return slog.New(slog.NewJSONHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))
}

func startHealthServer(worker *agents.Worker, port int, logger *slog.Logger) *http.Server {
	if port == 0 {
		return nil
	}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"status":      "ok",
			"worker_id":   worker.ID(),
			"active_jobs": worker.ActiveJobs(),
		})
	})
	srv := &http.Server{Addr: fmt.Sprintf(":%d", port), Handler: mux}
	go func() {
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Error("health server failed", "error", err)
		}
	}()
	logger.Info("health endpoint listening", "addr", srv.Addr)
	return srv
}

func usage() {
	fmt.Fprintf(os.Stderr, `Usage: %s <command> [flags]

Commands:
  dev     run with debug logs and local dev defaults
  start   run for production with JSON logs

Flags:
  -url string            LiveKit server URL (defaults to $LIVEKIT_URL)
  -api-key string        LiveKit API key (defaults to $LIVEKIT_API_KEY)
  -api-secret string     LiveKit API secret (defaults to $LIVEKIT_API_SECRET)
  -health-port int       HTTP health endpoint port, 0 disables (default 8081)
  -drain-timeout value   wait for active jobs on shutdown (default 1m0s)
`, os.Args[0])
}
