package agents

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"runtime/debug"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"github.com/livekit/protocol/auth"
	"github.com/livekit/protocol/livekit"
	"golang.org/x/sync/errgroup"
	"google.golang.org/protobuf/proto"
)

type JobType = livekit.JobType

const (
	JobTypeRoom        JobType = livekit.JobType_JT_ROOM
	JobTypePublisher   JobType = livekit.JobType_JT_PUBLISHER
	JobTypeParticipant JobType = livekit.JobType_JT_PARTICIPANT
)

type WorkerStatus = livekit.WorkerStatus

const (
	WorkerStatusAvailable WorkerStatus = livekit.WorkerStatus_WS_AVAILABLE
	WorkerStatusFull      WorkerStatus = livekit.WorkerStatus_WS_FULL
)

const (
	defaultPingInterval  = 30 * time.Second
	updateStatusInterval = 2500 * time.Millisecond
	maxReconnectBackoff  = 30 * time.Second
	assignmentTimeout    = 10 * time.Second
)

type EntrypointFunc func(job *JobContext) error

type RequestFunc func(req *JobRequest) error

type LoadFunc func(w *Worker) float32

type WorkerOptions struct {
	Entrypoint EntrypointFunc
	RequestFnc RequestFunc
	LoadFnc    LoadFunc

	AgentName     string
	JobType       JobType
	LoadThreshold float32
	Permissions   *livekit.ParticipantPermission

	URL       string
	APIKey    string
	APISecret string

	PingInterval time.Duration
	Logger       *slog.Logger
}

type Worker struct {
	opts WorkerOptions
	log  *slog.Logger

	mu       sync.Mutex
	id       string
	status   WorkerStatus
	draining bool
	pending  map[string]JobAcceptInfo
	jobs     map[string]*JobContext

	sendCh chan *livekit.WorkerMessage
}

func NewWorker(opts WorkerOptions) (*Worker, error) {
	if opts.Entrypoint == nil {
		return nil, errors.New("agents: WorkerOptions.Entrypoint is required")
	}
	if opts.URL == "" {
		opts.URL = os.Getenv("LIVEKIT_URL")
	}
	if opts.APIKey == "" {
		opts.APIKey = os.Getenv("LIVEKIT_API_KEY")
	}
	if opts.APISecret == "" {
		opts.APISecret = os.Getenv("LIVEKIT_API_SECRET")
	}
	if opts.URL == "" || opts.APIKey == "" || opts.APISecret == "" {
		return nil, errors.New("agents: URL, APIKey and APISecret are required (or set LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET)")
	}
	if opts.PingInterval <= 0 {
		opts.PingInterval = defaultPingInterval
	}
	if opts.LoadThreshold == 0 {
		opts.LoadThreshold = 0.75
	}
	if opts.Logger == nil {
		opts.Logger = slog.Default()
	}
	if opts.Permissions == nil {
		opts.Permissions = &livekit.ParticipantPermission{
			CanSubscribe:      true,
			CanPublish:        true,
			CanPublishData:    true,
			CanUpdateMetadata: true,
			Agent:             true,
		}
	}
	return &Worker{
		opts:    opts,
		log:     opts.Logger,
		status:  WorkerStatusAvailable,
		pending: make(map[string]JobAcceptInfo),
		jobs:    make(map[string]*JobContext),
		sendCh:  make(chan *livekit.WorkerMessage, 64),
	}, nil
}

func (w *Worker) ID() string {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.id
}

func (w *Worker) ActiveJobs() int {
	w.mu.Lock()
	defer w.mu.Unlock()
	return len(w.jobs)
}

func (w *Worker) Run(ctx context.Context) error {
	backoff := time.Second
	for {
		start := time.Now()
		err := w.runConn(ctx)
		if ctx.Err() != nil {
			return nil
		}
		if time.Since(start) > maxReconnectBackoff {
			backoff = time.Second
		}
		w.log.Warn("connection to LiveKit server lost, reconnecting",
			"error", err, "backoff", backoff.String())
		select {
		case <-ctx.Done():
			return nil
		case <-time.After(backoff):
		}
		if backoff < maxReconnectBackoff {
			backoff *= 2
		}
	}
}

func (w *Worker) Drain(ctx context.Context) error {
	w.mu.Lock()
	w.draining = true
	w.status = WorkerStatusFull
	active := len(w.jobs)
	w.mu.Unlock()
	w.sendStatus()
	if active == 0 {
		return nil
	}
	w.log.Info("draining, waiting for active jobs to finish", "active_jobs", active)

	ticker := time.NewTicker(200 * time.Millisecond)
	defer ticker.Stop()
	for {
		w.mu.Lock()
		remaining := len(w.jobs)
		w.mu.Unlock()
		if remaining == 0 {
			return nil
		}
		select {
		case <-ctx.Done():
			w.mu.Lock()
			for _, job := range w.jobs {
				job.cancel()
			}
			w.mu.Unlock()
			return ctx.Err()
		case <-ticker.C:
		}
	}
}

func (w *Worker) runConn(ctx context.Context) error {
	token, err := auth.NewAccessToken(w.opts.APIKey, w.opts.APISecret).
		SetVideoGrant(&auth.VideoGrant{Agent: true}).
		ToJWT()
	if err != nil {
		return fmt.Errorf("agents: build token: %w", err)
	}

	agentURL, err := buildAgentURL(w.opts.URL)
	if err != nil {
		return err
	}

	header := http.Header{"Authorization": {"Bearer " + token}}
	conn, resp, err := websocket.DefaultDialer.DialContext(ctx, agentURL, header)
	if err != nil {
		if resp != nil {
			return fmt.Errorf("agents: dial %s: %w (http %d)", agentURL, err, resp.StatusCode)
		}
		return fmt.Errorf("agents: dial %s: %w", agentURL, err)
	}
	defer conn.Close()
	w.log.Info("connected to LiveKit server", "url", agentURL)

	for len(w.sendCh) > 0 {
		<-w.sendCh
	}
	w.send(&livekit.WorkerMessage{Message: &livekit.WorkerMessage_Register{
		Register: &livekit.RegisterWorkerRequest{
			Type:               w.opts.JobType,
			AgentName:          w.opts.AgentName,
			Version:            Version,
			PingInterval:       uint32(w.opts.PingInterval / time.Second),
			AllowedPermissions: w.opts.Permissions,
		},
	}})

	g, ctx := errgroup.WithContext(ctx)
	g.Go(func() error {
		<-ctx.Done()
		conn.Close()
		return nil
	})
	g.Go(func() error { return w.writeLoop(ctx, conn) })
	g.Go(func() error { return w.readLoop(conn) })
	g.Go(func() error { return w.pingLoop(ctx) })
	g.Go(func() error { return w.statusLoop(ctx) })
	return g.Wait()
}

func (w *Worker) writeLoop(ctx context.Context, conn *websocket.Conn) error {
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case msg := <-w.sendCh:
			data, err := proto.Marshal(msg)
			if err != nil {
				return err
			}
			if err := conn.WriteMessage(websocket.BinaryMessage, data); err != nil {
				return err
			}
		}
	}
}

// this is the readloop which receives the message and will be the starting point for the worker to handle the message and take action based on the message type. The readloop will continuously read messages from the websocket connection and handle them accordingly.
func (w *Worker) readLoop(conn *websocket.Conn) error {
	for {
		msgType, data, err := conn.ReadMessage()
		if err != nil {
			return err
		}
		if msgType != websocket.BinaryMessage {
			continue
		}
		msg := &livekit.ServerMessage{}
		if err := proto.Unmarshal(data, msg); err != nil {
			w.log.Warn("failed to decode server message", "error", err)
			continue
		}
		w.handleMessage(msg)
	}
}

func (w *Worker) pingLoop(ctx context.Context) error {
	ticker := time.NewTicker(w.opts.PingInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			w.send(&livekit.WorkerMessage{Message: &livekit.WorkerMessage_Ping{
				Ping: &livekit.WorkerPing{Timestamp: time.Now().UnixMilli()},
			}})
		}
	}
}

func (w *Worker) statusLoop(ctx context.Context) error {
	ticker := time.NewTicker(updateStatusInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			w.sendStatus()
		}
	}
}

func (w *Worker) handleMessage(msg *livekit.ServerMessage) {
	switch m := msg.Message.(type) {
	case *livekit.ServerMessage_Register:
		w.handleRegister(m.Register)
	case *livekit.ServerMessage_Availability:
		w.handleAvailability(m.Availability)
	case *livekit.ServerMessage_Assignment:
		w.handleAssignment(m.Assignment)
	case *livekit.ServerMessage_Termination:
		w.handleTermination(m.Termination)
	case *livekit.ServerMessage_Pong:
		w.log.Debug("pong received",
			"rtt_ms", time.Now().UnixMilli()-m.Pong.GetLastTimestamp())
	}
}

func (w *Worker) handleRegister(res *livekit.RegisterWorkerResponse) {
	w.mu.Lock()
	w.id = res.GetWorkerId()
	w.mu.Unlock()
	w.log.Info("worker registered",
		"worker_id", res.GetWorkerId(),
		"server_version", res.GetServerInfo().GetVersion(),
		"protocol", res.GetServerInfo().GetProtocol())
	w.sendStatus()
}

func (w *Worker) handleAvailability(req *livekit.AvailabilityRequest) {
	job := req.GetJob()
	w.log.Info("job offer received",
		"job_id", job.GetId(),
		"job_type", job.GetType().String(),
		"room", job.GetRoom().GetName())

	w.mu.Lock()
	draining := w.draining
	w.mu.Unlock()

	r := &JobRequest{job: job}
	switch {
	case draining:
		_ = r.Reject()
	case w.opts.RequestFnc == nil:
		_ = r.Accept(JobAcceptInfo{})
	default:
		if err := w.opts.RequestFnc(r); err != nil {
			w.log.Warn("request handler failed, declining job", "job_id", job.GetId(), "error", err)
			r.answered = true
			r.accepted = false
		} else if !r.answered {
			_ = r.Accept(JobAcceptInfo{})
		}
	}

	if !r.accepted {
		w.send(&livekit.WorkerMessage{Message: &livekit.WorkerMessage_Availability{
			Availability: &livekit.AvailabilityResponse{JobId: job.GetId(), Available: false},
		}})
		w.log.Info("job declined", "job_id", job.GetId())
		return
	}

	info := r.accept
	w.mu.Lock()
	w.pending[job.GetId()] = info
	w.mu.Unlock()
	time.AfterFunc(assignmentTimeout, func() {
		w.mu.Lock()
		_, stale := w.pending[job.GetId()]
		delete(w.pending, job.GetId())
		w.mu.Unlock()
		if stale {
			w.log.Warn("job assignment never arrived", "job_id", job.GetId())
		}
	})

	w.send(&livekit.WorkerMessage{Message: &livekit.WorkerMessage_Availability{
		Availability: &livekit.AvailabilityResponse{
			JobId:                 job.GetId(),
			Available:             true,
			ParticipantIdentity:   info.Identity,
			ParticipantName:       info.Name,
			ParticipantMetadata:   info.Metadata,
			ParticipantAttributes: info.Attributes,
		},
	}})
	w.log.Info("job accepted", "job_id", job.GetId(), "identity", info.Identity)
}

func (w *Worker) handleAssignment(a *livekit.JobAssignment) {
	job := a.GetJob()
	id := job.GetId()

	w.mu.Lock()
	_, ok := w.pending[id]
	delete(w.pending, id)
	w.mu.Unlock()
	if !ok {
		w.log.Warn("assignment for unknown job, ignoring", "job_id", id)
		return
	}

	roomURL := a.GetUrl()
	if roomURL == "" {
		roomURL = w.opts.URL
	}
	ctx, cancel := context.WithCancel(context.Background())
	jobCtx := &JobContext{
		ctx:    ctx,
		cancel: cancel,
		job:    job,
		url:    roomURL,
		token:  a.GetToken(),
	}

	w.mu.Lock()
	w.jobs[id] = jobCtx
	w.mu.Unlock()
	w.log.Info("job assigned", "job_id", id, "room", job.GetRoom().GetName())

	go w.runJob(jobCtx)
}

func (w *Worker) handleTermination(t *livekit.JobTermination) {
	w.mu.Lock()
	jobCtx := w.jobs[t.GetJobId()]
	w.mu.Unlock()
	if jobCtx == nil {
		return
	}
	w.log.Info("job termination requested", "job_id", t.GetJobId())
	jobCtx.Shutdown()
}

func (w *Worker) runJob(jobCtx *JobContext) {
	id := jobCtx.job.GetId()
	defer func() {
		if r := recover(); r != nil {
			w.log.Error("job panicked", "job_id", id, "panic", r, "stack", string(debug.Stack()))
			w.updateJobStatus(id, livekit.JobStatus_JS_FAILED, fmt.Sprintf("panic: %v", r))
		}
		w.finishJob(jobCtx)
	}()

	w.updateJobStatus(id, livekit.JobStatus_JS_RUNNING, "")
	if err := w.opts.Entrypoint(jobCtx); err != nil {
		w.log.Error("job failed", "job_id", id, "error", err)
		w.updateJobStatus(id, livekit.JobStatus_JS_FAILED, err.Error())
		return
	}
	w.updateJobStatus(id, livekit.JobStatus_JS_SUCCESS, "")
	w.log.Info("job completed", "job_id", id)
}

func (w *Worker) finishJob(jobCtx *JobContext) {
	jobCtx.cancel()

	jobCtx.mu.Lock()
	fns := jobCtx.shutdownFns
	jobCtx.shutdownFns = nil
	room := jobCtx.room
	jobCtx.room = nil
	jobCtx.mu.Unlock()

	for _, fn := range fns {
		fn()
	}
	if room != nil {
		room.Disconnect()
	}

	w.mu.Lock()
	delete(w.jobs, jobCtx.job.GetId())
	w.mu.Unlock()
	w.sendStatus()
}

func (w *Worker) updateJobStatus(id string, status livekit.JobStatus, errMsg string) {
	w.send(&livekit.WorkerMessage{Message: &livekit.WorkerMessage_UpdateJob{
		UpdateJob: &livekit.UpdateJobStatus{JobId: id, Status: status, Error: errMsg},
	}})
}

func (w *Worker) sendStatus() {
	w.mu.Lock()
	status := w.status
	jobCount := uint32(len(w.jobs))
	w.mu.Unlock()
	var load float32
	if w.opts.LoadFnc != nil {
		load = w.opts.LoadFnc(w)
	}
	w.send(&livekit.WorkerMessage{Message: &livekit.WorkerMessage_UpdateWorker{
		UpdateWorker: &livekit.UpdateWorkerStatus{
			Status:   &status,
			Load:     load,
			JobCount: jobCount,
		},
	}})
}

func (w *Worker) send(msg *livekit.WorkerMessage) {
	select {
	case w.sendCh <- msg:
	default:
		w.log.Warn("send queue full, dropping message")
	}
}

func buildAgentURL(base string) (string, error) {
	u, err := url.Parse(base)
	if err != nil {
		return "", fmt.Errorf("agents: invalid url %q: %w", base, err)
	}
	switch u.Scheme {
	case "http":
		u.Scheme = "ws"
	case "https":
		u.Scheme = "wss"
	case "ws", "wss":
	default:
		return "", fmt.Errorf("agents: unsupported url scheme %q", u.Scheme)
	}
	u.Path = strings.TrimSuffix(u.Path, "/") + "/agent"
	return u.String(), nil
}
