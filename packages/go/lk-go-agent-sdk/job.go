package agents

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/livekit/protocol/livekit"
	lksdk "github.com/livekit/server-sdk-go/v2"
)

type JobAcceptInfo struct {
	Name       string
	Identity   string
	Metadata   string
	Attributes map[string]string
}

type JobRequest struct {
	job      *livekit.Job
	answered bool
	accepted bool
	accept   JobAcceptInfo
}

func (r *JobRequest) Job() *livekit.Job {
	return r.job
}

func (r *JobRequest) Room() *livekit.Room {
	return r.job.GetRoom()
}

func (r *JobRequest) Accept(info JobAcceptInfo) error {
	if r.answered {
		return errors.New("agents: job request already answered")
	}
	if info.Identity == "" {
		info.Identity = "agent-" + r.job.GetId()
	}
	r.answered = true
	r.accepted = true
	r.accept = info
	return nil
}

func (r *JobRequest) Reject() error {
	if r.answered {
		return errors.New("agents: job request already answered")
	}
	r.answered = true
	return nil
}

type JobContext struct {
	ctx    context.Context
	cancel context.CancelFunc

	job   *livekit.Job
	url   string
	token string

	mu          sync.Mutex
	room        *lksdk.Room
	shutdownFns []func()
}

func (c *JobContext) Context() context.Context {
	return c.ctx
}

func (c *JobContext) Done() <-chan struct{} {
	return c.ctx.Done()
}

func (c *JobContext) Job() *livekit.Job {
	return c.job
}

func (c *JobContext) Room() *lksdk.Room {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.room
}

func (c *JobContext) Connect(callbacks ...*lksdk.RoomCallback) (*lksdk.Room, error) {
	c.mu.Lock()
	if c.room != nil {
		room := c.room
		c.mu.Unlock()
		return room, nil
	}
	c.mu.Unlock()

	cb := lksdk.NewRoomCallback()
	for _, extra := range callbacks {
		cb.Merge(extra)
	}
	userDisconnect := cb.OnDisconnected
	cb.OnDisconnected = func() {
		if userDisconnect != nil {
			userDisconnect()
		}
		c.Shutdown()
	}

	room, err := lksdk.ConnectToRoomWithToken(c.url, c.token, cb)
	if err != nil {
		return nil, fmt.Errorf("agents: connect to room: %w", err)
	}
	c.mu.Lock()
	c.room = room
	c.mu.Unlock()
	return room, nil
}

func (c *JobContext) WaitForParticipant(ctx context.Context, identity string) (*lksdk.RemoteParticipant, error) {
	room := c.Room()
	if room == nil {
		return nil, errors.New("agents: not connected, call Connect first")
	}
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()
	for {
		for _, p := range room.GetRemoteParticipants() {
			if identity == "" || p.Identity() == identity {
				return p, nil
			}
		}
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-c.ctx.Done():
			return nil, errors.New("agents: job shut down")
		case <-ticker.C:
		}
	}
}

func (c *JobContext) AddShutdownCallback(fn func()) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.shutdownFns = append(c.shutdownFns, fn)
}

func (c *JobContext) Shutdown() {
	c.cancel()
}
