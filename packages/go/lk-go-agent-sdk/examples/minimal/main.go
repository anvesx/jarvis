package main

import (
	"log/slog"

	agents "github.com/beingPro007/lk-go-agent-sdk"
	"github.com/beingPro007/lk-go-agent-sdk/cli"
)

func main() {
	cli.Run(agents.WorkerOptions{
		Entrypoint: func(job *agents.JobContext) error {
			room, err := job.Connect()
			if err != nil {
				return err
			}
			slog.Info("agent joined room", "room", room.Name(), "job_id", job.Job().GetId())
			<-job.Done()
			return nil
		},
	})
}
