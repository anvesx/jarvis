package voice

import (
	"context"

	"github.com/beingPro007/lk-go-agent-sdk/audio"
	"github.com/beingPro007/lk-go-agent-sdk/llm"
	"github.com/beingPro007/lk-go-agent-sdk/stt"
	"github.com/beingPro007/lk-go-agent-sdk/tts"
)

type STTNode func(ctx context.Context, frames <-chan audio.Frame) (<-chan stt.SpeechEvent, error)

type LLMNode func(ctx context.Context, chatCtx *llm.ChatContext, tools []llm.Tool) (<-chan llm.ChatChunk, error)

type TTSNode func(ctx context.Context, text <-chan string) (<-chan tts.SynthesizedAudio, error)

type Agent struct {
	Instructions string
	Tools        []llm.Tool
	STTNode      STTNode
	LLMNode      LLMNode
	TTSNode      TTSNode
}

func (a *Agent) tool(name string) (llm.Tool, bool) {
	for _, t := range a.Tools {
		if t.Name == name {
			return t, true
		}
	}
	return llm.Tool{}, false
}
