package voice

import (
	"context"
	"errors"

	"github.com/beingPro007/lk-go-agent-sdk/audio"
	"github.com/beingPro007/lk-go-agent-sdk/llm"
	"github.com/beingPro007/lk-go-agent-sdk/stt"
	"github.com/beingPro007/lk-go-agent-sdk/tts"
)

func (s *AgentSession) sttNode(agent *Agent) STTNode {
	if agent != nil && agent.STTNode != nil {
		return agent.STTNode
	}
	return s.defaultSTTNode
}

func (s *AgentSession) llmNode(agent *Agent) LLMNode {
	if agent != nil && agent.LLMNode != nil {
		return agent.LLMNode
	}
	return s.defaultLLMNode
}

func (s *AgentSession) ttsNode(agent *Agent) TTSNode {
	if agent != nil && agent.TTSNode != nil {
		return agent.TTSNode
	}
	return s.defaultTTSNode
}

func (s *AgentSession) defaultSTTNode(ctx context.Context, frames <-chan audio.Frame) (<-chan stt.SpeechEvent, error) {
	if s.stt == nil {
		return nil, errors.New("voice: no STT configured")
	}
	stream := s.stt.Stream(ctx)
	if err := stream.Err(); err != nil {
		stream.Close()
		return nil, err
	}
	go func() {
		for f := range frames {
			stream.PushFrame(f)
		}
		stream.EndInput()
	}()
	out := make(chan stt.SpeechEvent, 32)
	go func() {
		defer close(out)
		defer stream.Close()
		for ev := range stream.Chan() {
			select {
			case out <- ev:
			case <-ctx.Done():
				return
			}
		}
		s.emitErr(stream.Err())
	}()
	return out, nil
}

func (s *AgentSession) defaultLLMNode(ctx context.Context, chatCtx *llm.ChatContext, tools []llm.Tool) (<-chan llm.ChatChunk, error) {
	if s.llm == nil {
		return nil, errors.New("voice: no LLM configured")
	}
	stream := s.llm.Chat(ctx, chatCtx, tools)
	out := make(chan llm.ChatChunk, 32)
	go func() {
		defer close(out)
		defer stream.Close()
		for c := range stream.Chan() {
			select {
			case out <- c:
			case <-ctx.Done():
				return
			}
		}
		s.emitErr(stream.Err())
	}()
	return out, nil
}

func (s *AgentSession) defaultTTSNode(ctx context.Context, text <-chan string) (<-chan tts.SynthesizedAudio, error) {
	if s.tts == nil {
		return nil, errors.New("voice: no TTS configured")
	}
	stream := s.tts.Stream(ctx)
	if err := stream.Err(); err != nil {
		stream.Close()
		return nil, err
	}
	go func() {
		for t := range text {
			stream.PushText(t)
		}
		stream.EndInput()
	}()
	out := make(chan tts.SynthesizedAudio, 32)
	go func() {
		defer close(out)
		defer stream.Close()
		for a := range stream.Chan() {
			select {
			case out <- a:
			case <-ctx.Done():
				return
			}
		}
		s.emitErr(stream.Err())
	}()
	return out, nil
}
