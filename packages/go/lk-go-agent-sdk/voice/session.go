package voice

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/beingPro007/lk-go-agent-sdk/audio"
	"github.com/beingPro007/lk-go-agent-sdk/llm"
	"github.com/beingPro007/lk-go-agent-sdk/stt"
	"github.com/beingPro007/lk-go-agent-sdk/tts"
	"github.com/beingPro007/lk-go-agent-sdk/vad"
)

const (
	defaultMaxToolSteps        = 5
	defaultTurnThreshold       = 0.5
	defaultMaxEndpointingDelay = 3 * time.Second
	defaultMinEndpointingDelay = 500 * time.Millisecond
	turnTranscribeTimeout      = 5 * time.Second
	resumeDelay                = time.Second // a cut, unheard turn is answered when no words follow in this time
	voicedLine                 = 0.5         // a VAD window at or above this probability has voice
	voicedRecent               = 100 * time.Millisecond // about three Silero windows
)

type AgentSession struct {
	stt stt.STT
	llm llm.LLM
	tts tts.TTS
	vad vad.VAD

	turnDetector        TurnDetector
	turnThreshold       float64
	minEndpointingDelay time.Duration
	maxEndpointingDelay time.Duration
	preemptive          bool
	maxToolSteps        int
	minInterruption     time.Duration

	dropped    atomic.Uint64 // input frames dropped because a consumer was too slow
	lastVoiced atomic.Int64  // unix nanoseconds of the last VAD window with voice; 0 before any

	events chan Event

	ctxMu   sync.RWMutex
	chatCtx *llm.ChatContext

	genMu sync.Mutex

	mu      sync.Mutex
	state   AgentState
	current *SpeechHandle
	agent   *Agent
	out     AudioOutput
}

type SessionOption func(*AgentSession)

func WithSTT(s stt.STT) SessionOption { return func(a *AgentSession) { a.stt = s } }
func WithLLM(l llm.LLM) SessionOption { return func(a *AgentSession) { a.llm = l } }
func WithTTS(t tts.TTS) SessionOption { return func(a *AgentSession) { a.tts = t } }
func WithVAD(v vad.VAD) SessionOption { return func(a *AgentSession) { a.vad = v } }

func WithTurnDetector(d TurnDetector, threshold float64) SessionOption {
	return func(a *AgentSession) {
		a.turnDetector = d
		if threshold > 0 {
			a.turnThreshold = threshold
		}
	}
}

func WithMaxEndpointingDelay(d time.Duration) SessionOption {
	return func(a *AgentSession) { a.maxEndpointingDelay = d }
}

func WithMinEndpointingDelay(d time.Duration) SessionOption {
	return func(a *AgentSession) { a.minEndpointingDelay = d }
}

func WithPreemptiveGeneration(enabled bool) SessionOption {
	return func(a *AgentSession) { a.preemptive = enabled }
}

func WithMaxToolSteps(n int) SessionOption {
	return func(a *AgentSession) { a.maxToolSteps = n }
}

// WithMinInterruptionDuration makes the user speak this long, by the VAD,
// before their speech interrupts an agent that thinks or speaks. Shorter
// speech (an echo of the agent, a cough, "mm-hmm") is ignored, and its
// transcript is dropped (EventUserSpeechIgnored). Zero, the default,
// interrupts at the first sign of speech. LiveKit Agents uses 500 ms.
func WithMinInterruptionDuration(d time.Duration) SessionOption {
	return func(a *AgentSession) { a.minInterruption = d }
}

func NewAgentSession(opts ...SessionOption) *AgentSession {
	s := &AgentSession{
		chatCtx:             &llm.ChatContext{},
		events:              make(chan Event, 128),
		state:               StateListening,
		turnThreshold:       defaultTurnThreshold,
		minEndpointingDelay: defaultMinEndpointingDelay,
		maxEndpointingDelay: defaultMaxEndpointingDelay,
		maxToolSteps:        defaultMaxToolSteps,
	}
	for _, o := range opts {
		o(s)
	}
	return s
}

func (s *AgentSession) Events() <-chan Event { return s.events }

// DroppedFrames counts the input frames that the session dropped because the
// STT or the VAD did not take them in time. More than zero means gaps in the
// audio that the STT hears.
func (s *AgentSession) DroppedFrames() uint64 { return s.dropped.Load() }

func (s *AgentSession) ChatContext() *llm.ChatContext { return s.chatCtx }

func (s *AgentSession) State() AgentState {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.state
}

func (s *AgentSession) CurrentSpeech() *SpeechHandle {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.current
}

func (s *AgentSession) Start(ctx context.Context, agent *Agent, in AudioInput, out AudioOutput) error {
	if s.stt == nil || s.llm == nil || s.tts == nil {
		return errors.New("voice: STT, LLM and TTS are required")
	}

	s.mu.Lock()
	s.agent = agent
	s.out = out
	s.mu.Unlock()

	if agent.Instructions != "" {
		s.ctxMu.Lock()
		s.chatCtx.Add(llm.RoleSystem, agent.Instructions)
		s.ctxMu.Unlock()
	}

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	sttFrames := make(chan audio.Frame, 64)
	var vadFrames chan audio.Frame
	if s.vad != nil {
		vadFrames = make(chan audio.Frame, 64)
	}
	// An AudioTurnDetector hears the turn, and an STT that can also recognize
	// a whole stretch of audio transcribes the turn in one call at commit:
	// both need the turn's audio.
	_, hears := s.turnDetector.(AudioTurnDetector)
	recognizer, _ := s.stt.(stt.Recognizer)
	var heard *turnAudio
	if hears || recognizer != nil {
		heard = newTurnAudio(turnAudioMax)
	}
	go func() {
		defer close(sttFrames)
		if vadFrames != nil {
			defer close(vadFrames)
		}
		for f := range in.Frames() {
			s.pushFrame(sttFrames, f)
			if vadFrames != nil {
				s.pushFrame(vadFrames, f)
			}
			if heard != nil {
				heard.push(f)
			}
		}
	}()

	vadTurns := make(chan vad.EventType, 8)
	if s.vad != nil {
		vadStream := s.vad.Stream(ctx)
		defer vadStream.Close()
		go func() {
			for f := range vadFrames {
				vadStream.PushFrame(f)
			}
			vadStream.EndInput()
		}()
		go s.runVAD(vadStream, vadTurns)
	}

	events, err := s.sttNode(agent)(ctx, sttFrames)
	if err != nil {
		return err
	}

	var userText strings.Builder
	var pending *SpeechHandle
	var pendingText string
	var timer *time.Timer
	var timerC <-chan time.Time
	userSpeaking := false

	// The endpointing decision. gen changes whenever the turn changes (new
	// text, new speech, a commit), so a detector answer about an older state
	// of the turn is dropped.
	var (
		gen            int
		inFlight       bool
		commitOnResult bool
		known          bool
		knownP         float64
	)
	results := make(chan turnResult, 4)
	bump := func() {
		gen++
		inFlight, known, commitOnResult = false, false, false
	}

	stopTimer := func() {
		if timer != nil {
			timer.Stop()
			timer = nil
			timerC = nil
		}
	}
	arm := func(d time.Duration) {
		stopTimer()
		timer = time.NewTimer(d)
		timerC = timer.C
	}
	defer stopTimer()

	// When the user talks again before any audio of the reply played, the
	// reply was never heard: carry holds that turn's text, and the next commit
	// sends both parts as one turn. lastHandle and lastText are the latest
	// committed turn.
	var carry, lastText string
	var lastHandle *SpeechHandle
	turnText := func() string { return strings.TrimSpace(carry + " " + strings.TrimSpace(userText.String())) }
	continues := func() {
		if lastHandle != nil && lastHandle.Interrupted() && !lastHandle.Played() {
			carry = lastText
		}
		lastHandle = nil
	}

	// While the agent thinks or speaks, speech must last minInterruption
	// before it counts. ignoring is true from such a start of speech until the
	// speech proves long enough; its transcript is dropped.
	ignoring := false
	var intTimer, resumeTimer *time.Timer
	var intC, resumeC <-chan time.Time
	stopInt := func() {
		if intTimer != nil {
			intTimer.Stop()
			intTimer, intC = nil, nil
		}
	}
	stopResume := func() {
		if resumeTimer != nil {
			resumeTimer.Stop()
			resumeTimer, resumeC = nil, nil
		}
	}
	defer stopInt()
	defer stopResume()
	agentActive := func() bool {
		h := s.CurrentSpeech()
		if h == nil || h.Interrupted() {
			return false
		}
		select {
		case <-h.Done():
			return false
		default:
			return true
		}
	}
	// voiced reports whether the user has voice right now, not only the VAD's
	// silence hangover after a short word. A VAD that sends no inference
	// events counts as voiced.
	voiced := func() bool {
		last := s.lastVoiced.Load()
		return last == 0 || time.Since(time.Unix(0, last)) < voicedRecent
	}
	var speechStart time.Time
	turnAudio := func() []audio.Frame {
		if heard == nil || recognizer == nil {
			return nil
		}
		return heard.snapshot()
	}

	commit := func() {
		stopTimer()
		stopResume()
		bump()
		if heard != nil && s.vad == nil {
			defer heard.keepLast(0) // without a VAD, the next turn starts now
		}
		hasText := strings.TrimSpace(userText.String()) != ""
		final := turnText()
		replaces := carry != ""
		userText.Reset()
		if !hasText {
			if pending != nil {
				pending.Interrupt()
				pending = nil
			}
			return
		}
		carry = ""
		if pending != nil && pendingText == final {
			pending.Commit()
			lastHandle, lastText = pending, final
			pending = nil
			return
		}
		if pending != nil {
			pending.Interrupt()
			pending = nil
		}
		in := turnInput{text: final, audio: turnAudio(), replaces: replaces}
		lastHandle, lastText = s.startGeneration(ctx, agent, out, in, true), final
	}

	// resume answers a turn whose reply was cut before anyone heard it, when
	// the speech that cut it brought no words (a false interruption).
	resume := func() {
		if carry == "" || userSpeaking || strings.TrimSpace(userText.String()) != "" {
			return
		}
		final := carry
		carry = ""
		bump()
		in := turnInput{text: final, audio: turnAudio(), replaces: true}
		lastHandle, lastText = s.startGeneration(ctx, agent, out, in, true), final
	}

	// apply turns a probability into a timer, or into a commit at once when
	// the STT has already ended the speech.
	apply := func(p float64, immediate bool) {
		if p >= s.turnThreshold {
			if immediate {
				commit()
			} else {
				arm(s.minEndpointingDelay)
			}
			return
		}
		arm(s.maxEndpointingDelay)
	}

	// decide runs when the user is quiet with text in hand. immediate is true
	// when the STT reported the end of speech. The detector runs off the
	// event loop; its answer comes back on results.
	decide := func(immediate bool) {
		if strings.TrimSpace(userText.String()) == "" {
			return
		}
		text := turnText()
		if s.turnDetector == nil {
			if immediate {
				commit()
			} else {
				arm(s.minEndpointingDelay)
			}
			return
		}
		if known {
			apply(knownP, immediate)
			return
		}
		if inFlight {
			commitOnResult = commitOnResult || immediate
			return
		}
		inFlight, commitOnResult = true, immediate
		arm(s.maxEndpointingDelay) // the turn still commits if the detector never answers
		g := gen
		var turn []audio.Frame
		if heard != nil {
			turn = heard.snapshot()
		}
		go func() {
			p, err := s.predict(ctx, text, turn)
			select {
			case results <- turnResult{gen: g, p: p, err: err}:
			case <-ctx.Done():
			}
		}()
	}

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()

		case <-timerC:
			commit()

		case <-intC:
			intTimer, intC = nil, nil
			if userSpeaking && ignoring && voiced() {
				ignoring = false // long enough: this is the user, not an echo
				s.interrupt()
				continues()
				if heard != nil && carry == "" && strings.TrimSpace(userText.String()) == "" {
					heard.keepLast(time.Since(speechStart) + turnAudioPreroll) // a new turn: keep only its audio
				}
			}

		case <-resumeC:
			resumeTimer, resumeC = nil, nil
			resume()

		case r := <-results:
			if r.gen != gen || userSpeaking {
				continue // the turn changed, or the user talks again: the answer is stale
			}
			inFlight = false
			if r.err != nil {
				s.emitErr(r.err)
				r.p = 1 // a failed detector must not hold the turn open
			}
			known, knownP = true, r.p
			apply(r.p, commitOnResult)

		case t := <-vadTurns:
			switch t {
			case vad.StartOfSpeech:
				userSpeaking = true
				stopTimer()
				stopResume()
				bump()
				speechStart = time.Now()
				if s.minInterruption > 0 && agentActive() {
					// the audio stays until the speech proves to be a turn: it may continue the unheard turn
					ignoring = true
					intTimer = time.NewTimer(s.minInterruption)
					intC = intTimer.C
				} else {
					ignoring = false
					s.interrupt()
					continues()
					if heard != nil && strings.TrimSpace(userText.String()) == "" && carry == "" {
						heard.keepLast(turnAudioPreroll) // a new turn starts: drop the audio before it
					}
				}
			case vad.EndOfSpeech:
				userSpeaking = false
				stopInt()
				if ignoring {
					continue // too short to interrupt the agent: not a turn
				}
				decide(false)
				if carry != "" && strings.TrimSpace(userText.String()) == "" {
					stopResume()
					resumeTimer = time.NewTimer(resumeDelay)
					resumeC = resumeTimer.C
				}
			}

		case ev, ok := <-events:
			if !ok {
				s.waitCurrent(ctx)
				return nil
			}
			text := firstAlt(ev)
			switch ev.Type {
			case stt.StartOfSpeech:
				if s.vad == nil {
					s.emit(Event{Type: EventUserStartedSpeaking})
					s.interrupt()
					continues()
					if inFlight || known {
						stopTimer() // the user talks on: the held turn waits for the next end of speech
						bump()
					}
				}

			case stt.InterimTranscript:
				s.emit(Event{Type: EventUserTranscript, Transcript: text, IsFinal: false})

			case stt.FinalTranscript:
				if text == "" {
					continue
				}
				if ignoring {
					s.emit(Event{Type: EventUserSpeechIgnored, Transcript: text, IsFinal: true})
					continue
				}
				stopTimer()
				stopResume()
				bump()
				userText.WriteString(text)
				userText.WriteString(" ")
				s.emit(Event{Type: EventUserTranscript, Transcript: text, IsFinal: true})
				if s.vad != nil && !userSpeaking {
					decide(false)
				}
				if s.preemptive {
					if pending != nil {
						pending.Interrupt()
						pending = nil
					}
					pendingText = turnText()
					pending = s.startGeneration(ctx, agent, out, turnInput{text: pendingText}, false)
				}

			case stt.EndOfSpeech:
				decide(true)
			}
		}
	}
}

func (s *AgentSession) Say(ctx context.Context, text string) (*SpeechHandle, error) {
	if s.tts == nil {
		return nil, errors.New("voice: no TTS configured")
	}
	s.mu.Lock()
	out, agent := s.out, s.agent
	s.mu.Unlock()
	if out == nil {
		return nil, errors.New("voice: session not started")
	}

	genCtx, cancel := context.WithCancel(ctx)
	h := newSpeechHandle(cancel, true)
	h.setText(text)
	s.setCurrent(h)

	go func() {
		defer cancel()
		defer h.finish()
		s.genMu.Lock()
		defer s.genMu.Unlock()
		if genCtx.Err() != nil {
			return
		}
		local := s.snapshot()
		spoken := s.speak(genCtx, h, agent, out, singleText(text))
		if spoken != "" {
			local.Add(llm.RoleAssistant, spoken)
			s.commitChat(local)
		}
		s.emit(Event{Type: EventAgentStoppedSpeaking, SpeechID: h.ID()})
		s.setState(StateListening)
	}()
	return h, nil
}

func (s *AgentSession) Interrupt() { s.interrupt() }

// turnInput is one committed user turn: its text from the STT pieces, its
// audio when the turn is transcribed again as a whole, and whether it replaces
// the previous turn (see llm.Message.Replaces).
type turnInput struct {
	text     string
	audio    []audio.Frame
	replaces bool
}

func (s *AgentSession) startGeneration(ctx context.Context, agent *Agent, out AudioOutput, in turnInput, committed bool) *SpeechHandle {
	genCtx, cancel := context.WithCancel(ctx)
	h := newSpeechHandle(cancel, committed)
	s.setCurrent(h)

	go func() {
		defer cancel()
		defer h.finish()
		s.generate(genCtx, h, agent, out, in)
	}()
	return h
}

// transcribe returns the text of the whole turn from its audio, or the text
// of the pieces when the STT cannot do that or fails.
func (s *AgentSession) transcribe(ctx context.Context, in turnInput) string {
	rec, ok := s.stt.(stt.Recognizer)
	if !ok || len(in.audio) == 0 {
		return in.text
	}
	tctx, cancel := context.WithTimeout(ctx, turnTranscribeTimeout)
	defer cancel()
	data, err := rec.Recognize(tctx, in.audio)
	if err != nil {
		if ctx.Err() == nil {
			s.emitErr(fmt.Errorf("voice: transcribe the whole turn (the text of the pieces is used): %w", err))
		}
		return in.text
	}
	if t := strings.TrimSpace(data.Text); t != "" {
		return t
	}
	return in.text
}

func (s *AgentSession) generate(ctx context.Context, h *SpeechHandle, agent *Agent, out AudioOutput, in turnInput) {
	s.genMu.Lock()
	defer s.genMu.Unlock()
	if ctx.Err() != nil {
		return
	}

	s.setState(StateThinking)
	userText := s.transcribe(ctx, in)
	if ctx.Err() != nil {
		s.setState(StateListening)
		return
	}
	s.emit(Event{Type: EventUserTurn, Transcript: userText, IsFinal: true, SpeechID: h.ID()})
	local := s.snapshot()
	local.Messages = append(local.Messages, llm.Message{Role: llm.RoleUser, Content: userText, Replaces: in.replaces})

	textCh := make(chan string, 64)
	audioCh, err := s.ttsNode(agent)(ctx, textCh)
	if err != nil {
		close(textCh)
		s.emitErr(err)
		s.setState(StateListening)
		return
	}

	audioDone := make(chan struct{})
	go s.pumpAudio(ctx, h, out, audioCh, audioDone)

	node := s.llmNode(agent)
	var spoken strings.Builder

	for step := 0; ; step++ {
		chunks, err := node(ctx, local, agent.Tools)
		if err != nil {
			s.emitErr(err)
			break
		}
		var round strings.Builder
		var calls []llm.ToolCall
		for c := range chunks {
			if ctx.Err() != nil {
				break
			}
			if d := c.Delta.Content; d != "" {
				round.WriteString(d)
				spoken.WriteString(d)
				s.emit(Event{Type: EventAgentResponse, Text: d, SpeechID: h.ID()})
				select {
				case textCh <- d:
				case <-ctx.Done():
				}
			}
			calls = append(calls, c.Delta.ToolCalls...)
		}
		if ctx.Err() != nil {
			if round.Len() > 0 {
				local.Add(llm.RoleAssistant, round.String())
			}
			break
		}
		if len(calls) == 0 {
			if round.Len() > 0 {
				local.Add(llm.RoleAssistant, round.String())
			}
			break
		}
		if step >= s.maxToolSteps {
			s.emitErr(errors.New("voice: max tool steps exceeded"))
			break
		}
		local.Messages = append(local.Messages, llm.Message{
			Role:      llm.RoleAssistant,
			Content:   round.String(),
			ToolCalls: calls,
		})
		for _, call := range calls {
			local.AddToolResult(call, s.runTool(ctx, agent, call))
		}
	}

	close(textCh)
	<-audioDone
	if ctx.Err() == nil {
		out.WaitForPlayout()
	}

	h.setText(spoken.String())
	// A reply cut off before any of its audio played was never heard. It stays
	// out of the chat, and its user text joins the next turn (see Start).
	if h.Committed() && (ctx.Err() == nil || h.Played()) {
		s.commitChat(local)
	}
	s.emit(Event{Type: EventAgentStoppedSpeaking, SpeechID: h.ID()})
	s.setState(StateListening)
}

func (s *AgentSession) speak(ctx context.Context, h *SpeechHandle, agent *Agent, out AudioOutput, text <-chan string) string {
	audioCh, err := s.ttsNode(agent)(ctx, text)
	if err != nil {
		s.emitErr(err)
		return ""
	}
	audioDone := make(chan struct{})
	go s.pumpAudio(ctx, h, out, audioCh, audioDone)
	<-audioDone
	if ctx.Err() != nil {
		return ""
	}
	out.WaitForPlayout()
	return h.Text()
}

func (s *AgentSession) pumpAudio(ctx context.Context, h *SpeechHandle, out AudioOutput, in <-chan tts.SynthesizedAudio, done chan struct{}) {
	defer close(done)
	started := false
	for a := range in {
		if len(a.Frame.Data) == 0 {
			continue
		}
		if !started {
			select {
			case <-h.committed:
			case <-ctx.Done():
				return
			}
			started = true
			h.played.Store(true)
			s.setState(StateSpeaking)
			s.emit(Event{Type: EventAgentStartedSpeaking, SpeechID: h.ID()})
		}
		if err := out.CaptureFrame(a.Frame); err != nil {
			s.emitErr(err)
			return
		}
	}
}

func (s *AgentSession) runTool(ctx context.Context, agent *Agent, call llm.ToolCall) string {
	s.emit(Event{Type: EventToolCall, ToolCall: call, Text: call.Arguments})
	tool, ok := agent.tool(call.Name)
	if !ok || tool.Handler == nil {
		result := "error: unknown tool " + call.Name
		s.emit(Event{Type: EventToolResult, ToolCall: call, Text: result})
		return result
	}
	result, err := tool.Handler(ctx, call.Arguments)
	if err != nil {
		result = "error: " + err.Error()
	}
	s.emit(Event{Type: EventToolResult, ToolCall: call, Text: result})
	return result
}

func (s *AgentSession) runVAD(stream vad.Stream, turns chan<- vad.EventType) {
	for ev := range stream.Chan() {
		switch ev.Type {
		case vad.InferenceDone:
			if ev.Probability >= voicedLine {
				s.lastVoiced.Store(time.Now().UnixNano())
			}
		case vad.StartOfSpeech:
			s.emit(Event{Type: EventUserStartedSpeaking})
			notifyTurn(turns, vad.StartOfSpeech) // the event loop decides whether to interrupt
		case vad.EndOfSpeech:
			s.emit(Event{Type: EventUserStoppedSpeaking})
			notifyTurn(turns, vad.EndOfSpeech)
		}
	}
	s.emitErr(stream.Err())
}

func notifyTurn(turns chan<- vad.EventType, t vad.EventType) {
	select {
	case turns <- t:
	default:
	}
}

func (s *AgentSession) interrupt() {
	s.mu.Lock()
	h, out := s.current, s.out
	s.mu.Unlock()
	if h == nil {
		return
	}
	select {
	case <-h.Done():
		return
	default:
	}
	h.Interrupt()
	if out != nil {
		out.ClearBuffer()
	}
}

func (s *AgentSession) setCurrent(h *SpeechHandle) {
	s.mu.Lock()
	prev := s.current
	s.current = h
	s.mu.Unlock()
	if prev != nil && prev != h {
		prev.Interrupt()
	}
}

func (s *AgentSession) waitCurrent(ctx context.Context) {
	s.mu.Lock()
	h := s.current
	s.mu.Unlock()
	if h == nil {
		return
	}
	select {
	case <-h.Done():
	case <-ctx.Done():
	}
}

func (s *AgentSession) snapshot() *llm.ChatContext {
	s.ctxMu.RLock()
	defer s.ctxMu.RUnlock()
	return s.chatCtx.Clone()
}

func (s *AgentSession) commitChat(local *llm.ChatContext) {
	s.ctxMu.Lock()
	s.chatCtx.Messages = local.Messages
	s.ctxMu.Unlock()
}

func (s *AgentSession) setState(st AgentState) {
	s.mu.Lock()
	changed := s.state != st
	s.state = st
	s.mu.Unlock()
	if changed {
		s.emit(Event{Type: EventStateChanged, State: st})
	}
}

func (s *AgentSession) emit(ev Event) {
	select {
	case s.events <- ev:
	default:
	}
}

func (s *AgentSession) emitErr(err error) {
	if err == nil {
		return
	}
	s.emit(Event{Type: EventError, Err: err})
}

func (s *AgentSession) pushFrame(ch chan audio.Frame, f audio.Frame) {
	select {
	case ch <- f:
	default:
		s.dropped.Add(1)
	}
}

func singleText(text string) <-chan string {
	ch := make(chan string, 1)
	ch <- text
	close(ch)
	return ch
}

func firstAlt(ev stt.SpeechEvent) string {
	if len(ev.Alternatives) == 0 {
		return ""
	}
	return ev.Alternatives[0].Text
}
