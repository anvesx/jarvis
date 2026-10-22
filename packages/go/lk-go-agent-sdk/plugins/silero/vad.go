package silero

import (
	"context"
	_ "embed"
	"errors"
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/beingPro007/lk-go-agent-sdk/audio"
	"github.com/beingPro007/lk-go-agent-sdk/vad"
	ort "github.com/yalue/onnxruntime_go"
)

//go:embed silero_vad.onnx
var modelData []byte

var (
	envOnce sync.Once
	envErr  error
)

func initEnv(libPath string) error {
	envOnce.Do(func() {
		if libPath != "" {
			ort.SetSharedLibraryPath(libPath)
		}
		envErr = ort.InitializeEnvironment()
	})
	return envErr
}

type config struct {
	libPath    string
	threshold  float32
	negThresh  float32
	sampleRate int
	minSpeech  time.Duration
	minSilence time.Duration
}

type Option func(*config)

func WithLibraryPath(p string) Option { return func(c *config) { c.libPath = p } }
func WithThreshold(t float32) Option  { return func(c *config) { c.threshold = t } }

// WithNegativeThreshold sets the line below which speech ends (hysteresis).
// The default is threshold - 0.15 (at least 0.01), as the Silero authors
// advise: a dip between words then does not end the speech.
func WithNegativeThreshold(t float32) Option        { return func(c *config) { c.negThresh = t } }
func WithSampleRate(r int) Option                   { return func(c *config) { c.sampleRate = r } }
func WithMinSpeechDuration(d time.Duration) Option  { return func(c *config) { c.minSpeech = d } }
func WithMinSilenceDuration(d time.Duration) Option { return func(c *config) { c.minSilence = d } }

type SileroVAD struct {
	session     *ort.DynamicAdvancedSession
	mu          sync.Mutex
	threshold   float32
	negThresh   float32
	sampleRate  int
	windowSize  int
	contextSize int

	minSpeechSamples  int
	minSilenceSamples int

	modelPath string
}

func New(opts ...Option) (*SileroVAD, error) {
	c := &config{
		libPath:    os.Getenv("ONNXRUNTIME_LIB_PATH"),
		threshold:  0.5,
		sampleRate: 16000,
		minSpeech:  250 * time.Millisecond,
		minSilence: 100 * time.Millisecond,
	}
	for _, o := range opts {
		o(c)
	}
	if c.negThresh <= 0 {
		c.negThresh = max(c.threshold-0.15, 0.01)
	}

	var windowSize, contextSize int
	switch c.sampleRate {
	case 16000:
		windowSize, contextSize = 512, 64
	case 8000:
		windowSize, contextSize = 256, 32
	default:
		return nil, errors.New("silero: sample rate must be 16000 or 8000")
	}

	if err := initEnv(c.libPath); err != nil {
		return nil, fmt.Errorf("silero: init onnxruntime (set ONNXRUNTIME_LIB_PATH or WithLibraryPath): %w", err)
	}

	modelPath, err := writeModel()
	if err != nil {
		return nil, err
	}
	session, err := ort.NewDynamicAdvancedSession(modelPath,
		[]string{"input", "state", "sr"},
		[]string{"output", "stateN"}, nil)
	if err != nil {
		os.Remove(modelPath)
		return nil, fmt.Errorf("silero: create session: %w", err)
	}

	return &SileroVAD{
		session:           session,
		threshold:         c.threshold,
		negThresh:         c.negThresh,
		sampleRate:        c.sampleRate,
		windowSize:        windowSize,
		contextSize:       contextSize,
		minSpeechSamples:  int(c.minSpeech.Seconds() * float64(c.sampleRate)),
		minSilenceSamples: int(c.minSilence.Seconds() * float64(c.sampleRate)),
		modelPath:         modelPath,
	}, nil
}

func (v *SileroVAD) Close() error {
	err := v.session.Destroy()
	os.Remove(v.modelPath)
	return err
}

func (v *SileroVAD) infer(input, state *ort.Tensor[float32], sr *ort.Tensor[int64], output, stateN *ort.Tensor[float32]) float32 {
	v.mu.Lock()
	defer v.mu.Unlock()
	if err := v.session.Run([]ort.Value{input, state, sr}, []ort.Value{output, stateN}); err != nil {
		return 0
	}
	return output.GetData()[0]
}

const (
	ctlFrame = iota
	ctlFlush
	ctlEnd
)

type item struct {
	frame audio.Frame
	kind  int
}

type stream struct {
	v       *SileroVAD
	det     detector
	events  chan vad.Event
	control chan item
	done    chan struct{}
	once    sync.Once

	mu  sync.Mutex
	err error

	input  *ort.Tensor[float32]
	state  *ort.Tensor[float32]
	stateN *ort.Tensor[float32]
	sr     *ort.Tensor[int64]
	output *ort.Tensor[float32]

	buf     []float32
	context []float32
}

func (v *SileroVAD) SampleRate() int { return v.sampleRate }

func (v *SileroVAD) Stream(ctx context.Context) vad.Stream {
	s := &stream{
		v:       v,
		events:  make(chan vad.Event, 64),
		control: make(chan item, 128),
		done:    make(chan struct{}),
		context: make([]float32, v.contextSize),
		det: detector{
			threshold:         v.threshold,
			negThreshold:      v.negThresh,
			sampleRate:        v.sampleRate,
			minSpeechSamples:  v.minSpeechSamples,
			minSilenceSamples: v.minSilenceSamples,
		},
	}
	if err := s.alloc(); err != nil {
		s.err = err
		close(s.events)
		return s
	}
	go s.run(ctx)
	return s
}

func (s *stream) alloc() error {
	var err error
	if s.input, err = ort.NewEmptyTensor[float32](ort.NewShape(1, int64(s.v.contextSize+s.v.windowSize))); err != nil {
		return err
	}
	if s.state, err = ort.NewTensor(ort.NewShape(2, 1, 128), make([]float32, 256)); err != nil {
		return err
	}
	if s.stateN, err = ort.NewEmptyTensor[float32](ort.NewShape(2, 1, 128)); err != nil {
		return err
	}
	if s.sr, err = ort.NewTensor(ort.NewShape(1), []int64{int64(s.v.sampleRate)}); err != nil {
		return err
	}
	if s.output, err = ort.NewEmptyTensor[float32](ort.NewShape(1, 1)); err != nil {
		return err
	}
	return nil
}

func (s *stream) PushFrame(f audio.Frame) { s.sendCtl(item{frame: f, kind: ctlFrame}) }
func (s *stream) Flush()                  { s.sendCtl(item{kind: ctlFlush}) }
func (s *stream) EndInput()               { s.sendCtl(item{kind: ctlEnd}) }

func (s *stream) sendCtl(it item) {
	select {
	case s.control <- it:
	case <-s.done:
	}
}

func (s *stream) Chan() <-chan vad.Event { return s.events }

func (s *stream) Err() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.err
}

func (s *stream) Close() error {
	s.once.Do(func() { close(s.done) })
	return nil
}

func (s *stream) run(ctx context.Context) {
	defer close(s.events)
	defer s.destroy()
	for {
		select {
		case <-ctx.Done():
			return
		case <-s.done:
			return
		case it := <-s.control:
			switch it.kind {
			case ctlFrame:
				s.feed(it.frame)
			case ctlFlush:
				s.emitAll(s.det.reset())
				s.resetModel()
			case ctlEnd:
				s.emitAll(s.det.reset())
				s.resetModel()
				return
			}
		}
	}
}

func (s *stream) feed(f audio.Frame) {
	for _, x := range f.Data {
		s.buf = append(s.buf, float32(x)/32768)
	}
	win := s.v.windowSize
	ctxN := s.v.contextSize
	processed := 0
	for len(s.buf)-processed >= win {
		data := s.input.GetData()
		copy(data, s.context)
		copy(data[ctxN:], s.buf[processed:processed+win])
		prob := s.v.infer(s.input, s.state, s.sr, s.output, s.stateN)
		copy(s.state.GetData(), s.stateN.GetData())
		copy(s.context, s.buf[processed+win-ctxN:processed+win])
		s.emitAll(s.det.step(prob, win))
		processed += win
	}
	if processed > 0 {
		n := copy(s.buf, s.buf[processed:])
		s.buf = s.buf[:n]
	}
}

func (s *stream) resetModel() {
	s.buf = s.buf[:0]
	clear(s.context)
	clear(s.state.GetData())
}

func (s *stream) emitAll(evs []vad.Event) {
	for _, ev := range evs {
		select {
		case s.events <- ev:
		case <-s.done:
			return
		}
	}
}

func (s *stream) destroy() {
	if s.input != nil {
		s.input.Destroy()
	}
	if s.state != nil {
		s.state.Destroy()
	}
	if s.stateN != nil {
		s.stateN.Destroy()
	}
	if s.sr != nil {
		s.sr.Destroy()
	}
	if s.output != nil {
		s.output.Destroy()
	}
}

func writeModel() (string, error) {
	f, err := os.CreateTemp("", "silero_vad_*.onnx")
	if err != nil {
		return "", err
	}
	if _, err := f.Write(modelData); err != nil {
		f.Close()
		os.Remove(f.Name())
		return "", err
	}
	if err := f.Close(); err != nil {
		os.Remove(f.Name())
		return "", err
	}
	return f.Name(), nil
}
