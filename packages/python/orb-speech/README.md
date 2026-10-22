# orb-speech

Local speech for orb: Whistle (speech to text), Kokoro (text to speech) and Smart Turn (end of turn), behind
one HTTP service on 127.0.0.1:8200. The design is in [LLD.md](LLD.md); the model choice is in
[decision 0003](../../../docs/decisions/0003-local-speech-models.md).

```sh
python -m orb_speech.models            # fetch the models and check their SHA-256 (about 170 MB)
HF_HUB_OFFLINE=1 python -m orb_speech.server
curl -s localhost:8200/health
```

The engines also work alone:
```python
from orb_speech import models
from orb_speech.stt import WhistleSTT
from orb_speech.tts import KokoroTTS

tts = KokoroTTS(models.path("kokoro-v1.0.int8.onnx"), models.path("voices-v1.0.bin"))
audio = tts.synthesize("Hello there.")          # 24 kHz float32
```
