# orb-speech

Speech for orb, behind one HTTP service on 127.0.0.1:8200. It is local, except for one optional cloud engine:
- **speech to text:** Parakeet TDT 0.6B v3 (the default, 2.5 GB, the final text), Whistle (17 MB, about 25 ms a
  call: the voice worker's interim words), or Cartesia Ink-Whisper (**a cloud service**: the audio goes to Cartesia; it needs `CARTESIA_API_KEY`)
- **text to speech:** Kokoro (the live voice) or two speech LMs for the bench: maya1 (English) and indic-speak
  (22 Indian languages)

The text of `POST /v1/stt` is put in written form ("pull request number 4821" becomes "pull request #4821", "3 30 pm"
becomes "3:30 PM") by `orb_speech.textfmt.written`. The answer also has `raw`, the engine's own text. `?format=raw`
or `ORB_STT_FORMAT=raw` turns it off. The rules are in [decision 0010](../../../docs/decisions/0010-written-form-formatter.md).

A request can name its engine, so the bench switches models without a restart. The design is in [LLD.md](LLD.md).
The model choices are in [decision 0005](../../../docs/decisions/0005-stt-parakeet.md) and
[decision 0007](../../../docs/decisions/0007-whistle-default-and-tts-results.md) and
[decision 0008](../../../docs/decisions/0008-cartesia-cloud-stt.md) (Cartesia) and
[decision 0012](../../../docs/decisions/0012-stt-final-and-interim-engines.md) (Parakeet final, Whistle interims).

```sh
python -m orb_speech.models            # fetch the models and check their SHA-256 (about 2.7 GB)
python -m orb_speech.models --all      # also the full Kokoro model, maya1, indic-speak and SNAC (4.3 GB more)
HF_HUB_OFFLINE=1 python -m orb_speech.server
curl -s localhost:8200/v1/engines
```

The engines also work alone:
```python
from orb_speech import models
from orb_speech.stt import ParakeetSTT, WhistleSTT
from orb_speech.tts import KokoroTTS

stt = WhistleSTT(weights=str(models.whistle_weights()))
tts = KokoroTTS(models.path("kokoro-v1.0.int8.onnx"), models.path("voices-v1.0.bin"))
audio = tts.synthesize("Hello there.")                      # 24 kHz float32
hindi = tts.synthesize("नमस्ते, आप कैसे हैं?", lang="hi")      # a Hindi voice when none is given
```

Cartesia: add the line `CARTESIA_API_KEY=...` to the `.env` file of the repo (it is not in git), then restart the
speech service. The engine is then in the bench's speech-to-text list, marked "(cloud)".

Model credits: Whistle by Cactus Compute (Apache-2.0); Parakeet TDT 0.6B v3 by NVIDIA (CC-BY-4.0); Kokoro-82M
(Apache-2.0); maya1 by Maya Research (Apache-2.0); indic-speak by Bodhan AI (Indic Open Model License); SNAC by Hubert
Siuzdak (MIT). maya1 and indic-speak are Llama 3.2 fine-tunes, so the Llama 3.2 Community License applies too.
