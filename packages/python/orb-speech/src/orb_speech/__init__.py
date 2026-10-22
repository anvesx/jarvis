"""Local speech for orb, on this machine only. No audio and no text leaves it.

- stt: Whistle (Cactus Compute), 16 kHz speech to text, about 12 to 35 ms for a sentence on an Apple M5.
- tts: Kokoro-82M through ONNX Runtime, text to 24 kHz speech, about 0.1 s for each second of speech (int8).
- turn: Smart Turn v3.2 (Pipecat), the probability that the speaker has finished the turn, from the audio.
- server: the HTTP service that the Go voice worker calls.

Model files and their checksums are in models.py. Fetch them with `python -m orb_speech.models`.
"""

__version__ = "0.1.0"
