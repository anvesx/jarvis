"""Local speech for orb, on this machine only. No audio and no text leaves it.

- stt: Whistle (Cactus Compute), about 20 ms a call, and Parakeet TDT 0.6B v3 (NVIDIA), more accurate, on the GPU.
- tts: Kokoro-82M through ONNX Runtime, about 0.1 s for each second of speech (int8), English and Hindi.
- tts_lm: maya1 and indic-speak, 3B speech LMs on MLX (bench engines: slower than real time on an Apple M5).
- engines: the registries that build an engine on first use; gpu: one lock for torch on the Apple GPU.
- server: the HTTP service that the LiveKit voice worker and the bench call.

Model files and their checksums are in models.py. Fetch them with `python -m orb_speech.models`.
"""

__version__ = "0.1.0"
