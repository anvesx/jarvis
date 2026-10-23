"""Text to speech with two speech LMs on MLX: maya1 (English; the voice is a description in words) and indic-speak
(22 Indian languages and English; 44 named voices and style tags).

Both are Llama 3.2 3B fine-tunes, here in 4 bits (1.9 GB each; models.py pins the builds), and both write SNAC 24 kHz
codes with the same token layout. One frame is 7 codes and 2048 samples (85 ms), so real time needs about 82 tokens
per second. The LM runs on MLX. The codes are decoded with torch on the GPU (MPS):
- maya1: the SNAC decoder.
- indic-speak: the SNAC quantizer, then its own Vocos decoder (vocos.py).

Streaming: when enough new frames are ready, they are decoded with context frames on the left and a lookahead on the
right, and only their own samples go out. So the joins do not click, and each sample goes out once.

One worker thread per engine does all the MLX and torch work, so the engine speaks one text at a time. When a stream
stops (the listener left, or the person cut in), the generation ends at the next token.

The text is split into sentences, because the 4-bit models drop words in long inputs. Each sentence has a token limit
of about twice its expected length, so a model that does not stop cannot go on for long.

Every engine of the speech service has the same contract (see server.py): `name`, `sample_rate`, `voices`,
`check(text, voice, speed, style, lang)` and an async `stream(...)` of float32 chunks.
"""
from __future__ import annotations

import asyncio
import gc
import logging
import re
import threading
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from orb_speech import gpu

log = logging.getLogger("orb.speech")

SAMPLE_RATE = 24000
FRAME = 2048                # samples per frame
CODES = 7                   # codes per frame, in the order c0, c1, c2, c2, c1, c2, c2
CODEBOOK = 4096
AUDIO_BASE = 128266         # the token of code 0 at frame position 0; position p adds p * 4096
START_OF_SPEECH, END_OF_SPEECH = 128257, 128258
START_OF_HUMAN, END_OF_HUMAN, START_OF_AI, END_OF_AI = 128259, 128260, 128261, 128262
BOS, EOT = 128000, 128009
MAX_TEXT = 2000
MAX_PIECE = 200             # characters per generation
GAP = 0.2                   # seconds of silence between two pieces
TOKENS_PER_SECOND = 82      # of audio
MAX_TOKENS = 2520           # about 30 s of audio for one piece
SENTENCE_END = re.compile(r"(?<=[.!?।॥])\s+")


def clean_text(text: str) -> str:
    text = " ".join((text or "").split())
    if not text:
        raise ValueError("The text is empty.")
    if len(text) > MAX_TEXT:
        raise ValueError(f"The text has {len(text)} characters. The limit is {MAX_TEXT}.")
    return text


def no_speed(name: str, speed: float | None) -> None:
    if speed not in (None, 1.0):
        raise ValueError(f"{name} has no speed control.")


def split_text(text: str, limit: int = MAX_PIECE) -> list[str]:
    """Sentences. A longer sentence is cut at a comma, a semicolon or a colon, else at a space, in its second half."""
    out = []
    for sentence in SENTENCE_END.split(" ".join(text.split())):
        while len(sentence) > limit:
            cut = max(sentence.rfind(c, limit // 2, limit) for c in ",;:")
            if cut < 0:
                cut = sentence.rfind(" ", limit // 2, limit)
            if cut < 0:
                cut = sentence.rfind(" ", 0, limit)
            if cut <= 0:
                cut = limit - 1
            out.append(sentence[:cut + 1].strip())
            sentence = sentence[cut + 1:].strip()
        if sentence:
            out.append(sentence)
    return out


def token_limit(text: str) -> int:
    """About twice the tokens that the text needs (12 characters per second of speech), plus 3 seconds."""
    return min(MAX_TOKENS, int((2 * len(text) / 12 + 3) * TOKENS_PER_SECOND))


class Frames:
    """SNAC frames from the generated tokens. A token's value tells its position in the frame. So a token out of
    place drops the frame that it is in, and the next frame starts at the next position-0 token."""

    def __init__(self):
        self.codes: list[list[int]] = []  # complete frames, 7 codes each
        self._part: list[int] = []
        self.dropped = 0                  # tokens dropped

    def __len__(self) -> int:
        return len(self.codes)

    def add(self, token: int) -> bool:
        """Adds a token. True when it completed a frame. Tokens that are not audio codes are ignored."""
        pos, code = divmod(token - AUDIO_BASE, CODEBOOK)
        if not 0 <= pos < CODES:
            return False
        if pos != len(self._part):
            self.dropped += len(self._part)
            self._part = []
            if pos != 0:
                self.dropped += 1
                return False
        self._part.append(code)
        if len(self._part) < CODES:
            return False
        self.codes.append(self._part)
        self._part = []
        return True

    def levels(self, start: int, end: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The three SNAC code levels of frames [start, end): 1, 2 and 4 codes per frame."""
        a = np.asarray(self.codes[start:end], dtype=np.int64).reshape(-1, CODES)
        return np.ascontiguousarray(a[:, 0]), a[:, [1, 4]].reshape(-1), a[:, [2, 3, 5, 6]].reshape(-1)


class Chunker:
    """Decides when frames go out, and decodes them. The first chunk goes out after `first` frames (a short wait),
    then `step` frames at a time. A chunk is decoded with up to `context` frames before it and `lookahead` frames
    after it. Only the chunk's own samples go out. The first `skip` frames never go out."""

    def __init__(self, decode: Callable[[int, int], np.ndarray], *, first: int, step: int, context: int,
                 lookahead: int, skip: int = 0):
        self.decode, self.first, self.step, self.context, self.lookahead = decode, first, step, context, lookahead
        self.sent = self.skip = skip

    def ready(self, n: int, final: bool = False) -> np.ndarray | None:
        """The next chunk when there are `n` complete frames, or None when it is too early.
        `final`: no more frames come, so the rest goes out."""
        until = n if final else n - self.lookahead
        need = 1 if final else self.first if self.sent == self.skip else self.step
        if until - self.sent < need:
            return None
        start = max(0, self.sent - self.context)
        wav = self.decode(start, n)
        out = wav[(self.sent - start) * FRAME:(until - start) * FRAME]
        self.sent = until
        return out


class SpeechLM:
    """The shared part of maya1 and indic-speak. A subclass sets the prompt, the sampling and the decoder."""

    name = ""
    sample_rate = SAMPLE_RATE
    voices: tuple[str, ...] = ()
    temperature, top_p, top_k, repetition_penalty = 0.6, 0.9, 0, 1.1
    first, step, context, lookahead, skip = 2, 6, 4, 2, 0
    max_prefixes = 8  # prompt caches kept, one per voice

    def __init__(self, lm_dir: Path, snac_dir: Path, device: str | None = None):
        self._pool = ThreadPoolExecutor(1, thread_name_prefix=f"tts-{self.name}")
        self._closed = threading.Event()
        self._prefixes: OrderedDict = OrderedDict()
        try:
            self._pool.submit(self._load, Path(lm_dir), Path(snac_dir), device).result()
        except BaseException:
            self._pool.shutdown(wait=False)
            raise

    # --- the contract
    @classmethod
    def check(cls, text: str, voice: str | None = None, speed: float | None = None, style: str | None = None,
              lang: str | None = None) -> tuple[str, str, float, str]:
        raise NotImplementedError

    @classmethod
    def style_of(cls, style: str | None) -> str:
        return ""

    async def stream(self, text: str, voice: str | None = None, speed: float | None = None, style: str | None = None,
                     lang: str | None = None) -> AsyncIterator[np.ndarray]:
        """Yields audio chunks while the model writes. Stopping the iteration stops the generation."""
        text, voice, _, _ = self.check(text, voice, speed, style, lang)
        style = self.style_of(style)
        loop, queue, stop = asyncio.get_running_loop(), asyncio.Queue(), threading.Event()

        def put(item) -> None:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, item)
            except RuntimeError:  # the event loop is closed
                stop.set()

        try:
            self._pool.submit(self._run, text, voice, style, put, stop)
        except RuntimeError as e:  # the engine was unloaded
            raise ValueError(f"{self.name} is unloaded. Send the request again.") from e
        try:
            while (item := await queue.get()) is not None:
                if isinstance(item, Exception):
                    raise RuntimeError(f"{self.name} failed: {item}") from item
                yield item
        finally:
            stop.set()

    def synthesize(self, text: str, voice: str | None = None, speed: float | None = None, style: str | None = None,
                   lang: str | None = None) -> np.ndarray:
        """All the audio at once (tests and tools)."""
        text, voice, _, _ = self.check(text, voice, speed, style, lang)
        items: list = []
        self._pool.submit(self._run, text, voice, self.style_of(style), items.append, threading.Event()).result()
        for item in items:
            if isinstance(item, Exception):
                raise RuntimeError(f"{self.name} failed: {item}") from item
        chunks = [x for x in items if x is not None]
        return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)

    def close(self) -> None:
        """Frees the model memory. A running generation stops at its next token."""
        self._closed.set()
        try:
            self._pool.submit(self._free).result(timeout=60)
        finally:
            self._pool.shutdown(wait=False)

    # --- the subclass parts
    def _voice_ids(self, voice: str, style: str) -> list[int]:
        raise NotImplementedError

    def _text_ids(self, text: str) -> list[int]:
        raise NotImplementedError

    def _load_decoder(self, lm_dir: Path) -> None:
        pass

    def _vocode(self, z):
        return self.snac.decoder(z)

    # --- the prompt
    def prompt_ids(self, text: str, voice: str, style: str) -> tuple[list[int], list[int]]:
        """(The voice part, the text part) of the prompt:
        <start_of_human> <bos> voice part, text <eot> <end_of_human> <start_of_ai> <start_of_speech>."""
        return ([START_OF_HUMAN, BOS] + self._voice_ids(voice, style),
                self._text_ids(text) + [EOT, END_OF_HUMAN, START_OF_AI, START_OF_SPEECH])

    # --- everything below runs in the worker thread
    def _load(self, lm_dir: Path, snac_dir: Path, device: str | None) -> None:
        import torch
        from mlx_lm import load
        from mlx_lm.sample_utils import make_logits_processors, make_sampler
        from snac import SNAC

        t0 = time.perf_counter()
        self.model, self.tok = load(str(lm_dir))
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        with gpu.MPS:  # torch on the GPU: one thread at a time (gpu.py)
            self.snac = SNAC.from_pretrained(str(snac_dir)).eval().to(self.device)  # torch.load: weights_only
            self._load_decoder(lm_dir)
        self._sampler = make_sampler(temp=self.temperature, top_p=self.top_p, top_k=self.top_k)
        self._processors = make_logits_processors(repetition_penalty=self.repetition_penalty)
        self._stops = {END_OF_SPEECH, END_OF_AI, *([self.tok.eos_token_id] if self.tok.eos_token_id else [])}
        text, voice, _, _ = self.check("Hello.")
        for _ in self._speak(text, voice, "", threading.Event()):  # the first run compiles the GPU kernels
            pass
        log.info("%s ready on %s in %.1f s", self.name, self.device, time.perf_counter() - t0)

    def _free(self) -> None:
        import mlx.core as mx
        import torch

        self._prefixes.clear()
        for attr in ("model", "tok", "snac", "vocos", "_sampler", "_processors"):
            self.__dict__.pop(attr, None)
        gc.collect()
        mx.clear_cache()
        if torch.backends.mps.is_available():
            with gpu.MPS:
                torch.mps.empty_cache()

    def _prefix(self, voice: str, style: str, ids: list[int]):
        """A prompt cache of the voice part, without its last token. Made once per voice and kept."""
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        key = (voice, style)
        if key in self._prefixes:
            self._prefixes.move_to_end(key)
            return self._prefixes[key]
        cache = make_prompt_cache(self.model)
        self.model(mx.array(ids[:-1])[None], cache=cache)
        mx.eval([x for c in cache for x in (c.keys, c.values)])
        self._prefixes[key] = (cache, len(ids) - 1)
        while len(self._prefixes) > self.max_prefixes:
            self._prefixes.popitem(last=False)
        return self._prefixes[key]

    def _decode(self, frames: Frames, start: int, end: int) -> np.ndarray:
        import torch

        levels = frames.levels(start, end)
        with gpu.MPS, torch.inference_mode():
            codes = [torch.from_numpy(c).to(self.device).unsqueeze(0) for c in levels]
            wav = self._vocode(self.snac.quantizer.from_codes(codes))
            return wav.reshape(-1).float().clamp(-1, 1).cpu().numpy()

    def _speak(self, text: str, voice: str, style: str, stop: threading.Event) -> Iterator[np.ndarray]:
        """The audio of one piece of text, chunk by chunk."""
        import mlx.core as mx
        from mlx_lm.generate import generate_step

        voice_ids, text_ids = self.prompt_ids(text, voice, style)
        cache, length = self._prefix(voice, style, voice_ids)
        frames = Frames()
        chunker = Chunker(lambda a, b: self._decode(frames, a, b), first=self.first, step=self.step,
                          context=self.context, lookahead=self.lookahead, skip=self.skip)
        steps = generate_step(mx.array(voice_ids[-1:] + text_ids), self.model, max_tokens=token_limit(text),
                              sampler=self._sampler, logits_processors=self._processors, prompt_cache=cache)
        tokens = 0
        try:
            for token, _ in steps:
                token = int(token)
                if token in self._stops or stop.is_set() or self._closed.is_set():
                    break
                tokens += 1
                if frames.add(token) and (out := chunker.ready(len(frames))) is not None:
                    yield out
        finally:
            steps.close()
            for c in cache:  # back to the voice part, for the next piece
                c.trim(c.offset - length)
        if frames.dropped:
            log.info("%s dropped %d tokens out of place (of %d)", self.name, frames.dropped, tokens)
        if tokens >= token_limit(text):
            log.warning("%s reached the token limit (%d): the model did not stop", self.name, tokens)
        if not stop.is_set() and (out := chunker.ready(len(frames), final=True)) is not None:
            yield out

    def _run(self, text: str, voice: str, style: str, put: Callable, stop: threading.Event) -> None:
        try:
            for i, piece in enumerate(split_text(text)):
                if stop.is_set() or self._closed.is_set():
                    break
                if i:
                    put(np.zeros(int(GAP * SAMPLE_RATE), np.float32))
                for chunk in self._speak(piece, voice, style, stop):
                    put(chunk)
        except Exception as e:  # the caller gets it; the worker thread lives on
            log.exception("%s failed", self.name)
            put(e)
        finally:
            put(None)


MAYA1_VOICES = (
    "Calm and clear female voice, in her 30s with an American accent, natural pacing.",  # the example of the build
    "Warm male voice in his 40s with an American accent, friendly and conversational, moderate pace.",
    "Young female voice with a British accent, bright and energetic, quick pace.",
)


class Maya1TTS(SpeechLM):
    """maya1: English. The voice is a description in words: age, gender, accent, pitch, pace, mood. Emotion tags
    such as <laugh> or <sigh> can go in the text. It has no style or speed control: `style` is ignored."""

    name = "maya1"
    voices = MAYA1_VOICES
    temperature, top_p, top_k, repetition_penalty = 0.4, 0.9, 0, 1.1  # the settings of the MLX build
    first, step, context, lookahead, skip = 2, 6, 4, 2, 1             # the build drops the first frame (2048 samples)

    @classmethod
    def check(cls, text, voice=None, speed=None, style=None, lang=None):
        text = clean_text(text)
        no_speed(cls.name, speed)
        if lang not in (None, "", "en", "en-us", "en-gb"):
            raise ValueError(f"maya1 speaks English only, not {lang!r}.")
        voice = " ".join((voice or cls.voices[0]).split())
        if len(voice) > 400:
            raise ValueError("The voice description has more than 400 characters.")
        if any(c in voice for c in '"<>'):
            raise ValueError('A maya1 voice description cannot contain ", < or >.')
        return text, voice, 1.0, lang or "en-us"

    def _voice_ids(self, voice, style):
        return self.tok.encode(f'<description="{voice}">', add_special_tokens=False)

    def _text_ids(self, text):
        return self.tok.encode(f" {text}", add_special_tokens=False)


# One female and one male voice per language, from the model card (voices.md). Every voice speaks every language, but
# a voice is best in its own language.
INDIC_VOICES = {
    "as": ("Prastuti", "Ankur"), "bn": ("Ishita", "Sourav"), "brx": ("Gwrbw", "Sansuma"), "doi": ("Preeti", "Sham"),
    "gu": ("Dhara", "Parth"), "hi": ("Kavya", "Amit"), "kn": ("Deepika", "Adarsh"), "ks": ("Zoon", "Ishfaq"),
    "kok": ("Anjali", "Sandeep"), "mai": ("Vaidehi", "Madhukar"), "ml": ("Lakshmi", "Kiran"),
    "mni": ("Thoibi", "Chaoba"), "mr": ("Anagha", "Chinmay"), "ne": ("Srijana", "Sagar"), "or": ("Itishree", "Akash"),
    "pa": ("Kaur", "Manpreet"), "sa": ("Bharati", "Aryaman"), "sat": ("Phulmani", "Sibu"), "sd": ("Moomal", "Rano"),
    "ta": ("Anitha", "Arun"), "te": ("Sravani", "Vamsi"), "ur": ("Saba", "Zaid"),
}
INDIC_STYLES = ("advertisements", "AIR style news", "TV style news", "single person narration audiobook",
                "children's stories", "educational lecture", "news", "Customer Care",
                "happy", "sad", "anger", "fear", "surprise", "disgust")
SPEAKER = (156938, 156939)  # <|speaker> and <speaker|> (token_contract.md of the model)
STYLE = (156940, 156941)    # <|style> and <style|>


class IndicSpeakTTS(SpeechLM):
    """indic-speak: 22 Indian languages and English. `voice` is a name of the model's roster, `style` an optional
    trained tag. `lang` only picks the default voice: the model reads the script of the text."""

    name = "indic-speak"
    voices = tuple(v for pair in INDIC_VOICES.values() for v in pair)
    styles = INDIC_STYLES
    langs = (*INDIC_VOICES, "en", "en-us", "en-gb")
    temperature, top_p, top_k, repetition_penalty = 0.6, 0.9, 50, 1.2  # the model card's production settings
    first, step, context, lookahead, skip = 2, 8, 12, 4, 0             # Vocos sees about 5 frames on each side

    @classmethod
    def check(cls, text, voice=None, speed=None, style=None, lang=None):
        text = clean_text(text)
        no_speed(cls.name, speed)
        lang = (lang or "hi").lower()
        if lang not in cls.langs:
            raise ValueError(f"indic-speak supports {', '.join(cls.langs)}, not {lang!r}.")
        if voice:
            match = [v for v in cls.voices if v.lower() == voice.strip().lower()]
            if not match:
                raise ValueError(f"Unknown voice {voice!r}. Hindi voices: Kavya, Amit. All: {', '.join(cls.voices)}.")
            voice = match[0]
        else:
            voice = INDIC_VOICES.get(lang, INDIC_VOICES["hi"])[0]
        cls.style_of(style)
        return text, voice, 1.0, lang

    @classmethod
    def style_of(cls, style):
        if not style or not style.strip():
            return ""
        match = [s for s in cls.styles if s.lower() == " ".join(style.split()).lower()]
        if not match:
            raise ValueError(f"Unknown style {style!r}. Styles: {', '.join(cls.styles)}.")
        return match[0]

    def _load_decoder(self, lm_dir):
        from orb_speech import vocos

        for token, want in (("<|speaker>", SPEAKER[0]), ("<style|>", STYLE[1]), ("<|snac_0|>", AUDIO_BASE),
                            ("<|end_of_speech|>", END_OF_SPEECH)):
            if self.tok.convert_tokens_to_ids(token) != want:
                raise ValueError(f"The tokenizer in {lm_dir} does not match the token contract ({token}).")
        self.vocos = vocos.load(lm_dir / "vocos" / "best.pt", self.device)

    def _vocode(self, z):
        return self.vocos(z.float())

    def _voice_ids(self, voice, style):
        enc = lambda s: self.tok.encode(s, add_special_tokens=False)
        ids = [SPEAKER[0], *enc(voice), SPEAKER[1], *enc("\n")]
        if style:
            ids += [STYLE[0], *enc(style), STYLE[1], *enc("\n")]
        return ids

    def _text_ids(self, text):
        return self.tok.encode(text, add_special_tokens=False)
