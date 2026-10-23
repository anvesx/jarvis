"""The speech-LM engines (maya1, indic-speak) without their models: text pieces, SNAC frames, the stream chunks, the
checks, the prompt layout and the Vocos decoder. The models themselves run only with ORB_TEST_TTS_LM=1 (2 GB each)."""
import os

import numpy as np
import pytest

from orb_speech import tts_lm
from orb_speech.tts_lm import (AUDIO_BASE, BOS, CODEBOOK, END_OF_HUMAN, EOT, FRAME, START_OF_AI, START_OF_HUMAN,
                               START_OF_SPEECH, Chunker, Frames, IndicSpeakTTS, Maya1TTS, split_text, token_limit)


def tokens(frame_codes):
    """The tokens of one frame: code c at position p is AUDIO_BASE + p * 4096 + c."""
    return [AUDIO_BASE + p * CODEBOOK + c for p, c in enumerate(frame_codes)]


def test_split_text_makes_sentences_and_short_pieces():
    assert split_text("Hello there.  How are you?") == ["Hello there.", "How are you?"]
    assert split_text("नमस्ते। आप कैसे हैं?") == ["नमस्ते।", "आप कैसे हैं?"]
    long = ", ".join(f"part number {i} of a long list" for i in range(20)) + "."
    pieces = split_text(long)
    assert all(len(p) <= tts_lm.MAX_PIECE for p in pieces) and len(pieces) > 1
    assert " ".join(pieces) == long
    assert all(len(p) > tts_lm.MAX_PIECE // 2 for p in pieces[:-1])  # cut in the second half, not at the first comma


def test_token_limit_grows_with_the_text_and_has_a_cap():
    assert token_limit("Hi.") < token_limit("A sentence of about forty characters...") < tts_lm.MAX_TOKENS
    assert token_limit("x" * 2000) == tts_lm.MAX_TOKENS


def test_frames_take_codes_in_order_and_skip_other_tokens():
    f = Frames()
    done = [f.add(t) for t in tokens([1, 2, 3, 4, 5, 6, 7]) + [EOT] + tokens([8, 9, 10, 11, 12, 13, 14])]
    assert done.count(True) == 2 and len(f) == 2 and f.dropped == 0
    c0, c1, c2 = f.levels(0, 2)
    assert c0.tolist() == [1, 8]
    assert c1.tolist() == [2, 5, 9, 12]                 # positions 1 and 4
    assert c2.tolist() == [3, 4, 6, 7, 10, 11, 13, 14]  # positions 2, 3, 5 and 6


def test_frames_drop_a_frame_with_a_token_out_of_place():
    f = Frames()
    bad = tokens([1, 2, 3])[:2] + tokens([0, 0, 9])[2:]  # positions 0, 1 and 2
    for t in bad:
        f.add(t)
    f.add(AUDIO_BASE + 5 * CODEBOOK)                    # position 5 where 3 is next: the frame is dropped
    for t in tokens([20, 21, 22, 23, 24, 25, 26]):      # the next frame starts at position 0
        f.add(t)
    assert len(f) == 1 and f.codes[0][0] == 20 and f.dropped == 4


def fake_decode(start, end):
    """Each frame decodes to its own index, so the test can see which frames went out."""
    return np.repeat(np.arange(start, end, dtype=np.float32), FRAME)


@pytest.mark.parametrize("skip", [0, 1])
@pytest.mark.parametrize("first,step,lookahead", [(2, 6, 2), (2, 8, 4), (1, 1, 0)])
def test_chunker_sends_every_frame_once_and_in_order(skip, first, step, lookahead):
    calls = []

    def decode(a, b):
        calls.append((a, b))
        return fake_decode(a, b)

    c = Chunker(decode, first=first, step=step, context=3, lookahead=lookahead, skip=skip)
    out = [x for n in range(1, 40) if (x := c.ready(n)) is not None]
    out.append(c.ready(39, final=True))
    sent = np.concatenate([x for x in out if x is not None])
    assert np.array_equal(sent, np.repeat(np.arange(skip, 39, dtype=np.float32), FRAME))
    assert all(b - a <= 3 + max(step, first) + lookahead for a, b in calls)  # the context stays small


def test_chunker_waits_for_the_lookahead_then_sends_the_first_chunk_early():
    c = Chunker(fake_decode, first=2, step=6, context=4, lookahead=2)
    assert c.ready(3) is None                      # 1 frame without its lookahead
    first = c.ready(4)
    assert len(first) == 2 * FRAME                 # 2 frames, decoded with 2 frames of lookahead
    assert c.ready(9) is None                      # 5 new frames: less than a step
    assert len(c.ready(10)) == 6 * FRAME
    assert c.ready(10, final=True) is not None and c.ready(10, final=True) is None


def test_maya1_check():
    text, voice, speed, lang = Maya1TTS.check("  Hello   there. ")
    assert (text, voice, speed, lang) == ("Hello there.", tts_lm.MAYA1_VOICES[0], 1.0, "en-us")
    assert Maya1TTS.check("Hi.", voice="Deep male voice, slow.")[1] == "Deep male voice, slow."
    for kwargs, msg in [({"voice": 'A "quoted" voice'}, "cannot contain"), ({"lang": "hi"}, "English only"),
                        ({"speed": 1.5}, "no speed"), ({"voice": "x" * 401}, "400 characters")]:
        with pytest.raises(ValueError, match=msg):
            Maya1TTS.check("Hello.", **kwargs)
    assert Maya1TTS.style_of("happy") == ""  # maya1 has no style tags: emotion tags go in the text


def test_indic_speak_check():
    assert IndicSpeakTTS.check("नमस्ते।")[1:] == ("Kavya", 1.0, "hi")
    assert IndicSpeakTTS.check("வணக்கம்.", lang="ta")[1] == "Anitha"
    assert IndicSpeakTTS.check("Hello.", voice="amit", lang="en-us")[1:] == ("Amit", 1.0, "en-us")
    assert IndicSpeakTTS.style_of("AIR Style  News") == "AIR style news" and IndicSpeakTTS.style_of(" ") == ""
    assert len(IndicSpeakTTS.voices) == 44 and len(set(IndicSpeakTTS.voices)) == 44
    for kwargs, msg in [({"voice": "Nobody"}, "Unknown voice"), ({"style": "angry robot"}, "Unknown style"),
                        ({"lang": "fr"}, "supports"), ({"speed": 0.8}, "no speed")]:
        with pytest.raises(ValueError, match=msg):
            IndicSpeakTTS.check("नमस्ते।", **kwargs)


class FakeTokenizer:
    """One token per character, from 1000 up, so the layout of a prompt is easy to read."""

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [1000 + ord(ch) for ch in text]


def enc(text):
    return FakeTokenizer().encode(text)


def test_prompt_layouts_follow_the_model_cards():
    indic = IndicSpeakTTS.__new__(IndicSpeakTTS)  # no model: only the tokenizer is used
    indic.tok = FakeTokenizer()
    head, tail = indic.prompt_ids("नमस्ते", "Amit", "happy")
    assert head == ([START_OF_HUMAN, BOS, 156938] + enc("Amit") + [156939] + enc("\n")
                    + [156940] + enc("happy") + [156941] + enc("\n"))
    assert tail == enc("नमस्ते") + [EOT, END_OF_HUMAN, START_OF_AI, START_OF_SPEECH]
    assert indic.prompt_ids("x", "Amit", "")[0] == [START_OF_HUMAN, BOS, 156938] + enc("Amit") + [156939] + enc("\n")

    maya = Maya1TTS.__new__(Maya1TTS)
    maya.tok = FakeTokenizer()
    head, tail = maya.prompt_ids("Hi there.", "Calm voice.", "")
    assert head == [START_OF_HUMAN, BOS] + enc('<description="Calm voice.">')
    assert tail == enc(" Hi there.") + [EOT, END_OF_HUMAN, START_OF_AI, START_OF_SPEECH]


def test_vocos_layout_and_length():
    torch = pytest.importorskip("torch")
    from orb_speech.vocos import VocosDecoder

    dec = VocosDecoder(dim=16, intermediate_dim=32, num_blocks_pre=1, num_blocks_post=2, upsample=4, n_fft=512,
                       hop=128).eval()
    with torch.no_grad():
        y = dec(torch.randn(1, 768, 8))
    assert tuple(y.shape) == (1, 1, 512 * 8)  # 512 samples per latent step, exactly
    names = set(dec.state_dict())
    assert {"stem.weight", "pre.0.gamma", "post.1.dwconv.weight", "final_norm.bias", "head.out.weight"} <= names
    assert "head.window" not in names  # a buffer of the code, not of the checkpoint
    with pytest.raises(ValueError, match="512"):
        VocosDecoder(dim=16, intermediate_dim=32, num_blocks_pre=1, num_blocks_post=1, upsample=2, n_fft=512, hop=128)


@pytest.mark.skipif(os.environ.get("ORB_TEST_TTS_LM") != "1", reason="set ORB_TEST_TTS_LM=1 (loads 2 GB models)")
@pytest.mark.parametrize("name", ["maya1", "indic-speak"])
def test_the_model_speaks(name):
    from orb_speech import models

    if name == "maya1":
        tts = Maya1TTS(models.maya1_dir(False), models.snac_dir(False))
        wav = tts.synthesize("Hello there, how are you today?")
    else:
        tts = IndicSpeakTTS(models.indic_speak_dir(False), models.snac_dir(False))
        wav = tts.synthesize("नमस्ते, आप कैसे हैं?")
    try:
        assert wav.dtype == np.float32 and 0.8 < len(wav) / tts.sample_rate < 8
        assert np.sqrt(np.mean(wav ** 2)) > 0.01  # speech, not silence
    finally:
        tts.close()


@pytest.mark.skipif(os.environ.get("ORB_TEST_TTS_LM") != "1", reason="set ORB_TEST_TTS_LM=1 (loads 2 GB models)")
def test_parakeet_and_a_speech_lm_share_the_gpu():
    """The crash of 2026-10-22: Parakeet and the maya1 decoder used torch on MPS from two threads at once, and the
    process aborted. With the lock of gpu.py they take turns."""
    import threading

    from orb_speech import models
    from orb_speech.stt import ParakeetSTT

    stt = ParakeetSTT(models.parakeet_dir(False), language="en")  # its keep-warm thread runs too
    tts = Maya1TTS(models.maya1_dir(False), models.snac_dir(False))
    stop, heard = threading.Event(), []

    def listen():
        while not stop.is_set():
            heard.append(stt.transcribe(np.zeros(16000, np.float32))["ms"])

    t = threading.Thread(target=listen)
    t.start()
    try:
        for _ in range(3):
            assert len(tts.synthesize("We should hash the passwords before we store them.")) > 24000
    finally:
        stop.set()
        t.join()
        tts.close()
        stt.close()
    assert len(heard) > 10
