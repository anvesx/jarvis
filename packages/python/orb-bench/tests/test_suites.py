"""The HTTP suites against fake services, on a tiny corpus in a temporary folder."""
import json

import httpx
import numpy as np
import pytest

from orb_bench import audio, corpus, report
from orb_bench.paths import corpus_dir


@pytest.fixture
def bench(tmp_path, monkeypatch):
    monkeypatch.setenv("ORB_TESTBENCH_DIR", str(tmp_path))
    for group, items in {"stt/synthetic": [{"id": "s1", "path": "s1.wav", "ref": "We should store passwords.",
                                            "lang": "en", "group": "synthetic", "variant": "clean", "sec": 1.0}],
                         "noise": [{"id": "n1", "path": "n1.wav", "ref": "", "lang": "en", "group": "noise", "sec": 1.0},
                                   {"id": "n2", "path": "n2.wav", "ref": "", "lang": "en", "group": "noise", "sec": 1.0}]}.items():
        for it in items:
            audio.write(corpus_dir(group) / it["path"], np.zeros(16000, np.float32))
        corpus._save(group, items)
    corpus._save("llm", [{"id": "c1", "turns": ["Hello there.", "We should store passwords."], "expect": {"persona": "Rival"}}])
    corpus._save("tts", [{"id": "t1", "text": "Hello there.", "lang": "en-us"}, {"id": "t2", "text": "नमस्ते।", "lang": "hi"}])
    return tmp_path


def test_stt_suite_scores_speech_and_counts_invented_words(bench):
    seen = []

    def handler(req):
        seen.append(dict(req.url.params))
        text = {"whistle": "we should store the passwords", "parakeet": "We should store passwords."}[req.url.params["engine"]]
        return httpx.Response(200, json={"text": text, "ms": 12.0, "audio_ms": 1000})

    from orb_bench.suites import stt
    res = stt.run(["whistle", "parakeet"], ["stt/synthetic", "noise"], "http://speech",
                  http=httpx.Client(transport=httpx.MockTransport(handler)))
    cells = {(r["engine"], r["corpus"], r["part"]): r for r in res["summary"]}
    assert cells[("whistle", "stt/synthetic", "clean")]["wer"] == 0.25
    assert cells[("parakeet", "stt/synthetic", "clean")]["wer"] == 0.0
    assert cells[("whistle", "noise", "noise")]["invented"] == 1.0  # the fake hears words in pure noise
    assert {s["engine"] for s in seen} == {"whistle", "parakeet"} and report.load("stt")["run"] == res["run"]


def test_llm_suite_checks_replies_and_personas(bench):
    def handler(req):
        if req.url.path == "/v1/sessions":
            return httpx.Response(201, json={"id": "s1"})
        body = json.loads(req.content)
        reply = "Persona: Rival. Why: test." if "passwords" not in body["text"] else "That is risky. Hash them instead."
        events = [{"type": "decision", "data": {"persona": "Rival"}}, {"type": "token", "data": reply},
                  {"type": "reply", "data": {"text": reply, "timing": {"llm_first_token": 0.2}}}]
        return httpx.Response(200, content="".join(json.dumps(e) + "\n" for e in events).encode())

    from orb_bench.suites import llm
    res = llm.run(["fake"], "http://agent", http=httpx.Client(transport=httpx.MockTransport(handler)))
    row = res["rows"][0]
    assert row["checks"] == {"has_reply": True, "short": True, "no_leak": True, "persona": True}
    assert res["summary"] == [{"model": "fake", "cases": 1, "passed": 1, "silent": 0, "first_token_p50": res["rows"][0]["first_token_s"],
                               "first_token_max": res["rows"][0]["first_token_s"]}]
    assert llm.checks("Persona: Rival. Why: test.", "Rival", {})["no_leak"] is False


def test_tts_suite_measures_and_judges(bench):
    pcm = (np.sin(np.linspace(0, 200, 24000)) * 8000).astype("<i2").tobytes()  # 1 s at 24 kHz

    spoken = []

    def handler(req):
        if req.url.path == "/v1/tts":
            spoken.append(json.loads(req.content)["text"])
            return httpx.Response(200, content=pcm, headers={"X-Sample-Rate": "24000"})
        return httpx.Response(200, json={"text": "Hello there."})  # the English judge (Parakeet)

    from orb_bench.suites import tts
    res = tts.run(tts.CONFIGS[:2], speech_url="http://speech",
                  http=httpx.Client(transport=httpx.MockTransport(handler)), hindi_judge=lambda x16: "नमस्ते")
    rows = {r["text_id"]: r for r in res["rows"]}
    assert len(res["rows"]) == 2 and spoken[0] == "Hello." and len(spoken) == 4  # one warm-up per config
    assert rows["t1"]["audio_s"] == 1.0 and rows["t1"]["wer"] == 0.0 and rows["t1"]["voice"] == "af_heart"
    assert rows["t2"]["voice"] == "hf_alpha" and rows["t2"]["wer"] == 0.0
    assert (bench / "results" / "tts" / rows["t1"]["wav"]).is_file()


def test_turn_summary_counts_each_failure_kind():
    from orb_bench.suites.turn import summarize

    rows = [{"case": "finished", "first_audio_s": 1.5, "commit_s": 0.4},
            {"case": "finished", "first_audio_s": 2.0, "commit_s": 0.5},
            {"case": "unfinished", "early_reply": True, "first_audio_s": 2.5, "commit_s": 0.6},
            {"case": "noise", "new_turn": False, "agent_audio": False},
            {"case": "noise", "new_turn": True, "agent_audio": False},
            {"case": "backchannel", "agent_kept_talking": True, "became_turn": False},
            {"case": "echo", "agent_audio_continued": False, "became_turn": True},
            {"case": "barge_in", "stop_s": 0.97, "two_words_s": 0.95, "agent_paused_s": 0.98}]
    s = summarize(rows)
    assert s["first_audio_p50"] == 2.0 and s["commit_p50"] == 0.5
    assert s["early_replies"] == 1 and s["unfinished"] == 1
    assert s["noise_false_turns"] == 1 and s["noise_clips"] == 2
    assert s["backchannel_ok"] is True and s["echo_became_turn"] is True and s["barge_in_stop_s"] == 0.97


def test_llm_suite_repeats_a_case_and_counts_silent_replies(bench):
    """The failure of 2026-10-22: a model that writes hidden text first may end a turn with no word at all."""
    corpus._save("llm", [{"id": "mic_check", "turns": ["Can you hear me clearly?"], "expect": {}, "repeat": 3}])
    asked = []

    def handler(req):
        if req.url.path == "/v1/sessions":
            return httpx.Response(201, json={"id": f"s{len(asked)}"})
        asked.append(json.loads(req.content)["text"])
        reply = [] if len(asked) == 3 else [{"type": "token", "data": "I can hear you."}]  # the 3rd call, after the warm-up
        events = [{"type": "decision", "data": {"persona": "Teacher"}}, *reply,
                  {"type": "reply", "data": {"text": "".join(e["data"] for e in reply), "timing": {}}}]
        return httpx.Response(200, content="".join(json.dumps(e) + "\n" for e in events).encode())

    from orb_bench.suites import llm
    res = llm.run(["fake"], "http://agent", http=httpx.Client(transport=httpx.MockTransport(handler)))
    assert [r["case"] for r in res["rows"]] == ["mic_check#1", "mic_check#2", "mic_check#3"]
    assert [r["checks"]["has_reply"] for r in res["rows"]] == [True, False, True]
    s = res["summary"][0]
    assert (s["cases"], s["passed"], s["silent"]) == (3, 2, 1)  # the silent reply is a failure, and it is counted
    assert s["first_token_p50"] is not None and s["first_token_max"] >= s["first_token_p50"]
    assert len(asked) == 4  # one warm-up turn, then three runs of the case
