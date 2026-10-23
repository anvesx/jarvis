"""The stt suite's written-form summary and the offline format suite, on a saved run and no service."""
import json

import httpx
import numpy as np
import pytest

from orb_bench import audio, corpus, report
from orb_bench.paths import corpus_dir
from orb_bench.suites import format as fmt
from orb_bench.suites import stt

REF = "Please review pull request #4821 at 3:30 PM."
SPOKEN_FORM = "please review pull request number 4821 at 3 30 pm"


@pytest.fixture
def bench(tmp_path, monkeypatch):
    monkeypatch.setenv("ORB_TESTBENCH_DIR", str(tmp_path))
    items = {"stt/dictation": [{"id": "d1", "path": "d1.wav", "ref": REF, "keys": ["#4821", "3:30 PM"], "lang": "en",
                                "group": "dictation", "variant": "clean", "cls": "number", "written": True, "sec": 1.0}],
             "stt/edacc": [{"id": "e1", "path": "e1.wav", "ref": "we should store passwords", "lang": "en",
                            "group": "south_asian", "sec": 1.0}],
             "noise": [{"id": "n1", "path": "n1.wav", "ref": "", "lang": "en", "group": "noise", "sec": 1.0}]}
    for group, rows in items.items():
        for it in rows:
            audio.write(corpus_dir(group) / it["path"], np.zeros(16000, np.float32))
        corpus._save(group, rows)
    return tmp_path


def fake_speech(hyps: dict):
    def handler(req):
        return httpx.Response(200, json={"text": hyps[req.url.params["engine"]], "ms": 20.0})
    return httpx.Client(transport=httpx.MockTransport(handler))


def run_stt(hyp):
    return stt.run(["e"], ["stt/dictation", "stt/edacc", "noise"], "http://speech", http=fake_speech({"e": hyp}))


def cell(res, corpus_name, part, **where):
    return next(r for r in res["summary"] if r["corpus"] == corpus_name and r["part"] == part
                and all(r.get(k) == v for k, v in where.items()))


def test_stt_suite_adds_written_metrics_only_for_written_form_groups(bench):
    res = run_stt(SPOKEN_FORM)
    d = cell(res, "stt/dictation", "all")
    assert d["fwer"] > d["wer"] > 0 and d["entity"] == 0.0 and d["punct_f1"] == 0.0
    assert cell(res, "stt/dictation", "cls:number")["clips"] == 1 and cell(res, "stt/dictation", "clean")["clips"] == 1
    e = cell(res, "stt/edacc", "all")
    assert e["wer"] == 0.0 or e["wer"] > 0
    assert "fwer" not in e and "entity" not in e and "punct_f1" not in e
    assert cell(res, "noise", "noise")["invented"] == 1.0
    row = next(r for r in res["rows"] if r["id"] == "d1")
    assert row["raw"] == SPOKEN_FORM and row["keys"] == ["#4821", "3:30 PM"]


def test_stt_suite_scores_a_perfect_written_transcript(bench):
    d = cell(run_stt(REF), "stt/dictation", "all")
    assert (d["wer"], d["fwer"], d["entity"], d["punct_f1"]) == (0.0, 0.0, 1.0, 1.0)


def test_format_suite_with_raw_reproduces_the_stt_run_without_a_service(bench):
    src = run_stt(SPOKEN_FORM)
    res = fmt.run(src["run"], ["raw"])
    assert res["source"] == src["run"] and report.load("format")["run"] == res["run"]
    for part in ("all", "cls:number"):
        a, b = cell(src, "stt/dictation", part), cell(res, "stt/dictation", part)
        assert {k: b[k] for k in ("wer", "fwer", "entity", "punct_f1")} == {k: a[k] for k in ("wer", "fwer", "entity", "punct_f1")}
    assert cell(res, "stt/dictation", "all")["added"] == 0 and cell(res, "stt/dictation", "all")["stages"] == "raw"
    assert cell(res, "noise", "noise")["invented"] == 1.0


def test_format_suite_runs_registered_stages_in_order_and_scores_them(bench, monkeypatch):
    monkeypatch.setitem(fmt.STAGES, "numbers", lambda t: t.replace("number 4821", "#4821").replace("3 30 pm", "3:30 PM"))
    monkeypatch.setitem(fmt.STAGES, "dot", lambda t: t.capitalize() + ".")
    monkeypatch.setitem(fmt.STAGES, "liar", lambda t: t + " bananas")
    src = run_stt(SPOKEN_FORM)
    res = fmt.run(src["run"], ["raw", "numbers", "numbers+dot", "liar"], engines=["e"], groups=["stt/dictation"])
    by = {r["stages"]: r for r in res["summary"] if r["part"] == "all"}
    assert by["raw"]["entity"] == 0.0 and by["numbers"]["entity"] == 1.0
    assert by["numbers+dot"]["fwer"] < by["numbers"]["fwer"] < by["raw"]["fwer"] and by["numbers+dot"]["punct_f1"] == 1.0
    assert by["numbers"]["added"] == 0 and by["liar"]["added"] == 1  # the hallucination check
    row = next(r for r in res["rows"] if r["stages"] == "numbers+dot")
    assert list(row["stage_ms"]) == ["numbers", "dot"] and row["stage_ms_total"] > 0
    assert {r["corpus"] for r in res["rows"]} == {"stt/dictation"}


def test_format_suite_names_the_unknown_stage_and_loads_a_dotted_path_lazily(bench):
    src = run_stt(REF)
    with pytest.raises(KeyError, match="no stage 'written'"):
        fmt.run(src["run"], ["written"])
    fmt.register("upper", "string:capwords")
    try:
        res = fmt.run(src["run"], ["upper"], groups=["stt/dictation"])
        assert cell(res, "stt/dictation", "all")["fwer"] > 0
    finally:
        fmt.STAGES.pop("upper")
    assert fmt.parse("raw, written,vocab+written") == ["raw", "written", "vocab+written"]


def test_format_suite_needs_an_stt_run(bench):
    with pytest.raises(FileNotFoundError, match="run stt"):
        fmt.run(None, ["raw"])
