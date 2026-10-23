"""The bench API: corpus and result paths stay inside their folders, names come from fixed lists, one run at a time."""
import json
import threading
import time

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from orb_agent.bench import inside, make_router


@pytest.fixture()
def bench(tmp_path, monkeypatch):
    monkeypatch.setenv("ORB_TESTBENCH_DIR", str(tmp_path))
    group = tmp_path / "corpus" / "noise"
    group.mkdir(parents=True)
    import soundfile as sf
    sf.write(group / "white.wav", np.zeros(1600, dtype=np.float32), 16000)
    (group / "manifest.json").write_text(json.dumps([{"id": "white", "path": "white.wav", "ref": ""}]))
    (tmp_path / "secret.txt").write_text("not for the page")
    app = FastAPI()
    app.include_router(make_router("http://speech.invalid", "http://agent.invalid"))
    return TestClient(app), tmp_path


def test_inside_refuses_paths_that_leave_the_folder(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.wav").write_bytes(b"x")
    (tmp_path / "y.txt").write_bytes(b"y")
    assert inside(tmp_path / "a", "x.wav") == (tmp_path / "a" / "x.wav").resolve()
    for rel in ("../y.txt", "../../etc/passwd", "missing.wav", "."):
        with pytest.raises(Exception) as e:
            inside(tmp_path / "a", rel)
        assert getattr(e.value, "status_code", None) == 404, rel


def test_corpus_and_clips_come_only_from_known_groups(bench):
    c, _ = bench
    assert c.get("/v1/bench/corpus/noise").json()[0]["id"] == "white"
    assert c.get("/v1/bench/clip/noise/white.wav").status_code == 200
    assert c.get("/v1/bench/corpus/secrets").status_code == 404           # not a corpus group
    assert c.get("/v1/bench/clip/noise/..%2F..%2Fsecret.txt").status_code == 404
    assert c.get("/v1/bench/clip/..%2Fsecret.txt/x").status_code == 404
    assert c.get("/v1/bench/corpus/echo").status_code == 404              # a known group that is not built


def test_results_come_only_from_known_suites_and_runs(bench):
    c, root = bench
    (root / "results" / "turn").mkdir(parents=True)
    (root / "results" / "turn" / "20261022-205034.json").write_text(json.dumps({"suite": "turn", "rows": []}))
    assert c.get("/v1/bench/results/turn").json()["runs"] == ["20261022-205034"]
    assert c.get("/v1/bench/results/turn/20261022-205034").json()["suite"] == "turn"
    assert c.get("/v1/bench/results/turn/missing").status_code == 404
    assert c.get("/v1/bench/results/secrets").status_code == 404
    assert c.get("/v1/bench/audio/..%2F..%2F/secret.txt").status_code == 404


def test_one_run_at_a_time_and_unknown_suites_are_refused(bench, monkeypatch):
    c, _ = bench
    from orb_bench.suites import stt

    release = threading.Event()

    def slow_run(*args, **kwargs):
        release.wait(5)
        return {"run": "20261022-000000", "summary": []}

    monkeypatch.setattr(stt, "run", slow_run)
    assert c.post("/v1/bench/runs", json={"suite": "secrets"}).status_code == 400
    first = c.post("/v1/bench/runs", json={"suite": "stt", "limit": 1})
    assert first.status_code == 202
    assert c.post("/v1/bench/runs", json={"suite": "stt"}).status_code == 409  # the first run still goes
    release.set()
    job = first.json()["id"]
    for _ in range(50):
        status = c.get(f"/v1/bench/runs/{job}").json()
        if status["status"] != "running":
            break
        time.sleep(0.05)
    assert status["status"] == "done" and status["run"] == "20261022-000000"
    assert c.post("/v1/bench/runs", json={"suite": "stt", "limit": 1000}).status_code == 202  # free again
    release.set()
    assert c.post("/v1/bench/runs", json={"suite": "stt", "limit": 1001}).status_code == 422  # the clip limit
