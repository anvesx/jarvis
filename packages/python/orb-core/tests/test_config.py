import os

import pytest

from orb_core import config


def test_parse_env_text_handles_comments_quotes_and_export():
    text = '# comment\n\nA=1\nexport B="two words"\nC=\'3\'\nBAD LINE\nD=a=b\n'
    assert config.parse_env_text(text) == {"A": "1", "B": "two words", "C": "3", "D": "a=b"}


def test_load_env_files_never_overrides_the_process(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("ORB_T_SET=from-file\nORB_T_NEW=new\nORB_T_EMPTY=\n")
    monkeypatch.setenv("ORB_T_SET", "from-process")
    monkeypatch.delenv("ORB_T_NEW", raising=False)
    monkeypatch.delenv("ORB_T_EMPTY", raising=False)
    loaded = config.load_env_files(f, tmp_path / "missing.env", None)
    assert loaded == ["ORB_T_NEW"]
    assert os.environ["ORB_T_SET"] == "from-process"
    assert os.environ["ORB_T_NEW"] == "new"
    assert "ORB_T_EMPTY" not in os.environ
    monkeypatch.delenv("ORB_T_NEW")


def test_describe_never_returns_values(monkeypatch):
    monkeypatch.setenv("ORB_T_API_KEY", "super-secret-value")
    monkeypatch.delenv("ORB_T_MISSING", raising=False)
    out = config.describe(["ORB_T_API_KEY", "ORB_T_MISSING"])
    assert out == {"ORB_T_API_KEY": "set", "ORB_T_MISSING": "missing"}
    assert "super-secret-value" not in repr(out)


def test_env_required_names_the_variable_but_not_a_value(monkeypatch):
    monkeypatch.delenv("ORB_T_REQUIRED", raising=False)
    with pytest.raises(RuntimeError, match="ORB_T_REQUIRED is not set"):
        config.env("ORB_T_REQUIRED", required=True)


def test_is_secret():
    assert config.is_secret("JEV_API_KEY") and config.is_secret("CF_WORKERS_TOKEN")
    assert not config.is_secret("LIVEKIT_URL")


def test_repo_root_finds_go_work(tmp_path, monkeypatch):
    (tmp_path / "go.work").write_text("go 1.26\n")
    (tmp_path / "a" / "b").mkdir(parents=True)
    monkeypatch.delenv("ORB_ROOT", raising=False)
    monkeypatch.chdir(tmp_path / "a" / "b")
    assert config.repo_root() == tmp_path.resolve()
