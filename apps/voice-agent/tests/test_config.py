"""The room-to-session rule, the .env filter, the LiveKit credentials rule and the settings."""
import pytest

from orb_voice.config import SESSION_ID, Settings, load_repo_env, session_for


def test_session_for_rooms_of_the_web_page_and_others():
    assert session_for("orb-20261021-191500-a1b2c3") == "20261021-191500-a1b2c3"
    a, b = session_for("Some Room!"), session_for("Some Room!")
    assert a == b and SESSION_ID.match(a) and a.startswith("room-")  # stable and valid
    assert session_for("orb-BAD ID") != "BAD ID"  # an invalid ID after the prefix does not pass


def test_repo_env_copies_only_orb_and_livekit_keys_and_the_gemini_key(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("ORB_TTS_VOICE=am_michael\nLIVEKIT_URL=ws://127.0.0.1:7880\n"
                                   "CF_WORKERS_TOKEN=do-not-copy\nORB_STT_LANGUAGE=de\nGOOGLE_API_KEY=g-key\n"
                                   "GOOGLE_APPLICATION_CREDENTIALS=/do/not/copy\nGOOGLE_API_KEY_OLD=old\n")
    for k in ("ORB_TTS_VOICE", "LIVEKIT_URL", "CF_WORKERS_TOKEN", "GOOGLE_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS",
              "GOOGLE_API_KEY_OLD"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("ORB_STT_LANGUAGE", "en")  # a set variable wins over the file
    loaded = load_repo_env(tmp_path)
    assert sorted(loaded) == ["GOOGLE_API_KEY", "LIVEKIT_URL", "ORB_TTS_VOICE"]
    import os
    assert "CF_WORKERS_TOKEN" not in os.environ and "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ
    assert "GOOGLE_API_KEY_OLD" not in os.environ and os.environ["ORB_STT_LANGUAGE"] == "en"


def test_settings_defaults_and_overrides(monkeypatch):
    for k in ("ORB_MIN_INTERRUPTION_WORDS", "ORB_PREEMPTIVE", "ORB_AGENT_URL", "ORB_LOG_TEXT"):
        monkeypatch.delenv(k, raising=False)
    s = Settings.from_env()
    assert (s.min_interruption_words, s.preemptive, s.min_delay, s.max_delay) == (2, True, 0.3, 2.5)
    assert (s.vad_threshold, s.vad_min_silence) == (0.5, 0.55) and s.log_text is False
    monkeypatch.setenv("ORB_PREEMPTIVE", "0")
    monkeypatch.setenv("ORB_AGENT_URL", "http://127.0.0.1:9100/")
    s = Settings.from_env()
    assert s.preemptive is False and s.agent_url == "http://127.0.0.1:9100"


def test_dev_credentials_only_on_this_machine():
    from orb_voice.config import check_livekit

    check_livekit({"LIVEKIT_URL": "ws://127.0.0.1:7880", "LIVEKIT_API_KEY": "devkey", "LIVEKIT_API_SECRET": "secret"})
    for env in [{"LIVEKIT_URL": "wss://lk.example.com", "LIVEKIT_API_KEY": "devkey", "LIVEKIT_API_SECRET": "x" * 40},
                {"LIVEKIT_URL": "wss://lk.example.com", "LIVEKIT_API_KEY": "k", "LIVEKIT_API_SECRET": "short"}]:
        with pytest.raises(ValueError, match="32 or more"):
            check_livekit(env)
    check_livekit({"LIVEKIT_URL": "wss://lk.example.com", "LIVEKIT_API_KEY": "k", "LIVEKIT_API_SECRET": "x" * 40})


def test_a_cloud_stt_gets_a_slower_interim_cadence(monkeypatch):
    monkeypatch.delenv("ORB_CLOUD_INTERIM_EVERY_S", raising=False)
    assert Settings.from_env().cloud_interim_every == 1.0 > Settings.from_env().interim_every
    monkeypatch.setenv("ORB_CLOUD_INTERIM_EVERY_S", "2.5")
    assert Settings.from_env().cloud_interim_every == 2.5
