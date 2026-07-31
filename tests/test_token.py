"""The token lookup is the difference between working and not working from
another project: a bare load_dotenv() searches upward from the caller's cwd,
finds that project's .env or nothing, and reports 'HF_TOKEN is not set'."""
import pytest

from audio_to_text.transcribe import resolve_hf_token


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """Every test starts with no ambient token and a cwd with no .env."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.chdir(tmp_path)


def test_environment_variable_wins(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("HF_TOKEN=from_cwd\n")
    monkeypatch.setenv("HF_TOKEN", "from_env")

    assert resolve_hf_token() == "from_env"


def test_cwd_dotenv_beats_global(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("HF_TOKEN=from_cwd\n")
    config = tmp_path / "config"
    config.mkdir()
    (config / ".env").write_text("HF_TOKEN=from_global\n")
    monkeypatch.setattr("audio_to_text.transcribe.CONFIG_DIR", config)

    assert resolve_hf_token() == "from_cwd"


def test_falls_back_to_global_config(monkeypatch, tmp_path):
    """The case that makes this whole feature work: called from a project that
    has no .env of its own."""
    config = tmp_path / "config"
    config.mkdir()
    (config / ".env").write_text("HF_TOKEN=from_global\n")
    monkeypatch.setattr("audio_to_text.transcribe.CONFIG_DIR", config)

    assert resolve_hf_token() == "from_global"


def test_returns_none_when_nowhere(monkeypatch, tmp_path):
    monkeypatch.setattr("audio_to_text.transcribe.CONFIG_DIR", tmp_path / "nonexistent")

    assert resolve_hf_token() is None


def test_empty_token_is_not_accepted(monkeypatch, tmp_path):
    """An exported-but-empty HF_TOKEN must fall through to the files rather than
    being treated as a real credential -- otherwise the failure surfaces much
    later, as an opaque 401 from Hugging Face."""
    monkeypatch.setenv("HF_TOKEN", "")
    config = tmp_path / "config"
    config.mkdir()
    (config / ".env").write_text("HF_TOKEN=from_global\n")
    monkeypatch.setattr("audio_to_text.transcribe.CONFIG_DIR", config)

    assert resolve_hf_token() == "from_global"


def test_main_passes_the_resolved_token_to_the_pipeline(monkeypatch, tmp_path):
    """The headline feature, pinned end-to-end through main().

    Every other test here exercises resolve_hf_token() directly, and the six
    main() tests elsewhere *stub* it -- so reverting both call sites to the old
    os.environ.get("HF_TOKEN") left the whole suite green. This is the only test
    that fails if main() stops using the resolution chain.
    """
    import audio_to_text.transcribe as t

    monkeypatch.chdir(tmp_path)          # no ./.env here
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "meeting.m4a").touch()

    config = tmp_path / "config"
    config.mkdir()
    (config / ".env").write_text("HF_TOKEN=from_global\n")
    monkeypatch.setattr(t, "CONFIG_DIR", config)

    seen: dict[str, object] = {}

    def capture(token):
        seen["token"] = token
        return object()

    fake_result = {
        "segments": [
            {"words": [{"word": "hi", "start": 0.0, "end": 0.5, "probability": 0.9}]},
        ],
    }

    monkeypatch.setattr(t, "ensure_apple_silicon", lambda: None)
    monkeypatch.setattr(t, "load_diarization_pipeline", capture)
    monkeypatch.setattr(t, "preprocess_audio", lambda media_path, tmp_dir, audio_filter: media_path)
    monkeypatch.setattr(t, "run_whisper", lambda *a, **k: fake_result)
    monkeypatch.setattr(
        t, "run_diarization",
        lambda source, pipeline, *, num_speakers=None: (
            [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}], {}
        ),
    )

    assert t.main([]) == 0
    assert seen["token"] == "from_global"


def test_does_not_mutate_process_environment(monkeypatch, tmp_path):
    """Reading a .env must not leak into os.environ -- that would make the
    lookup order depend on whatever ran earlier in the process."""
    import os

    (tmp_path / ".env").write_text("HF_TOKEN=from_cwd\n")
    monkeypatch.setattr("audio_to_text.transcribe.CONFIG_DIR", tmp_path / "nonexistent")

    resolve_hf_token()

    assert "HF_TOKEN" not in os.environ
