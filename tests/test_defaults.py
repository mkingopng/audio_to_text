"""Defaults must follow the caller, not this checkout. PROJECT_ROOT-based
defaults meant a no-arg run from another project scanned *this* repo's data/."""
from audio_to_text.transcribe import (
    default_input_dir,
    gather_media,
    resolve_output_dir,
)


def test_default_input_dir_follows_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    assert default_input_dir() == tmp_path / "data"


def test_default_output_is_data_transcriptions(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    assert resolve_output_dir(None) == (tmp_path / "data" / "transcriptions").resolve()


def test_default_output_dir_is_created(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    out = resolve_output_dir(None)

    assert out.is_dir()


def test_explicit_output_dir_overrides_and_is_created(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "somewhere" / "else"

    out = resolve_output_dir(target)

    assert out == target.resolve()
    assert out.is_dir()


def test_gather_media_defaults_to_cwd_data(monkeypatch, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "meeting.m4a").touch()
    monkeypatch.chdir(tmp_path)

    assert gather_media(None) == [data / "meeting.m4a"]


def test_gather_media_ignores_the_transcriptions_subdir(monkeypatch, tmp_path):
    """Output nests inside input, so a second batch run must not descend into
    it. Safe only because gather_media uses iterdir() + is_file(); a switch to
    rglob() would silently break this."""
    data = tmp_path / "data"
    (data / "transcriptions").mkdir(parents=True)
    (data / "meeting.m4a").touch()
    (data / "transcriptions" / "meeting.md").touch()
    (data / "transcriptions" / "stray.m4a").touch()
    monkeypatch.chdir(tmp_path)

    assert gather_media(None) == [data / "meeting.m4a"]


def test_gather_media_returns_nothing_when_no_data_dir(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    assert gather_media(None) == []


def test_gather_media_still_returns_an_explicitly_named_missing_file(monkeypatch, tmp_path):
    """Existence of a named file is checked later, in run_whisper, so that the
    error names the file rather than being swallowed here as 'nothing found'."""
    monkeypatch.chdir(tmp_path)
    missing = tmp_path / "nope.m4a"

    assert gather_media(missing) == [missing]
