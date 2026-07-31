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


def test_main_fuse_defaults_to_data_transcriptions(monkeypatch, tmp_path):
    """The --fuse branch's default output dir, which no other test reaches.

    Every other --fuse test passes an explicit --output-dir and mocks run_fusion
    to raise, so reverting this site to the old PROJECT_ROOT/output default left
    the suite green.
    """
    import audio_to_text.transcribe as t

    monkeypatch.chdir(tmp_path)
    primary = tmp_path / "teams.mp4"
    primary.touch()
    secondary = tmp_path / "phone.m4a"
    secondary.touch()

    seen: dict[str, object] = {}

    def fake_run_fusion(a, b, **kwargs):
        seen["output_dir"] = kwargs["output_dir"]
        out = kwargs["output_dir"] / "teams.md"
        out.write_text("x")
        return out

    monkeypatch.setattr(t, "ensure_apple_silicon", lambda: None)
    monkeypatch.setattr(t, "resolve_hf_token", lambda: "fake-token")
    monkeypatch.setattr(t, "load_diarization_pipeline", lambda token: object())
    monkeypatch.setattr("audio_to_text.fusion.run_fusion", fake_run_fusion)

    exit_code = t.main([str(primary), "--fuse", str(secondary)])

    assert exit_code == 0
    assert seen["output_dir"] == (tmp_path / "data" / "transcriptions").resolve()
    assert (tmp_path / "data" / "transcriptions").is_dir()


def test_main_reports_missing_data_dir_with_actionable_guidance(monkeypatch, tmp_path, capsys):
    """The 'no data/ directory' branch is a user-facing deliverable of this change
    and was entirely unpinned -- replacing its condition with `elif False:` left
    the suite green."""
    import audio_to_text.transcribe as t

    monkeypatch.chdir(tmp_path)          # no ./data here
    monkeypatch.setattr(t, "ensure_apple_silicon", lambda: None)

    exit_code = t.main([])

    err = capsys.readouterr().err
    assert exit_code == 1
    assert "no 'data/' directory" in err
    assert str(tmp_path) in err          # names the CALLER's cwd, not the checkout
    assert "audio-to-text path/to/recording.m4a" in err


def test_main_writes_into_data_transcriptions_by_default(monkeypatch, tmp_path):
    """End-to-end through main() with no --output-dir: the whole point of the
    feature. The unit tests above cover resolve_output_dir in isolation, but only
    this one proves main() actually routes through it -- a main() that kept
    writing to a stale constant would pass every other test in this file.
    """
    import audio_to_text.transcribe as t

    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "meeting.m4a").touch()

    fake_result = {
        "segments": [
            {"words": [{"word": "hi", "start": 0.0, "end": 0.5, "probability": 0.9}]},
        ],
    }

    with monkeypatch.context() as m:
        m.setattr(t, "ensure_apple_silicon", lambda: None)
        m.setattr(t, "resolve_hf_token", lambda: "fake-token")
        m.setattr(t, "load_diarization_pipeline", lambda token: object())
        m.setattr(t, "preprocess_audio", lambda media_path, tmp_dir, audio_filter: media_path)
        m.setattr(t, "run_whisper", lambda *a, **k: fake_result)
        m.setattr(
            t, "run_diarization",
            lambda source, pipeline, *, num_speakers=None: (
                [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}], {}
            ),
        )
        exit_code = t.main([])

    assert exit_code == 0
    assert (tmp_path / "data" / "transcriptions" / "meeting.md").is_file()
    # and NOT in the project root, which is what it would do if the default were cwd
    assert not (tmp_path / "meeting.md").exists()


def test_main_warns_on_repetition_loop_in_the_single_file_path(monkeypatch, tmp_path, capsys):
    """Pins the single-file WIRING of the loop warning.

    Both paths produce hallucination loops, so both must report them. Without
    this, deleting main()'s warn_on_repetition_loops call left the whole suite
    green -- the fusion path's test does not reach this call site.
    """
    import audio_to_text.transcribe as t

    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "meeting.m4a").touch()

    fake_result = {
        "segments": [
            {"words": [
                {"word": " Lars.", "start": float(i) / 10, "end": float(i) / 10 + 0.05,
                 "probability": 0.4}
                for i in range(30)
            ]},
        ],
    }

    with monkeypatch.context() as m:
        m.setattr(t, "ensure_apple_silicon", lambda: None)
        m.setattr(t, "resolve_hf_token", lambda: "fake-token")
        m.setattr(t, "load_diarization_pipeline", lambda token: object())
        m.setattr(t, "preprocess_audio", lambda media_path, tmp_dir, audio_filter: media_path)
        m.setattr(t, "run_whisper", lambda *a, **k: fake_result)
        m.setattr(
            t, "run_diarization",
            lambda source, pipeline, *, num_speakers=None: (
                [{"start": 0.0, "end": 10.0, "speaker": "SPEAKER_00"}], {}
            ),
        )
        exit_code = t.main([])

    assert exit_code == 0
    err = capsys.readouterr().err
    assert "repetition loop" in err.lower()
    assert "lars" in err.lower()
