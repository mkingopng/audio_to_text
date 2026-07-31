"""Unit tests for the pure speaker-attribution/grouping/rendering logic in transcribe.py."""
import subprocess
from pathlib import Path
from unittest.mock import patch

from audio_to_text.transcribe import align_words_to_speakers
from audio_to_text.transcribe import (
    _group_consecutive,
    group_into_turns,
    relabel_speakers,
    render_markdown,
)
from audio_to_text.transcribe import build_ffmpeg_args
from audio_to_text.transcribe import extract_words, run_whisper


def test_align_words_to_speakers_assigns_by_max_overlap():
    turns = [
        {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
        {"start": 5.0, "end": 10.0, "speaker": "SPEAKER_01"},
    ]
    words = [
        {"word": "Hello", "start": 0.5, "end": 1.0, "probability": 0.9},
        {"word": " world", "start": 6.0, "end": 6.5, "probability": 0.9},
        # straddles the turn boundary at 5.0s, but spends more time (0.6s) in
        # turn 2 than turn 1 (0.2s) -- must be assigned to SPEAKER_01, proving
        # word-level (not whole-segment) attribution.
        {"word": " boundary", "start": 4.8, "end": 5.6, "probability": 0.8},
    ]

    result = align_words_to_speakers(words, turns)

    assert [w["speaker"] for w in result] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_01"]
    # original word fields are preserved
    assert result[0]["word"] == "Hello"
    assert result[0]["probability"] == 0.9


def test_align_words_to_speakers_handles_gap_with_no_overlapping_turn():
    turns = [
        {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
        {"start": 4.0, "end": 6.0, "speaker": "SPEAKER_01"},
    ]
    # falls in the silent gap between the two turns, closer to turn 2
    words = [{"word": "gap", "start": 3.6, "end": 3.8, "probability": 0.7}]

    result = align_words_to_speakers(words, turns)

    assert result[0]["speaker"] == "SPEAKER_01"


def test_group_consecutive_merges_same_speaker_words_into_one_turn():
    aligned = [
        {"word": "Hello", "start": 0.0, "end": 0.5, "probability": 0.9, "speaker": "SPEAKER_00"},
        {"word": " there", "start": 0.5, "end": 1.0, "probability": 0.8, "speaker": "SPEAKER_00"},
        {"word": " hi", "start": 1.0, "end": 1.3, "probability": 0.7, "speaker": "SPEAKER_01"},
    ]

    turns = _group_consecutive(aligned)

    assert len(turns) == 2
    assert turns[0]["speaker"] == "SPEAKER_00"
    assert turns[0]["text"] == "Hello there"
    assert turns[0]["start"] == 0.0
    assert turns[0]["end"] == 1.0
    assert turns[0]["confidence"] == (0.9 + 0.8) / 2
    assert turns[1]["speaker"] == "SPEAKER_01"
    assert turns[1]["text"] == "hi"


def test_relabel_speakers_is_stable_and_does_not_renumber_on_return():
    turns = [
        {"speaker": "SPEAKER_05", "start": 0.0, "end": 1.0, "text": "a", "confidence": 0.9},
        {"speaker": "SPEAKER_02", "start": 1.0, "end": 2.0, "text": "b", "confidence": 0.9},
        # SPEAKER_05 talks again later -- must stay "Person 1", not become "Person 3"
        {"speaker": "SPEAKER_05", "start": 2.0, "end": 3.0, "text": "c", "confidence": 0.9},
    ]

    relabeled = relabel_speakers(turns)

    assert [t["speaker"] for t in relabeled] == ["Person 1", "Person 2", "Person 1"]


def test_group_into_turns_combines_grouping_and_relabeling():
    aligned = [
        {"word": "Hi", "start": 0.0, "end": 0.5, "probability": 0.9, "speaker": "SPEAKER_03"},
        {"word": " yo", "start": 0.5, "end": 1.0, "probability": 0.9, "speaker": "SPEAKER_01"},
    ]

    turns = group_into_turns(aligned)

    assert turns[0]["speaker"] == "Person 1"
    assert turns[1]["speaker"] == "Person 2"


def test_render_markdown_formats_heading_and_timestamp():
    turns = [
        {"speaker": "Person 1", "start": 3.0, "text": "Guidelines, but here you can see."},
        {"speaker": "Person 3", "start": 102.0, "text": "Yep, we have both choices."},
        # past the 1-hour mark -- must render hh:mm:ss
        {"speaker": "Person 1", "start": 3725.0, "text": "Back again."},
    ]

    markdown = render_markdown(turns)

    assert "## Person 1 — 00:03\n\nGuidelines, but here you can see.\n" in markdown
    assert "## Person 3 — 01:42\n\nYep, we have both choices.\n" in markdown
    assert "## Person 1 — 01:02:05\n\nBack again.\n" in markdown


def test_render_markdown_clamps_negative_timestamp_to_zero():
    """_shift_and_remap (audio_to_text/fusion.py) clips negative starts to 0.0, but this is a
    defensive belt-and-braces check: negative seconds must never render as the
    divmod-garbage hh:mm:ss a naive negative int() would produce (e.g. -1:59:59)."""
    turns = [{"speaker": "Person 1", "start": -2.5, "text": "should clamp to zero"}]

    markdown = render_markdown(turns)

    assert "## Person 1 — 00:00\n\nshould clamp to zero\n" in markdown


def test_build_ffmpeg_args_without_filter():
    args = build_ffmpeg_args(Path("in.mp4"), Path("/tmp/out.wav"), None)

    assert args == [
        "ffmpeg", "-nostdin", "-y", "-i", "in.mp4",
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", "/tmp/out.wav",
    ]


def test_build_ffmpeg_args_with_filter():
    args = build_ffmpeg_args(Path("in.m4a"), Path("/tmp/out.wav"), "highpass=f=80,loudnorm=I=-16:TP=-1.5:LRA=11")

    assert "-af" in args
    assert args[args.index("-af") + 1] == "highpass=f=80,loudnorm=I=-16:TP=-1.5:LRA=11"
    assert args[-1] == "/tmp/out.wav"


def test_run_whisper_requests_word_timestamps_by_default(tmp_path):
    media = tmp_path / "clip.wav"
    media.write_bytes(b"fake wav bytes")
    fake_result = {"text": "hi", "segments": [], "language": "en"}

    with patch("audio_to_text.transcribe.mlx_whisper.transcribe", return_value=fake_result) as mock_transcribe:
        result = run_whisper(media, model_repo="mlx-community/whisper-large-v3-turbo")

    assert result == fake_result
    assert mock_transcribe.call_args.kwargs["word_timestamps"] is True


def test_run_whisper_word_timestamps_can_be_disabled(tmp_path):
    media = tmp_path / "clip.wav"
    media.write_bytes(b"fake wav bytes")

    with patch("audio_to_text.transcribe.mlx_whisper.transcribe", return_value={}) as mock_transcribe:
        run_whisper(media, model_repo="turbo", word_timestamps=False)

    assert mock_transcribe.call_args.kwargs["word_timestamps"] is False


def test_extract_words_flattens_segments_in_order():
    result = {
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "Hi", "words": [
                {"word": "Hi", "start": 0.0, "end": 0.5, "probability": 0.9},
            ]},
            {"start": 1.0, "end": 2.0, "text": " there", "words": [
                {"word": " there", "start": 1.0, "end": 1.5, "probability": 0.8},
            ]},
        ],
    }

    words = extract_words(result)

    assert [w["word"] for w in words] == ["Hi", " there"]


import numpy as np
import pytest

from audio_to_text.transcribe import load_diarization_pipeline, run_diarization


class _FakeTurn:
    def __init__(self, start, end):
        self.start = start
        self.end = end


class _FakeDiarization:
    def __init__(self, tracks):
        self._tracks = tracks  # list of (start, end, label)

    def itertracks(self, yield_label=True):
        for start, end, label in self._tracks:
            yield _FakeTurn(start, end), None, label

    def labels(self):
        return sorted({label for _, _, label in self._tracks})


class _FakeDiarizeOutput:
    """Mirrors pyannote 4.x's DiarizeOutput: `.speaker_diarization` /
    `.speaker_embeddings` attributes, not a `(diarization, embeddings)` tuple."""

    def __init__(self, speaker_diarization, speaker_embeddings):
        self.speaker_diarization = speaker_diarization
        self.speaker_embeddings = speaker_embeddings


def test_run_diarization_parses_pipeline_output_sorted_by_start():
    tracks = [(5.0, 9.0, "SPEAKER_00"), (0.0, 5.0, "SPEAKER_01")]
    embeddings_array = np.array([[1.0, 0.0], [0.0, 1.0]])  # row 0 -> SPEAKER_00, row 1 -> SPEAKER_01 (sorted order)

    def fake_pipeline(path, **kwargs):
        assert kwargs == {"num_speakers": 2}
        return _FakeDiarizeOutput(_FakeDiarization(tracks), embeddings_array)

    turns, embeddings = run_diarization(Path("fake.wav"), fake_pipeline, num_speakers=2)

    assert turns == [
        {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_01"},
        {"start": 5.0, "end": 9.0, "speaker": "SPEAKER_00"},
    ]
    assert set(embeddings.keys()) == {"SPEAKER_00", "SPEAKER_01"}
    assert np.array_equal(embeddings["SPEAKER_00"], np.array([1.0, 0.0]))
    assert np.array_equal(embeddings["SPEAKER_01"], np.array([0.0, 1.0]))


def test_run_diarization_omits_num_speakers_when_not_given():
    def fake_pipeline(path, **kwargs):
        assert kwargs == {}
        return _FakeDiarizeOutput(_FakeDiarization([(0.0, 1.0, "SPEAKER_00")]), np.array([[1.0]]))

    run_diarization(Path("fake.wav"), fake_pipeline, num_speakers=None)


def test_run_diarization_warns_on_speaker_count_mismatch(capsys):
    def fake_pipeline(path, **kwargs):
        return _FakeDiarizeOutput(_FakeDiarization([(0.0, 1.0, "SPEAKER_00")]), np.array([[1.0]]))

    run_diarization(Path("fake.wav"), fake_pipeline, num_speakers=6)

    assert "warning" in capsys.readouterr().err.lower()


def test_load_diarization_pipeline_raises_clear_error_without_token():
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        load_diarization_pipeline(None)


def test_main_continues_batch_after_diarization_failure(tmp_path, capsys):
    """A diarization/alignment failure on one file must not abort the rest of the batch.

    Mirrors the existing FileNotFoundError/CalledProcessError per-file resilience:
    the failing file is counted and skipped, the next file still gets written.
    """
    import audio_to_text.transcribe as t

    media_files = [Path("fake1.m4a"), Path("fake2.m4a")]
    fake_result = {
        "segments": [
            {"words": [{"word": "hi", "start": 0.0, "end": 0.5, "probability": 0.9}]},
        ],
    }

    call_count = {"n": 0}

    def fake_run_diarization(source, pipeline, *, num_speakers=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # simulate a near-silent/short file: pyannote detects zero turns,
            # which align_words_to_speakers turns into a ValueError.
            raise ValueError("align_words_to_speakers: no diarization turns to align against")
        return [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}], {}

    with patch.object(t, "ensure_apple_silicon"), \
         patch.object(t, "gather_media", return_value=media_files), \
         patch.object(t, "resolve_hf_token", return_value="fake-token"), \
         patch.object(t, "load_diarization_pipeline", return_value=object()), \
         patch.object(t, "preprocess_audio", side_effect=lambda media_path, tmp_dir, audio_filter: media_path), \
         patch.object(t, "run_whisper", return_value=fake_result), \
         patch.object(t, "run_diarization", side_effect=fake_run_diarization):
        exit_code = t.main(["ignored.m4a", "--output-dir", str(tmp_path)])

    assert exit_code == 1  # one of two files failed
    assert not (tmp_path / "fake1.md").exists()
    assert (tmp_path / "fake2.md").exists()
    assert "error: diarization failed for 'fake1.m4a'" in capsys.readouterr().err


def _run_main_capturing_audio_filter(argv, tmp_path):
    """Drive main()'s batch path with everything heavy mocked, returning the
    audio_filter value it passed to preprocess_audio."""
    import audio_to_text.transcribe as t

    seen = {}
    fake_result = {
        "segments": [{"words": [{"word": "hi", "start": 0.0, "end": 0.5, "probability": 0.9}]}],
    }

    def fake_preprocess(media_path, tmp_dir, audio_filter):
        seen["audio_filter"] = audio_filter
        return media_path

    with patch.object(t, "ensure_apple_silicon"), \
         patch.object(t, "gather_media", return_value=[Path("fake.m4a")]), \
         patch.object(t, "resolve_hf_token", return_value="fake-token"), \
         patch.object(t, "load_diarization_pipeline", return_value=object()), \
         patch.object(t, "preprocess_audio", side_effect=fake_preprocess), \
         patch.object(t, "run_whisper", return_value=fake_result), \
         patch.object(t, "run_diarization",
                       return_value=([{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}], {})):
        exit_code = t.main([*argv, "--output-dir", str(tmp_path)])

    assert exit_code == 0
    return seen["audio_filter"]


def test_main_applies_no_audio_filter_by_default(tmp_path):
    """Regression guard on previously-working behaviour: --preprocess/--denoise are
    opt-in. Task 4 made *extraction* unconditional (diarization needs the WAV
    regardless), and it would be easy for a later change to make the filter chain
    unconditional along with it -- which would silently alter the audio every
    existing no-flag invocation feeds to Whisper."""
    assert _run_main_capturing_audio_filter(["fake.m4a"], tmp_path) is None


def test_main_applies_default_filter_chain_with_preprocess_flag(tmp_path):
    """Regression guard: --preprocess still builds and passes the highpass+loudnorm
    chain (no denoise stage)."""
    audio_filter = _run_main_capturing_audio_filter(["fake.m4a", "--preprocess"], tmp_path)

    assert audio_filter == "highpass=f=80,loudnorm=I=-16:TP=-1.5:LRA=11"


def test_main_adds_denoise_stage_and_implies_preprocess(tmp_path):
    """Regression guard: --denoise alone implies preprocessing and inserts the
    afftdn stage between the highpass and loudnorm stages."""
    audio_filter = _run_main_capturing_audio_filter(["fake.m4a", "--denoise"], tmp_path)

    assert audio_filter == "highpass=f=80,afftdn=nf=-25,loudnorm=I=-16:TP=-1.5:LRA=11"


def test_main_audio_filter_override_wins(tmp_path):
    """Regression guard: an explicit --audio-filter overrides the built chain
    (and implies preprocessing) rather than being combined with it."""
    audio_filter = _run_main_capturing_audio_filter(
        ["fake.m4a", "--audio-filter", "highpass=f=200", "--denoise"], tmp_path
    )

    assert audio_filter == "highpass=f=200"


def test_main_fuse_reports_clean_error_on_missing_media_file(tmp_path, capsys):
    """--fuse must fail cleanly, before running any pipeline, if the primary media
    file doesn't exist. This is a real (unmocked) pre-flight check -- run_fusion
    is never reached, so there's nothing to mock."""
    import audio_to_text.transcribe as t

    missing_primary = tmp_path / "primary.m4a"  # deliberately not created
    secondary = tmp_path / "secondary.m4a"
    secondary.touch()

    with patch.object(t, "ensure_apple_silicon"), \
         patch.object(t, "gather_media", return_value=[missing_primary]):
        exit_code = t.main([
            str(missing_primary), "--fuse", str(secondary), "--output-dir", str(tmp_path),
        ])

    assert exit_code == 1
    assert f"error: no such file: '{missing_primary}'" in capsys.readouterr().err


def test_main_fuse_reports_clean_error_on_missing_fuse_file(tmp_path, capsys):
    """Same pre-flight existence check, for the secondary (--fuse) file."""
    import audio_to_text.transcribe as t

    primary = tmp_path / "primary.m4a"
    primary.touch()
    missing_secondary = tmp_path / "secondary.m4a"  # deliberately not created

    with patch.object(t, "ensure_apple_silicon"), \
         patch.object(t, "gather_media", return_value=[primary]):
        exit_code = t.main([
            str(primary), "--fuse", str(missing_secondary), "--output-dir", str(tmp_path),
        ])

    assert exit_code == 1
    assert f"error: no such file: '{missing_secondary}'" in capsys.readouterr().err


def test_main_fuse_reports_clean_error_when_ffmpeg_missing(tmp_path, capsys):
    """--fuse must fail cleanly if ffmpeg itself isn't on PATH (the actual scenario
    that produces a FileNotFoundError from run_fusion/preprocess_audio -- a plain
    missing source file is caught earlier by the pre-flight existence check above,
    before run_fusion is ever called)."""
    import audio_to_text.transcribe as t

    primary = tmp_path / "primary.m4a"
    primary.touch()
    secondary = tmp_path / "secondary.m4a"
    secondary.touch()

    with patch.object(t, "ensure_apple_silicon"), \
         patch.object(t, "gather_media", return_value=[primary]), \
         patch.object(t, "resolve_hf_token", return_value="fake-token"), \
         patch.object(t, "load_diarization_pipeline", return_value=object()), \
         patch("audio_to_text.fusion.run_fusion",
               side_effect=FileNotFoundError("ffmpeg not found on PATH; cannot extract audio.")):
        exit_code = t.main([
            str(primary), "--fuse", str(secondary), "--output-dir", str(tmp_path),
        ])

    assert exit_code == 1
    assert "error: ffmpeg not found on PATH" in capsys.readouterr().err


def test_main_fuse_reports_clean_error_on_ffmpeg_failure(tmp_path, capsys):
    """--fuse must fail cleanly if ffmpeg preprocessing fails for either source."""
    import audio_to_text.transcribe as t

    primary = tmp_path / "primary.m4a"
    primary.touch()
    secondary = tmp_path / "secondary.m4a"
    secondary.touch()

    ffmpeg_error = subprocess.CalledProcessError(
        returncode=1, cmd=["ffmpeg"], stderr=b"invalid data found when processing input"
    )

    with patch.object(t, "ensure_apple_silicon"), \
         patch.object(t, "gather_media", return_value=[primary]), \
         patch.object(t, "resolve_hf_token", return_value="fake-token"), \
         patch.object(t, "load_diarization_pipeline", return_value=object()), \
         patch("audio_to_text.fusion.run_fusion", side_effect=ffmpeg_error):
        exit_code = t.main([
            str(primary), "--fuse", str(secondary), "--output-dir", str(tmp_path),
        ])

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "error: ffmpeg preprocessing failed" in err
    assert "invalid data found when processing input" in err


def test_main_fuse_reports_clean_error_on_speaker_count_mismatch(tmp_path, capsys):
    """match_speakers raises ValueError when the two sources' detected speaker
    counts differ -- a real, expected outcome (not a corner case) when running
    without --num-speakers. Before this fix this reached main() as an unhandled
    traceback after both sources' full ASR + diarization passes had already run;
    it must instead print a clean error and return 1."""
    import audio_to_text.transcribe as t

    primary = tmp_path / "primary.m4a"
    primary.touch()
    secondary = tmp_path / "secondary.m4a"
    secondary.touch()

    with patch.object(t, "ensure_apple_silicon"), \
         patch.object(t, "gather_media", return_value=[primary]), \
         patch.object(t, "resolve_hf_token", return_value="fake-token"), \
         patch.object(t, "load_diarization_pipeline", return_value=object()), \
         patch("audio_to_text.fusion.run_fusion",
               side_effect=ValueError("len(embeddings_a) != len(embeddings_b): 6 vs 5")):
        exit_code = t.main([
            str(primary), "--fuse", str(secondary), "--output-dir", str(tmp_path),
        ])

    assert exit_code == 1
    assert "error: len(embeddings_a) != len(embeddings_b)" in capsys.readouterr().err


def test_main_fuse_reports_clean_error_when_diarization_pipeline_fails_to_load(tmp_path, capsys):
    """load_diarization_pipeline's own clear RuntimeError (missing/invalid HF_TOKEN,
    gated model terms not accepted) must surface as a clean error+return 1 in the
    --fuse path too, not an unhandled traceback."""
    import audio_to_text.transcribe as t

    primary = tmp_path / "primary.m4a"
    primary.touch()
    secondary = tmp_path / "secondary.m4a"
    secondary.touch()

    with patch.object(t, "ensure_apple_silicon"), \
         patch.object(t, "gather_media", return_value=[primary]), \
         patch.object(t, "resolve_hf_token", return_value="fake-token"), \
         patch.object(t, "load_diarization_pipeline",
                       side_effect=RuntimeError("HF_TOKEN is not set. ...")):
        exit_code = t.main([
            str(primary), "--fuse", str(secondary), "--output-dir", str(tmp_path),
        ])

    assert exit_code == 1
    assert "error: HF_TOKEN is not set" in capsys.readouterr().err


def test_fuse_module_invocation_resolves_fusion_import(tmp_path):
    """Regression: main()'s --fuse branch imports run_fusion lazily, inside the
    function body. That import is deferred because fusion.py imports nine names
    from transcribe.py at module level, so a top-level import here would be a
    circular import -- but a deferred import also defers its failure to runtime,
    after both sources' ASR passes would normally have run.

    This runs the real entry point as a subprocess (not an in-process import), so
    it is actually sensitive to import resolution rather than to sys.modules
    caching -- an in-process test would succeed via the already-imported
    audio_to_text.fusion module regardless of whether the import line is correct.

    Runs from an empty cwd with HOME redirected, so no .env on this machine can
    satisfy the token lookup; the run then fails fast at
    load_diarization_pipeline's own clean RuntimeError (no network, no model
    download), proving execution got past the import line.
    """
    import os
    import sys

    repo_root = Path(__file__).resolve().parent.parent
    primary = repo_root / "tests" / "__init__.py"  # any real, harmless file
    env = {**os.environ, "HF_TOKEN": "", "HOME": str(tmp_path)}

    result = subprocess.run(
        [sys.executable, "-m", "audio_to_text.transcribe",
         str(primary), "--fuse", str(primary)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60,
    )

    assert "ModuleNotFoundError" not in result.stderr
    assert "HF_TOKEN is not set" in result.stderr
    assert result.returncode == 1


def test_detect_repetition_loops_finds_a_loop_shredded_across_one_word_turns():
    """The instrument-blindness case, and the reason this scan is doc-wide.

    A Whisper repetition loop on ambiguous audio gets split across dozens of
    one-word turns by diarization jitter (~31% of blocks in the reference output
    hold a single word), so NO single block ever contains a repeat. The original
    within-block scan reported "1 block in 714, negligible" and declared the
    problem safely deferrable; re-measured doc-wide, one run in two carried a
    183-token loop of a word absent from the other run entirely.
    """
    from audio_to_text.transcribe import detect_repetition_loops

    turns = [
        {"speaker": f"S{i % 3}", "start": float(i), "end": float(i) + 0.1,
         "text": "Lars." if 5 <= i < 45 else "some ordinary words here"}
        for i in range(60)
    ]

    # No individual turn holds a repeat -- a within-block scan sees nothing.
    assert all(len(set(t["text"].split())) == len(t["text"].split()) for t in turns)

    loops = detect_repetition_loops(turns)

    assert len(loops) == 1
    assert loops[0]["token"] == "lars"
    assert loops[0]["count"] == 40
    assert loops[0]["start"] == 5.0


def test_detect_repetition_loops_ignores_ordinary_repeated_speech():
    """Threshold calibration, measured rather than assumed: across the 13,454
    tokens of the real reference transcript the longest legitimate consecutive
    run is 4 ('okay', 'yeah', 'easier' -- natural emphasis). Warning on those
    would cry wolf on every run.
    """
    from audio_to_text.transcribe import detect_repetition_loops

    turns = [
        {"speaker": "S0", "start": 0.0, "end": 1.0, "text": "okay okay okay okay"},
        {"speaker": "S1", "start": 1.0, "end": 2.0, "text": "yeah yeah"},
        {"speaker": "S0", "start": 2.0, "end": 3.0, "text": "that makes it easier easier easier"},
    ]

    assert detect_repetition_loops(turns) == []


def test_run_fusion_warns_when_the_transcript_contains_a_repetition_loop(tmp_path, monkeypatch, capsys):
    """Pins the WIRING, not just the detector. A pure function nothing calls is
    a feature that can be deleted with the suite still green.
    """
    from audio_to_text import fusion
    import numpy as np

    def fake_process_source(media_path, tmp_dir, **kwargs):
        wav_path = tmp_dir / (media_path.stem + ".clean.wav")
        wav_path.write_bytes(b"")
        turns = [
            {"speaker": "SPEAKER_00", "start": float(i), "end": float(i) + 0.1,
             "text": "Lars.", "confidence": 0.5} for i in range(30)
        ]
        return wav_path, turns, {"SPEAKER_00": np.array([1.0, 0.0])}

    monkeypatch.setattr(fusion, "_process_source", fake_process_source)
    monkeypatch.setattr(fusion, "_correlate_envelopes", lambda a, b: (0.0, 9.9))

    fusion.run_fusion(
        tmp_path / "a.mp4", tmp_path / "b.m4a",
        model_repo="x", language="en", initial_prompt=None, num_speakers=None,
        output_dir=tmp_path / "out", diarization_pipeline=object(),
    )

    err = capsys.readouterr().err
    assert "repetition" in err.lower()
    assert "lars" in err.lower()


def test_smooth_micro_turns_reattributes_a_sandwiched_fragment_and_keeps_every_word():
    """Diarization jitter steals a word or two out of one speaker's sentence and
    emits it as its own block under another speaker -- ~23% of blocks in the
    reference output hold a single word, half the headings introducing fragments
    like "So", "the", "it?".

    The fix RE-ATTRIBUTES rather than deletes: the fragment rejoins the sentence
    it came from, and no word is ever lost. That matters because the discriminator
    is imperfect, so the failure mode must be a mis-attributed word, not a
    missing one.
    """
    from audio_to_text.transcribe import smooth_micro_turns

    turns = [
        {"speaker": "S0", "start": 0.0, "end": 5.0, "confidence": 0.9,
         "text": "we can extend it to some of"},
        # zero-duration one-word fragment attributed to a different speaker
        {"speaker": "S1", "start": 5.0, "end": 5.0, "confidence": 0.4, "text": "the"},
        {"speaker": "S0", "start": 5.0, "end": 9.0, "confidence": 0.9,
         "text": "other things"},
    ]

    result = smooth_micro_turns(turns)

    assert len(result) == 1
    assert result[0]["speaker"] == "S0"
    assert result[0]["text"] == "we can extend it to some of the other things"
    assert (result[0]["start"], result[0]["end"]) == (0.0, 9.0)
    # every word survives
    before = sorted(w for t in turns for w in t["text"].split())
    assert sorted(result[0]["text"].split()) == before


def test_smooth_micro_turns_keeps_backchannel_turns():
    """"yeah"/"mm-hmm" are GENUINE one-word turns in a meeting, not jitter. Of the
    short blocks in the reference output 27% are lexical backchannel, and a length
    threshold alone cannot tell the two populations apart -- absorbing one
    misattributes real speech.
    """
    from audio_to_text.transcribe import smooth_micro_turns

    turns = [
        {"speaker": "S0", "start": 0.0, "end": 5.0, "confidence": 0.9, "text": "does that work"},
        {"speaker": "S1", "start": 5.0, "end": 5.0, "confidence": 0.9, "text": "Yeah"},
        {"speaker": "S0", "start": 5.0, "end": 9.0, "confidence": 0.9, "text": "good"},
    ]

    result = smooth_micro_turns(turns)

    assert [t["speaker"] for t in result] == ["S0", "S1", "S0"]


def test_smooth_micro_turns_keeps_a_short_turn_that_took_real_time():
    """Exact-zero duration is the jitter signal, not shortness. A short turn that
    actually occupies time on the clock is someone speaking.

    The 0.3s duration is deliberate: it sits inside the tempting "< 0.5s" band.
    That threshold was measured and REJECTED -- rules using it absorb genuine
    turns, including a 93-word one, because merged-turn durations were themselves
    corrupt. A test using a comfortably long turn would let the clause be relaxed
    from `== 0` to `< 0.5` with the suite still green; this one catches it.
    """
    from audio_to_text.transcribe import smooth_micro_turns

    turns = [
        {"speaker": "S0", "start": 0.0, "end": 5.0, "confidence": 0.9, "text": "so then"},
        {"speaker": "S1", "start": 5.0, "end": 5.3, "confidence": 0.9, "text": "Paul did"},
        {"speaker": "S0", "start": 5.3, "end": 9.0, "confidence": 0.9, "text": "right"},
    ]

    result = smooth_micro_turns(turns)

    assert [t["speaker"] for t in result] == ["S0", "S1", "S0"]


def test_smooth_micro_turns_never_moves_speech_across_a_speaker_boundary():
    """Only a SANDWICHED fragment is absorbed -- same speaker on both sides. If the
    neighbours differ, absorbing would move words from one person to another, which
    is a correctness bug rather than a readability fix.
    """
    from audio_to_text.transcribe import smooth_micro_turns

    turns = [
        {"speaker": "S0", "start": 0.0, "end": 5.0, "confidence": 0.9, "text": "first speaker"},
        {"speaker": "S1", "start": 5.0, "end": 5.0, "confidence": 0.4, "text": "and"},
        {"speaker": "S2", "start": 5.0, "end": 9.0, "confidence": 0.9, "text": "third speaker"},
    ]

    result = smooth_micro_turns(turns)

    assert [t["speaker"] for t in result] == ["S0", "S1", "S2"]


def test_group_into_turns_smooths_micro_turns_in_the_single_file_path():
    """Pins the single-file WIRING (main() routes through group_into_turns).

    Opt-in: smoothing re-attributes words between speakers, so group_into_turns
    only smooths when asked. test_group_into_turns_does_not_smooth_unless_asked
    pins the other direction.
    """
    from audio_to_text.transcribe import group_into_turns

    aligned = [
        {"word": "some", "start": 0.0, "end": 1.0, "probability": 0.9, "speaker": "S0"},
        {"word": " of", "start": 1.0, "end": 2.0, "probability": 0.9, "speaker": "S0"},
        # zero-duration jitter word attributed to another speaker
        {"word": " the", "start": 2.0, "end": 2.0, "probability": 0.4, "speaker": "S1"},
        {"word": " other", "start": 2.0, "end": 3.0, "probability": 0.9, "speaker": "S0"},
        {"word": " things", "start": 3.0, "end": 4.0, "probability": 0.9, "speaker": "S0"},
    ]

    turns = group_into_turns(aligned, smooth=True)

    assert len(turns) == 1
    assert turns[0]["text"] == "some of the other things"


def test_smooth_micro_turns_keeps_multi_word_backchannel():
    """"Yeah yeah" / "no no" are the commonest real two-word backchannels, and the
    rule admits turns of up to two words -- so they are squarely in scope.

    Keying the whole string at once squashed them to "yeahyeah", which is not in
    the token set, so the guard that exists to protect backchannel let the most
    common backchannel of all straight through.
    """
    from audio_to_text.transcribe import smooth_micro_turns

    for text in ("Yeah yeah", "no no", "yeah okay"):
        turns = [
            {"speaker": "S0", "start": 0.0, "end": 5.0, "confidence": 0.9, "text": "does that work"},
            {"speaker": "S1", "start": 5.0, "end": 5.0, "confidence": 0.9, "text": text},
            {"speaker": "S0", "start": 5.0, "end": 9.0, "confidence": 0.9, "text": "good"},
        ]

        result = smooth_micro_turns(turns)

        assert [t["speaker"] for t in result] == ["S0", "S1", "S0"], (
            f"{text!r} was absorbed as jitter"
        )


def test_smooth_micro_turns_keeps_hyphenated_backchannel():
    """The hyphen must survive normalisation, or "mm-hmm" and "uh-huh" fall out of
    the token set and get absorbed -- they are among the most frequent genuine
    one-word turns in the reference output.
    """
    from audio_to_text.transcribe import smooth_micro_turns

    for text in ("mm-hmm", "uh-huh"):
        turns = [
            {"speaker": "S0", "start": 0.0, "end": 5.0, "confidence": 0.9, "text": "and then we ship"},
            {"speaker": "S1", "start": 5.0, "end": 5.0, "confidence": 0.9, "text": text},
            {"speaker": "S0", "start": 5.0, "end": 9.0, "confidence": 0.9, "text": "right"},
        ]

        assert [t["speaker"] for t in smooth_micro_turns(turns)] == ["S0", "S1", "S0"], (
            f"{text!r} was absorbed as jitter"
        )


def test_smooth_micro_turns_word_limit_is_pinned_on_both_sides():
    """Two-sided pin on the word-count clause.

    Relaxing it to <= 4 absorbs 3- and 4-word REAL turns -- the destructive
    direction. Tightening it to <= 1 silently drops the two-word fragments the
    rule is documented to handle ("be a" inside "going to be a new season").
    """
    from audio_to_text.transcribe import smooth_micro_turns

    def run(fragment):
        turns = [
            {"speaker": "S0", "start": 0.0, "end": 5.0, "confidence": 0.9, "text": "going to"},
            {"speaker": "S1", "start": 5.0, "end": 5.0, "confidence": 0.4, "text": fragment},
            {"speaker": "S0", "start": 5.0, "end": 9.0, "confidence": 0.9, "text": "new season"},
        ]
        return smooth_micro_turns(turns)

    # two words -> absorbed (the documented "be a" case)
    assert len(run("be a")) == 1
    # three words -> a real turn, must survive
    assert [t["speaker"] for t in run("be a whole")] == ["S0", "S1", "S0"]


def test_repetition_loop_threshold_ignores_real_emphasis_and_catches_a_loop():
    """Two-sided pin anchored to MEASURED literals, not to the constant.

    Deriving the run lengths from REPETITION_LOOP_THRESHOLD makes the test track
    the constant, so it cannot fail when the constant moves. These numbers are
    measured: across the 13,454 tokens of the reference transcript the longest
    legitimate consecutive run is 4 ("okay", "yeah", "easier"), and an observed
    hallucination ran to 183. Warning on 4 would cry wolf on every real run.
    """
    from audio_to_text.transcribe import detect_repetition_loops

    def run(count, token="Lars."):
        return detect_repetition_loops([
            {"speaker": "S0", "start": float(i), "end": float(i) + 0.1, "text": token}
            for i in range(count)
        ])

    # the longest run real speech produced in the reference transcript
    assert run(4, "okay") == [], "warned on ordinary emphasis (4 repeats)"
    assert run(6) == [], "warned on 6 repeats -- below anything measured as a loop"
    # a genuine loop
    assert len(run(10)) == 1, "missed a 10-token loop"
    assert run(183)[0]["count"] == 183


def test_detect_repetition_loops_sees_through_mixed_punctuation():
    """A real Whisper loop is not one repeated string -- it comes out as
    "Lars. Lars, Lars!" with the punctuation varying. Tests that reuse one identical
    token never exercise the stripping, so it could be narrowed to "." unnoticed.
    """
    from audio_to_text.transcribe import detect_repetition_loops

    variants = ["Lars.", "Lars,", "Lars!", "Lars?", '"Lars"', "Lars'"]
    turns = [
        {"speaker": "S0", "start": float(i), "end": float(i) + 0.1,
         "text": variants[i % len(variants)]}
        for i in range(20)
    ]

    loops = detect_repetition_loops(turns)

    assert len(loops) == 1
    assert loops[0]["token"] == "lars"
    assert loops[0]["count"] == 20


def test_smooth_micro_turns_keeps_the_least_confident_score_when_absorbing():
    """An absorbed turn's confidence must be the MINIMUM of the three, not the
    maximum. The merged block now contains a fragment the diarizer was unsure
    about, so the combined turn is at most as trustworthy as its weakest part --
    taking the max would launder low-confidence text into a confident-looking one.
    """
    from audio_to_text.transcribe import smooth_micro_turns

    turns = [
        {"speaker": "S0", "start": 0.0, "end": 5.0, "confidence": 0.9, "text": "some of"},
        {"speaker": "S1", "start": 5.0, "end": 5.0, "confidence": 0.2, "text": "the"},
        {"speaker": "S0", "start": 5.0, "end": 9.0, "confidence": 0.8, "text": "other things"},
    ]

    result = smooth_micro_turns(turns)

    assert len(result) == 1
    assert result[0]["confidence"] == 0.2


def _jitter_fragment_words() -> list[dict]:
    """Aligned words that group into a sandwiched, zero-duration, non-backchannel
    fragment -- exactly what smooth_micro_turns absorbs. " so" has start == end,
    which is the jitter signal; shortness alone is not.
    """
    return [
        {"word": "Hi", "start": 0.0, "end": 1.0, "probability": 0.9, "speaker": "SPEAKER_00"},
        {"word": " there", "start": 1.0, "end": 2.0, "probability": 0.9, "speaker": "SPEAKER_00"},
        {"word": " so", "start": 2.0, "end": 2.0, "probability": 0.4, "speaker": "SPEAKER_01"},
        {"word": " we", "start": 2.0, "end": 3.0, "probability": 0.9, "speaker": "SPEAKER_00"},
        {"word": " continue", "start": 3.0, "end": 4.0, "probability": 0.9, "speaker": "SPEAKER_00"},
    ]


def test_group_into_turns_does_not_smooth_unless_asked():
    """Smoothing RE-ATTRIBUTES words between speakers, so it must not fire on a
    run that did not ask for it. It is the only change in this pipeline that can
    put one person's word under another person's heading, and its discriminator
    is admittedly imperfect -- so the default has to be the honest one, leaving
    diarization's own attribution alone.
    """
    turns = group_into_turns(_jitter_fragment_words())

    assert [t["speaker"] for t in turns] == ["Person 1", "Person 2", "Person 1"]
    assert turns[1]["text"] == "so"


def test_group_into_turns_smooths_when_asked():
    """The opt-in path: same words, smooth=True, and the fragment rejoins its
    sentence. Every word survives -- absorption moves words, never drops them.
    """
    turns = group_into_turns(_jitter_fragment_words(), smooth=True)

    assert [t["speaker"] for t in turns] == ["Person 1"]
    assert turns[0]["text"] == "Hi there so we continue"
