"""Unit tests for the pure speaker-attribution/grouping/rendering logic in transcribe.py."""
import subprocess
from pathlib import Path
from unittest.mock import patch

from src.transcribe import align_words_to_speakers
from src.transcribe import (
    _group_consecutive,
    group_into_turns,
    relabel_speakers,
    render_markdown,
)
from src.transcribe import build_ffmpeg_args
from src.transcribe import extract_words, run_whisper


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

    with patch("src.transcribe.mlx_whisper.transcribe", return_value=fake_result) as mock_transcribe:
        result = run_whisper(media, model_repo="mlx-community/whisper-large-v3-turbo")

    assert result == fake_result
    assert mock_transcribe.call_args.kwargs["word_timestamps"] is True


def test_run_whisper_word_timestamps_can_be_disabled(tmp_path):
    media = tmp_path / "clip.wav"
    media.write_bytes(b"fake wav bytes")

    with patch("src.transcribe.mlx_whisper.transcribe", return_value={}) as mock_transcribe:
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

from src.transcribe import load_diarization_pipeline, run_diarization


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
    import src.transcribe as t

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
         patch.object(t, "load_dotenv"), \
         patch.object(t, "load_diarization_pipeline", return_value=object()), \
         patch.object(t, "preprocess_audio", side_effect=lambda media_path, tmp_dir, audio_filter: media_path), \
         patch.object(t, "run_whisper", return_value=fake_result), \
         patch.object(t, "run_diarization", side_effect=fake_run_diarization):
        exit_code = t.main(["ignored.m4a", "--output-dir", str(tmp_path)])

    assert exit_code == 1  # one of two files failed
    assert not (tmp_path / "fake1.md").exists()
    assert (tmp_path / "fake2.md").exists()
    assert "error: diarization failed for 'fake1.m4a'" in capsys.readouterr().err


def test_main_fuse_reports_clean_error_on_missing_file(tmp_path, capsys):
    """--fuse must fail cleanly (no raw traceback) if a source file is missing.

    Unlike the batch loop, --fuse processes exactly one fusion attempt per
    invocation, so there's no "skip and continue" here -- just a clean
    'error: ...' message on stderr and a non-zero exit code.
    """
    import src.transcribe as t

    with patch.object(t, "ensure_apple_silicon"), \
         patch.object(t, "gather_media", return_value=[Path("primary.m4a")]), \
         patch.object(t, "load_dotenv"), \
         patch.object(t, "load_diarization_pipeline", return_value=object()), \
         patch("src.fusion.run_fusion", side_effect=FileNotFoundError("no such file: 'secondary.m4a'")):
        exit_code = t.main([
            "primary.m4a", "--fuse", "secondary.m4a", "--output-dir", str(tmp_path),
        ])

    assert exit_code == 1
    assert "error: no such file: 'secondary.m4a'" in capsys.readouterr().err


def test_main_fuse_reports_clean_error_on_ffmpeg_failure(tmp_path, capsys):
    """--fuse must fail cleanly if ffmpeg preprocessing fails for either source."""
    import src.transcribe as t

    ffmpeg_error = subprocess.CalledProcessError(
        returncode=1, cmd=["ffmpeg"], stderr=b"invalid data found when processing input"
    )

    with patch.object(t, "ensure_apple_silicon"), \
         patch.object(t, "gather_media", return_value=[Path("primary.m4a")]), \
         patch.object(t, "load_dotenv"), \
         patch.object(t, "load_diarization_pipeline", return_value=object()), \
         patch("src.fusion.run_fusion", side_effect=ffmpeg_error):
        exit_code = t.main([
            "primary.m4a", "--fuse", "secondary.m4a", "--output-dir", str(tmp_path),
        ])

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "error: ffmpeg preprocessing failed" in err
    assert "invalid data found when processing input" in err
