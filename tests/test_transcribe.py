"""Unit tests for the pure speaker-attribution/grouping/rendering logic in transcribe.py."""
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
