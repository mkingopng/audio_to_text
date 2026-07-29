"""Unit tests for the pure speaker-attribution/grouping/rendering logic in transcribe.py."""
from src.transcribe import align_words_to_speakers
from src.transcribe import (
    _group_consecutive,
    group_into_turns,
    relabel_speakers,
    render_markdown,
)


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
