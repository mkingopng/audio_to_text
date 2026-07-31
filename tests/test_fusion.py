"""Unit tests for audio_to_text/fusion.py -- dual-source sync, speaker matching, turn merging."""
import pytest
import numpy as np
from scipy.io import wavfile

from audio_to_text.fusion import find_offset, match_speakers


def test_find_offset_recovers_known_delay(tmp_path):
    rate = 1000  # low sample rate keeps this test fast; the algorithm is rate-agnostic
    rng = np.random.default_rng(0)
    world = rng.normal(0, 0.01, 6000).astype(np.float32)
    world[2000:2100] += 5.0  # a distinctive "event" both sources should pick up

    delay_seconds = 1.5
    delay_samples = int(delay_seconds * rate)
    samples_a = world  # source A's clock starts at world-time 0
    samples_b = world[delay_samples:]  # source B started recording 1.5s later

    wav_a = tmp_path / "a.wav"
    wav_b = tmp_path / "b.wav"
    wavfile.write(wav_a, rate, samples_a)
    wavfile.write(wav_b, rate, samples_b)

    offset = find_offset(wav_a, wav_b)

    assert abs(offset - delay_seconds) < 0.15  # within one default window_seconds


def _unit_vector(angle_degrees: float) -> np.ndarray:
    radians = np.radians(angle_degrees)
    return np.array([np.cos(radians), np.sin(radians)])


def test_match_speakers_recovers_permutation_across_relabeled_sources():
    embeddings_a = {
        "SPEAKER_00": _unit_vector(0),
        "SPEAKER_01": _unit_vector(90),
        "SPEAKER_02": _unit_vector(200),
    }
    # same 3 physical speakers, different local labels, slightly perturbed angles
    embeddings_b = {
        "SPK_B": _unit_vector(92),
        "SPK_A": _unit_vector(2),
        "SPK_C": _unit_vector(198),
    }

    mapping = match_speakers(embeddings_a, embeddings_b)

    assert mapping == {"SPEAKER_00": "SPK_A", "SPEAKER_01": "SPK_B", "SPEAKER_02": "SPK_C"}


def test_match_speakers_uses_hungarian_not_greedy_nearest_match():
    """Greedy nearest-match would assign A_0→B_0, A_1→B_2, A_2→B_1 (cost 0.0641).
    Hungarian finds the globally optimal A_0→B_2, A_1→B_0, A_2→B_1 (cost 0.019).
    This test verifies that match_speakers does NOT use greedy."""
    embeddings_a = {
        "A_0": _unit_vector(0),
        "A_1": _unit_vector(10),
        "A_2": _unit_vector(160),
    }
    embeddings_b = {
        "B_0": _unit_vector(5),
        "B_1": _unit_vector(160),
        "B_2": _unit_vector(350),
    }

    mapping = match_speakers(embeddings_a, embeddings_b)

    # Greedy (processing A in order) would pick: A_0→B_0, A_1→B_2, A_2→B_1
    # Hungarian (optimal) picks: A_0→B_2, A_1→B_0, A_2→B_1
    assert mapping == {"A_0": "B_2", "A_1": "B_0", "A_2": "B_1"}


def test_match_speakers_raises_on_mismatched_sizes():
    """linear_sum_assignment silently returns partial mappings for rectangular matrices.
    match_speakers must explicitly guard against this, raising ValueError."""
    embeddings_a = {
        "A_0": _unit_vector(0),
        "A_1": _unit_vector(120),
    }
    embeddings_b = {
        "B_0": _unit_vector(0),
        "B_1": _unit_vector(120),
        "B_2": _unit_vector(240),
    }

    with pytest.raises(ValueError, match="len\\(embeddings_a\\) != len\\(embeddings_b\\)"):
        match_speakers(embeddings_a, embeddings_b)


from audio_to_text.fusion import _shift_and_remap, merge_turns


def test_shift_and_remap_applies_offset_and_speaker_map():
    turns = [{"speaker": "SPEAKER_00", "start": 1.0, "end": 2.0, "text": "hi", "confidence": 0.5}]

    result = _shift_and_remap(turns, offset=10.0, speaker_map={"SPEAKER_00": "SPEAKER_01"})

    assert result == [{"speaker": "SPEAKER_01", "start": 11.0, "end": 12.0, "text": "hi", "confidence": 0.5}]


def test_shift_and_remap_drops_turns_entirely_before_zero_and_clips_straddling_turn():
    """A negative offset (B's recording started before A's) means some of B's early
    turns predate A's timeline zero and have no valid position on the merged
    timeline. A turn ending before zero must be dropped entirely; one straddling
    zero must be clipped to start at 0, not shifted to a negative timestamp."""
    turns = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 3.0, "text": "before the call started", "confidence": 0.5},
        {"speaker": "SPEAKER_00", "start": 4.0, "end": 6.0, "text": "straddles zero", "confidence": 0.5},
        {"speaker": "SPEAKER_00", "start": 10.0, "end": 12.0, "text": "well after zero", "confidence": 0.5},
    ]

    result = _shift_and_remap(turns, offset=-5.0, speaker_map={"SPEAKER_00": "SPEAKER_01"})

    assert result == [
        {"speaker": "SPEAKER_01", "start": 0.0, "end": 1.0, "text": "straddles zero", "confidence": 0.5},
        {"speaker": "SPEAKER_01", "start": 5.0, "end": 7.0, "text": "well after zero", "confidence": 0.5},
    ]


def test_merge_turns_prefers_higher_confidence_source_and_appends_gaps():
    turns_a = [
        {"speaker": "A0", "start": 0.0, "end": 5.0, "text": "hello from a", "confidence": 0.5},
        {"speaker": "A1", "start": 5.0, "end": 9.0, "text": "garbled b word", "confidence": 0.3},
    ]
    turns_b_shifted = [
        # overlaps turn 2, HIGHER confidence -> should replace turn 2's text
        {"speaker": "A1", "start": 5.1, "end": 8.9, "text": "clear phone audio", "confidence": 0.9},
        # overlaps turn 1, LOWER confidence -> must NOT replace turn 1's text
        {"speaker": "A0", "start": 0.2, "end": 4.8, "text": "quieter mic", "confidence": 0.2},
        # doesn't overlap any A turn at all -> appended as a gap-fill
        {"speaker": "A2", "start": 9.0, "end": 11.0, "text": "only caught by phone", "confidence": 0.8},
    ]

    merged = merge_turns(turns_a, turns_b_shifted)

    assert merged == [
        {"speaker": "A0", "start": 0.0, "end": 5.0, "text": "hello from a", "confidence": 0.5},
        {"speaker": "A1", "start": 5.0, "end": 9.0, "text": "clear phone audio", "confidence": 0.9},
        {"speaker": "A2", "start": 9.0, "end": 11.0, "text": "only caught by phone", "confidence": 0.8},
    ]


def test_merge_turns_does_not_duplicate_a_b_turn_spanning_two_a_turns():
    """Regression: a single B turn overlapping two same-speaker A turns must
    replace at most one of them, not get copy-pasted into both (observed on
    the real Task 13 fusion run: identical text appeared under two separate
    Person N headings a few seconds apart)."""
    turns_a = [
        {"speaker": "A0", "start": 0.0, "end": 5.0, "text": "a turn one", "confidence": 0.3},
        {"speaker": "A1", "start": 5.0, "end": 6.0, "text": "interjection", "confidence": 0.9},
        {"speaker": "A0", "start": 6.0, "end": 11.0, "text": "a turn two", "confidence": 0.3},
    ]
    turns_b_shifted = [
        # one continuous B turn spanning both of A0's turns (and the interjection gap)
        {"speaker": "A0", "start": 0.1, "end": 10.9, "text": "one continuous phone turn", "confidence": 0.8},
    ]

    merged = merge_turns(turns_a, turns_b_shifted)

    a0_texts = [t["text"] for t in merged if t["speaker"] == "A0"]
    assert a0_texts == ["one continuous phone turn", "a turn two"]


def test_merge_turns_appends_b_turn_overlapping_a_different_speakers_a_turn():
    """A B turn that overlaps in TIME with a different speaker's A turn (e.g. one
    mic caught cross-talk the other missed entirely) is not a replacement
    candidate (speaker mismatch) but must still be appended as new content --
    not silently dropped just because some other speaker's A turn occupies
    that time range."""
    turns_a = [
        {"speaker": "A0", "start": 0.0, "end": 5.0, "text": "a says this", "confidence": 0.9},
    ]
    turns_b_shifted = [
        {"speaker": "A1", "start": 1.0, "end": 4.0, "text": "only b's mic caught this", "confidence": 0.95},
    ]

    merged = merge_turns(turns_a, turns_b_shifted)

    assert merged == [
        {"speaker": "A0", "start": 0.0, "end": 5.0, "text": "a says this", "confidence": 0.9},
        {"speaker": "A1", "start": 1.0, "end": 4.0, "text": "only b's mic caught this", "confidence": 0.95},
    ]


def test_run_fusion_uses_separate_tmp_dirs_for_each_source(tmp_path, monkeypatch):
    """Regression: two sources sharing a filename stem (e.g. two recordings each
    named "meeting", one .mp4 and one .m4a, or two files from different folders
    that happen to share a name) must not collide. preprocess_audio names its
    output after the stem alone, so if both sources were preprocessed into the
    same shared tmp_dir, the second ffmpeg run would silently overwrite the
    first's WAV -- corrupting find_offset into comparing a file against itself
    and returning a plausible-looking but meaningless 0.0 offset."""
    from audio_to_text import fusion

    seen_dirs = []

    def fake_process_source(media_path, tmp_dir, **kwargs):
        seen_dirs.append(tmp_dir)
        wav_path = tmp_dir / (media_path.stem + ".clean.wav")
        wav_path.write_bytes(b"")
        speaker = "SPEAKER_00"
        turns = [{"speaker": speaker, "start": 0.0, "end": 1.0, "text": "hi", "confidence": 0.9}]
        return wav_path, turns, {speaker: np.array([1.0, 0.0])}

    monkeypatch.setattr(fusion, "_process_source", fake_process_source)
    monkeypatch.setattr(fusion, "find_offset", lambda wav_a, wav_b: 0.0)

    out_path = fusion.run_fusion(
        tmp_path / "meeting.mp4", tmp_path / "meeting.m4a",
        model_repo="x", language="en", initial_prompt=None, num_speakers=None,
        output_dir=tmp_path / "out", diarization_pipeline=object(),
    )

    assert len(seen_dirs) == 2
    assert seen_dirs[0] != seen_dirs[1]
    assert out_path.exists()
