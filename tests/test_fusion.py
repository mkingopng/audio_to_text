"""Unit tests for audio_to_text/fusion.py -- dual-source sync, speaker matching, turn merging."""
from pathlib import Path

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

    # The replaced turn carries B's span (5.1-8.9) as well as B's text, not A's
    # (5.0-9.0): a turn's timestamps must describe the words it actually holds.
    assert merged == [
        {"speaker": "A0", "start": 0.0, "end": 5.0, "text": "hello from a", "confidence": 0.5},
        {"speaker": "A1", "start": 5.1, "end": 8.9, "text": "clear phone audio", "confidence": 0.9},
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


def test_merge_turns_consumes_a_sibling_a_turn_contained_in_the_winning_b_text():
    """The residual redundancy left behind after the used_b_ids fix.

    When B's diarization keeps as one turn what A's split into two same-speaker
    turns, B's text wins one of them -- and B's spanning text CONTAINS what the
    sibling A turn says, so the sibling is emitted again underneath. The earlier
    fix removed exact duplication; this is containment duplication, and it
    restated whole paragraphs (up to 325 shared characters) under a second
    heading in the reference output.
    """
    sibling = "we can approve this version, let us say we are happy"
    turns_a = [
        {"speaker": "A0", "start": 0.0, "end": 5.0, "text": "right okay", "confidence": 0.3},
        {"speaker": "A1", "start": 5.0, "end": 5.2, "text": "to", "confidence": 0.9},
        {"speaker": "A0", "start": 5.2, "end": 11.0, "text": sibling, "confidence": 0.3},
    ]
    turns_b_shifted = [
        {"speaker": "A0", "start": 0.1, "end": 10.9, "confidence": 0.95,
         "text": f"right okay {sibling}"},
    ]

    merged = merge_turns(turns_a, turns_b_shifted)

    texts = [t["text"] for t in merged]
    assert sum(sibling in t for t in texts) == 1, (
        f"the sibling turn is emitted twice -- once inside B's winning text and "
        f"once on its own: {texts}"
    )


def test_containment_guard_does_not_fire_across_speakers():
    """Restricted to same-speaker on purpose. A cross-speaker near-duplicate means
    one of the two headings is WRONG, and the guard cannot tell which -- firing
    there consumes the correctly attributed copy and keeps the misattributed one,
    worsening the defect. Measured on the real pair: at radius 2 an unrestricted
    guard fires on 28 cross-speaker cases.
    """
    shared = "the budget for the second quarter is what we agreed"
    turns_a = [
        {"speaker": "A0", "start": 0.0, "end": 5.0, "text": "opening remark", "confidence": 0.3},
        {"speaker": "A1", "start": 5.0, "end": 5.2, "text": "mm", "confidence": 0.9},
        # DIFFERENT speaker from the replaced turn
        {"speaker": "A2", "start": 5.2, "end": 11.0, "text": shared, "confidence": 0.3},
    ]
    turns_b_shifted = [
        {"speaker": "A0", "start": 0.1, "end": 10.9, "confidence": 0.95,
         "text": f"opening remark {shared}"},
    ]

    merged = merge_turns(turns_a, turns_b_shifted)

    assert any(t["speaker"] == "A2" and t["text"] == shared for t in merged), (
        "the guard consumed a different speaker's turn -- that deletes the copy "
        "that may well be the correctly attributed one"
    )


def test_containment_guard_does_not_swallow_short_micro_turns():
    """A minimum length, so the guard fixes redundancy and does not quietly become
    a fragmentation smoother.

    Measured on the real pair, the fires are bimodal: a cluster at <=8 normalized
    chars ('I', 'for', 'And', 'as', 'you know', 'Daniel') which are micro-turns,
    and a cluster at >=20 which are genuinely restated content. The floor sits in
    the empty gap between them. Absorbing micro-turns is a separate, riskier piece
    of work -- 'Daniel' may be a real one-word answer.
    """
    turns_a = [
        {"speaker": "A0", "start": 0.0, "end": 5.0, "text": "so anyway", "confidence": 0.3},
        {"speaker": "A1", "start": 5.0, "end": 5.2, "text": "mm", "confidence": 0.9},
        {"speaker": "A0", "start": 5.2, "end": 6.0, "text": "Daniel", "confidence": 0.3},
    ]
    turns_b_shifted = [
        {"speaker": "A0", "start": 0.1, "end": 10.9, "confidence": 0.95,
         "text": "so anyway Daniel is the one who owns that piece of work"},
    ]

    merged = merge_turns(turns_a, turns_b_shifted)

    assert any(t["text"] == "Daniel" for t in merged), (
        "a 6-character micro-turn was consumed; the guard must not smooth "
        "fragmentation as a side effect"
    )


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


def test_merge_turns_never_mixes_one_sources_span_with_anothers_text():
    """A merged turn's (start, end, text) must all come from a SINGLE source turn.

    The confidence-replacement branch used to emit B's text under A's start/end.
    When B's turn spans more speech than A's, the timestamps stop describing the
    text: the shipped reference output contains a 93-word block with a 0.4s
    duration, and the rendered mm:ss heading points at the wrong moment.

    Stated as an invariant rather than a single example, because the defect is a
    property of the branch, not of one input.
    """
    turns_a = [
        {"speaker": "S0", "start": 5.0, "end": 5.4, "text": "yeah", "confidence": 0.30},
        {"speaker": "S1", "start": 6.0, "end": 8.0, "text": "a different speaker", "confidence": 0.95},
    ]
    turns_b_shifted = [
        # B heard this as one continuous 90-second turn; A's diarization caught
        # only a 0.4s sliver of it.
        {"speaker": "S0", "start": 3.0, "end": 93.0, "confidence": 0.95,
         "text": "ninety seconds of speech that source B captured as a single turn"},
    ]

    merged = merge_turns(turns_a, turns_b_shifted)

    sources = {(t["start"], t["end"], t["text"]) for t in turns_a + turns_b_shifted}
    for turn in merged:
        assert (turn["start"], turn["end"], turn["text"]) in sources, (
            f"merged turn {turn['text']!r} carries span "
            f"[{turn['start']}, {turn['end']}] which belongs to a different source turn"
        )


def test_merge_turns_replacement_does_not_produce_impossible_speaking_rate():
    """The user-visible face of the timestamp corruption.

    93 words cannot be spoken in 0.4 seconds. Any merged turn implying a rate no
    human reaches means its timestamps describe different speech than its text.
    """
    text = " ".join(f"word{i}" for i in range(93))
    turns_a = [{"speaker": "S0", "start": 100.0, "end": 100.4, "text": "mm", "confidence": 0.3}]
    turns_b_shifted = [
        {"speaker": "S0", "start": 60.0, "end": 140.0, "text": text, "confidence": 0.9},
    ]

    merged = merge_turns(turns_a, turns_b_shifted)

    replaced = next(t for t in merged if t["text"] == text)
    duration = replaced["end"] - replaced["start"]
    rate = len(replaced["text"].split()) / duration
    assert rate < 10.0, (
        f"{len(text.split())} words in {duration}s = {rate:.0f} words/sec; "
        "the merged turn kept A's span while taking B's text"
    )


def _envelope_pair_wavs(tmp_path, rate=1000, seconds=120, shared=True, delay_seconds=8.0):
    """Two WAVs that either do (shared) or do not (not shared) record the same room."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    n = rate * seconds
    world = rng.normal(0, 0.01, n).astype(np.float32)
    # speech-like bursts, so the RMS envelope has structure to correlate on
    for start in range(0, n - rate, rate * 3):
        world[start:start + rng.integers(rate // 2, rate)] += rng.normal(0, 1.0)

    delay = int(delay_seconds * rate)
    a = world[:-delay] if delay else world
    if shared:
        b = world[delay:]
    else:
        other = rng.normal(0, 0.01, n).astype(np.float32)
        for start in range(0, n - rate, rate * 3):
            other[start:start + rng.integers(rate // 2, rate)] += rng.normal(0, 1.0)
        b = other[delay:]

    wav_a, wav_b = tmp_path / "a.wav", tmp_path / "b.wav"
    wavfile.write(wav_a, rate, a)
    wavfile.write(wav_b, rate, b)
    return wav_a, wav_b


def test_offset_confidence_separates_a_true_pair_from_unrelated_recordings():
    """find_offset returns the argmax lag with no measure of how peaked the
    correlation actually is, so two recordings that share no acoustic content
    still yield a confident-looking number.

    peak/best_rival (the best peak more than 5s from the argmax) is the metric
    that separates them. Measured on the real 70-minute pair: 1.5162 for the true
    pair against a null cluster at 1.0002-1.0050, built from those same two
    recordings so the separation is attributable to alignment, not to recording
    character. peak/median and the z-score were also measured and rejected --
    they separate by less than a factor of two.
    """
    from audio_to_text.fusion import _correlate_envelopes

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        true_a, true_b = _envelope_pair_wavs(tmp_path / "t", shared=True)
        _, true_confidence = _correlate_envelopes(true_a, true_b)

        false_a, false_b = _envelope_pair_wavs(tmp_path / "f", shared=False)
        _, false_confidence = _correlate_envelopes(false_a, false_b)

    assert true_confidence > 1.2, f"true pair scored only {true_confidence:.4f}"
    assert false_confidence < 1.2, f"unrelated pair scored {false_confidence:.4f}"


def test_run_fusion_reports_the_offset_and_warns_when_alignment_is_weak(
    tmp_path, monkeypatch, capsys
):
    """Pins the WIRING. src/fusion.py contained zero print statements and
    run_fusion never surfaced the offset, so every run got no visibility and no
    peak-quality signal. The recorded mitigation ("a human sanity-checks the
    offset") was a one-off validation script run once on one recording pair --
    a procedure, not a property of the tool.
    """
    from audio_to_text import fusion

    def fake_process_source(media_path, tmp_dir, **kwargs):
        wav_path = tmp_dir / (media_path.stem + ".clean.wav")
        wav_path.write_bytes(b"")
        turns = [{"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0, "text": "hi", "confidence": 0.9}]
        return wav_path, turns, {"SPEAKER_00": np.array([1.0, 0.0])}

    monkeypatch.setattr(fusion, "_process_source", fake_process_source)
    # a confident-looking lag with a flat correlation behind it
    monkeypatch.setattr(fusion, "_correlate_envelopes", lambda a, b: (26.1, 1.002))

    fusion.run_fusion(
        tmp_path / "a.mp4", tmp_path / "b.m4a",
        model_repo="x", language="en", initial_prompt=None, num_speakers=None,
        output_dir=tmp_path / "out", diarization_pipeline=object(),
    )

    out, err = capsys.readouterr()
    assert "26.1" in out, "the offset must be surfaced on every run"
    assert "warning" in err.lower()
    assert "overlap" in err.lower() or "align" in err.lower()


def _stub_fusion(monkeypatch, tmp_path):
    """Patch out the ASR/diarization/offset work so run_fusion's output naming
    can be tested without a real 70-minute pipeline run."""
    from audio_to_text import fusion

    def fake_process_source(media_path, tmp_dir, **kwargs):
        wav_path = tmp_dir / (media_path.stem + ".clean.wav")
        wav_path.write_bytes(b"")
        speaker = "SPEAKER_00"
        turns = [{"speaker": speaker, "start": 0.0, "end": 1.0, "text": "hi", "confidence": 0.9}]
        return wav_path, turns, {speaker: np.array([1.0, 0.0])}

    monkeypatch.setattr(fusion, "_process_source", fake_process_source)
    monkeypatch.setattr(fusion, "_correlate_envelopes", lambda a, b: (0.0, 9.9))
    return fusion


def test_run_fusion_writes_fused_suffix_and_leaves_single_file_transcript_intact(
    tmp_path, monkeypatch
):
    """A fused run must not silently destroy a single-file transcript of the same
    primary. Both modes named the output after the primary's stem, so transcribing
    teams.mp4 and then fusing teams.mp4 --fuse phone.m4a wrote teams.md twice, the
    second run replacing the first with no warning.

    Fusion now writes <stem>.fused.md, so the two modes cannot collide and the
    filename records which pipeline produced it.
    """
    fusion = _stub_fusion(monkeypatch, tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    # a transcript from an earlier single-file run of the SAME primary
    single_file = out_dir / "meeting.md"
    single_file.write_text("the single-file transcript", encoding="utf-8")

    out_path = fusion.run_fusion(
        tmp_path / "meeting.mp4", tmp_path / "phone.m4a",
        model_repo="x", language="en", initial_prompt=None, num_speakers=None,
        output_dir=out_dir, diarization_pipeline=object(),
    )

    assert out_path.name == "meeting.fused.md"
    assert out_path.is_file()
    assert single_file.read_text(encoding="utf-8") == "the single-file transcript"


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
    monkeypatch.setattr(fusion, "_correlate_envelopes", lambda a, b: (0.0, 9.9))

    out_path = fusion.run_fusion(
        tmp_path / "meeting.mp4", tmp_path / "meeting.m4a",
        model_repo="x", language="en", initial_prompt=None, num_speakers=None,
        output_dir=tmp_path / "out", diarization_pipeline=object(),
    )

    assert len(seen_dirs) == 2
    assert seen_dirs[0] != seen_dirs[1]
    assert out_path.exists()


def test_run_fusion_smooths_micro_turns(tmp_path, monkeypatch):
    """Pins the fusion WIRING of micro-turn smoothing. Fusion roughly doubles
    fragmentation (16% one-word blocks single-file vs 31% fused), so this path
    needs it at least as much as the single-file one.
    """
    from audio_to_text import fusion

    def fake_process_source(media_path, tmp_dir, **kwargs):
        wav_path = tmp_dir / (media_path.stem + ".clean.wav")
        wav_path.write_bytes(b"")
        embeddings = {"S0": np.array([1.0, 0.0]), "S1": np.array([0.0, 1.0])}
        if media_path.stem == "a":
            turns = [
                {"speaker": "S0", "start": 0.0, "end": 5.0, "confidence": 0.9,
                 "text": "we can extend it to some of"},
                {"speaker": "S1", "start": 5.0, "end": 5.0, "confidence": 0.4, "text": "the"},
                {"speaker": "S0", "start": 5.0, "end": 9.0, "confidence": 0.9, "text": "other things"},
            ]
        else:
            # the second mic heard the same stretch, less clearly -- so it never
            # wins a replacement and never gap-fills, isolating the smoothing
            turns = [
                {"speaker": "S0", "start": 0.0, "end": 9.0, "confidence": 0.1,
                 "text": "muffled version of the same thing"},
            ]
        return wav_path, turns, embeddings

    monkeypatch.setattr(fusion, "_process_source", fake_process_source)
    monkeypatch.setattr(fusion, "_correlate_envelopes", lambda a, b: (0.0, 9.9))

    out_path = fusion.run_fusion(
        tmp_path / "a.mp4", tmp_path / "b.m4a",
        model_repo="x", language="en", initial_prompt=None, num_speakers=None,
        output_dir=tmp_path / "out", diarization_pipeline=object(),
    )

    text = out_path.read_text(encoding="utf-8")
    assert "some of the other things" in text
    assert text.count("## Person") == 1


def test_offset_confidence_reports_no_confidence_when_clips_are_too_short_to_judge(tmp_path):
    """Fail-safe, not fail-silent.

    The confidence metric masks out everything within 5s of the peak before
    looking for a rival. If BOTH recordings are shorter than that window, every
    candidate rival is masked away and there is nothing left to compare the peak
    against. Treating that as infinite confidence would make two entirely
    unrelated clips report a perfect alignment and suppress the warning -- the
    exact failure this feature exists to catch, inverted.

    An unmeasurable alignment must score as untrustworthy.
    """
    from audio_to_text.fusion import _correlate_envelopes, OFFSET_CONFIDENCE_THRESHOLD

    rate = 1000
    rng = np.random.default_rng(0)
    for name in ("a", "b"):
        # 4 seconds each -- shorter than the 5s rival-exclusion window
        wavfile.write(tmp_path / f"{name}.wav", rate,
                      rng.normal(0, 0.01, 4 * rate).astype(np.float32))

    _offset, confidence = _correlate_envelopes(tmp_path / "a.wav", tmp_path / "b.wav")

    assert confidence < OFFSET_CONFIDENCE_THRESHOLD, (
        f"two unrelated 4s clips scored {confidence}; an alignment that cannot be "
        "measured must not be reported as a confident one"
    )


def test_containment_guard_never_consumes_a_turn_that_itself_won_a_replacement():
    """Only an untouched A turn may be consumed. Consuming a turn that itself won
    a replacement throws away B TEXT -- content no other block carries -- rather
    than a duplicate. Silent, unrecoverable transcript loss.
    """
    turns_a = [
        {"speaker": "S0", "start": 0.0, "end": 5.0, "text": "right okay", "confidence": 0.3},
        {"speaker": "S1", "start": 5.0, "end": 5.2, "text": "to", "confidence": 0.9},
        {"speaker": "S0", "start": 5.2, "end": 11.0, "confidence": 0.3,
         "text": "the second quarter budget is agreed"},
    ]
    turns_b_shifted = [
        {"speaker": "S0", "start": 0.1, "end": 10.9, "confidence": 0.95,
         "text": "right okay the second quarter budget is agreed"},
        # this one wins turns_a[2] -- so turns_a[2] must NOT be treated as a
        # consumable sibling of turns_a[0], or B's unique text disappears
        {"speaker": "S0", "start": 5.3, "end": 10.8, "confidence": 0.9,
         "text": "only source B heard this sentence at all"},
    ]

    merged = merge_turns(turns_a, turns_b_shifted)

    assert any("only source B heard this sentence at all" in t["text"] for t in merged), (
        f"B's unique text was discarded: {[t['text'] for t in merged]}"
    )


def test_containment_guard_ignores_a_sibling_b_never_covered_in_time():
    """Index adjacency is not temporal adjacency.

    A-turn indices say nothing about elapsed time, so a turn twenty minutes later
    can sit two indices away. Consuming it deletes speech and makes its words
    reappear under a heading timestamped at the start of B's span -- the same
    "span and text describe different speech" defect the replacement branch was
    fixed for, re-entering through another door.
    """
    distant = "we can approve this version now"
    turns_a = [
        {"speaker": "S0", "start": 0.0, "end": 5.0, "text": "right okay", "confidence": 0.3},
        {"speaker": "S1", "start": 5.0, "end": 1200.0, "confidence": 0.9,
         "text": "a twenty minute monologue from someone else"},
        # 20 minutes after B's turn ended
        {"speaker": "S0", "start": 1200.0, "end": 1260.0, "text": distant, "confidence": 0.3},
    ]
    turns_b_shifted = [
        {"speaker": "S0", "start": 0.1, "end": 10.9, "confidence": 0.95,
         "text": f"right okay {distant}"},
    ]

    merged = merge_turns(turns_a, turns_b_shifted)

    survivor = [t for t in merged if t["text"] == distant]
    assert survivor, (
        "a turn 20 minutes outside B's span was consumed; its speech is now lost "
        f"and only appears under a block starting at 0.1s: {[t['text'] for t in merged]}"
    )
    assert survivor[0]["start"] == 1200.0


def test_containment_guard_matches_across_casing_and_punctuation_differences():
    """The normalisation is the point, not decoration.

    Two independent ASR passes punctuate and capitalise the same speech
    differently, so a raw substring test would miss almost every real duplication.
    Pins that: B's text carries the same words with different case and punctuation.
    """
    sibling = "We can approve this version -- let's say we're happy."
    turns_a = [
        {"speaker": "S0", "start": 0.0, "end": 5.0, "text": "Right, okay.", "confidence": 0.3},
        {"speaker": "S1", "start": 5.0, "end": 5.2, "text": "to", "confidence": 0.9},
        {"speaker": "S0", "start": 5.2, "end": 11.0, "text": sibling, "confidence": 0.3},
    ]
    turns_b_shifted = [
        # same words, different casing and punctuation entirely
        {"speaker": "S0", "start": 0.1, "end": 10.9, "confidence": 0.95,
         "text": "right okay WE CAN APPROVE THIS VERSION, let's say we're happy!"},
    ]

    merged = merge_turns(turns_a, turns_b_shifted)

    assert not any(t["text"] == sibling for t in merged), (
        "the sibling survived, so containment is being judged on raw text rather "
        f"than normalised words: {[t['text'] for t in merged]}"
    )


def test_containment_floor_protects_micro_turns_and_consumes_restated_content():
    """Two-sided pin anchored to the MEASURED bimodality, not to the constant.

    Deriving the fixture from _CONTAINMENT_MIN_CHARS makes the test move with the
    constant, so it can never fail when the constant does -- a self-referential
    pin is no pin at all. These lengths come from the real data: fires cluster at
    <= 8 normalized chars (micro-turns like "you know") and >= 20 (restated
    content), with an empty gap between. Both ends must hold.
    """
    def run(sibling_text):
        turns_a = [
            {"speaker": "S0", "start": 0.0, "end": 5.0, "text": "opening", "confidence": 0.3},
            {"speaker": "S1", "start": 5.0, "end": 5.2, "text": "to", "confidence": 0.9},
            {"speaker": "S0", "start": 5.2, "end": 11.0, "text": sibling_text, "confidence": 0.3},
        ]
        turns_b = [{"speaker": "S0", "start": 0.1, "end": 10.9, "confidence": 0.95,
                    "text": f"opening {sibling_text} and then some more"}]
        return [t["text"] for t in merge_turns(turns_a, turns_b)]

    # 8 chars -- the micro-turn cluster. Consuming these is fragmentation
    # smoothing by the back door, and "Daniel" may be a real one-word answer.
    assert "you know" in run("you know"), "an 8-char micro-turn was consumed"
    assert "Daniel" in run("Daniel"), "a 6-char one-word turn was consumed"
    # 20+ chars -- genuinely restated content, must go
    restated = "along the same lines"
    assert restated not in run(restated), "a 20-char restated block was NOT consumed"


def test_containment_radius_does_not_reach_beyond_two_a_turns():
    """Two-sided pin on the radius, anchored to literal distances.

    Radius must be >= 2 (same-speaker A turns are never adjacent) but must not
    creep wider: the further it reaches, the more unrelated same-speaker turns
    become deletable on a text match alone.
    """
    sibling = "the number we finally agreed on today"

    def run(intervening):
        turns_a = [{"speaker": "S0", "start": 0.0, "end": 5.0, "text": "opening", "confidence": 0.3}]
        for i in range(intervening):
            turns_a.append({"speaker": f"S{i+1}", "start": 5.0 + i * 0.1,
                            "end": 5.1 + i * 0.1, "text": "uh", "confidence": 0.9})
        turns_a.append({"speaker": "S0", "start": 6.0, "end": 10.0,
                        "text": sibling, "confidence": 0.3})
        turns_b = [{"speaker": "S0", "start": 0.1, "end": 10.9, "confidence": 0.95,
                    "text": f"opening {sibling}"}]
        return [t["text"] for t in merge_turns(turns_a, turns_b)]

    # exactly 2 A-indices apart -> consumed (the real same-speaker case)
    assert sibling not in run(1), "the radius-2 case must fire"
    # 3 apart -> beyond reach, must survive
    assert sibling in run(2), "the guard reached further than two A turns"


def test_offset_confidence_test_tracks_the_shipped_threshold_and_measured_band():
    """Pins the calibration itself, not a magic number.

    Asserting against a hardcoded 1.2 lets the shipped threshold move anywhere in
    a wide band with the suite green -- to 5.0, where every legitimate fusion
    warns, or to 1.003, below the measured null ceiling of 1.0050. Anchored here
    to the values actually measured on the real pair.
    """
    from audio_to_text.fusion import OFFSET_CONFIDENCE_THRESHOLD

    measured_true_pair = 1.5162
    measured_null_ceiling = 1.0050

    assert measured_null_ceiling < OFFSET_CONFIDENCE_THRESHOLD < measured_true_pair, (
        f"threshold {OFFSET_CONFIDENCE_THRESHOLD} no longer separates the measured "
        f"null ceiling ({measured_null_ceiling}) from the measured true pair "
        f"({measured_true_pair})"
    )


def test_run_fusion_warns_about_cross_speaker_duplicate_attribution(tmp_path, monkeypatch, capsys):
    """The remaining known defect is now VISIBLE rather than silent.

    Near-identical text under two different speakers means one heading is wrong.
    The two sources' diarizations disagreed and no arbiter here can settle it, so
    the tool reports the timestamps instead of silently picking a copy -- picking
    would turn an artifact a reader notices into a misattribution they do not.
    """
    from audio_to_text import fusion

    shared = "the budget for the second quarter is exactly what we agreed on"

    def fake_process_source(media_path, tmp_dir, **kwargs):
        wav_path = tmp_dir / (media_path.stem + ".clean.wav")
        wav_path.write_bytes(b"")
        embeddings = {"S0": np.array([1.0, 0.0]), "S1": np.array([0.0, 1.0])}
        if media_path.stem == "a":
            turns = [{"speaker": "S0", "start": 0.0, "end": 9.0, "confidence": 0.9, "text": shared}]
        else:
            # source B heard the SAME words but attributed them to the other speaker
            turns = [{"speaker": "S1", "start": 0.0, "end": 9.0, "confidence": 0.95, "text": shared}]
        return wav_path, turns, embeddings

    monkeypatch.setattr(fusion, "_process_source", fake_process_source)
    monkeypatch.setattr(fusion, "_correlate_envelopes", lambda a, b: (0.0, 9.9))

    fusion.run_fusion(
        tmp_path / "a.mp4", tmp_path / "b.m4a",
        model_repo="x", language="en", initial_prompt=None, num_speakers=None,
        output_dir=tmp_path / "out", diarization_pipeline=object(),
    )

    err = capsys.readouterr().err
    assert "two different speakers" in err
    assert "Person 1 vs Person 2" in err


def test_cross_speaker_duplicate_warning_ignores_hallucination_loops():
    """Degenerate repeated text trivially "duplicates" itself between any two
    blocks. On the real pair that accounted for 5 of 16 apparent pairs -- warning
    on them would report a speaker problem where the actual fault is Whisper.
    """
    from audio_to_text.transcribe import detect_cross_speaker_duplicates

    turns = [
        {"speaker": "Person 1", "start": float(i), "end": float(i) + 0.1, "text": "Paul."}
        for i in range(30)
    ]
    for i, turn in enumerate(turns):
        turn["speaker"] = f"Person {1 + i % 3}"   # loop shredded across speakers

    assert detect_cross_speaker_duplicates(turns) == []
