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


def _speech_like(n: int, rate: int, rng, floor: float) -> np.ndarray:
    """A non-negative, DC-heavy signal: bursts of "speech" over a steady noise floor.

    This is what real preprocessed meeting audio produces once _rms_envelope runs
    over it. The distinction matters -- see
    test_find_offset_survives_a_dc_heavy_envelope.
    """
    t = np.arange(n) / rate
    gate = np.zeros(n)
    position = 0
    while position < n:
        speaking = int(rng.uniform(0.4, 1.2) * rate)
        pause = int(rng.uniform(0.2, 0.8) * rate)
        gate[position:position + speaking] = 1.0
        position += speaking + pause
    syllables = 0.5 * (1 + np.sin(2 * np.pi * 5 * t))
    return np.abs(rng.normal(0, 1.0, n)) * syllables * gate + floor


def _write_scaled(path: Path, rate: int, samples: np.ndarray) -> None:
    wavfile.write(path, rate, (samples / max(samples.max(), 1e-9) * 0.5 * 32767).astype(np.int16))


def test_find_offset_survives_a_dc_heavy_envelope(tmp_path):
    """RMS envelopes are strictly non-negative with a large DC component, so a raw
    cross-correlation is dominated by the triangular overlap-count term
    (mean_a * mean_b * overlap_length) rather than by acoustic content. That term
    peaks at MAXIMUM OVERLAP regardless of what was said, so a short quiet source
    aligns to wherever it overlaps most -- not to where it actually belongs.

    This is distinct from the confidence check: that validates the answer, this
    decides where the peak lands, and a prominence test cannot catch it because
    the wrong peak is broad and high.

    The fixture is the documented use case -- a phone that joined late, recording
    quietly across the room. Correlating the envelopes as-is recovers +6.8s
    against a true +20.0s; subtracting the means recovers +20.0s.
    """
    rate = 1000  # low rate keeps this fast; the algorithm is rate-agnostic
    rng = np.random.default_rng(0)
    true_offset = 20.0

    full = _speech_like(60 * rate, rate, rng, floor=0.05)
    start = int(true_offset * rate)
    window = full[start:start + 8 * rate]
    # Quieter, and sitting on its own room-noise floor: a second mic across the room.
    secondary = np.abs(window * 0.3 + 0.5 + rng.normal(0, 0.01, len(window)))

    wav_a = tmp_path / "a.wav"
    wav_b = tmp_path / "b.wav"
    _write_scaled(wav_a, rate, full)
    _write_scaled(wav_b, rate, secondary)

    assert abs(find_offset(wav_a, wav_b) - true_offset) < 0.2


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


def test_match_speakers_names_the_speaker_when_pyannote_returns_a_zero_embedding():
    """pyannote pads with zero-vector embeddings when it finds fewer voice
    clusters than speakers -- documented in its speaker_diarization pipeline, and
    likeliest in exactly the crowded cross-talking meeting fusion is for.

    A zero norm divides to an all-NaN row. NaN is not an exception: it
    propagates silently into the cost matrix and only surfaces as
    linear_sum_assignment's "matrix contains invalid numeric entries", several
    steps and (in production) two full ASR + diarization passes later. The user
    is then told nothing they can act on.
    """
    embeddings_a = {"A_0": _unit_vector(0), "A_1": _unit_vector(90)}
    embeddings_b = {"B_0": _unit_vector(0), "B_1": np.array([0.0, 0.0])}

    with pytest.raises(ValueError, match="no usable voice embedding.*B_1"):
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

    def fuse(**extra) -> str:
        out_path = fusion.run_fusion(
            tmp_path / "a.mp4", tmp_path / "b.m4a",
            model_repo="x", language="en", initial_prompt=None, num_speakers=None,
            output_dir=tmp_path / "out", diarization_pipeline=object(), **extra,
        )
        return out_path.read_text(encoding="utf-8")

    smoothed = fuse(smooth=True)
    assert "some of the other things" in smoothed
    assert smoothed.count("## Person") == 1

    # Off by default: smoothing re-attributes "the" from S1 to S0, so a run that
    # did not ask for it must leave diarization's own attribution alone. Asserting
    # only the smooth=True direction would stay green on a run_fusion that ignored
    # the parameter and smoothed unconditionally.
    assert fuse().count("## Person") == 3


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


def test_gap_fill_keeps_a_b_turn_that_only_grazes_a_same_speaker_a_turn():
    """The gap-fill test was binary: ANY overlap with a same-speaker A turn meant
    "already represented", so one shared millisecond suppressed a whole B turn.

    The case is the one the feature exists for. The Teams capture drops out
    mid-sentence at 5s; the phone recorded straight through to 30s. They share
    0.1s. B loses the confidence comparison, so A's fragment is kept -- and B is
    then withheld as redundant, silently discarding 25 seconds of speech only one
    microphone caught. README promises the opposite: "Speech only one microphone
    caught is appended rather than dropped."

    Appending re-states the few overlapping words. That is the deliberate trade:
    a visible duplicate is recoverable by whoever reads the transcript, silent
    deletion is not.
    """
    tail = "and then twenty five more seconds of speech nobody else recorded"
    turns_a = [
        {"speaker": "A0", "start": 0.0, "end": 5.0, "confidence": 0.9,
         "text": "so the thing is"},
    ]
    turns_b_shifted = [
        {"speaker": "A0", "start": 4.9, "end": 30.0, "confidence": 0.6,
         "text": f"so the thing is {tail}"},
    ]

    merged = merge_turns(turns_a, turns_b_shifted)

    assert any(tail in turn["text"] for turn in merged), (
        "25s of B-only speech was dropped because it grazed an A turn by 0.1s: "
        f"{[t['text'] for t in merged]}"
    )


def test_gap_fill_still_withholds_a_b_turn_a_already_covers():
    """The other side of the ratio bound: a B turn genuinely representing the same
    speech as an A turn must NOT be appended, or every well-covered moment is
    emitted twice. This is what the original binary test got right."""
    turns_a = [
        {"speaker": "A0", "start": 0.0, "end": 10.0, "confidence": 0.9,
         "text": "the whole utterance as source a heard it"},
    ]
    turns_b_shifted = [
        {"speaker": "A0", "start": 0.2, "end": 9.8, "confidence": 0.6,
         "text": "the whole utterance as source b heard it"},
    ]

    merged = merge_turns(turns_a, turns_b_shifted)

    assert len(merged) == 1, f"B duplicated a moment A already covers: {merged}"


def test_containment_guard_ignores_a_distant_sibling_inside_b_span():
    """Being inside B's span is not the same as being near the turn B replaced.

    test_containment_guard_ignores_a_sibling_b_never_covered_in_time pins the
    sibling sitting OUTSIDE B's span. But a long B turn makes that span wide, and
    inside it the sibling can still be arbitrarily far away in time -- so the
    guard's own stated principle ("A-turn indices say nothing about elapsed
    time") was only enforced at the span boundary.

    The duplication this guard exists to remove is one utterance that A's
    diarization split in two and B's kept whole, so its halves are temporally
    adjacent. A speaker RESTATING a stock phrase half a minute later is new
    speech, not a duplicate -- and here it is A's own higher-confidence text,
    deleted while the lower-confidence turn that displaced it survives.
    """
    restatement = "we should ship it on friday"
    turns_a = [
        {"speaker": "S0", "start": 0.0, "end": 5.0, "confidence": 0.3,
         "text": "we should ship it on friday i think"},
        {"speaker": "S1", "start": 5.0, "end": 6.0, "text": "hmm", "confidence": 0.9},
        # 25 seconds later, but still inside B's 0.1-40.0 span, and only two
        # A-turn indices away -- which is the NORMAL distance between two
        # same-speaker turns, since _group_consecutive flushes on every change.
        {"speaker": "S0", "start": 30.0, "end": 33.0, "text": restatement, "confidence": 0.9},
    ]
    turns_b_shifted = [
        {"speaker": "S0", "start": 0.1, "end": 40.0, "confidence": 0.95,
         "text": "we should ship it on friday i think"},
    ]

    merged = merge_turns(turns_a, turns_b_shifted)

    survivor = [t for t in merged if t["text"] == restatement]
    assert survivor, (
        "a restatement 25s from the turn B replaced was consumed as a duplicate; "
        f"that speech is now lost: {[t['text'] for t in merged]}"
    )
    assert survivor[0]["start"] == 30.0


def test_containment_guard_spares_a_sibling_only_partially_present_in_b():
    """ENTIRE containment is the condition, not resemblance.

    A sibling that merely shares a long opening with B's text is not a duplicate
    of it -- the words B did not capture exist nowhere else, so consuming the
    sibling deletes them outright. Any relaxation of `sibling in b_text` to a
    similarity or prefix rule (SequenceMatcher over some ratio, `sibling[:20] in
    b_text`, or accepting `b_text in sibling`) silently destroys real speech,
    which is why the guard tests exact containment and nothing looser.

    This is the axis the other four guard conditions' tests do not cover: they
    pin the speaker, the radius, the length floor and the temporal overlap, and
    every one of those can hold while the texts merely resemble each other.
    """
    sibling = "we can approve this version and send it to legal on friday"
    turns_a = [
        {"speaker": "S0", "start": 0.0, "end": 5.0, "text": "right okay", "confidence": 0.3},
        # Same speaker, adjacent index, inside B's span, well over the length
        # floor -- every other condition is satisfied. Only containment is not:
        # B heard the first half and missed "send it to legal on friday".
        {"speaker": "S0", "start": 5.0, "end": 9.0, "text": sibling, "confidence": 0.3},
    ]
    turns_b_shifted = [
        {"speaker": "S0", "start": 0.1, "end": 9.5, "confidence": 0.95,
         "text": "right okay we can approve this version and"},
    ]

    merged = merge_turns(turns_a, turns_b_shifted)

    surviving_words = {word for turn in merged for word in turn["text"].lower().split()}
    lost = {"send", "it", "to", "legal", "on", "friday"} - surviving_words
    assert not lost, (
        f"the guard consumed a sibling B only partially contained, deleting {sorted(lost)}. "
        f"Merged output: {[t['text'] for t in merged]}"
    )


def test_containment_guard_spares_a_sibling_that_says_more_than_b_did():
    """Containment is directional: the SIBLING must be inside B's text, never the
    reverse.

    When B captured less than the A sibling did, B's text is contained in the
    sibling rather than the other way round. The sibling is then the fuller
    record, and consuming it discards exactly the words B missed. Relaxing the
    condition to `sibling in b_text or b_text in sibling` reads as symmetric and
    harmless; it deletes the tail of every turn the second mic under-heard.
    """
    sibling = "we can approve this version and send it to legal on friday"
    turns_a = [
        {"speaker": "S0", "start": 0.0, "end": 5.0, "text": "right okay", "confidence": 0.3},
        {"speaker": "S0", "start": 5.0, "end": 9.0, "text": sibling, "confidence": 0.3},
    ]
    turns_b_shifted = [
        # B is more confident but heard only the opening -- its text sits ENTIRELY
        # inside the sibling's.
        {"speaker": "S0", "start": 0.1, "end": 9.5, "confidence": 0.95,
         "text": "we can approve this version"},
    ]

    merged = merge_turns(turns_a, turns_b_shifted)

    surviving_words = {word for turn in merged for word in turn["text"].lower().split()}
    lost = {"and", "send", "it", "to", "legal", "on", "friday"} - surviving_words
    assert not lost, (
        f"the guard consumed the fuller sibling because B's text was inside IT, "
        f"deleting {sorted(lost)}. Merged output: {[t['text'] for t in merged]}"
    )


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


def test_merge_turns_keeps_a_turn_when_confidence_only_TIES():
    """Pre-existing behaviour: B replaces A only on STRICTLY higher confidence.

    Source A defines the canonical paragraph boundaries, so a tie must leave A's
    text in place. Relaxing the comparison to >= silently hands every tie to the
    secondary recording -- a change of which microphone wins, invisible in any
    other test.
    """
    turns_a = [{"speaker": "S0", "start": 0.0, "end": 5.0, "text": "A's wording", "confidence": 0.7}]
    turns_b_shifted = [
        {"speaker": "S0", "start": 0.1, "end": 4.9, "text": "B's wording", "confidence": 0.7},
    ]

    merged = merge_turns(turns_a, turns_b_shifted)

    assert [t["text"] for t in merged] == ["A's wording"], (
        "an equal-confidence B turn replaced A's text; only strictly higher may win"
    )


def test_merge_turns_returns_turns_in_chronological_order():
    """Pre-existing behaviour: the merged list is sorted by start.

    Gap-filled B turns are appended after A's, so without the final sort the
    output interleaves wrongly -- the transcript renders out of order, and
    micro-turn smoothing's "sandwiched by the same speaker" test reads the wrong
    neighbours because it depends entirely on this ordering.
    """
    turns_a = [
        {"speaker": "S0", "start": 100.0, "end": 105.0, "text": "late A turn", "confidence": 0.9},
    ]
    turns_b_shifted = [
        # no same-speaker A turn overlaps these, so both are gap-filled and appended
        {"speaker": "S1", "start": 10.0, "end": 12.0, "text": "early B turn", "confidence": 0.9},
        {"speaker": "S1", "start": 200.0, "end": 202.0, "text": "later B turn", "confidence": 0.9},
    ]

    merged = merge_turns(turns_a, turns_b_shifted)

    starts = [t["start"] for t in merged]
    assert starts == sorted(starts), f"merged output is not chronological: {starts}"
    assert [t["text"] for t in merged] == ["early B turn", "late A turn", "later B turn"]


def test_run_fusion_end_to_end_over_the_real_pipeline(tmp_path, monkeypatch):
    """The one test that actually runs run_fusion's pipeline.

    Every other run_fusion test mocks out _process_source and the offset search,
    so the body composing the pipeline was invisible: a mutation pass found that
    flipping the offset's sign, skipping merge_turns entirely, skipping
    relabel_speakers, or hardcoding offset=0.0 each left the whole suite green.
    Skipping relabel_speakers is the headline product feature -- transcripts
    would ship with raw "## SPEAKER_00" headings and nothing would notice.

    Worse, the existing fixture returned "SPEAKER_00" for BOTH sources, so the
    speaker map and its inverse were identical and the map-inversion at the
    single highest-consequence line could not be got wrong.

    So: mock only ffmpeg (preprocess_audio) and ASR (run_whisper), give the two
    sources DIFFERENT speaker-label namespaces, and let the real find_offset,
    match_speakers, _shift_and_remap, merge_turns, relabel_speakers and
    render_markdown run over real audio with a known offset.
    """
    from audio_to_text import fusion

    rate = 1000
    true_offset = 20.0
    rng = np.random.default_rng(0)
    world = _speech_like(60 * rate, rate, rng, floor=0.05)
    # B started 20s late: its local t=0 is A's t=20.
    secondary_audio = np.abs(world[int(true_offset * rate):int(28 * rate)] * 0.3 + 0.5)

    wav_for = {}

    def fake_preprocess(media_path, tmp_dir, audio_filter):
        wav = tmp_dir / (media_path.stem + ".clean.wav")
        _write_scaled(wav, rate, world if media_path.stem == "teams" else secondary_audio)
        wav_for[wav] = media_path.stem
        return wav

    def words(pairs):
        return {"segments": [
            {"words": [{"word": w, "start": s, "end": e, "probability": p}
                       for w, s, e, p in pairs]}
        ]}

    def fake_run_whisper(wav_path, **kwargs):
        if wav_for[wav_path] == "teams":
            # A's clock. Muffled -- low probability, so B's text should win.
            return words([("alpha", 20.0, 21.0, 0.30), ("mumble", 21.0, 22.0, 0.30),
                          ("bravo", 23.0, 24.0, 0.30), ("garble", 24.0, 25.0, 0.30)])
        # B's LOCAL clock: 0.0 here is 20.0 on A's timeline.
        return words([("alpha", 0.0, 1.0, 0.95), ("clear", 1.0, 2.0, 0.95),
                      ("bravo", 3.0, 4.0, 0.95), ("crisp", 4.0, 5.0, 0.95)])

    class FakeDiarization:
        def __init__(self, tracks):
            self._tracks = tracks

        def itertracks(self, yield_label=True):
            for start, end, label in self._tracks:
                yield type("T", (), {"start": start, "end": end})(), None, label

        def labels(self):
            return sorted({label for _, _, label in self._tracks}, key=str)

    class FakeOutput:
        def __init__(self, tracks, embeddings):
            self.speaker_diarization = FakeDiarization(tracks)
            self.speaker_embeddings = embeddings

    def fake_pipeline(path, **kwargs):
        # DIFFERENT label namespaces per source, so the a->b map and its inverse
        # are distinguishable and passing the wrong one cannot go unnoticed.
        if wav_for[Path(path)] == "teams":
            return FakeOutput(
                [(20.0, 22.0, "SPEAKER_00"), (23.0, 25.0, "SPEAKER_01")],
                np.array([_unit_vector(0), _unit_vector(90)]),
            )
        return FakeOutput(
            [(0.0, 2.0, "SPK_X"), (3.0, 5.0, "SPK_Y")],
            np.array([_unit_vector(2), _unit_vector(92)]),  # X~=SPEAKER_00, Y~=SPEAKER_01
        )

    monkeypatch.setattr(fusion, "preprocess_audio", fake_preprocess)
    monkeypatch.setattr(fusion, "run_whisper", fake_run_whisper)

    out_path = fusion.run_fusion(
        tmp_path / "teams.mp4", tmp_path / "phone.m4a",
        model_repo="x", language="en", initial_prompt=None, num_speakers=None,
        output_dir=tmp_path / "out", diarization_pipeline=fake_pipeline,
    )
    rendered = out_path.read_text(encoding="utf-8")

    # Speakers are relabeled -- raw diarization ids must never reach the file.
    assert "SPEAKER_" not in rendered and "SPK_" not in rendered, rendered
    assert "## Person 1" in rendered and "## Person 2" in rendered, rendered

    # B's clearer text won both turns, which can only happen if the offset put
    # B's turns on top of A's. Wrong sign or 0.0 and they never overlap.
    assert "clear" in rendered and "crisp" in rendered, rendered
    assert "mumble" not in rendered and "garble" not in rendered, rendered

    # ...and landed on A's timeline, at 00:20 and 00:23 rather than 00:00/00:03.
    assert "## Person 1 — 00:20" in rendered, rendered
    assert "## Person 2 — 00:23" in rendered, rendered


# --- cross-speaker duplicate detection: the constants, pinned two-sided --------
#
# Previously one happy-path wiring test covered all of this, so every threshold
# was free to move in the destructive direction and the loop exclusion was
# unreachable. Each test below names the mutation it kills.

def _dup_turns(texts, *, speakers=None, gap=10.0):
    """Blocks far enough apart in time that repetition-loop detection is not
    triggered by the fixture itself."""
    speakers = speakers or [f"Person {1 + i % 2}" for i in range(len(texts))]
    return [
        {"speaker": speakers[i], "start": i * gap, "end": i * gap + 1.0, "text": text}
        for i, text in enumerate(texts)
    ]


_DUP_SENTENCE = "we should approve the budget before friday afternoon"  # 51 normalized chars


def test_cross_speaker_duplicate_detection_fires_on_a_real_duplicate():
    """Baseline. Kills _DUPLICATE_MIN_SHARED_CHARS raised (40 -> 200) and
    _DUPLICATE_MIN_SHARE raised (0.5 -> 0.99)."""
    from audio_to_text.transcribe import detect_cross_speaker_duplicates

    found = detect_cross_speaker_duplicates(_dup_turns([_DUP_SENTENCE, _DUP_SENTENCE]))

    assert len(found) == 1
    assert found[0]["speakers"] == ("Person 1", "Person 2")


def test_cross_speaker_duplicate_detection_ignores_a_short_shared_span():
    """Kills _DUPLICATE_MIN_SHARED_CHARS lowered (40 -> 5), which would flood the
    warning with every pair of blocks sharing a common phrase."""
    from audio_to_text.transcribe import detect_cross_speaker_duplicates

    # 29 shared characters -- a stock opening, not a duplicated utterance.
    found = detect_cross_speaker_duplicates(
        _dup_turns(["we should approve the budget xx", "we should approve the budget yy"])
    )

    assert found == []


def test_cross_speaker_duplicate_detection_requires_the_span_to_dominate_the_block():
    """The shared span must cover most of the SHORTER block, not just clear the
    absolute floor. Kills _DUPLICATE_MIN_SHARE lowered (0.5 -> 0.0): a 45-char
    quotation inside two otherwise different 100+ char blocks is someone
    repeating a phrase, not one utterance rendered twice.
    """
    from audio_to_text.transcribe import detect_cross_speaker_duplicates

    quoted = "we should approve the budget before friday ab"  # 45 chars: over the floor
    first = quoted + " and then i went on at considerable length about something else entirely"
    second = quoted + " but she disagreed and said we ought to wait until the new quarter"

    found = detect_cross_speaker_duplicates(_dup_turns([first, second]))

    assert found == []


def test_cross_speaker_duplicate_detection_spans_exactly_five_blocks():
    """Two-sided pin on _DUPLICATE_MAX_DISTANCE. A duplicate five blocks apart
    must be found (kills 5 -> 1); one six apart must not (kills 5 -> 50, which
    would pair blocks minutes apart that merely share a stock sentence).

    The filler blocks are all DIFFERENT from each other on purpose. Identical
    filler pairs with itself at distance 1, which would satisfy this test's
    assertion for entirely the wrong reason and let 5 -> 1 survive.
    """
    from audio_to_text.transcribe import detect_cross_speaker_duplicates

    filler = [
        "unrelated chatter about the weather this morning and the traffic",
        "someone asking whether anybody had seen the latest quarterly figures",
        "a question about parking validation that nobody seems able to answer",
        "the sound of a door closing and a chair being pulled out loudly",
        "a completely separate tangent concerning the office coffee machine",
    ]

    at_five = _dup_turns([_DUP_SENTENCE] + filler[:4] + [_DUP_SENTENCE])
    found_five = detect_cross_speaker_duplicates(at_five)
    assert any(f["start"] == 0.0 for f in found_five), (
        f"a duplicate five blocks apart was not found: {found_five}"
    )

    # Explicit speakers: with seven blocks the alternating default puts index 0
    # and index 6 under the SAME speaker, so the pair would be skipped by the
    # same-speaker rule before distance was ever consulted -- and 5 -> 50 would
    # survive for a reason that has nothing to do with distance.
    at_six = _dup_turns(
        [_DUP_SENTENCE] + filler + [_DUP_SENTENCE],
        speakers=["Person 1"] + ["Person 2"] * 5 + ["Person 2"],
    )
    found_six = detect_cross_speaker_duplicates(at_six)
    assert found_six == [], f"paired blocks six apart: {found_six}"


def test_cross_speaker_duplicate_detection_fires_below_total_identity():
    """The share bound must admit a real duplicate that is not character-identical
    -- two ASR passes rarely produce byte-identical text. Kills
    _DUPLICATE_MIN_SHARE raised (0.5 -> 0.99), which the identical-text baseline
    above cannot: there shared == len, so any share threshold under 1.0 passes.
    """
    from audio_to_text.transcribe import detect_cross_speaker_duplicates

    shared = "we should approve the budget before friday ab"  # 45 chars, over the floor
    # ~0.75 of the shorter block -- clearly one utterance, not byte-identical.
    turns = _dup_turns([shared + " i think", shared + " i guess"])

    assert len(detect_cross_speaker_duplicates(turns)) == 1


def test_cross_speaker_duplicate_detection_skips_same_speaker_pairs():
    """The warning's text says "under two different speakers", and the whole
    premise is that one of the two attributions must be wrong. Two blocks under
    the SAME speaker are a redundancy question, not an attribution one. Kills
    dropping the first["speaker"] == second["speaker"] skip.
    """
    from audio_to_text.transcribe import detect_cross_speaker_duplicates

    turns = _dup_turns([_DUP_SENTENCE, _DUP_SENTENCE], speakers=["Person 1", "Person 1"])

    assert detect_cross_speaker_duplicates(turns) == []


def test_cross_speaker_duplicate_detection_normalizes_case_and_punctuation():
    """Two ASR passes punctuate and capitalise differently; the same sentence
    must still match across that. Kills normalize() -> identity."""
    from audio_to_text.transcribe import detect_cross_speaker_duplicates

    turns = _dup_turns([
        "We should approve the budget, before Friday afternoon!",
        "we should approve the budget before friday afternoon",
    ])

    assert len(detect_cross_speaker_duplicates(turns)) == 1


def test_cross_speaker_duplicate_detection_excludes_a_real_repetition_loop():
    """Rewritten: the previous fixture used "Paul." -- 4 normalized characters
    against a 40-character floor, so its empty result was guaranteed by the floor
    alone and was completely insensitive to the loop exclusion it claimed to
    test. Deleting the exclusion left the suite green.

    A real Whisper loop is a long run of one token, which diarization jitter then
    shreds across blocks attributed to different speakers. Each block is well
    over the floor and matches its neighbours almost exactly, so without the
    exclusion every adjacent pair is reported as a speaker-attribution problem
    when the actual fault is upstream in Whisper.
    """
    from audio_to_text.transcribe import detect_cross_speaker_duplicates

    block = " ".join(["paul"] * 12)  # 59 normalized chars, and part of a 72-token run
    turns = _dup_turns([block] * 6, gap=1.0)

    assert detect_cross_speaker_duplicates(turns) == []
