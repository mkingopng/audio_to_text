"""Unit tests for src/fusion.py -- dual-source sync, speaker matching, turn merging."""
import pytest
import numpy as np
from scipy.io import wavfile

from src.fusion import find_offset, match_speakers


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
