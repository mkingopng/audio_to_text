"""Unit tests for src/fusion.py -- dual-source sync, speaker matching, turn merging."""
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
