"""Fuse two recordings of the same meeting into one speaker-attributed transcript.

Reuses transcribe.py's per-source pipeline (ASR + diarization + word-level
speaker alignment), then synchronizes the two sources' timelines, matches
their independently-clustered speaker identities to each other, and picks
the clearer source's text per overlapping turn.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.optimize import linear_sum_assignment


def _rms_envelope(samples: np.ndarray, sample_rate: int, window_seconds: float) -> np.ndarray:
    """Windowed RMS energy envelope -- correlating on this is faster and more robust
    to speech-content differences between the two mics than correlating raw samples."""
    window = max(1, int(sample_rate * window_seconds))
    usable_length = len(samples) - (len(samples) % window)
    reshaped = samples[:usable_length].reshape(-1, window)
    return np.sqrt(np.mean(reshaped.astype(np.float64) ** 2, axis=1))


def find_offset(wav_a: Path, wav_b: Path, *, window_seconds: float = 0.1) -> float:
    """Seconds to ADD to source B's timestamps to align them onto source A's timeline."""
    rate_a, samples_a = wavfile.read(wav_a)
    rate_b, samples_b = wavfile.read(wav_b)
    if rate_a != rate_b:
        raise ValueError(f"sample rate mismatch between sources: {rate_a} vs {rate_b}")

    envelope_a = _rms_envelope(samples_a, rate_a, window_seconds)
    envelope_b = _rms_envelope(samples_b, rate_b, window_seconds)

    correlation = signal.correlate(envelope_a, envelope_b, mode="full", method="fft")
    lag_index = int(np.argmax(correlation)) - (len(envelope_b) - 1)
    return lag_index * window_seconds


def match_speakers(embeddings_a: dict[str, np.ndarray], embeddings_b: dict[str, np.ndarray]) -> dict[str, str]:
    """Match source A's speaker embeddings to source B's via the Hungarian algorithm.

    Greedy nearest-match can be led astray when two voices are close together
    (assigning both of B's closest speakers to the same A speaker, then being
    forced into a bad leftover pairing); the Hungarian algorithm finds the
    globally optimal one-to-one assignment instead.
    """
    labels_a = list(embeddings_a.keys())
    labels_b = list(embeddings_b.keys())
    matrix_a = np.stack([embeddings_a[label] for label in labels_a])
    matrix_b = np.stack([embeddings_b[label] for label in labels_b])

    normalized_a = matrix_a / np.linalg.norm(matrix_a, axis=1, keepdims=True)
    normalized_b = matrix_b / np.linalg.norm(matrix_b, axis=1, keepdims=True)
    similarity = normalized_a @ normalized_b.T
    cost = 1.0 - similarity

    row_indices, col_indices = linear_sum_assignment(cost)
    return {labels_a[row]: labels_b[col] for row, col in zip(row_indices, col_indices)}
