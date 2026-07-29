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
