"""Measure the offset search against a DC-heavy envelope, the shipped way and raw.

Reproduces the numbers in bugs.md's OFFSET_CONFIDENCE_THRESHOLD entry. Builds
pairs of WAVs from the same synthetic speech with a known offset, where the
secondary is a quieter, shorter window from the middle of the primary -- the
documented "phone that joined late, sitting across the room" case -- and compares
the shipped correlation against one without mean-subtraction.

Synthetic on purpose: it needs the ground-truth offset, and the real 70-minute
pair is not committed. It measures the DIRECTION of the effect, not a calibration.

    uv run python tools/measure_dc_bias.py
"""
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from audio_to_text.fusion import _correlate_envelopes, _rms_envelope  # noqa: E402

RATE = 16_000
TMP = Path(tempfile.mkdtemp(prefix="dc_bias_"))


def speech_like(duration_s, rng):
    """Amplitude-modulated noise: bursts of 'speech' separated by pauses."""
    n = int(duration_s * RATE)
    carrier = rng.normal(0, 1.0, n)
    # syllable-rate envelope (~4 Hz) gated into utterances of a few seconds
    t = np.arange(n) / RATE
    syllable = 0.5 * (1 + np.sin(2 * np.pi * 4.0 * t))
    gate = np.zeros(n)
    pos = 0
    while pos < n:
        on = int(rng.uniform(1.5, 4.0) * RATE)
        off = int(rng.uniform(0.3, 1.5) * RATE)
        gate[pos:pos + on] = 1.0
        pos += on + off
    return carrier * syllable * gate


def write(path, samples):
    peak = np.max(np.abs(samples)) or 1.0
    wavfile.write(path, RATE, (samples / peak * 0.5 * 32767).astype(np.int16))


def mean_removed_offset(wav_a, wav_b, window_seconds=0.1):
    from scipy import signal
    ra, sa = wavfile.read(wav_a)
    rb, sb = wavfile.read(wav_b)
    ea = _rms_envelope(sa.astype(np.float64), ra, window_seconds)
    eb = _rms_envelope(sb.astype(np.float64), rb, window_seconds)
    corr = signal.correlate(ea - ea.mean(), eb - eb.mean(), mode="full", method="fft")
    return (int(np.argmax(corr)) - (len(eb) - 1)) * window_seconds


print(f"{'trial':>5} {'true':>9} {'as-shipped':>12} {'mean-removed':>14}")
shipped_ok = removed_ok = 0
for trial in range(20):
    rng = np.random.default_rng(trial)
    full = speech_like(300, rng)                 # 5 min primary
    true_offset = 60.0                           # B starts 60 s into A
    start = int(true_offset * RATE)
    window = full[start:start + 30 * RATE]       # 30 s clip from the middle
    # B: quieter, with its own room-noise floor -- a phone across the room
    b = window * 0.35 + rng.normal(0, 0.02, len(window))

    wav_a, wav_b = TMP / f"a{trial}.wav", TMP / f"b{trial}.wav"
    write(wav_a, full)
    write(wav_b, b)

    shipped, conf = _correlate_envelopes(wav_a, wav_b)
    removed = mean_removed_offset(wav_a, wav_b)
    shipped_ok += abs(shipped - true_offset) < 1.0
    removed_ok += abs(removed - true_offset) < 1.0
    print(f"{trial:>5} {true_offset:>+9.2f} {shipped:>+12.2f} {removed:>+14.2f}   (confidence {conf:.3f})")

print(f"\nas-shipped correct:   {shipped_ok}/20")
print(f"mean-removed correct: {removed_ok}/20")
