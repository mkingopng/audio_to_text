"""Re-derive the offset peak-quality metric and its null floor.

The prior triage reported peak/best_rival = 1.5162 for the true pair against a
null cluster at 1.000-1.005, but the script behind it is gone. This rebuilds the
measurement from the captured WAVs so the warn threshold is chosen from
separation I measured rather than a number I inherited.

Null controls are built from the SAME two recordings, so any separation is
attributable to alignment rather than to recording character.
"""
import numpy as np
from pathlib import Path
from scipy import signal
from scipy.io import wavfile

from audio_to_text.fusion import _rms_envelope

CAP = Path(__file__).parent / "capture"
WINDOW = 0.1
EXCLUSION_S = 5.0  # a rival peak must be this far from the argmax to count


def metrics(env_a: np.ndarray, env_b: np.ndarray) -> tuple[float, float, float]:
    """(lag_seconds, peak/best_rival, peak/median)."""
    corr = signal.correlate(env_a, env_b, mode="full", method="fft")
    arg = int(np.argmax(corr))
    peak = float(corr[arg])
    lag = (arg - (len(env_b) - 1)) * WINDOW

    exclusion = int(EXCLUSION_S / WINDOW)
    masked = corr.copy()
    lo, hi = max(0, arg - exclusion), min(len(corr), arg + exclusion + 1)
    masked[lo:hi] = -np.inf
    best_rival = float(np.max(masked))

    median = float(np.median(corr))
    return lag, peak / best_rival, peak / median


def main() -> None:
    wavs = sorted(CAP.glob("*.clean.wav"))
    a_path = [w for w in wavs if "Meeting" in w.name][0]
    b_path = [w for w in wavs if "Tag5" in w.name][0]
    rate_a, sa = wavfile.read(a_path)
    rate_b, sb = wavfile.read(b_path)
    assert rate_a == rate_b, (rate_a, rate_b)
    print(f"A: {len(sa)/rate_a/60:.1f} min   B: {len(sb)/rate_b/60:.1f} min   rate={rate_a}\n")

    ea = _rms_envelope(sa, rate_a, WINDOW)
    eb = _rms_envelope(sb, rate_b, WINDOW)
    rng = np.random.default_rng(0)

    ha, hb = len(ea) // 2, len(eb) // 2
    shuffled = eb.copy()
    rng.shuffle(shuffled)
    cases = {
        "TRUE PAIR (A vs B)":            (ea, eb),
        "A 1st half vs B 2nd half":      (ea[:ha], eb[hb:]),
        "A 2nd half vs B 1st half":      (ea[ha:], eb[:hb]),
        "A vs shuffled B":               (ea, shuffled),
        "A vs reversed B":               (ea, eb[::-1]),
        "A vs gaussian noise":           (ea, np.abs(rng.normal(0, eb.std(), len(eb)))),
    }

    print(f"{'case':<28} {'lag (s)':>9} {'peak/best_rival':>17} {'peak/median':>13}")
    print("-" * 70)
    results = {}
    for name, (x, y) in cases.items():
        lag, ratio, pmed = metrics(x, y)
        results[name] = ratio
        print(f"{name:<28} {lag:>9.1f} {ratio:>17.4f} {pmed:>13.3f}")

    true = results["TRUE PAIR (A vs B)"]
    nulls = [v for k, v in results.items() if k != "TRUE PAIR (A vs B)"]
    print("-" * 70)
    print(f"true = {true:.4f}   null max = {max(nulls):.4f}   "
          f"separation = {true/max(nulls):.2f}x")
    print(f"\nmidpoint threshold would be {(true + max(nulls))/2:.3f}")
    print("NOTE: n=1 for the true pair. Warn, do not gate.")


if __name__ == "__main__":
    main()
