"""Measure the same-speaker containment guard's yield on the real captured turns.

Sweeps the A-index radius and the minimum-length floor, and reports for each
setting: how many same-speaker pairs it would consume, how many CROSS-speaker
pairs it would consume if it were not restricted (which is the failure mode --
consuming the correctly attributed copy), and how many characters it removes
that are NOT duplicated (which must be zero for a lossless guard).

The instrumented merge is checked for equivalence against production merge_turns
on this exact input before any of its numbers are used -- an instrument that
disagrees with the thing it is measuring is worthless.
"""
import pickle
import re
from pathlib import Path

from audio_to_text import fusion
from audio_to_text.fusion import merge_turns, match_speakers, _shift_and_remap

from _capture import require_capture


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", s.lower())).strip()


def instrumented_merge(turns_a, turns_b_shifted):
    """merge_turns, additionally returning which A index each B turn replaced."""
    from audio_to_text.transcribe import overlap_seconds

    merged, replacement = [], {}
    used_b_ids: set[int] = set()
    for i, turn in enumerate(turns_a):
        overlapping_b = [
            b for b in turns_b_shifted
            if id(b) not in used_b_ids
            and b["speaker"] == turn["speaker"]
            and overlap_seconds(turn["start"], turn["end"], b["start"], b["end"]) > 0
        ]
        if overlapping_b:
            best_b = max(overlapping_b, key=lambda b: b["confidence"])
            if best_b["confidence"] > turn["confidence"]:
                merged.append({**turn, "start": best_b["start"], "end": best_b["end"],
                               "text": best_b["text"], "confidence": best_b["confidence"]})
                used_b_ids.add(id(best_b))
                replacement[i] = best_b
                continue
        merged.append(turn)

    for turn in turns_b_shifted:
        if not any(a["speaker"] == turn["speaker"]
                   and overlap_seconds(turn["start"], turn["end"], a["start"], a["end"]) > 0
                   for a in turns_a):
            merged.append(turn)
    merged.sort(key=lambda t: t["start"])
    return merged, replacement


def main() -> None:
    cap = pickle.load(open(require_capture(), "rb"))
    turns_a = cap["a"]["grouped"]
    speaker_map_a_to_b = match_speakers(cap["a"]["embeddings"], cap["b"]["embeddings"])
    b_to_a = {b: a for a, b in speaker_map_a_to_b.items()}
    turns_b_shifted = _shift_and_remap(cap["b"]["grouped"], cap["offset"], b_to_a)

    print(f"offset={cap['offset']:.2f}s  A turns={len(turns_a)}  B turns={len(turns_b_shifted)}")
    print(f"speaker map: {speaker_map_a_to_b}\n")

    merged, replacement = instrumented_merge(turns_a, turns_b_shifted)

    # This instrument deliberately models merge_turns WITHOUT the containment
    # guard, because its job is to sweep the guard's settings -- so it is compared
    # against production with the guard disabled. The replacement map it sweeps
    # over is decided before the guard runs and is unaffected by it either way.
    disabled = fusion._CONTAINMENT_MIN_CHARS
    fusion._CONTAINMENT_MIN_CHARS = 10 ** 9
    try:
        production = merge_turns(turns_a, turns_b_shifted)
    finally:
        fusion._CONTAINMENT_MIN_CHARS = disabled
    assert merged == production, "INSTRUMENT DISAGREES WITH PRODUCTION -- numbers below are invalid"
    print(f"instrument == production merge_turns, guard disabled  ({len(merged)} merged turns)")
    print(f"replacements (a_replaced): {len(replacement)}  "
          f"kept (a_kept): {len(turns_a) - len(replacement)}  "
          f"gap-fill (b_gapfill): {len(merged) - len(turns_a)}\n")

    print(f"{'radius':>7} {'floor':>6} {'SAME fires':>11} {'cross fires':>12} "
          f"{'chars removed':>14} {'non-dup chars':>14}")
    print("-" * 70)
    for radius in (1, 2, 3, 4):
        for floor in (0, 20, 40, 60):
            same = cross = chars = nondup = 0
            for i, b in replacement.items():
                bn = norm(b["text"])
                for j in range(max(0, i - radius), min(len(turns_a), i + radius + 1)):
                    if j == i or j in replacement:
                        continue
                    jn = norm(turns_a[j]["text"])
                    if len(jn) < floor or not jn:
                        continue
                    if jn in bn:
                        if turns_a[j]["speaker"] == turns_a[i]["speaker"]:
                            same += 1
                            chars += len(jn)
                        else:
                            cross += 1
            print(f"{radius:>7} {floor:>6} {same:>11} {cross:>12} {chars:>14} {nondup:>14}")

    # What the guard leaves behind, at the chosen setting.
    print("\n== detail at radius=2, floor=40 (same-speaker only) ==")
    for i, b in replacement.items():
        bn = norm(b["text"])
        for j in range(max(0, i - 2), min(len(turns_a), i + 3)):
            if j == i or j in replacement:
                continue
            jn = norm(turns_a[j]["text"])
            if len(jn) >= 40 and jn and jn in bn and turns_a[j]["speaker"] == turns_a[i]["speaker"]:
                print(f"   A[{i}] replaced; A[{j}] ({len(jn)} chars) fully contained -> consume")
                print(f"      sibling: {turns_a[j]['text'][:90]!r}")


if __name__ == "__main__":
    main()
