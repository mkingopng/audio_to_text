"""Measure the same-speaker containment guard's yield on the real captured turns.

Sweeps the A-index radius and the minimum-length floor, and reports for each
setting: how many same-speaker pairs it would consume, how many CROSS-speaker
pairs it would consume if it were not restricted (which is the failure mode --
consuming the correctly attributed copy), and how many characters it removes.

Full containment is lossless by construction -- the consumed text is a substring
of the block that remains -- so this tool does not attempt to report "characters
destroyed": for this criterion it is always zero, and a column that is zero by
construction is theatre. tools/verify_lossless.py measures the property end to end
on rendered output instead, where it can actually fail.

The instrumented merge is checked for equivalence against production merge_turns
on this exact input before any of its numbers are used -- an instrument that
disagrees with the thing it is measuring is worthless.
"""
import pickle
import re
from pathlib import Path

from audio_to_text import fusion
from audio_to_text.fusion import merge_turns, match_speakers, _shift_and_remap
from audio_to_text.transcribe import overlap_seconds

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


def _temporally_eligible(turns_a, i, j, b):
    """The two TIME conditions production applies, which this sweep omitted.

    Without these the instrument models a guard that has not shipped since
    5b91665, and every count it prints overstates what production would consume
    -- the exact charge bugs.md levels at the prior triage.
    """
    if overlap_seconds(turns_a[j]["start"], turns_a[j]["end"], b["start"], b["end"]) <= 0.0:
        return False
    gap = max(turns_a[j]["start"] - turns_a[i]["end"],
              turns_a[i]["start"] - turns_a[j]["end"], 0.0)
    return gap <= fusion._CONTAINMENT_MAX_GAP_SECONDS


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
          f"{'chars removed':>14}")
    print("-" * 70)
    # floor 8 and 20 bracket the measured bimodal gap (micro-turns <= 8 chars,
    # restated content >= 20), which is the argument the shipped value rests on.
    for radius in (1, 2, 3, 4):
        for floor in (0, 8, 20, 40):
            same = cross = chars = 0
            for i, b in replacement.items():
                bn = norm(b["text"])
                for j in range(max(0, i - radius), min(len(turns_a), i + radius + 1)):
                    if j == i or j in replacement:
                        continue
                    jn = norm(turns_a[j]["text"])
                    if len(jn) < floor or not jn:
                        continue
                    if not _temporally_eligible(turns_a, i, j, b):
                        continue
                    if jn in bn:
                        if turns_a[j]["speaker"] == turns_a[i]["speaker"]:
                            same += 1
                            chars += len(jn)
                        else:
                            cross += 1
            print(f"{radius:>7} {floor:>6} {same:>11} {cross:>12} {chars:>14}")

    # What the guard leaves behind, at the chosen setting.
    print(f"\n== detail at the SHIPPED setting: radius={fusion._CONTAINMENT_RADIUS}, "
          f"floor={fusion._CONTAINMENT_MIN_CHARS} (same-speaker only) ==")
    for i, b in replacement.items():
        bn = norm(b["text"])
        for j in range(max(0, i - fusion._CONTAINMENT_RADIUS),
                       min(len(turns_a), i + fusion._CONTAINMENT_RADIUS + 1)):
            if j == i or j in replacement:
                continue
            jn = norm(turns_a[j]["text"])
            if (len(jn) >= fusion._CONTAINMENT_MIN_CHARS and jn in bn
                    and turns_a[j]["speaker"] == turns_a[i]["speaker"]
                    and _temporally_eligible(turns_a, i, j, b)):
                print(f"   A[{i}] replaced; A[{j}] ({len(jn)} chars) fully contained -> consume")
                print(f"      sibling: {turns_a[j]['text'][:90]!r}")


if __name__ == "__main__":
    main()
