"""Establish the CAUSE of the cross-speaker duplicate pairs, or fail honestly.

The prior triage logged them as "cause not established -- may be
speaker-attribution error, or genuine cross-talk both microphones captured",
and said distinguishing the two requires listening to the recordings.

There is a structural discriminator that does not require listening, and this
script measures it. merge_turns only ever replaces an A turn with a B turn of
the SAME (mapped) speaker, and only ever gap-fill-appends a B turn when NO
same-speaker A turn overlaps it. So:

  * If the two sources AGREED on who was speaking, B's turn either replaces A's
    turn (one output block) or is dropped as already-represented. Agreement
    cannot produce two blocks.
  * Therefore any cross-speaker duplicate involving a gap-filled B turn PROVES
    the two sources disagreed about the speaker.
  * A cross-speaker duplicate between two A turns proves source A's own
    diarization split one stretch of speech between two speaker clusters.

The rival hypothesis -- genuine cross-talk, i.e. two people really saying
near-identical words -- predicts the duplicated spans sit at DIFFERENT times.
Attribution error predicts they sit at the SAME time. That is measurable.

INSTRUMENT CHECK: the tagged merge is asserted equal to production merge_turns
on this exact input before any number below is used.
"""
import pickle
import re
from difflib import SequenceMatcher
from pathlib import Path

from audio_to_text.fusion import (
    _CONTAINMENT_MIN_CHARS, _CONTAINMENT_RADIUS, _normalize_for_containment,
    _shift_and_remap, match_speakers, merge_turns,
)
from audio_to_text.transcribe import overlap_seconds, relabel_speakers

from _capture import require_capture
MIN_SHARED, MIN_FRACTION, MAX_DISTANCE = 40, 0.5, 5


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower())


def tagged_merge(turns_a, turns_b_shifted):
    """Production merge_turns, with each emitted turn tagged by its origin branch."""
    replacement, used = {}, set()
    for i, turn in enumerate(turns_a):
        cands = [b for b in turns_b_shifted
                 if id(b) not in used and b["speaker"] == turn["speaker"]
                 and overlap_seconds(turn["start"], turn["end"], b["start"], b["end"]) > 0]
        if cands:
            best = max(cands, key=lambda b: b["confidence"])
            if best["confidence"] > turn["confidence"]:
                replacement[i] = best
                used.add(id(best))

    consumed = set()
    for i, best in replacement.items():
        bt = _normalize_for_containment(best["text"])
        for j in range(max(0, i - _CONTAINMENT_RADIUS),
                       min(len(turns_a), i + _CONTAINMENT_RADIUS + 1)):
            if j == i or j in replacement or turns_a[j]["speaker"] != turns_a[i]["speaker"]:
                continue
            sj = _normalize_for_containment(turns_a[j]["text"])
            if len(sj) >= _CONTAINMENT_MIN_CHARS and sj in bt:
                consumed.add(j)

    merged = []
    for i, turn in enumerate(turns_a):
        if i in consumed:
            continue
        best = replacement.get(i)
        if best is not None:
            merged.append({**turn, "start": best["start"], "end": best["end"],
                           "text": best["text"], "confidence": best["confidence"],
                           "_origin": "a_replaced", "_a_index": i})
        else:
            merged.append({**turn, "_origin": "a_kept", "_a_index": i})

    for turn in turns_b_shifted:
        if not any(a["speaker"] == turn["speaker"]
                   and overlap_seconds(turn["start"], turn["end"], a["start"], a["end"]) > 0
                   for a in turns_a):
            merged.append({**turn, "_origin": "b_gapfill", "_a_index": None})
    merged.sort(key=lambda t: t["start"])
    return merged


def main() -> None:
    cap = pickle.load(open(require_capture(), "rb"))
    turns_a = cap["a"]["grouped"]
    a_to_b = match_speakers(cap["a"]["embeddings"], cap["b"]["embeddings"])
    turns_b = _shift_and_remap(cap["b"]["grouped"], cap["offset"], {v: k for k, v in a_to_b.items()})

    tagged = tagged_merge(turns_a, turns_b)
    production = merge_turns(turns_a, turns_b)
    stripped = [{k: v for k, v in t.items() if not k.startswith("_")} for t in tagged]
    assert stripped == production, "INSTRUMENT DISAGREES WITH PRODUCTION -- numbers invalid"
    print(f"instrument == production merge_turns ({len(tagged)} turns)\n")

    blocks = relabel_speakers(tagged)
    print(f"origin mix: " + str({o: sum(1 for b in blocks if b['_origin'] == o)
                                 for o in ('a_kept', 'a_replaced', 'b_gapfill')}) + "\n")

    pairs = []
    for d in range(1, MAX_DISTANCE + 1):
        for i in range(len(blocks) - d):
            x, y = blocks[i], blocks[i + d]
            nx, ny = norm(x["text"]), norm(y["text"])
            if not nx or not ny:
                continue
            m = SequenceMatcher(None, nx, ny, autojunk=False).find_longest_match(0, len(nx), 0, len(ny))
            if m.size > MIN_SHARED and m.size > MIN_FRACTION * min(len(nx), len(ny)):
                if x["speaker"] != y["speaker"]:
                    pairs.append((i, d, x, y, m.size))

    print(f"== {len(pairs)} CROSS-SPEAKER duplicate pairs ==\n")
    origin_mix, timing = {}, {"same_time": 0, "different_time": 0}
    for i, d, x, y, shared in pairs:
        key = " + ".join(sorted([x["_origin"], y["_origin"]]))
        origin_mix[key] = origin_mix.get(key, 0) + 1
        ov = overlap_seconds(x["start"], x["end"], y["start"], y["end"])
        shorter = min(x["end"] - x["start"], y["end"] - y["start"])
        same_time = shorter > 0 and ov / shorter > 0.5
        timing["same_time" if same_time else "different_time"] += 1
        print(f"  idx={i:>3} d={d}  {x['speaker']}({x['_origin']}) -> {y['speaker']}({y['_origin']})")
        print(f"       shared={shared:>3}ch  spans [{x['start']:.1f},{x['end']:.1f}] vs "
              f"[{y['start']:.1f},{y['end']:.1f}]  overlap={ov:.1f}s "
              f"({'SAME TIME' if same_time else 'different time'})")

    print(f"\n== origin mix of the cross-speaker pairs ==")
    for k, v in sorted(origin_mix.items(), key=lambda kv: -kv[1]):
        print(f"   {k:<28} {v}")
    print(f"\n== timing ==\n   {timing}")
    print("\nInterpretation guide:")
    print("  any pair involving b_gapfill  -> the two sources DISAGREED on the speaker")
    print("     (agreement cannot emit two blocks: B either replaces A's turn or is dropped)")
    print("  a_kept + a_kept               -> source A's own diarization split one")
    print("     stretch of speech across two speaker clusters, with no fusion involvement")
    print("  SAME TIME                     -> one stretch of speech rendered twice")
    print("  different time                -> consistent with genuine repetition/cross-talk")


if __name__ == "__main__":
    main()
