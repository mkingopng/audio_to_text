"""Prove that the containment guard and micro-turn smoothing lose no speech.

Produces the before/after figures bugs.md quotes, from the shipped code paths
rather than an ad-hoc script:

    guard   OFF -> ON   block delta, word delta, word TYPES lost
    smooth  OFF -> ON   block delta, word delta, word TYPES lost

"Lossless" is checked per BLOCK -- words are collected from parsed block bodies and
never joined across a block boundary. Flattening the whole document to one string
glues neighbours together ("months" + "the" -> "monthsthe") and invents vanished
word types that were never lost; that instrument reports failures that are its own.

    uv run python tools/verify_lossless.py
"""
from __future__ import annotations

import collections
import pickle
import re

from audio_to_text import fusion
from audio_to_text.fusion import match_speakers, merge_turns, _shift_and_remap
from audio_to_text.transcribe import relabel_speakers, render_markdown, smooth_micro_turns

from _capture import require_capture


def words_per_block(turns) -> collections.Counter:
    return collections.Counter(
        word
        for turn in turns
        for word in re.sub(r"[^a-z0-9 ]", " ", turn["text"].lower()).split()
    )


def compare(label: str, before, after) -> bool:
    wb, wa = words_per_block(before), words_per_block(after)
    lost_types = {w: c for w, c in (wb - wa).items() if w not in wa}
    gained = sum((wa - wb).values())
    print(f"\n== {label} ==")
    print(f"   blocks: {len(before)} -> {len(after)}   "
          f"({(len(before)-len(after))/max(1,len(before)):.1%} reduction)")
    print(f"   words:  {sum(wb.values())} -> {sum(wa.values())}   "
          f"({sum((wb-wa).values())} occurrences removed, {gained} added)")
    print(f"   word TYPES lost from the document: {lost_types or 'NONE'}")
    ok = not lost_types and gained == 0
    print(f"   LOSSLESS: {'yes' if ok else 'NO -- speech was destroyed'}")
    return ok


def main() -> int:
    cap = pickle.load(open(require_capture(), "rb"))
    turns_a = cap["a"]["grouped"]
    a_to_b = match_speakers(cap["a"]["embeddings"], cap["b"]["embeddings"])
    turns_b = _shift_and_remap(cap["b"]["grouped"], cap["offset"],
                               {v: k for k, v in a_to_b.items()})

    shipped_floor = fusion._CONTAINMENT_MIN_CHARS
    fusion._CONTAINMENT_MIN_CHARS = 10 ** 9          # guard disabled
    try:
        guard_off = merge_turns(turns_a, turns_b)
    finally:
        fusion._CONTAINMENT_MIN_CHARS = shipped_floor
    guard_on = merge_turns(turns_a, turns_b)

    ok = compare("containment guard: OFF -> ON", guard_off, guard_on)
    ok &= compare("micro-turn smoothing: OFF -> ON", guard_on, smooth_micro_turns(guard_on))

    final = relabel_speakers(smooth_micro_turns(guard_on))
    print(f"\n== shipped pipeline end to end ==")
    print(f"   {len(guard_off)} merged turns -> {len(final)} rendered blocks")
    print(f"   rendered characters: {len(render_markdown(final))}")
    print(f"\nOVERALL: {'lossless' if ok else 'SPEECH WAS DESTROYED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
