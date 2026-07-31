"""Render the captured real merge to Markdown, with the containment guard on or off,
so the guard's effect can be measured with tools/analyze_transcript.py rather than
asserted.

    uv run python render_capture.py <out.md> [--no-guard]
"""
import pickle
import sys
from pathlib import Path

from audio_to_text import fusion
from audio_to_text.fusion import match_speakers, merge_turns, _shift_and_remap
from audio_to_text.transcribe import relabel_speakers, render_markdown

from _capture import require_capture


def main() -> None:
    out = Path(sys.argv[1])
    if "--no-guard" in sys.argv:
        # a floor nothing can reach == guard disabled, without touching the source
        fusion._CONTAINMENT_MIN_CHARS = 10 ** 9

    cap = pickle.load(open(require_capture(), "rb"))
    turns_a = cap["a"]["grouped"]
    a_to_b = match_speakers(cap["a"]["embeddings"], cap["b"]["embeddings"])
    turns_b = _shift_and_remap(cap["b"]["grouped"], cap["offset"], {v: k for k, v in a_to_b.items()})

    merged = merge_turns(turns_a, turns_b)
    out.write_text(render_markdown(relabel_speakers(merged)), encoding="utf-8")
    print(f"{out.name}: {len(merged)} turns")


if __name__ == "__main__":
    main()
