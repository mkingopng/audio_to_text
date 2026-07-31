"""Re-run the real fusion pipeline and persist every intermediate.

The triage doc's derived numbers (branch attribution, guard yield, timestamp
corruption scale) came from a scratchpad/ that no longer exists, so none of them
can currently be checked. This rebuilds the evidence base by running the SAME
production functions and keeping what run_fusion throws away.

Writes to OUT: turns_a / turns_b (grouped, B pre-shift), raw diarization turns,
speaker embeddings, the offset, and both cleaned WAVs.

    uv run python <this> 2>&1 | tee capture.log
"""
from __future__ import annotations

import pickle
import shutil
import sys
import time
from pathlib import Path

REPO = Path("/Users/mkingomac.com/Developer/GitHub/audio_to_text")
OUT = Path(__file__).parent / "capture"

from audio_to_text.transcribe import (  # noqa: E402
    align_words_to_speakers,
    _group_consecutive,
    extract_words,
    load_diarization_pipeline,
    preprocess_audio,
    resolve_hf_token,
    run_diarization,
    run_whisper,
)
from audio_to_text.fusion import find_offset  # noqa: E402

PRIMARY = REPO / "data/Meeting with Michael Kingston-20260729_130839-Meeting Recording.mp4"
SECONDARY = REPO / "data/Tag5_29_July_2026_10903_pm.m4a"
MODEL = "mlx-community/whisper-large-v3-turbo"
NUM_SPEAKERS = 6  # the shipped reference output has exactly Person 1..6


def process(tag: str, media: Path, pipeline) -> dict:
    """Same call sequence as fusion._process_source, but retaining intermediates."""
    t0 = time.time()
    wav = preprocess_audio(media, OUT, None)
    print(f"[{tag}] wav {wav.name} ({time.time()-t0:.0f}s)", flush=True)

    t0 = time.time()
    result = run_whisper(wav, model_repo=MODEL, language="en", initial_prompt=None)
    words = extract_words(result)
    print(f"[{tag}] asr: {len(words)} words ({time.time()-t0:.0f}s)", flush=True)

    t0 = time.time()
    diar_turns, embeddings = run_diarization(wav, pipeline, num_speakers=NUM_SPEAKERS)
    print(f"[{tag}] diarization: {len(diar_turns)} turns ({time.time()-t0:.0f}s)", flush=True)

    aligned = align_words_to_speakers(words, diar_turns)
    grouped = _group_consecutive(aligned)
    print(f"[{tag}] grouped: {len(grouped)} turns", flush=True)
    return {
        "wav": wav, "words": words, "diar_turns": diar_turns,
        "embeddings": embeddings, "grouped": grouped,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (PRIMARY, SECONDARY):
        if not f.exists():
            print(f"missing: {f}", file=sys.stderr)
            return 1

    pipeline = load_diarization_pipeline(resolve_hf_token())
    a = process("A", PRIMARY, pipeline)
    b = process("B", SECONDARY, pipeline)

    offset = find_offset(a["wav"], b["wav"])
    print(f"offset = {offset:.3f}s", flush=True)

    with open(OUT / "capture.pkl", "wb") as fh:
        pickle.dump({
            "offset": offset,
            "a": {k: v for k, v in a.items() if k != "wav"},
            "b": {k: v for k, v in b.items() if k != "wav"},
            "wav_a": str(a["wav"]), "wav_b": str(b["wav"]),
            "model": MODEL, "num_speakers": NUM_SPEAKERS,
        }, fh)
    print(f"wrote {OUT/'capture.pkl'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
