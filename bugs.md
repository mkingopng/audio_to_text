# Known issues / deferred findings

Bugs and limitations noticed during development but deliberately not fixed, with
the reasoning. Each is a conscious deferral, not an oversight.

## Fusion: residual redundancy when one B turn spans two A turns

**Status:** open, cosmetic
**Found:** 2026-07-31, `/review` of the speaker-diarization-fusion feature

`merge_turns` (`src/fusion.py`) selects text at whole-turn granularity. When the
secondary source's diarization produces one long turn spanning two of the primary
source's turns, the winning B turn's text may include content that also appears in
the *next* A turn, which is emitted separately. The result reads as mild repetition
across two adjacent blocks — attribution is still correct.

The exact-duplicate case (identical text under two headings) was fixed via the
`used_b_ids` set. This residual case is narrower.

**Why deferred:** a real fix needs either per-word splicing — explicitly rejected in
the design spec, since splicing two independent ASR passes word-by-word garbles
sentences where the passes segment speech differently — or a containment-ratio
heuristic that risks discarding legitimately-better B transcription near turn
boundaries. That's a design tradeoff, not a mechanical patch. Observed once in
~70 minutes of real audio.

## Fusion: no confidence check on the cross-correlation offset peak

**Status:** open, mitigated by process
**Found:** 2026-07-29, task-level review of `find_offset`

`find_offset` (`src/fusion.py`) returns the argmax lag with no measure of how
peaked the correlation actually is. Two recordings that don't share enough
acoustic content would still yield a confident-looking number.

**Why deferred:** the workflow has a human sanity-check the offset against the
recordings before trusting fusion output. Worth revisiting if fusion is ever run
unattended.

## Whisper hallucination artifacts on ambiguous audio

**Status:** open, upstream
**Found:** 2026-07-30, real-recording validation

Whisper occasionally emits degenerate word repetition (e.g. `"In In In In..."`)
on quiet or ambiguous passages. Not a fusion bug — both sources produce it
independently. An upstream ASR limitation, out of scope for this project.
