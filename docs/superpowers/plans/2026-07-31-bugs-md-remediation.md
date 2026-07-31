# Plan — close out `bugs.md`

**Date:** 2026-07-31
**Branch:** `feature/global-tool`
**Spec:** all seven `bugs.md` entries either fixed with a mutation-tested test, or
logged with a re-measured, evidence-backed status.

**Evidence base.** `docs/validate/2026-07-31-bugs-md-triage.md` is the prior triage, but
its `scratchpad/` no longer exists, so every derived number in it (branch attribution
z=+4.4, guard yield 6/23, "113 wrong timestamps") is currently unreproducible. Numbers
used here are re-measured: transcript-level via `tools/analyze_transcript.py` against the
shipped 714-block output, mechanism-level via deterministic unit reproduction against
production `merge_turns`, and pipeline-level via a fresh capture of the real 70-minute
pair.

**Non-negotiable for every fix:** the test must FAIL before the change, PASS after, and
be mutation-tested — revert the fix, confirm the test goes red again. A prior review
found three behaviours in this repo whose tests stayed green with the feature deleted, so
"the suite passes" is not evidence on its own.

**Ordering is causal.** Task 1 re-sorts merged blocks, so Tasks 2 and 6 must be measured
after it lands, never before.

**Out of scope:** the packaging/CLI surface (src-layout package, `resolve_hf_token`, the
`./data/transcriptions` default) — finished and verified. Task 4 changes the fused
output's filename only, not the CLI argument surface.

---

## Task 1 — `merge_turns` carries B's span with B's text

- [x] 1.1 Test: a merged turn's `(start, end, text)` must all originate from one source
      turn. Pins the real behaviour — today the replacement branch emits B's text under
      A's span, so a 90-second B turn renders with a 0.4 s duration.
- [x] 1.2 Test: the concrete user-visible defect — an impossible words-per-second block.
- [x] 1.3 Fix: carry `best_b`'s `start`/`end` through the replacement branch.
- [x] 1.4 Mutation-test both tests; confirm the existing 60 stay green.

## Task 2 — same-speaker containment guard

- [ ] 2.1 Measure the guard's yield on the captured real turns, at the full-containment
      criterion, sweeping the minimum-length floor. Record the chosen floor and why.
- [ ] 2.2 Test: a non-replaced same-speaker A turn whose text is fully contained in the
      winning B text is consumed, not emitted twice.
- [ ] 2.3 Test: the guard does NOT fire cross-speaker (it would delete the correctly
      attributed copy and keep the wrong one).
- [ ] 2.4 Test: the guard does NOT fire below the length floor (no back-door smoothing).
- [ ] 2.5 Fix: implement the guard in `merge_turns`.
- [ ] 2.6 Mutation-test; re-measure redundancy on a real fused run.

## Task 3 — surface the fusion offset and its confidence

- [ ] 3.1 Re-derive `peak/best_rival` on the captured WAVs plus self-built null controls.
      Choose the warn threshold from measured separation, not from the prior doc.
- [ ] 3.2 Test: the confidence ratio separates a true pair from an unrelated pair.
- [ ] 3.3 Fix: `_correlate_envelopes()` returns `(lag, ratio)`; `find_offset` keeps its
      signature; `run_fusion` reports the offset and warns below threshold.
- [ ] 3.4 Mutation-test.

## Task 4 — fused output takes a `.fused.md` suffix

- [x] 4.1 Test: `run_fusion` writes `<stem>.fused.md`, and a pre-existing `<stem>.md`
      from a single-file run survives untouched.
- [x] 4.2 Fix: change the output filename in `run_fusion`.
- [x] 4.3 Mutation-test; update README.

## Task 5 — warn on Whisper repetition loops

- [x] 5.1 Test: a transcript containing a doc-wide repetition loop is detected, including
      when the loop is shredded across many one-word turns (the case the original
      within-block instrument was blind to).
- [x] 5.2 Test: an ordinary transcript produces no warning.
- [x] 5.3 Fix: `detect_repetition_loops(turns)` in `transcribe.py`, called from both the
      single-file and fusion paths.
- [x] 5.4 Mutation-test.

## Task 6 — micro-turn smoothing (Rule G2)  **[highest risk]**

- [ ] 6.1 Re-measure G2's eligible set on the captured turns AFTER Task 1, since Task 1
      changes block order and "sandwiched" depends on it.
- [ ] 6.2 Hand-check every absorption on the real transcript; report genuine turns
      damaged. If it damages real speech, do NOT land it — say so and leave it logged.
- [ ] 6.3 Test: a sandwiched ≤2-word zero-duration non-backchannel turn is re-attributed
      into the surrounding same-speaker turn, with every word preserved.
- [ ] 6.4 Test: a backchannel token is NOT absorbed.
- [ ] 6.5 Test: a turn with real duration is NOT absorbed.
- [ ] 6.6 Test: a turn NOT sandwiched by one speaker is never moved across a speaker
      boundary.
- [ ] 6.7 Fix: `smooth_micro_turns(turns)`, called from both paths.
- [ ] 6.8 Mutation-test; re-measure fragmentation end-to-end.

## Task 7 — cross-speaker duplicate pairs: establish cause or log it

- [ ] 7.1 Trace each cross-speaker duplicate to its originating branch and source using
      the captured intermediates.
- [ ] 7.2 Test the candidate discriminators (speaker-map disagreement vs genuine
      cross-talk), including whether each instrument could observe the alternative.
- [ ] 7.3 Either fix with a test, or log in `bugs.md` with the cause established and the
      falsification that failed. Do not guess.

## Task 8 — reconcile the record

- [ ] 8.1 Rewrite `bugs.md`: each entry closed or logged, every number one I measured.
- [ ] 8.2 Write up the re-measurement in `docs/validate/`, including where it corrects
      the prior triage.
- [ ] 8.3 Full suite green; fused run driven end-to-end; before/after measured.

---

## Acceptance checks

1. Every landed fix has a test that was red before, green after, and red again on revert
   — with the failing output pasted, not asserted.
2. `uv run pytest` green (60 baseline + new).
3. A real `--fuse` run completes and its output is measured with `analyze_transcript.py`.
4. Task 6 lands only if hand-checking shows it destroys no genuine speech.
5. `bugs.md` contains no number that cannot be reproduced from a committed script.
