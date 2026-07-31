# Known issues / deferred findings

Bugs and limitations noticed during development but deliberately not fixed, with
the reasoning. Each is a conscious deferral, not an oversight.

> **2026-07-31 — all three original entries were re-triaged against measured data**
> (`docs/validate/2026-07-31-bugs-md-triage.md`). None survived its recorded reasoning
> unchanged, and three further defects were found that had never been logged. Reproduce
> any number below with `uv run python tools/analyze_transcript.py <transcript.md>`.

## Fusion: residual redundancy when one B turn spans two A turns

**Status:** open — mechanism confirmed, **severity and scale were understated**
**Found:** 2026-07-31, `/review` of the speaker-diarization-fusion feature
**Re-triaged:** 2026-07-31 against the 714-block reference output

`merge_turns` (`src/fusion.py`) selects text at whole-turn granularity. When the
secondary source's diarization produces one long turn spanning two of the primary
source's turns, the winning B turn's text may include content that also appears in
the *next* A turn, which is emitted separately.

The exact-duplicate case (identical text under two headings) was fixed via the
`used_b_ids` set. That fix eliminated *exact* duplication and left **containment**
duplication: B's spanning text still contains what the sibling A turn says, and the
sibling is emitted unchanged.

**Corrections to the original entry:**

- **"Observed once in ~70 minutes" → 16 same-speaker instances** (26 including
  cross-speaker), with shared spans up to 325 characters — whole paragraphs restated.
  The original count came from a scan of *adjacent* blocks, which is structurally
  blind to this bug: `_group_consecutive` flushes on every speaker change, so two
  same-speaker turns are never adjacent. Scanning at block distances 1–5 finds them.
- **"Cosmetic" → true only for the same-speaker cases.** 10 cross-speaker duplicates
  place a paragraph under a person who did not say it. That is a correctness defect.
- **The prescribed fix was wrong.** Branch attribution (origin-tagging every merged
  turn, permutation-tested against base rate at z = +4.4) shows **31 of 35 redundant
  pairs come from the confidence-replacement branch**, only 4 from gap-fill.
- **A real fix does not need per-word splicing.** A containment guard on the
  replacement branch suffices — but it must use strict full containment (a looser
  similarity criterion destroys 840 characters of non-duplicated transcript), must be
  same-speaker-only (otherwise it deletes the correctly-attributed copy), and must
  search at A-index radius ≥ 2 (at radius 1 it fires on zero same-speaker cases).
  Measured lossless yield: 6 of 23 pairs.

## Fusion: no confidence check on the cross-correlation offset peak

**Status:** open — **not surfaced per run** (the recorded mitigation is out-of-band)
**Found:** 2026-07-29, task-level review of `find_offset`
**Re-triaged:** 2026-07-31

`find_offset` (`src/fusion.py`) returns the argmax lag with no measure of how
peaked the correlation actually is. Two recordings that don't share enough
acoustic content would still yield a confident-looking number.

**Correction to the original entry.** "Mitigated by process — the workflow has a human
sanity-check the offset" describes a one-off validation script (plan Task 12 Step 1),
run once on the reference pair. `src/fusion.py` contains **zero** `print` statements and
`run_fusion` never surfaces the offset, so every *future* run gets no visibility and no
peak-quality signal. The mitigation does not protect subsequent runs.

**A usable metric exists and is cheap.** `peak / best_rival` (best correlation peak more
than 5 s from the argmax), measured against five negative controls built from the same
recordings:

| pair | peak/best_rival |
|---|---|
| true pair | **1.5162** |
| A first half vs B second half | 1.0034 |
| A second half vs B first half | 1.0026 |
| A vs shuffled B | 1.0032 |
| A vs reversed B | 1.0049 |
| A vs gaussian noise | 1.0002 |

Note `peak/median` (3.135 vs nulls to 2.094) and the z-score (3.69 vs 1.80) separate far
less cleanly and should not be used. **Report and warn — do not gate:** the null floor is
well characterised but the true-pair distribution is n = 1.

## Whisper hallucination artifacts on ambiguous audio

**Status:** **REOPENED** — incidence unquantified; the original deferral rested on a
measurement that could not detect the problem
**Found:** 2026-07-30, real-recording validation
**Re-triaged:** 2026-07-31

Whisper occasionally emits degenerate word repetition on quiet or ambiguous passages.
Not a fusion bug — both sources produce it independently.

**Why the deferral is no longer justified as recorded.** The evidence for "rare" was a
count of identical consecutive words *within a block*. But ~31% of blocks hold a single
word (see fragmentation, below), so a repetition loop is shredded across dozens of blocks
and no single block ever contains a repeat. The instrument could only see loops that
fragmentation had not already scattered.

Re-measured doc-wide across block boundaries, on two ASR runs of the same audio:

```
run 1: within-block 4+ repeats = 3    doc-wide: nothing notable
run 2: within-block 4+ repeats = 13   doc-wide: 'lars' x183 across 34 blocks (17:45-18:01)
```

One run in two contains a **183-token hallucination loop** — a word appearing nowhere in
the other run. Still upstream and still arguably out of scope to *fix*, but the incidence
is **unknown**, not negligible, and it is bursty rather than diffuse. `tools/analyze_transcript.py`
now runs both the within-block and doc-wide passes.

## Output: ~31% of blocks are a single word (micro-turn fragmentation)

**Status:** open, unlogged until 2026-07-31
**Found:** 2026-07-31, measured triage

Of 714 blocks in the reference output, **220 (31%) hold one word** and **347 (49%) hold
five or fewer**, together carrying 4.8% of the text. Half the document's
`## Person N — mm:ss` headings introduce fragments like `"So"`, `"the"`, `"it?"`.

**Cause is upstream of grouping.** pyannote emits **42 of 96 turns shorter than 0.5 s**
(p10 = 0.02 s) on a sampled 4-minute slice; `align_words_to_speakers` assigns words to
these micro-turns and `_group_consecutive` faithfully starts a new block at each switch.
Present in the single-file path too (16% one-word), roughly doubled by fusion.

**Why not yet fixed.** A naive length threshold is destructive: 27% of short blocks are
genuine backchannel (`yeah`, `mm-hmm`, `gotcha`), and absorbing one misattributes real
speech. Duration alone is also unsafe — see timestamp corruption below. The most
conservative rule tested (≤2 words **and** exact-zero duration **and** not a backchannel
token) removes ~19–20% of blocks with zero hand-labelled genuine turns destroyed across
two runs, but it must be re-measured after the timestamp-corruption fix, which perturbs
block ordering.

## Fusion: `merge_turns` corrupts turn timestamps when B's text wins

**Status:** open, unlogged until 2026-07-31
**Found:** 2026-07-31, while hand-labelling smoothing candidates

In the confidence-replacement branch (`src/fusion.py`), the merged turn keeps **A's**
`start`/`end` while taking **B's** text:

```python
merged.append({**turn, "text": best_b["text"], "confidence": best_b["confidence"]})
```

When B's turn spans more speech than A's, the timestamps stop describing the text — the
reference output contains a **93-word block with a 0.4 s duration**. Over 206 replaced
turns: `start` differs from B's for 203 (max 42.6 s), `end` for 196 (max 84.8 s), and
**113 render a visibly wrong mm:ss heading**.

Consequences: wrong rendered timestamps, and any duration-based downstream logic is
unsafe on merged turns. **Not risk-free to fix** — `merge_turns` re-sorts by `start`, so
correcting it moves 109 of 684 blocks and changes which blocks are adjacent.

## Fusion: 12 cross-speaker duplicate pairs, cause unmeasured

**Status:** open, unlogged until 2026-07-31
**Found:** 2026-07-31, measured triage

Ten to twelve block pairs (run-dependent) carry near-identical text under **different**
`Person` headings, three of them at an identical timestamp. Not addressed by the
containment guard proposed for the redundancy bug above, which must be same-speaker-only.

**Cause is not established.** It may be speaker-attribution error (diarization or
Hungarian-matching disagreement between sources), or genuine cross-talk that both
microphones captured. Distinguishing the two requires listening to the recordings; no
automated discriminator has been validated. Logged rather than guessed at.

## Note: transcript metrics carry run-to-run variance

Whisper's temperature fallback makes ASR non-deterministic. Two runs over the same audio,
reusing the same diarization, gave 684 vs 756 blocks (**11%**, or 6% after excising a
hallucination span) and 23 vs 19 same-speaker redundant pairs (**17%**). Any single block
or pair count above should be read as approximate. Effects that survive this: branch
attribution (z = +4.4), the offset null-control separation, and the fragmentation shares.

## Output: a fused run silently overwrites a single-file transcript of the same primary

**Status:** open, minor — pre-existing, surfaced by the global-tool work
**Found:** 2026-07-31, verifying the `audio-to-text` wrapper end-to-end

Both modes name the output after the *primary* input's stem, so transcribing `teams.mp4`
and then fusing `teams.mp4 --fuse phone.m4a` writes `teams.md` twice, the second silently
replacing the first. Observed directly during verification: a single-file run produced
`data/transcriptions/teams.md`, and the subsequent fused run overwrote it with no warning.

Pre-existing behaviour, not introduced by the packaging change — but that change makes it
easier to hit, because output now defaults to a per-project `data/transcriptions/` that
accumulates across runs rather than a single `output/` folder the author was watching.

Not fixed here: the packaging work deliberately did not touch output naming, and the right
answer isn't obvious — a `.fused.md` suffix, a `--force` guard, and just letting it
overwrite are all defensible. Needs a decision, not a patch.
