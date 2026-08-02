# Known issues / deferred findings

Bugs and limitations noticed during development, with the reasoning. Each entry is
either **CLOSED** with the commit that fixed it and the test that pins it, or **OPEN**
with a measured status.

> **2026-08-02 — a two-pass review of this branch (code review + contrarian, per PR)
> added six fixes and reopened three questions.** New entries are dated 2026-08-02.
> The counts in the containment-guard entry are flagged as stale where a later
> condition invalidated them.
>
> **2026-07-31 — all seven entries were worked through and re-measured.**
> The prior triage (`docs/validate/2026-07-31-bugs-md-triage.md`) remains the narrative
> record, but its `scratchpad/` no longer exists, so none of its *derived* numbers
> (branch attribution z=+4.4, guard yield 6/23, "113 wrong timestamps") could be
> reproduced. Everything below was re-measured from scratch against
> the shipped reference output and a **fresh capture of the real 70-minute pair**.
> Where a number here disagrees with the prior triage, the disagreement is stated.
>
> Reproduce with:
> ```bash
> # transcript-level numbers, straight from a rendered .md
> uv run python tools/analyze_transcript.py <transcript.md>
>
> # pipeline-level numbers need a capture: re-runs ASR + diarization on both
> # sources and persists every intermediate. Slow (hours), and writes ~270 MB of
> # WAVs to data/fusion-capture/ -- set FUSION_CAPTURE_DIR to put it elsewhere.
> uv run python tools/capture_fusion_intermediates.py
>
> uv run python tools/measure_offset_confidence.py   # offset + null controls
> uv run python tools/measure_dc_bias.py            # offset search vs a DC-heavy envelope
> uv run python tools/measure_containment_guard.py   # guard yield by radius/floor
> uv run python tools/measure_smoothing.py           # micro-turn candidates, with context
> uv run python tools/analyze_cross_speaker.py       # cross-speaker pair origins
> uv run python tools/render_capture.py out.md       # render a capture, to diff before/after
> ```
> Each analysis tool asserts its instrumented merge equals production `merge_turns`
> on the same input before reporting anything — an instrument that disagrees with
> what it measures is worthless.
>
> **Honest limits on reproducibility.** The capture itself is not committed (two ~135 MB
> WAVs), and Whisper's temperature fallback means a fresh capture will not reproduce these
> counts exactly — see the variance note at the end. What *is* reproducible is the method:
> every figure quoted here is emitted by one of the scripts above rather than by an unsaved
> heredoc, which is the specific failure that made the prior triage's numbers uncheckable.
> Structural claims (0 same-speaker fires at radius 1; gap-fill cannot desynchronise
> timestamps; agreement cannot emit two blocks) are exact and hold on any input.

---

## CLOSED — `merge_turns` corrupted turn timestamps when B's text won

**Fixed:** `dfa7136` · **Severity:** correctness

The confidence-replacement branch emitted B's **text** under A's **start/end**, so a
merged turn's timestamps described different speech than its words. The reference output
contains a 93-word block with a 0.4 s duration, and 113 turns rendered a visibly wrong
`mm:ss` heading.

A merged turn is now B's turn outright when B's text wins.

*Mechanism confirmed exactly, not statistically.* The rival explanation on record — that
the impossible durations came from the gap-fill branch — is refuted by construction:
gap-fill appends B's turn **wholesale**, so it cannot desynchronise text from timestamps.
Only the replacement branch mixed the two.

Pinned by an invariant (`(start, end, text)` must all originate from one source turn)
plus the user-visible symptom (no impossible speaking rate). Both failed before the fix:
*"93 words in 0.4 s = 232 words/sec"*.

## CLOSED — fusion redundancy when one B turn spans two A turns

**Fixed:** `51f5bef` · **Severity:** cosmetic

The earlier `used_b_ids` fix removed *exact* duplication and left **containment**
duplication: B's spanning text still contained what the sibling A turn said, and the
sibling was emitted again underneath — whole paragraphs restated under a second heading.

A same-speaker containment guard now consumes the sibling. Three constraints, each
measured on the fresh capture and each independently mutation-tested:

| constraint | why, measured |
|---|---|
| **radius ≥ 2** in A-index terms | `_group_consecutive` flushes on every speaker change, so two same-speaker A turns are *never* adjacent. **0** same-speaker fires at radius 1, **13** at radius 2 |
| **same-speaker only** | an unrestricted guard fires on **28** cross-speaker cases, each time deleting the copy that may be the correctly attributed one |
| **≥ 20 normalised chars** | fires are bimodal — ≤8 chars are micro-turns (`I`, `for`, `And`, `you know`, `Daniel`), ≥20 are restated content, and the gap between is empty |

Full containment only, never a similarity threshold — a looser criterion consumes blocks
that are only *partly* duplicated and deletes the remainder.

> **The counts in the table above predate two later conditions and now overstate the
> guard's reach.** They were measured before the temporal-overlap condition (`5b91665`)
> and before the `_CONTAINMENT_MAX_GAP_SECONDS` bound, and `tools/measure_containment_guard.py`
> applied neither — so the instrument disagreed with production, which is the specific
> charge this file levels at the prior triage. The tool now applies both; the numbers
> above have **not** been re-measured, because that needs the 70-minute capture and it is
> not committed. Treat 13/28 as upper bounds on same-speaker/cross-speaker fires, not as
> current values. Re-run `tools/measure_containment_guard.py` against a fresh capture to
> replace them.

**Measured effect:** 592 → 587 blocks; same-speaker redundant pairs **16 → 14**;
cross-speaker unchanged at 16 (by design). 41 duplicate word occurrences removed, **zero
word types lost from the document**, nothing added.

## CLOSED — no confidence check on the cross-correlation offset peak

**Fixed:** `3cb36ed`

`find_offset` returned the argmax lag with no measure of how peaked the correlation was,
and `run_fusion` never surfaced it — `src/fusion.py` contained zero `print` statements. A
misaligned pair fused into a plausible-looking transcript with no signal.

The recorded mitigation ("the workflow has a human sanity-check the offset") was a one-off
validation script run once on one recording pair: a procedure, not a property of the tool.

`run_fusion` now reports the offset and `peak/best_rival`, warning below 1.2. Re-derived
against five negative controls built from the **same** two recordings, so any separation is
attributable to alignment rather than recording character:

| pair | peak/best_rival |
|---|---|
| true pair | **1.5162** |
| A 1st half vs B 2nd half | 1.0050 |
| A 2nd half vs B 1st half | 1.0019 |
| A vs shuffled B | 1.0032 |
| A vs reversed B | 1.0049 |
| A vs gaussian noise | 1.0002 |

`peak/median` (3.135 vs nulls to 2.098) and the z-score separate by less than 2× and were
rejected. **Warn, never gate** — the null floor is well characterised but the true-pair
distribution is n=1.

Validated end-to-end on 3-minute real slices the threshold was never tuned on: a correctly
aligned pair scored **1.43** (silent, offset recovered as +0.0 s), a partially overlapping
pair scored **1.19** (warned, offset +52.2 s — arithmetically correct for the cut).

## CLOSED — a fused run silently overwrote a single-file transcript of the same primary

**Fixed:** `4b101dd`

Both modes named the output after the primary's stem, so transcribing `teams.mp4` and then
fusing it wrote `teams.md` twice, the second run silently replacing the first.

Fusion now writes **`<stem>.fused.md`**. Of the three defensible answers (suffix, a
`--force` guard, or letting it overwrite) the suffix stops the collision by construction,
needs no new CLI flag, and records which pipeline produced the transcript.

Verified end-to-end: a single-file run then a fused run of the same primary now leave both
`teams.md` and `teams.fused.md` on disk.

## PARTLY CLOSED — micro-turn fragmentation

**Improved:** `a4c3dc1` · ~23% of blocks still hold a single word

Diarization emits very short turns (42 of 96 under 0.5 s on a sampled slice, p10 = 0.02 s);
`_group_consecutive` faithfully starts a new block at each switch, so half the document's
headings introduce fragments like `"So"`, `"the"`, `"it?"`.

Sandwiched ≤2-word, **exactly-zero-duration**, non-backchannel turns are
**re-attributed** into the surrounding speaker's sentence, **behind `--smooth` and off by
default** (`eb0e67d`). It is the only step in the pipeline that overrides diarization's
attribution rather than preserving or improving it, and its discriminator was hand-checked
on one meeting by the author of the word list it consults — so it is offered, not imposed.
**Re-attribution, not deletion**
— every word survives; only the spurious heading goes. That matters because the
discriminator is imperfect, so its failure mode must be a misattributed word rather than a
missing one.

**Measured:** 587 → 547 blocks (**6.8%**), words 13,797 → 13,797 — **zero lost, zero
gained**. All 19 clean candidates were hand-checked with context and every one is a
fragment stolen from the surrounding sentence (`the` inside "some of **the** other things",
`be a` inside "going to **be a** new season"). Zero genuine turns among them.

**Two corrections to the prior triage:**

- Its claimed yield was **19–20%** of blocks (132 of 684). Measured here the rule is
  eligible on 20 blocks (3.4%) and reduces the document by **6.8%** — real, but roughly a
  third of what was recorded.
- Its stated limitation *"G2 carries a fixed English word list, so it will not generalise"*
  overstates the list's role: on this run the word list excludes **zero** blocks. All 50
  exclusions among sandwiched ≤2-word blocks come from the **duration** clause. The list is
  untested insurance here, not load-bearing.

**Still open:** the remaining one-word blocks are not sandwiched, or have non-zero
duration. Going after them needs a real acoustic discriminator, not a longer word list.

## OPEN — Whisper hallucination loops (upstream), now detected

**Detection added:** `ed8da41` · **incidence: 2 of 3 runs**

Whisper emits degenerate repetition on quiet or ambiguous passages. Upstream and not
fixable here — but a run that hallucinates now says so instead of looking clean.

The original deferral rested on a blind instrument: it counted identical consecutive words
*within a block*, but ~23–31% of blocks hold a single word, so a loop is shredded across
dozens of blocks and no block ever contains a repeat. The scan is now **doc-wide**, across
turn boundaries.

| run | loops found |
|---|---|
| shipped reference | none |
| prior triage run 2 | `lars` ×183 |
| this capture | `paul` ×200, `whether` ×65 |

Threshold 10, measured: across the 13,454 tokens of the reference transcript the longest
*legitimate* consecutive run is **4** (`okay`, `yeah`, `easier` — natural emphasis).
Warns only, never gates.

**Why still open:** the fix belongs in Whisper. Incidence is now known to be non-negligible
and bursty rather than diffuse.

## PARTLY CLOSED — cross-speaker duplicate pairs (11, not 12)

**Cause established** + **now reported to the user** rather than left silent.

**Analysis:** `78eee79`, `docs/validate/2026-07-31-cross-speaker-cause.md`

Previously logged as *"cause unmeasured — may be attribution error or genuine cross-talk;
distinguishing them requires listening to the recordings."* There is a structural
discriminator that needs no listening, and it settles the question.

`merge_turns` only replaces an A turn with a **same-speaker** B turn, and only gap-fills a
B turn when **no** same-speaker A turn overlaps. So if the sources *agreed* on the speaker,
B either replaces A's turn or is dropped — **agreement cannot emit two blocks**.

Of 16 apparent pairs on the fresh capture:

- **5 are not duplication at all** — they sit inside the `paul` ×200 hallucination span, and
  degenerate text trivially "duplicates" itself between any two blocks. The honest count is
  **11**.
- **4** involve a gap-filled B turn → the two sources **disagreed** on the speaker (proven).
- **7** are `a_replaced + a_kept` → source A's own diarization split one stretch of speech
  across two clusters, and B's spanning turn covers both sides.

**Cross-talk is refuted:** all 11 real pairs sit at **gap 0.0 s** between their spans — one
stretch of speech rendered twice. Two speakers cannot both produce the same 46–126
characters simultaneously.

**What was done.** A fused run now *warns*, listing the timestamps where two speakers carry
near-identical text:

```
warning: 11 block pair(s) carry near-identical text under two different speakers, so
one heading in each pair is wrong. ... Check the speaker labels at:
  04:06  Person 1 vs Person 4 (55 shared characters)
  14:12  Person 4 vs Person 1 (101 shared characters)
  31:45  Person 1 vs Person 4 (126 shared characters)
  ...
```

**Why the attribution itself is not auto-resolved.** It needs an arbiter deciding which
source's diarization to trust, and the Hungarian embedding match is already the best signal
available — it is precisely what disagreed. Silently picking one copy would convert a
*visible* artifact (the same sentence under two names, which a reader notices) into a
confident-looking misattribution (which they do not), and risks deleting the correctly
attributed copy. Reporting is strictly better than guessing here.

Blocks inside a repetition loop are excluded, or degenerate text would trivially match
itself — that alone accounted for 5 of the 16 apparent pairs.

Scope: **11 pairs, ~1.9% of 587 blocks**, all now flagged by timestamp.

## CLOSED — the containment guard could delete speech minutes away

**Fixed:** `5b91665` · found by the contrarian review of this branch's own work

The guard matched on A-turn **index** adjacency and text alone. Indices say nothing about
elapsed time, so a turn twenty minutes later can sit two indices away: 60 seconds of speech
at t=1200 was deleted and its words reappeared under a heading stamped 0.1 s. That is the
same "span and text describe different speech" defect the timestamp fix eliminated,
re-entering through another door.

The sibling must now be speech B's turn actually covers. Costs nothing on real data —
identical 587 blocks, identical words, redundancy still 14.

## CLOSED — two-word backchannels bypassed the guard protecting them

**Fixed:** `5b91665`

`"Yeah yeah"` was keyed as one string (`"yeahyeah"`), which is not in the token set — so the
commonest two-word backchannels (`yeah yeah`, `no no`, `yeah okay`) were absorbed as jitter
by the very rule whose word list exists to protect them. Checked per word now, with the
hyphen preserved so `mm-hmm` and `uh-huh` stay protected too.

## OPEN — zero-duration B turns are always gap-filled

**Found:** 2026-07-31, while testing micro-turn smoothing · **minor**

`overlap_seconds` requires strictly positive overlap, and a zero-duration interval can
never produce that. So a B turn with `start == end` never counts as overlapping a
same-speaker A turn, and is appended as a gap-fill even when an A turn covers that instant —
producing a duplicate one-word block.

Not fixed: the fix (treating a zero-length interval as overlapping if it falls inside the
other) is small but changes gap-fill behaviour, and it was found while verifying an
unrelated change. Logged rather than folded in.

**Wider than first logged.** The same strict `> 0` also governs the *replacement* branch,
so a zero-duration **A** turn can never be replaced by a clearer B turn either — same root
cause, opposite direction. The gap-fill move to an overlap *ratio* did not change this:
a zero-duration turn has no duration to take a ratio of, so it keeps the historical
any-overlap treatment explicitly.

## OPEN — fusion aborts entirely on a speaker-count mismatch

**Found:** 2026-08-02, review of this branch · **moderate**

`match_speakers` raises when the two sources' diarizations find different numbers of
speakers, and `main` catches it and exits 1 — *after* both sources have been through a
full ASR and diarization pass, which on a real pair is the expensive part of the run. This
is not a corner case: it is the normal outcome without `--num-speakers`, which is why
`README` always passes `--num-speakers 6`.

Not fixed: the raise is correct — `linear_sum_assignment` on a rectangular matrix silently
returns only `min(len(A), len(B))` pairs, which corrupts fusion with no error, and that is
strictly worse. A fallback (match the `min(|A|, |B|)` confident pairs, drop the rest) is
plausible but changes what fusion *means* when the sources disagree about how many people
were in the room, and there is no measurement here to say it does better. Logged with the
workaround rather than guessed at.

## OPEN — `_shift_and_remap` clips a straddling turn's start but keeps its whole text

**Found:** 2026-08-02, review of this branch · **minor**

A B turn straddling A's timeline zero is clipped to `start = 0.0` while keeping its full
text, so its timestamp then claims speech began at 0.0 that actually began before A's
recording did. Turns ending at or before zero are dropped outright, which loses that
speech entirely.

Both are deliberate — there is no valid position on the merged timeline for either — but
the consequence (a timestamp that misdescribes its own text, the same class of defect the
`merge_turns` span fix addressed) was never written down. Only relevant with a negative
offset, i.e. when the secondary started recording first.

## OPEN — `OFFSET_CONFIDENCE_THRESHOLD`'s calibration predates mean-subtraction

**Found:** 2026-08-02, fixing the DC bias · **moderate**

The 1.2 threshold was measured against the real pair on an **un-centred** correlation:
true pair 1.5162, nulls 1.0002–1.0050. `_correlate_envelopes` now mean-subtracts both
envelopes and restricts the lag search to lags with real overlap, which changes the
correlation entirely — so both of those figures describe a computation that no longer
runs, and the separation they justify has not been re-measured.

Not fixed: re-measuring needs the 70-minute pair, which is not committed. The threshold
only ever *warns*, so a mis-set value costs a spurious warning or a missing one rather
than a wrong transcript.

**The change appears to have improved the metric, not just moved it.** On 20 synthetic
speech-like pairs (5-minute primary, 30-second secondary attenuated to 0.35 with its own
noise floor, true offset +60 s), measured before and after with
`tools/measure_dc_bias.py`:

| | offsets recovered | confidence range | would warn |
|---|---|---|---|
| before | 19/20 | 1.005–1.14 | 20/20 |
| after | **20/20** | **1.276–1.652** | **0/20** |

Before the fix the threshold fired on every run in this regime, including the 19 correct
ones — the warning was noise. After it, correct alignments clear 1.2 with margin. The one
pre-fix failure scored 1.005, so the warning did catch it.

This is one synthetic distribution and does **not** re-validate the 1.2 line against real
audio; the true-pair figure of 1.5162 still describes the old computation. It does mean
the threshold is no longer suspected of being too high, which is what the first version of
this entry claimed.

## OPEN — `find_offset` is no longer on the production path

**Found:** 2026-07-31, reviewing this branch · **trivial**

`run_fusion` calls `_correlate_envelopes` (which returns the offset *and* its confidence);
`find_offset` is now a thin wrapper over it, used only by `tools/` and its own test. It is
retained as the documented public "just give me the offset" entry point rather than
deleted, but be aware `test_find_offset_recovers_known_delay` now exercises a wrapper, not
the code `run_fusion` runs. `_correlate_envelopes` has its own tests.

---

## Note: transcript metrics carry run-to-run variance

Whisper's temperature fallback makes ASR non-deterministic. Three runs over the same audio
gave 714, 684/756, and 592 merged blocks. **Any single count above is approximate.**

What survives the variance: the structural facts (same-speaker A turns are never adjacent;
gap-fill cannot desynchronise timestamps; agreement cannot emit two blocks), the offset
null-control separation (1.52 vs 1.000–1.005), and the losslessness of both fixes (zero
word types lost, measured per-block).
