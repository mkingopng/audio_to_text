# VALIDATE — cause of the cross-speaker duplicate pairs

**Date:** 2026-07-31
**Status:** cause **ESTABLISHED**. Not fixed — see [Why no fix](#why-no-fix-yet).
**Supersedes:** the `bugs.md` entry "12 cross-speaker duplicate pairs, cause unmeasured",
which recorded that distinguishing attribution error from genuine cross-talk "requires
listening to the recordings; no automated discriminator has been validated."

**Evidence base:** a fresh capture of the real 70-minute pair (both sources re-run through
ASR + diarization, all intermediates persisted). Reproduce with
`tools/capture_fusion_intermediates.py` then `tools/analyze_cross_speaker.py`. The tagged
merge is asserted equal to production `merge_turns` on this exact input before any number
below is used.

---

## There is a structural discriminator, and it does not require listening

`merge_turns` only ever replaces an A turn with a B turn of the **same** (mapped) speaker,
and only ever gap-fill-appends a B turn when **no** same-speaker A turn overlaps it. So if
the two sources **agreed** on who was speaking, B's turn either replaces A's turn (one
output block) or is dropped as already represented. **Agreement cannot emit two blocks.**

Therefore a cross-speaker duplicate involving a gap-filled B turn *proves* the sources
disagreed, and one between two A turns proves source A's own diarization split one stretch
of speech across two clusters. No listening required.

## Measured: 16 cross-speaker pairs, and 5 of them are not a fusion defect at all

| origin | count | what it proves |
|---|---|---|
| `a_kept` + `a_replaced` | 7 | A's diarization split the speech; B's spanning turn covers both sides |
| `a_kept` + `a_kept` | 5 | **all five are the Whisper hallucination loop** — see below |
| `a_kept` + `b_gapfill` | 4 | the two sources **disagreed** on the speaker |

**The 5 `a_kept + a_kept` pairs are contamination, not duplication.** They sit in a single
span at t≈1079–1092 s where Whisper emitted `"Paul"` 200 consecutive times, shredded across
speakers by diarization jitter:

```
Person 4 [1079.3,1079.3] 'Paul Paul Paul Paul Paul Paul Paul Paul Paul ...'
Person 2 [1079.3,1079.7] 'Paul'
Person 4 [1079.7,1079.7] 'Paul Paul Paul Paul Paul'
Person 2 [1080.6,1091.9] 'Paul Paul Paul Paul Paul Paul Paul Paul Paul ...'
Person 5 [1091.9,1091.9] 'Paul Paul Paul Paul Paul Paul Paul Paul Paul'
```

Degenerate repeated text trivially "duplicates" itself between any two blocks, so the
redundancy scan counts it as cross-speaker duplication. **The honest cross-speaker count on
this run is 11, not 16.** This run also carried `'whether'` ×65; the repetition-loop warning
added in this branch catches both.

## The cross-talk hypothesis is refuted

The rival explanation on record was "genuine cross-talk that both microphones captured" —
i.e. two people really saying near-identical words, correctly attributed.

Measured as the **gap** between the two spans (0 s when they touch or overlap): **13 of 16
pairs sit at gap 0.0 s**, and all 11 non-hallucination pairs do. That is one stretch of
speech rendered twice, not two utterances. Two speakers cannot both produce the same 46–126
characters simultaneously.

The only two pairs with a real gap (11.3 s) were both hallucination artifacts.

### Instrument check — the first version of this measurement was blind

The first pass scored timing as *interval overlap ÷ shorter duration > 50%* and reported 6
of 16 as "different time". Five of those six involved **zero-duration turns**
(`[1080.6, 1080.6]`), where interval overlap is 0 by construction — the instrument could not
observe co-occurrence in exactly the cases it was being asked about. The sixth overlapped by
3.7 s and was only excluded by the 50% threshold. Re-measured as a gap, which is defined on
zero-duration spans, the picture inverts. Recorded because this document's predecessor
failed its gate four times for exactly this class of error.

## Verdict

**Cause: speaker-attribution disagreement, not cross-talk.** Split:

- **4 pairs** — the two sources disagreed about who spoke (structurally proven).
- **7 pairs** — source A's diarization split one stretch of speech across two speaker
  clusters, and B's spanning turn then covers both sides. Fusion does not create the
  misattribution; it makes it visible, and duplicates the content while doing so.
- **5 pairs** — not duplication at all: a Whisper hallucination loop.

## Why no fix yet

Both real mechanisms need something this codebase does not have: **an arbiter that decides
which source's speaker assignment to trust.**

- The 7 `a_replaced + a_kept` cases are the same shape as the same-speaker containment
  redundancy, but cross-speaker. The containment guard deliberately excludes them: a
  cross-speaker near-duplicate means one of the two headings is wrong, the guard cannot tell
  which, and firing there consumes the correctly attributed copy while keeping the
  misattributed one. That restriction is now better evidenced, not weaker.
- The 4 `b_gapfill` cases would need to overrule one source's diarization with the other's.
  The Hungarian embedding match is already the best available signal and it is precisely
  what disagreed here.

Logged rather than guessed at. The scope of the remaining problem is now known — **11 pairs,
~1.9% of 587 blocks** — and its cause is established, which is what the previous entry
lacked.
