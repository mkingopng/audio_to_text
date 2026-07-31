# VALIDATE — triage of the three open items in `bugs.md`

**Date:** 2026-07-31
**Branch:** `feature/mlx-whisper-gpu`
**Status:** revision 4 — revisions 1 and 3 both **FAILED** the contrarian premise gate. Each
failure came from an instrument that could not observe the alternative it claimed to rule out.
See [Corrections](#corrections-from-the-failed-first-attempt) for the full log; the measured
findings below are what survived.
**Scope:** confirm which of `bugs.md`'s three deferred items are real, whether each deferral
rationale still holds, and rank fixes by measured impact — before any spec.

**Evidence base:** the shipped reference output
`output/Meeting with Michael Kingston-20260729_130839-Meeting Recording.md`
(714 blocks, 13,454 words, ~70 min, produced by the fusion path), plus deterministic
unit-level reproduction against `src/fusion.py`. Test suite: 40 passed.

**Reproducibility:** every transcript statistic below comes from
`tools/analyze_transcript.py`, committed alongside this doc:

```bash
uv run python tools/analyze_transcript.py "output/Meeting with Michael Kingston-20260729_130839-Meeting Recording.md"
```

Revision 1 quoted numbers from an ad-hoc script that was never saved, so they could not be
checked. That is fixed.

---

## Summary of verdicts

| # | `bugs.md` item | Verdict | Deferral rationale still valid? |
|---|---|---|---|
| 1 | `merge_turns` residual redundancy | **Real — `bugs.md`'s mechanism CONFIRMED**, incidence understated (16, not 1) | **Partly** — mechanism right; "cosmetic" and "observed once" are both wrong |
| 2 | No confidence check on offset peak | **Real** | **Partly** — the mitigation was a one-off validation step; it does not protect future runs |
| 3 | Whisper hallucination artifacts | **Real — REOPENED**, incidence unquantified (`lars`×183 in run 2) | **No** — the deferral was confirmed by a blind instrument |
| — | *(unlogged)* micro-turn fragmentation | **Real, and causally upstream of #1** | n/a — never recorded |
| — | *(unlogged)* timestamp corruption in `merge_turns` | **Real** — 113 turns render a wrong mm:ss | n/a — never recorded |
| — | *(unlogged)* 12 cross-speaker duplicate pairs | **Real** — cause unmeasured | n/a — never recorded |

Net: **none of the three logged items survives its recorded reasoning intact.** #1's mechanism is
confirmed but its scale and severity are understated; #2's mitigation is out-of-band; #3's
deferral was confirmed by a measurement that could not have detected the problem. Three further
real defects are **not logged at all**.

Revision 1 also quoted numbers from an ad-hoc script that was never saved. Fixed — but the same
failure recurred twice more (corrections #11, #18).

**The load-bearing finding:** items #1 and the unlogged fragmentation problem are **causally
linked**. A one-word micro-turn splits a speaker's turn in two; the redundancy in #1 is what
happens next ([Finding 4](#finding-4--fragmentation-is-the-upstream-cause-of-finding-1)). They
are not, however, a single fix: smoothing resolves 43% of the redundancy, so #1 needs its own fix
in `merge_turns`'s **replacement** branch ([Finding 5](#finding-5-a--branch-attribution-it-is-the-replacement-branch-not-gap-fill)).

## Recommended work, with measured yield

| # | fix | measured yield | cost/risk |
|---|---|---|---|
| 1 | **Fix timestamp corruption** in `merge_turns`'s replacement branch (Finding 7) — carry B's span with B's text | **113** turns render a visibly wrong mm:ss; 196 carry a corrupted `end` | **not risk-free**: re-sorting moves **109 of 684 blocks** and shifts G2's eligible set 41→34, so #3 must be re-measured afterwards |
| 2 | **Same-speaker containment guard, full-containment criterion only**, radius ≥2 in A-index terms (Finding 5) | **6/23 (run 1), 4/19 (run 2)** — lossless | the lossy detector variant reaches 20/23 but destroys 840 chars of real transcript; rejected |
| 3 | **Micro-turn smoothing, Rule G2** (≤2 w, zero duration, not backchannel) | −19–20% blocks; **0** genuine turns destroyed in either run (22 of 24 exclusions duration-driven, 2 word-list-driven) | English word list; **no redundancy benefit claimed** (6 vs 1 across runs = noise); numbers must be re-measured after #1 |
| 4 | **Surface offset + `peak/best_rival`, warn-only** | closes the observability gap; 1.52 vs null 1.000–1.005 | ~10 lines; warn not gate — true-pair distribution is n=1 |
| — | **`bugs.md` #3 (Whisper hallucination) — REOPENED** | `lars`×183 across 34 blocks in run 2, absent in run 1 | **deferral no longer justified as recorded**; incidence unquantified, prior measurement was blind |
| — | **12 cross-speaker duplicate pairs** | not addressed by any fix above | cause **unmeasured** (attribution error vs genuine cross-talk); log to `bugs.md` |

Ordering is causal: #1 perturbs the block order #3 is measured against, so #1 precedes #3. The
revision-4 claim that *#3 may collapse #2's radius-2 pairs to radius 1* is **still unmeasured** and
is not load-bearing for the ordering. #4 is independent of all three.

---

## Finding 1 — `bugs.md` #1 is real and its mechanism is CONFIRMED

### What `bugs.md` claims

> When the secondary source's diarization produces one long turn spanning two of the primary
> source's turns, the winning B turn's text may include content that also appears in the *next*
> A turn … attribution is still correct.

Status recorded as **cosmetic**, incidence as "**observed once** in ~70 minutes".

### Measured

Scanning for a shared substring >40 chars covering >50% of the shorter block, at block
**distances 1–5**:

```
distance=1: SAME-speaker= 0   cross-speaker= 9
distance=2: SAME-speaker=12   cross-speaker= 1
distance=3: SAME-speaker= 3   cross-speaker= 0
distance=4: SAME-speaker= 0   cross-speaker= 0
distance=5: SAME-speaker= 1   cross-speaker= 0
--------------------------------------------------
TOTAL:      SAME-speaker=16   cross-speaker=10
```

**`bugs.md`'s mechanism is confirmed.** The same-speaker duplications are real, they are the
majority (16 vs 10), and they are *large* — shared spans of 278, 309, 325 chars, i.e. whole
paragraphs restated verbatim under a second heading.

Two of `bugs.md`'s three claims about it are nonetheless wrong:

- **"Observed once"** → 16 same-speaker instances (26 total including cross-speaker).
- **"Cosmetic"** → true for the same-speaker cases, but the 10 cross-speaker cases put a
  paragraph under a *person who did not say it*. That is a correctness defect, and it is not
  recorded anywhere.

### Why distance matters — and why revision 1 got this exactly backwards

`_group_consecutive` (`src/transcribe.py:209-218`) flushes a turn on **every** speaker change.
Two same-speaker turns are therefore **always** separated by at least one intervening turn, so a
same-speaker duplication can never appear as an *adjacent* pair. Revision 1 scanned adjacent
pairs only, found `SAME=0`, and concluded `bugs.md` was misdiagnosed. **The scan was structurally
incapable of observing the thing it claimed to rule out.** Credit to the contrarian gate for
catching it.

---

## Finding 2 — `bugs.md` #2 is real; the mitigation is out-of-band, not absent

`bugs.md` marks the missing correlation-peak confidence check as *"open, mitigated by process"*:

> the workflow has a human sanity-check the offset against the recordings before trusting
> fusion output.

**That mitigation exists, but only out-of-band.** The plan doc's Task 12 Step 1
(`docs/superpowers/plans/2026-07-29-speaker-diarization-fusion.md:1421-1439`) is a documented,
executed `print('offset (seconds):', find_offset(...))` plus a by-ear check. A human did verify
the offset — **once**, on this recording pair, via a throwaway script.

The accurate, narrower finding: **the check is a one-off procedure, not a property of the tool.**
`src/fusion.py` contains **zero** `print` statements (`grep -c 'print(' src/fusion.py` → 0);
`run_fusion` computes `offset = find_offset(wav_a, wav_b)` (`src/fusion.py:203`) and never
surfaces it. Every *future* `--fuse` run — on new pairs, where the single-constant-offset
assumption is untested — gets no offset visibility and no peak-quality signal.

### Calibration — measured, not assumed

Revision 1 asserted "~1.0 expected for unrelated audio" without measuring it. Measured now, with
five negative controls built from the same two recordings (so any separation is attributable to
alignment, not to recording character):

| pair | lag | peak/median | z | **peak/best_rival** |
|---|---|---|---|---|
| **A vs B (true pair)** | 26.1 s | 3.135 | 3.69 | **1.5162** |
| A first half vs B second half | −16.8 s | 2.094 | 1.80 | 1.0034 |
| A second half vs B first half | −3.9 s | 2.062 | 1.79 | 1.0026 |
| A vs shuffled B | 0.3 s | 2.019 | 1.75 | 1.0032 |
| A vs reversed B | 50.9 s | 2.057 | 1.79 | 1.0049 |
| A vs gaussian noise | 4.5 s | 2.006 | 1.74 | 1.0002 |

Conclusions:

- **`peak/best_rival` is the right metric.** True 1.5162 against a null cluster at
  1.0002–1.0049 — clean separation.
- **The other two metrics are weak and should not be used.** `peak/median` 3.135 vs nulls up to
  2.094, and z 3.69 vs nulls up to 1.80, separate by less than a factor of two. Revision 1
  reported all three without flagging this.
- **Still only one positive sample.** The null floor is well characterised; the *true-pair*
  distribution is not. So the fix must **report the metric and warn** below a conservative
  threshold — not gate on it.

---

## Finding 3 — REOPENED: the deferral was confirmed by a blind instrument

Revisions 1–4 said:

> 6+ identical consecutive words: 0 blocks; 4+ identical: 1 block (0.14%). Real, genuinely
> upstream, negligible incidence. **The deferral holds — no action.**

**That measurement was structurally incapable of finding the thing it was measuring.** The scan
counted identical consecutive words *within a block*. Finding 4 establishes that **31% of blocks
hold a single word** — so a repetition loop is shredded across dozens of blocks and no single
block ever contains a repeat. The instrument could only see loops that fragmentation had not
already scattered.

Re-measured doc-wide (token stream flattened across block boundaries;
`tools/analyze_transcript.py` now does both passes):

```
run 1: within-block 4+ repeats = 3    doc-wide: nothing notable
run 2: within-block 4+ repeats = 13   doc-wide: 'lars' x183 across 34 blocks
                                                (indices 121-154, 17:45-18:01)
```

**Run 2 contains a 183-token Whisper hallucination loop** — the word "Lars", which appears
nowhere in run 1, repeated 183 times across 34 consecutive blocks of the same recording. One run
in two. That is not "1 block in 714, negligible".

**Verdict changed: `bugs.md` #3 is NOT safely deferrable as recorded.** The incidence is
unknown (n=2, one occurrence), it is bursty rather than diffuse, and it is invisible to the
detection the earlier revisions relied on. It remains upstream — but "we can ignore it" was
justified by a blind measurement, and the honest status is *unquantified*, not *negligible*.

**This is the fourth instance of the same error in this document** — a falsification performed
with an instrument that could not observe the alternative — and it landed on the one item
declared safe to ignore. See [Corrections](#corrections-from-the-failed-first-attempt).

---

## Finding 4 — fragmentation is the upstream cause of Finding 1

### The scale of it

| | count | % of blocks | % of total text | sandwiched by same speaker |
|---|---|---|---|---|
| ≤1 word | 220 | **31%** | 1.6% | 73 (33%) |
| ≤2 words | 262 | 37% | 2.3% | 89 (34%) |
| ≤3 words | 291 | 41% | 2.9% | 104 (36%) |
| ≤5 words | 347 | **49%** | 4.8% | 126 (36%) |

Median block: **31 characters**. Nearly half the document's headings introduce ≤5 words.

### The causal link to Finding 1 — measured

For each of the 12 same-speaker redundant pairs at distance 2, what sits between them?

```
intervening block word counts: [1, 1, 1, 1, 1, 1, 1, 1, 14, 26, 27, 41]
median = 1     one-word intervening block: 8 of 12
```

Concretely — idx 84, `Person 1` … `'to'` (Person 4, **one word**) … `Person 1`, sharing **278
characters**; idx 175, `Person 2` … `'for'` (Person 1, one word) … `Person 2`, sharing **325**.

**So the mechanism is a chain:** a one-word micro-turn splits one A turn into two same-speaker A
turns → a B turn spanning both gets applied to one of them → the shared text is emitted twice.
**Eight of the twelve same-speaker redundancies exist only because a micro-turn split the turn.**
The remaining four (14–41-word intervening blocks) are the genuine standalone case `bugs.md`
describes.

This reframes the work: **fixing fragmentation should eliminate most of Finding 1 as a
side-effect.** They are not independent line items to be ranked against each other.

### Where fragmentation enters — NOT solely `_group_consecutive`

Running the **single-file** path on a 4-minute slice (`27:00–31:00` of the phone recording,
`--num-speakers 6`):

```
words=937   diarization_turns=96   grouped_blocks=31
diarization turn dur: p10=0.02s  p50=0.62s | turns <0.5s: 42/96
grouped blocks <=1 word: 5/31 (16%)   <=5 words: 12/31 (39%)
```

1. **The diarization signal is itself fragmented** — pyannote emits **42 of 96 turns shorter than
   0.5 s**, p10 = **0.02 s**. `align_words_to_speakers` assigns words to these micro-turns, so
   the flicker is *imported* before grouping runs. `_group_consecutive` faithfully groups what it
   is given; a floor there absorbs the symptom into a neighbour rather than correcting the
   attribution.
2. **Fragmentation is not a fusion artifact, but fusion roughly doubles it** — single-file 16%
   one-word vs fused 31%.
3. **A third source exists that grouping cannot reach.** The output contains **76 adjacent
   same-speaker block pairs**, which `_group_consecutive` can never emit (it flushes on speaker
   change). These come from `merge_turns` appending gap-fill B turns after grouping and never
   re-grouping (`src/fusion.py:140-153`). No smoothing inside `_group_consecutive` touches them.

### Why a naive length threshold is destructive

Of the 262 blocks of ≤2 words, **70 (27%) are lexical backchannel** — genuine agreement turns
that carry meaning in a meeting:

```
backchannel: yeah×34, yep×8, -hmm×8, yes×5, great×4, mm-hmm×3, no×2, okay×2
jitter:      you×15, so×13, it×8, i×7, the×5, and×5, a×4, to×3
```

Both populations appear as short blocks sandwiched between same-speaker neighbours. **A length
threshold cannot separate them**, so a naive floor would misattribute roughly a quarter of what
it touches — trading a readability win for new correctness bugs. Any smoothing rule needs a
real discriminator (e.g. duration/energy, or the sub-0.5 s diarization-turn signal), and must be
simulated against the real blocks before adoption.

---

## Finding 5 (A) — branch attribution: it is the REPLACEMENT branch, not gap-fill

Revision 1 blamed `merge_turns`'s **gap-fill** branch. The contrarian showed the
**confidence-replacement** branch produces an identical signature and that no output-side
evidence separated them. Resolved by re-running the real merge with each emitted turn tagged by
its originating branch.

**Method.** ASR + diarization were re-run on both sources and all intermediates persisted
(`capture.py`), then merged by a tagged re-implementation of `merge_turns`. That
re-implementation was **property-tested against production `merge_turns` over 3,000 randomized
inputs — 0 mismatches** — so the tag is the only difference.

Tags: `a_kept` (A's turn unchanged), `a_replaced` (A's boundaries, B's text won on confidence,
`fusion.py:132-137`), `b_gapfill` (B turn appended wholesale, `fusion.py:140-153`).

```
origin mix: a_kept=296  a_replaced=206  b_gapfill=182

== redundant pairs by originating branch ==
  SAME  a_replaced + a_kept      16
  SAME  a_kept + a_replaced       7
  cross a_kept + a_replaced       5
  cross b_gapfill + a_kept        4
  cross a_replaced + a_kept       3
                          TOTAL  35
```

**31 of 35 redundant pairs involve `a_replaced`. Only 4 involve `b_gapfill`.**

The contrarian's challenge was correct and consequential: **revision 1's prescribed fix — a text
similarity guard on the gap-fill branch — would have addressed 4 of 35 cases (11%)** and left the
dominant mechanism untouched.

### Is that just base rate? No — tested

`a_replaced` covers 206 of 684 turns, so it would appear in many pairs by chance.
Label-permutation test (2,000 shuffles of the origin tags, pair set held fixed —
`scratchpad/base_rate_test.py`):

```
a_replaced  observed=31  null_mean=17.9  sd=2.99  z=+4.40  p~0.0000
b_gapfill   observed= 4  null_mean=16.3  sd=2.93  z=-4.19  p~0.0000
a_kept      observed=35  null_mean=23.7  sd=2.81  z=+4.02  p~0.0000
```

The enrichment is real, and **`b_gapfill` is actively *depleted*** — it participates in fewer
pairs than chance would give it. Note also that `a_kept` appears in **35 of 35** pairs: every
duplication has an untouched A turn as its other half, which is precisely the containment
signature.

### What the guard can actually achieve — measured, not asserted

"31 of 35 (89%)" is the share of pairs *involving* `a_replaced`, **not** the share a containment
guard can act on. The guard fires only where the `a_replaced` turn is the **container**, and
"neighbouring" must be defined over **A-turn indices**. Measured on **both runs**
(`scratchpad/guard_full.py` — see correction #18; the earlier citation pointed at a script that
did not produce these numbers):

```
A-index gap between same-speaker A turns:  distance 2: 250   distance 3: 37   distance 4: 194
                                           distance 1: NONE

                                    SAME  cross  lossy fires  chars destroyed  worst
run1 radius=1  full containment        0      1            0                0      -
run1 radius=1  detector criterion      0      8            0                0      -
run1 radius=2  full containment        6      1            0                0      -
run1 radius=2  detector criterion     20      8           14              840    120
run2 radius=2  full containment        4      1            0                0      -
run2 radius=2  detector criterion     14      8           10              394     86
```

**The detector criterion is lossy, and revision 4 sold its yield without that cost.** It fires
whenever >40 chars and >50% of the neighbour is duplicated — so it consumes blocks that are only
*partly* duplicates, deleting the remainder. On run 1, **14 of the 20 fires destroy real
transcript text: 840 characters, worst single case 120**. Only the 6 full-containment fires are
lossless.

**Corrected yield, stated both ways:**

| variant | run 1 | run 2 | text destroyed |
|---|---|---|---|
| **Full containment (lossless)** | **6/23 (26%)** | 4/19 (21%) | **0** |
| Detector criterion (lossy) | 20/23 (87%) | 14/19 (74%) | 840 / 394 chars |

**A radius-1 guard fires on zero same-speaker cases.** `_group_consecutive` flushes on speaker
change, so two same-speaker A turns are *never* adjacent — the same structural fact this document
uses to demolish revision 1, which revision 3's own prescription then ignored. The guard needs
**radius ≥ 2 in A-index terms**, and must decide what to do with the intervening other-speaker
turn (which is usually the micro-turn from Finding 4 — so ordering the smoothing fix first may
collapse these pairs to radius 1 and simplify the guard).

**The guard must be same-speaker-only.** At radius 2 it would otherwise fire on 8 cross-speaker
cases, and in each it consumes a *correctly attributed* block — deleting the right copy and
keeping the wrong one, worsening the very correctness defect Finding 1 identifies.

**Recommendation: full containment only.** 6/23 with zero text loss beats 20/23 that shreds 840
characters of transcript — this is a transcription tool, and silently deleting speech to tidy up
duplication is the wrong trade. That is a real reduction in the fix's value versus revision 4's
claim, and it is the honest number.

**The 12 cross-speaker pairs are NOT addressed by this guard.** Revision 4 called them "a
speaker-attribution problem"; that was **asserted, not measured** — they could equally be genuine
cross-talk both microphones captured. Unresolved, and logged to `bugs.md` as such.

### The mechanism, now precisely characterised

`a_replaced + a_kept` is exactly what `bugs.md` described. A B turn spanning two A turns is
written into **one** of them (the `used_b_ids` set, added 2026-07-31, correctly stops it
replacing both). But B's spanning text *contains* what the sibling A turn says, and that sibling
is still emitted, unchanged, as `a_kept`. The earlier fix eliminated *exact* duplication and left
*containment* duplication behind — which is precisely the residue `bugs.md` recorded.

**The fix belongs in the replacement branch:** when B's text replaces an A turn, any neighbouring
A turn whose text is contained in that B turn must be consumed rather than emitted separately.

---

## Finding 6 (B) — smoothing simulation: duration does not discriminate; only the lexical guard does

Six candidate rules simulated against the real merged turns. A rule absorbs only *sandwiched*
blocks (same speaker both sides), so it can never move speech across a speaker boundary.

### The scoring had to be rebuilt: revision 3's was circular

Revision 3 scored each rule by counting how many absorbed blocks were in the `BACKCHANNEL` word
list — **the same list three of the rules use as their predicate**. Rules B, D and E therefore
scored exactly zero *by construction*, and the column carried no information for the three rules
it was used to choose among.

Replaced with a *more* independent discriminator: all **91** candidate blocks (the union of every
rule's eligible set) were dumped with surrounding context and **hand-labelled** GENUINE vs
JITTER — 32 genuine, 59 jitter (`scratchpad/labels.py`, `dump_candidates.py`).

**The independence claim must be qualified — it was too strong.** Measured:

```
GENUINE candidates <=2 words:            24
   ... of which in the BACKCHANNEL list: 16
JITTER candidates in the BACKCHANNEL list: 0 / 59
```

**No backchannel-list token was ever labelled JITTER.** The labelling was done by the same author
who wrote the list, after seeing which rules were under test, so for short blocks the labels
partially re-derive the predicate rather than testing it independently. That is one-directional
confirmation bias and it inflates confidence in any rule using the list.

**How much does it contaminate G2's "0 destroyed"?** Measured:

```
GENUINE <=2w blocks G2 would absorb but for the WORD LIST:  2   ('Yeah,', 'no.')
GENUINE <=2w blocks excluded by the DURATION clause alone:  22
```

So **22 of G2's 24 exclusions are duration-driven and only 2 are word-list-driven** — G2's result
is ~92% independent of the contaminated criterion. It survives, but "independent of every rule
predicate, which is what makes the comparison valid" was false as written, and the ~8% that is
circular is now on the record rather than implied.

*Also caveat: these are my judgements from reading context, not verified by ear. Not ground truth.*

| rule | blocks | absorbed | **GENUINE destroyed** | worst | redundancy 23 → |
|---|---|---|---|---|---|
| A — sandwiched ≤1 word | −168 | 59 | 17 (29%) | 1 w | 10 |
| A2 — sandwiched ≤2 words | −190 | 70 | 21 (30%) | 2 w | 8 |
| B — ≤2 w, not backchannel | −166 | 58 | **8 (14%)** | 2 w | 13 |
| C — duration < 0.5 s | −188 | 69 | 24 (35%) | **93 w** | 9 |
| D — < 0.5 s, not backchannel | −162 | 56 | 10 (18%) | **93 w** | 14 |
| E — ≤2 w, < 0.5 s, not bc | −148 | 49 | 4 (8%) | 2 w | 14 |
| F — duration == 0 | −140 | 45 | 2 (4%) | 1 w | 17 |
| G — ≤2 w AND duration == 0 | −136 | 43 | 2 (5%) | 1 w | 17 |
| H — duration < 0.05 s | −142 | 46 | 3 (7%) | 1 w | 16 |
| **G2 — ≤2 w, duration == 0, not bc** | **−132** | 41 | **0 (0%)** | — | 17 |

Three results, two of which reverse revision 3:

1. **Rule B's "0 destroyed" was false.** Against hand labels it destroys **8 genuine turns
   (14%)** — the same order as the naive rules it was chosen over. Revision 3's recommendation is
   **withdrawn**.
2. **Duration DOES discriminate — but at zero, not 0.5 s.** Revision 3 tested one threshold and
   concluded "duration separates nothing." Wrong: exact-zero duration is a strong jitter signal
   (11% of backchannel vs 58% of jitter), and the zero-duration family (F/G/H) destroys 4–7%
   against Rule B's 14%. Correction #8 in revision 3 is **retracted**.
3. **The `< 0.5 s` rules are dangerous in a way a count hides.** Rules C and D each absorb a
   **93-word** genuine turn. See Finding 7 — duration is not merely a weak signal on merged
   turns, it is *corrupt*.

**Recommended: Rule G2** — ≤2 words AND exact-zero duration AND not a backchannel token.
~19–20% of blocks removed, **zero genuine turns destroyed in both runs**, and the most stable
block reduction of any rule tested (12% run-to-run swing vs 30–37%). It is the most conservative
rule tried and the only one with no measured correctness cost.

**Stated limitations.** G2 carries the fixed English word list, so it will not generalise across
languages. And per the variance section, **no redundancy benefit should be claimed for it** —
that measured 6 pairs in run 1 and 1 in run 2.

### Correcting the causal claim — a third time

Revision 2 said fixing fragmentation would "eliminate most of Finding 1." Revision 3 revised that
to 43%. With the safe rule measured over two runs it is **6 and then 1 pair — indistinguishable
from noise**. The honest position: **smoothing and the redundancy fix are separate pieces of
work.** They share a root cause (Finding 4's micro-turns), and removing micro-turns may still
simplify the guard by collapsing radius-2 pairs to radius 1 — but smoothing cannot be sold as a
fix for the redundancy.

---

## Finding 7 (new) — `merge_turns` corrupts turn duration when B's text wins

Discovered while hand-labelling. Blocks with impossible duration/text ratios:

```
block 110:  58 words, duration 0.30s
block 526:  93 words, duration 0.40s
block  59:  22 words, duration 0.14s
```

Ninety-three words cannot be spoken in 0.4 seconds. Cause: in the replacement branch
(`src/fusion.py:132-137`), the merged turn keeps **A's** `start`/`end` while taking **B's** text:

```python
merged.append({**turn, "text": best_b["text"], "confidence": best_b["confidence"]})
```

When B's turn spans materially more speech than A's, the timestamps no longer describe the text.
Measured over the 206 `a_replaced` turns:

```
start differs at all:              203 / 206   (median 0.48s, p90 7.5s, max 42.6s)
start differs at RENDERED mm:ss:   113 / 206   <-- the user-visible count
end   differs at all:              196 / 206   (median 0.49s, p90 13.5s, max 84.8s)
```

- **113 turns render a visibly wrong timestamp** — not 206. Only `start` is rendered
  (`src/transcribe.py:256`), at mm:ss granularity, so sub-second drift is invisible. Revision 4's
  "~206" conflated the corrupted-`end` count with the user-visible defect.
- **The corrupted `end` (196 turns) is still real** — it makes any duration-based logic unsafe on
  merged turns, which is why Rules C and D absorb a 93-word turn while looking conservative.

### Correcting revision 4's prerequisite rationale — it was mechanically wrong

Revision 4 argued Finding 7 must land first because G2's `duration == 0` predicate rests on the
corrupted field. **False.** Zero-duration blocks by origin:

```
b_gapfill=60   a_kept=46   a_replaced=0
```

The corruption touches only `a_replaced`, which contributes **zero** zero-duration blocks — so
fixing it cannot change which blocks satisfy `dur == 0`.

The real coupling is different and was missed: fixing Finding 7 changes `start`, and `merge_turns`
**re-sorts by `start`**. Simulated, that moves **109 of 684 blocks** and shrinks G2's eligible
sandwiched set from **41 to 34 (−17%)** — because "sandwiched" depends on block order. So the
ordering conclusion survives, for a completely different reason: **every G2 number in this
document was measured on the pre-fix pipeline, and must be re-measured after Finding 7 lands.**

Also unexplained: source B's grouped turns include a **21-word turn with `dur == 0`** — Whisper
emitted identical timestamps for 21 consecutive words. `dur == 0` is therefore an ASR timestamp
artifact whose mechanism is not established. Calling it "a strong jitter signal" is a measured
*correlation* with no known cause; that is enough to build a conservative rule on, but it should
be logged, not assumed stable.

---

## Run-to-run variance — and what it does and does not undermine

The captured re-run is **not bit-identical** to the shipped output: 684 merged blocks vs 714, and
23 same-speaker / 12 cross-speaker redundant pairs vs 16 / 10. The offset reproduced exactly
(26.1 s) and the speaker map is a clean 1:1, so the divergence is ASR temperature-fallback
variance, not a pipeline difference.

Revision 3 waved this away as "every effect above is far larger than the variance." **That was
false for the smoothing payoff:** same-speaker redundant pairs swing 16 → 23 (**+44%**) between
runs, which is larger than the 6–10 pairs any smoothing rule claims to eliminate. A second ASR
run (reusing run 1's diarization, which is deterministic for a fixed WAV) was captured to put
n = 2 behind the affected numbers — results in the table below.

Run 2 reused run 1's diarization and re-ran only ASR. Word counts barely moved (12,407 → 12,543
and 12,602 → 12,669, ~+1%) but **block counts moved far more** (684 → 756, +11%): small ASR
differences amplify into turn boundaries.

A second methodological fault surfaced here, in my own scoring. `rescore.py` keyed the hand
labels to run 1's **block indices**, so they did not transfer to run 2 and its
"genuine destroyed" column was uninterpretable. Re-keyed by normalized text
(`scratchpad/rescore2.py`), with texts absent from run 1 counted as *unlabelled* rather than
silently scored safe:

```
                        run1                    run2                  swing
                 removed  genuine  redund   removed genuine redund  (removed)
A   <=1 word         168      17      10        226     19     11      35%
B   <=2w, not bc     166       8      13        216      5     14      30%
C   dur < 0.5s       188      24       9        244     19      8      30%
F   dur == 0         140       2      17        192      4     13      37%
G2  <=2w,dur0,not bc 132       0      17        148      0     18      12%

baseline blocks           684  ->  756   (+11%)
baseline redundant pairs   23  ->   19   (-17%)
```

### Run 2 is a contaminated replicate

Run 2's ASR contains the `lars`×183 hallucination loop from Finding 3 — 34 consecutive blocks
absent from run 1. That single artifact is **34 of the +72 block difference**, so the raw swing
column overstates true variance. Re-measured with the span excised (verified independently):

```
rule                run1   run2  swing |  run2*  swing*
baseline blocks      684    756   11%  |   722     6%
A  <=1w              168    226   35%  |   198    18%
B  <=2w notbc        166    216   30%  |   186    12%
C  dur<0.5           188    244   30%  |   210    12%
F  dur==0            140    192   37%  |   158    13%
G2 <=2w,d0,notbc     132    148   12%  |   140     6%
```

**Three conclusions:**

1. **G2 still holds, but by a smaller margin than revision 4 claimed.** Zero genuine turns
   destroyed in *both* runs (of the labelled subset — 20 of run 2's absorptions are unlabelled;
   spot-checked as 14 clear fragments, 4 hallucination tokens, 2 arguable). It remains the most
   stable rule, but at **6% swing versus 12–18%** — a factor of ~2, not the ~3 revision 4 reported
   off contaminated numbers.
2. **The smoothing redundancy payoff must be withdrawn as a claimed benefit.** G2 eliminates
   **6 pairs in run 1 and 1 in run 2**. That is noise. Revision 3 built its ranking partly on this
   number; it does not survive n = 2.
3. **Baseline metrics carry ±6–17% run-to-run** (11% before excising the hallucination, 6% after),
   so no block or pair count in this document is a constant.

**What variance does not undermine:** the branch attribution (z = +4.4 against a permutation
null), the null-control separation for `peak/best_rival` (1.52 vs 1.000–1.005), the fragmentation
shares (31% / 49%), and Finding 7's duration corruption — all far outside the observed swing.

---

## Corrections from the failed first attempt

Revision 1 failed the contrarian premise gate. Recorded because the failure mode is instructive,
not to be self-flagellating:

1. **Inverted Finding 1.** Claimed `bugs.md` was misdiagnosed and its prescribed fix wrong.
   Wrong: the mechanism is confirmed, and same-speaker cases *outnumber* cross-speaker ones. The
   cause was a detection heuristic (adjacent pairs only) structurally blind to the mechanism
   being tested. **Lesson: a falsification is worthless if the instrument cannot observe the
   thing being falsified.**
2. **Confused possibility with attribution.** A deterministic repro of the gap-fill branch was
   reported as "high confidence" for the real-world cause, without excluding the
   confidence-replacement branch, which produces the same signature.
3. **Mis-stated Finding 2** as "a control that was never implemented" (self-corrected before the
   verdict arrived, on finding the print in plan Task 12).
4. **Asserted the null baseline** ("~1.0 expected") instead of measuring it. Now measured.
5. **Ranked incomparable units** — "49% of blocks" vs "7 pairs" vs "0 prints" — with no per-fix
   yield estimate. Superseded: Finding 4 shows the top two items are one causal chain, so the
   ranking question was partly malformed.
6. **Unpublished analysis script**, making the numbers uncheckable. Now `tools/analyze_transcript.py`.

### Added in revision 3 — what the required evidence overturned

7. **Revision 1's fix prescription was wrong, as the contrarian suspected.** Branch attribution
   shows 31 of 35 redundant pairs come from the **replacement** branch and only 4 from gap-fill.
   The prescribed gap-fill guard would have fixed 11% of the problem. **Lesson: "I reproduced it"
   establishes that a mechanism *can* produce the signature, never that it *did*.**
8. **Revision 2's own proposal was refuted by its own simulation.** Revision 2 proposed using the
   sub-0.5 s diarization signal to separate jitter from genuine backchannel. Measured, both
   populations sit under 0.5 s (medians 0.00 s vs 0.22 s) and duration-based Rule C destroys as
   much real speech as the naive rule. Only the lexical guard works — and it is admittedly crude.
9. **Revision 2 overstated the causal link.** "Fixing fragmentation eliminates most of Finding 1"
   → measured at 43% for the non-destructive rule.

### Added in revision 4 — the second contrarian FAIL

Revision 3 also failed. Six findings, all upheld; two were repeats of lessons this very document
claims to have learned:

10. **Repeated its own structural insight against itself.** Revision 2 demolished revision 1 using
    the fact that same-speaker turns are never adjacent — then revision 3 prescribed a guard on
    "neighbouring" A turns, which at radius 1 fires on **zero** same-speaker cases. Corrected to
    radius ≥2 in A-index terms.
11. **Repeated the unpublished-script failure.** Cited a "3,000-trial property test" while
    correction #6 in the same document lists unpublished evidence as a lesson. Now published at
    `scratchpad/test_tagged_merge_equivalence.py` and passing.
12. **Overstated guard yield by conflating "involves" with "can act on."** 31/35 (89%) → **20/35
    (57%)**, or 20/23 (87%) of the same-speaker pairs it actually targets.
13. **Missed that the guard, as specified, worsens the defect it fixes.** At radius 2 it fires on
    8 cross-speaker pairs, consuming the *correctly attributed* copy each time. Now restricted to
    same-speaker only.
14. **Circular scoring.** Rule B's "0 backchannel destroyed" used the same word list that defines
    the rule — zero by construction. Against independent hand labels it destroys **14%**.
    Recommendation withdrawn.
15. **Refuted its own "duration separates nothing."** Only one threshold (0.5 s) was tested; the
    signal is at exact zero. The untested zero-duration family is strictly better, and led to
    **Rule G2** and to **Finding 7** (duration corruption), neither of which existed before.
16. **Dismissed run-to-run variance too quickly.** Rule B's claimed 10-pair payoff is smaller than
    the 16→23 swing in the metric itself. A second ASR run confirmed the objection: the smoothing
    redundancy payoff is 6 pairs in run 1 and 1 in run 2, and has been **withdrawn** as a claimed
    benefit. Block reduction survived (19–20%, the most stable metric of any rule).
17. **A third instrument fault, found by running the check.** The hand labels were keyed to run
    1's block *indices*, so they silently failed to transfer to run 2 — every unmatched
    absorption would have scored as "safe." Re-keyed by text, with unmatched blocks reported as
    unlabelled rather than assumed harmless.

### Added in revision 5 — the third contrarian FAIL

18. **Cited a script that does not produce the cited numbers — the third repeat of the
    unpublished-evidence failure**, in a document that lists it as corrections #6 and #11.
    `guard_yield.py` implements full containment and emits 6/1; the headline "20/8" came from an
    unsaved heredoc. Now published as `scratchpad/guard_full.py`, which emits both criteria on
    both runs.
19. **Sold the guard's lossy yield as if it were free — a repeat of correction #12's error.**
    14 of the 20 detector-criterion fires delete text that is *not* duplicated: **840 characters**,
    worst case 120. The lossless yield is **6/23 (26%)**, not 87%. Recommendation switched to
    full-containment-only.
20. **Overstated label independence.** No backchannel-list token was ever labelled JITTER (0/59) —
    one-directional confirmation bias by the same author who wrote the list. G2 survives because
    22 of its 24 exclusions are duration-driven, but the blanket independence claim is retracted.
21. **Built a stability claim on a contaminated replicate.** Run 2's `lars`×183 hallucination is
    34 of the +72 block delta; excising it moves G2's advantage from ~3× to ~2×.
22. **THE FOURTH INSTRUMENT-BLINDNESS ERROR — and on the one item declared safe.** Finding 3's
    "no action, deferral confirmed" rested on a within-block repeat scan, while Finding 4 of the
    same document establishes that 31% of blocks hold one word. A loop is shredded across blocks
    and becomes invisible. Doc-wide detection finds a 183-token hallucination in run 2.
    `bugs.md` #3 is **reopened**.
23. **Finding 7 inflated and mis-rationalised.** 113 user-visible wrong timestamps, not ~206; and
    the stated prerequisite ("G2's `dur==0` rests on the corrupted field") is mechanically false —
    `a_replaced` contributes zero zero-duration blocks. The real coupling is re-sorting (109
    blocks move). "No behavioural risk identified" retracted.

**Four gate failures, and the single recurring cause is unchanged:** confident claims from
instruments that could not observe the alternative. Each fix has been to check the instrument
first — which is now a standing requirement for any measurement in the follow-up work.

**The pattern across all three failures is one thing:** confident causal claims from instruments
that could not have detected the alternative. Every reversal came from checking the instrument,
not from new data.
