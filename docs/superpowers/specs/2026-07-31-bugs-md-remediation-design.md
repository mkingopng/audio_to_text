# Design — remediating the open items in `bugs.md`

> **Reconciled 2026-08-01 against what shipped.** Three things in this document were
> superseded by measurement during implementation; the code and `bugs.md` are authoritative
> where they differ.
>
> 1. **The containment floor is 20 normalized characters, not 40.** This document copied 40
>    from the redundancy detector's criterion. Measured on the real pair, the guard's fires
>    are bimodal — micro-turns at ≤8 chars, genuinely restated content at ≥20 — and 20 sits
>    in the empty gap between them. A threshold derived from the fire distribution beats one
>    borrowed from a different instrument.
> 2. **The cross-speaker duplicates were NOT out of scope.** This document claimed
>    distinguishing attribution error from genuine cross-talk "requires listening to the
>    recordings". False: `merge_turns` cannot emit two blocks when the sources agree, so a
>    gap-filled B turn in a cross-speaker pair *proves* they disagreed. The cause is now
>    established without audio — see `2026-07-31-cross-speaker-cause.md`.
> 3. **The guard also needed a time bound.** Index adjacency says nothing about elapsed time,
>    so a radius-2 sibling can sit twenty minutes away; the first implementation deleted 60
>    seconds of speech at t=1200. The sibling must be speech B's turn actually covers.
>
> The smoothing rule shipped unconditional and was returned to opt-in (`eb0e67d`), as
> specified below.

**Date:** 2026-07-31
**Branch:** `feature/global-tool`
**Validation basis:** `docs/validate/2026-07-31-bugs-md-triage.md` (revision 5, passed the
contrarian premise gate after four failures). Every defect below was measured there; this
document decides what to do about each, and nothing here re-opens a diagnosis.

---

## Goal

Close the seven items recorded in `bugs.md` — fixing the six that have a validated fix or a
decidable behaviour, and leaving the one that cannot honestly be fixed marked as such.

## Scope

| `bugs.md` item | Disposition |
|---|---|
| `merge_turns` corrupts turn timestamps | **Fix** — carry B's span with B's text |
| Residual containment redundancy | **Fix** — same-speaker full-containment guard, radius 2 |
| No confidence check on the offset peak | **Fix** — surface offset + `peak/best_rival`, warn-only |
| Fused run overwrites a single-file transcript | **Fix** — fused output becomes `<primary>.fused.md` |
| ~31% single-word blocks (fragmentation) | **Fix, opt-in** — Rule G2 behind `--smooth` |
| Whisper hallucination loops | **Detect and warn** — upstream, not repairable here |
| 12 cross-speaker duplicate pairs | **Out of scope** — cause unmeasured |

## Ordering, and why it is not negotiable

The timestamp fix changes `start` on 203 of 206 replaced turns, and `merge_turns` re-sorts by
`start`. Simulated, that moves **109 of 684 blocks** and shrinks the smoothing rule's eligible
sandwiched set from **41 to 34**. Every fragmentation and redundancy number in the triage was
measured on the pre-fix pipeline. So:

```
1. timestamp fix  ->  2. containment guard  ->  3. offset confidence  ->  4. output naming
                  ->  [re-run the reference pair]  ->  5. smoothing  ->  6. loop warning
```

Items 3, 4 and 6 are independent of the others; they are sequenced late only to keep the
re-measurement run clean of unrelated change.

---

## 1. `merge_turns` carries B's span with B's text

**Defect.** In the confidence-replacement branch (`src/audio_to_text/fusion.py:135`) the merged
turn keeps A's `start`/`end` while taking B's text. When B's turn spans more speech than A's, the
timestamps stop describing the text: the reference output holds a 93-word block with a 0.4 s
duration, and **113 of 206 replaced turns render a visibly wrong `mm:ss` heading**.

**Fix.**

```python
merged.append({
    **turn,
    "start": best_b["start"],
    "end": best_b["end"],
    "text": best_b["text"],
    "confidence": best_b["confidence"],
})
```

The speaker is provably unchanged — the candidate filter already requires
`b["speaker"] == turn["speaker"]` — so only the span moves.

**Contract change.** `merge_turns`'s docstring claims "Source A's turns define the canonical
paragraph boundaries." That stops being true and the docstring is corrected: boundaries come from
whichever source's text won, which is the only arrangement in which a timestamp describes the
text it labels.

**Risk, accepted.** Merged output reorders. `merge_turns` already ends with
`merged.sort(key=lambda t: t["start"])`, so ordering stays correct; what changes is *which* blocks
are adjacent, which is why re-measurement precedes the smoothing work.

**Tests.**

- A replaced turn carries B's `start`/`end`, not A's.
- A B turn starting earlier than the A turn it replaces sorts ahead of that A turn's predecessor
  in the merged output.

---

## 2. Same-speaker containment guard

**Defect.** A B turn spanning two same-speaker A turns is written into one of them; the sibling A
turn, whose text B's spanning text *contains*, is still emitted unchanged. The `used_b_ids` set
fixed *exact* duplication and left *containment* duplication. Measured: 16 same-speaker instances
in the reference output, shared spans up to 325 characters. 31 of 35 redundant pairs involve the
replacement branch (permutation-tested, z = +4.40).

**Fix.** Two passes inside `merge_turns`. The first records, per A index, which B turn (if any)
replaced it. The second drops any A turn that a neighbouring replacement's B text fully contains.

Constraints, each measured rather than chosen:

- **Same speaker only.** At radius 2 a speaker-agnostic guard fires on 8 cross-speaker pairs, and
  in each it consumes the *correctly attributed* copy — worsening the defect it exists to fix.
- **Full containment only.** The looser "detector criterion" (>40 shared chars and >50% of the
  shorter block) reaches 20/23 pairs but destroys 840 characters of non-duplicated transcript,
  worst single case 120. Rejected: this is a transcription tool, and deleting speech to tidy
  duplication is the wrong trade.
- **Radius 2 in A-index terms.** `_group_consecutive` flushes on every speaker change, so two
  same-speaker A turns are never adjacent; a radius-1 guard fires on **zero** same-speaker cases.
  Radius > 2 is unmeasured and is not used.
- **Only unreplaced A turns can be consumed.** A turn that itself won a B replacement carries text
  from the other source; dropping it would delete content rather than a duplicate. This matches the
  measured signature exactly — `a_kept` appears in 35 of 35 redundant pairs.

**Correction to the triage's prescription.** Its "6 fires, 0 chars destroyed" was measured over
the 23 pairs the redundancy detector had already flagged, all of which clear a >40-character
shared-substring bar. A bare substring test applied to *all* neighbours also fires wherever a
short A turn's text incidentally appears in B's paragraph — `"so"` is contained in almost any
sentence — and consuming those deletes genuine turns the measurement never counted. The guard
therefore requires the **consumed turn's normalized text to be ≥ 40 characters**, matching the
criterion the yield was measured under. Spurious-fire count is checked on the re-run.

**Expected yield.** 6 of 23 same-speaker pairs (run 1), 4 of 19 (run 2), zero characters
destroyed. Explicitly *not* the 87% the lossy variant would claim.

**Tests.**

- A B turn spanning two same-speaker A turns emits the B text once and drops the contained sibling.
- A cross-speaker A turn contained in B's text is **kept** (the guard must not fire).
- A short (< 40-char) same-speaker A turn incidentally contained in B's text is **kept**.
- A same-speaker A turn only *partially* overlapping B's text is **kept**.

---

## 3. Offset confidence, reported and warned on

**Defect.** `find_offset` returns the argmax lag with no measure of how peaked the correlation is,
and `run_fusion` never surfaces it — `src/audio_to_text/fusion.py` contains zero `print`
statements. The recorded mitigation ("a human sanity-checks the offset") was a one-off validation
script run once on the reference pair; it does not protect any future run.

**Fix.** `find_offset` returns a `NamedTuple`:

```python
class Offset(NamedTuple):
    seconds: float
    peak_ratio: float   # peak / best rival more than 5 s from the argmax
```

`run_fusion` prints the offset and ratio unconditionally, and warns to stderr below **1.10**.

**Metric choice and threshold, both measured.** Against five negative controls built from the same
two recordings:

| pair | peak/best_rival |
|---|---|
| true pair | **1.5162** |
| A first half vs B second half | 1.0034 |
| A second half vs B first half | 1.0026 |
| A vs shuffled B | 1.0032 |
| A vs reversed B | 1.0049 |
| A vs gaussian noise | 1.0002 |

`peak/median` (3.135 vs nulls to 2.094) and the z-score (3.69 vs 1.80) separate by less than a
factor of two and are not used. The 1.10 threshold sits ~20× above the null spread and ~5× below
the one true observation.

**Warn, never gate.** The null floor is well characterised; the true-pair distribution is n = 1.
A gate would refuse to run on a legitimate pair whose ratio simply lands lower, with no evidence
about how low a legitimate pair can go.

**Tests.**

- The known-delay fixture yields a `peak_ratio` well above the threshold, and `.seconds` still
  recovers the delay (the existing assertion, re-pointed at the field).
- Correlating a signal against unrelated noise yields a `peak_ratio` near 1.0.
- `run_fusion` on a low-ratio pair emits a warning to stderr and still produces a transcript.

---

## 4. Fused output becomes `<primary>.fused.md`

**Defect.** Both modes name the output after the primary input's stem, so transcribing
`teams.mp4` and then fusing `teams.mp4 --fuse phone.m4a` writes `teams.md` twice, the second
silently replacing the first. Observed directly during verification.

**Fix.** `run_fusion` writes `f"{primary_path.stem}.fused.md"`. The two modes stop colliding by
construction and both outputs survive, with no new flags and no prompt.

**Rejected alternatives.** Refuse-unless-`--force` protects against all overwrites but breaks
re-running to regenerate, which is the normal way to iterate on a transcript. Warn-and-overwrite
is the smallest change but still loses the earlier transcript.

**Ripple.** README and the `transcribe-recording` skill document the output filename; both are
updated.

**Test.** A fused run writes `<stem>.fused.md` and leaves a pre-existing `<stem>.md` byte-identical.

---

## 5. Micro-turn smoothing behind `--smooth`

**Defect.** Of 714 blocks in the reference output, 220 (31%) hold one word and 347 (49%) hold five
or fewer, together carrying 4.8% of the text. Half the document's headings introduce fragments
like `"So"`, `"the"`, `"it?"`. The cause is upstream of grouping: pyannote emits 42 of 96 turns
shorter than 0.5 s (p10 = 0.02 s) on a sampled slice, and `_group_consecutive` faithfully starts a
new block at each switch. Present in the single-file path too (16% one-word), roughly doubled by
fusion.

**Fix.** `smooth_micro_turns(turns)` in `transcribe.py`, applied before `relabel_speakers` in both
the single-file and fused paths, gated on a new `--smooth` flag. Rule G2:

> Absorb a turn into its neighbours when it holds **≤ 2 words** *and* has **exact-zero duration**
> *and* its normalized text is **not a backchannel token** *and* it is **sandwiched** — the turns
> either side share a speaker with each other and differ from this one.

Absorbing merges the three turns into one turn of the neighbours' speaker: text joined, `start`
from the first, `end` from the last. No words are deleted; the absorbed 1–2 words are
**reattributed** to the neighbouring speaker. That reattribution is the rule's entire risk.

**Why opt-in.** Measured across two runs, G2 destroys **zero** hand-labelled genuine turns and is
the most stable rule tested (6% run-to-run swing vs 12–18%). But the labels were produced by the
same author who wrote the backchannel list, after seeing which rules were under test — no
backchannel token was ever labelled JITTER (0 of 59), which is one-directional confirmation bias.
G2 survives because 22 of its 24 exclusions are duration-driven and only 2 are word-list-driven,
so it is ~92% independent of the contaminated criterion — but "~92% independent" is not a licence
to reattribute speech silently. Default off; flip it later once a smoothed transcript has been
read.

**Rejected alternatives.** Every less conservative rule destroys genuine turns against the hand
labels: sandwiched-≤1-word 29%, `< 0.5 s` 35%, `≤2 w not-backchannel` 14%. The `< 0.5 s` family
absorbs a **93-word** genuine turn, because duration on merged turns is corrupt (item 1) — which
is a further reason item 1 lands first.

**Shared vocabulary.** The `BACKCHANNEL` set moves from `tools/analyze_transcript.py` into
`transcribe.py`; the tool imports it. One definition, so the rule and the measurement of the rule
cannot drift apart.

**Stated limitation, carried into `bugs.md`.** The list is English-only, so the rule will not
generalise across languages. And **no redundancy benefit is claimed** — G2 eliminated 6 redundant
pairs in run 1 and 1 in run 2, which is noise.

**Re-measurement is part of the work, not a follow-up.** Every G2 number above was measured on the
pre-fix pipeline. After items 1–3 land, the reference pair is re-run and the numbers are
re-measured before any claim is made about them.

**Tests.**

- A sandwiched zero-duration ≤2-word non-backchannel turn is absorbed, and its words appear in the
  merged neighbour.
- A sandwiched `"yeah"` is **kept** (backchannel).
- A sandwiched ≤2-word turn with non-zero duration is **kept**.
- A ≤2-word zero-duration turn that is *not* sandwiched (different speakers either side) is **kept**.
- Without `--smooth`, block counts are unchanged.

---

## 6. Whisper hallucination-loop warning

**Defect, and why the earlier deferral fails.** Whisper emits degenerate word repetition on quiet
or ambiguous passages. The evidence for "rare" was a count of identical consecutive words *within
a block* — but ~31% of blocks hold a single word, so a repetition loop is shredded across dozens
of blocks and no single block ever contains a repeat. The instrument could only see loops that
fragmentation had not already scattered. Re-measured doc-wide, one run in two contained
`lars`×183 across 34 consecutive blocks, a word appearing nowhere in the other run.

**Fix — detection only.** `detect_repetition_loops(turns)` flattens the turns to a doc-wide token
stream and returns runs of identical consecutive tokens above a threshold, with the token, the
count and the timestamp where the run starts. Both pipeline paths call it after grouping and warn
to stderr:

```
warning: possible Whisper hallucination loop: 'lars' x183 starting at 17:45
```

The transcript file itself stays clean — the marker would travel with the file, but it would also
put a tool's guess into a document people read as a record of what was said.

**Threshold picked by measurement, not guess.** The analysis tool's doc-wide pass uses 4+, which
is tuned for investigation, not for a warning a user sees on every run. The threshold is chosen by
running the detector over the existing reference transcript at a range of values and picking the
lowest one that produces no false alarms there, then stated in the code with that justification.

**Not fixed: the loop itself.** It is upstream in Whisper. Changing decode parameters
(`condition_on_previous_text`, temperature fallback) is the standard mitigation and is
deliberately **not** attempted: its effect on this material is unmeasured, and it trades
transcription accuracy for a symptom this project has no measurement to price.

**Test.** A synthetic turn list containing a long token run produces a warning naming the token,
the count and a timestamp; a normal turn list produces none.

---

## Out of scope

**The 12 cross-speaker duplicate pairs.** Ten to twelve block pairs carry near-identical text
under different `Person` headings, three at an identical timestamp. The cause is genuinely
unmeasured: it may be speaker-attribution error (diarization or Hungarian-matching disagreement
between sources), or genuine cross-talk both microphones captured. Distinguishing the two requires
listening to the recordings, and no automated discriminator has been validated. The containment
guard above deliberately does not touch them — a same-speaker-only guard is what keeps it from
deleting the correctly-attributed copy. Stays in `bugs.md` with its cause marked unmeasured.

**Repairing hallucination loops**, and any Whisper decode-parameter change — see item 6.

**Anything that trades accuracy.** No smaller or quantized models, no accuracy-for-speed changes.

---

## Done-criteria

1. The full test suite passes, with new tests pinning each of the six behaviours above.
2. A post-fix fused run of the reference pair completes, and
   `uv run python tools/analyze_transcript.py` on its output reports: same-speaker redundant pairs
   reduced, **zero** spurious containment fires, and re-measured fragmentation and G2 figures.
3. `--smooth` off reproduces the unsmoothed block count; `--smooth` on shows the re-measured
   reduction, with the figure taken from the post-fix run rather than the triage.
4. An end-to-end CLI run writes `<primary>.fused.md` and leaves a pre-existing `<primary>.md`
   untouched.
5. `bugs.md` is rewritten to record what shipped, what remains open (the cross-speaker pairs, the
   English-only word list, the unquantified hallucination incidence), and what the measured
   post-fix numbers are.
