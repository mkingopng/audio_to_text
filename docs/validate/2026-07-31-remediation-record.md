# What shipped — `bugs.md` remediation

**Date:** 2026-07-31 · **Branch:** `feature/global-tool` · **Commits:** `e32de8f..HEAD`
**Tests:** 60 → 95, all green.

A plain-language record of what changed and why, for someone reading this later without
the conversation. The measured evidence is in `bugs.md`; the cross-speaker analysis is in
`2026-07-31-cross-speaker-cause.md`; the prior triage is `2026-07-31-bugs-md-triage.md`.

## The shape of the problem

All seven entries were about **fusion telling the user something false with a straight
face** — wrong timestamps, duplicated paragraphs, a fabricated speaker, an alignment that
was never checked. The through-line of every fix is the same: either make the output true,
or make the uncertainty visible. Nothing was made to *look* better than it is.

## What changed, and the reasoning

**Merged turns now carry their own timestamps.** When the second recording's text won a
turn, the merged block kept the *first* recording's start/end — so a 90-second sentence
could claim to have been spoken in 0.4 seconds, and its heading pointed at the wrong
minute. A merged turn is now simply the winning source's turn. The alternative — trimming
the text to fit the old timestamps — would have meant splicing two independent
transcriptions word by word, which garbles sentences where they segment speech
differently.

**Duplicated paragraphs are removed, conservatively.** When one recording heard as a single
turn what the other split in two, the shared text was emitted twice. The fix deletes the
duplicate copy only when it is *entirely* contained in the surviving one, only for the same
speaker, only within two turns, only above 20 characters, and only when the surviving turn
actually covers that moment in time. Each of those five conditions exists because relaxing
it was measured to destroy real speech. A looser "these look similar" rule was tested and
rejected: it deleted 840 characters of transcript that were not duplicates.

**Jitter fragments rejoin their sentence, if asked.** Diarization scatters single words
("the", "so") into their own blocks under the wrong speaker. `--smooth` folds them back into
the surrounding sentence — *moved*, never deleted, so a mistake costs a misattributed word
rather than a lost one. Genuine one-word turns ("yeah", "mm-hmm") are protected, because
absorbing those would silently put words in someone else's mouth. It is **off by default**:
every other step here either preserves diarization's attribution or improves it against
measured evidence, and this one overrides it on a discriminator hand-checked over a single
meeting. That asymmetry is what makes it an offer rather than a default.

**Three things the tool now admits to.** Fusion reports its alignment confidence; any run
reports Whisper repetition loops; fused runs report where the two recordings disagreed
about who spoke. The last one is deliberately *not* auto-resolved — see below.

**Fused output is named `<stem>.fused.md`** so it stops overwriting the single-file
transcript of the same recording.

## The judgement call worth knowing about

Eleven times in a 70-minute meeting, the two recordings disagree about who said something,
and the same sentence appears under two names. We could pick one. We chose not to: there is
no evidence available to say which is right (the speaker-embedding match is already the
best signal, and it is what disagreed), and picking would convert an oddity a reader
notices into a wrong answer they do not. So the tool lists the timestamps and leaves the
judgement to the person who was in the room.

## What is still open, honestly

Whisper's hallucination loops are upstream and unfixed — detected, not cured, and they
occurred in 2 of 3 runs. About 23% of blocks still hold a single word. Zero-duration
turns from the second source are always appended even when already represented. All are
recorded in `bugs.md` with measured status rather than left implicit.

## A note on the evidence

The prior triage's numbers could not be reproduced — the scripts behind them were never
saved. Everything here was re-measured, and the scripts are committed under `tools/`. That
does not make the counts reproducible to the character (Whisper is non-deterministic and
the 270 MB capture is not committed), but it makes the *method* checkable, which is the
part that failed last time.
