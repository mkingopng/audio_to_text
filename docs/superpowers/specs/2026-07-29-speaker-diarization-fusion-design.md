# Speaker diarization + dual-source fusion — design

**Date:** 2026-07-29
**Status:** Approved (design) — pending spec review
**Author:** Michael Kingston (with Claude)

## 1. Context & problem

`src/transcribe.py` currently writes one flat `.txt` per file — a single
unbroken block of text with no indication of who said what. For a 6-person
meeting this is hard to read back and impossible to attribute to individuals.

The project's `data/` folder currently holds two recordings of the **same**
in-person meeting (6 participants, one room):

- `Meeting with Michael Kingston-20260729_130839-Meeting Recording.mp4` — the
  Teams video recording, captured via the meeting organizer's laptop.
- `Tag5_29_July_2026_10903_pm.m4a` — an independent phone recording made from
  a different position in the (large) room.
- `Meeting with Michael Kingston.docx` — the Teams-generated transcript for
  the same session. **Not usable as multi-speaker ground truth**: because
  everyone else in the room was captured through one mic rather than joining
  Teams individually, Teams attributed the entire transcript to a single
  speaker label ("Michael Kingston"). It is not a reliable per-speaker or
  even necessarily accurate word-level reference.

There is no automatic speaker-name information available in any source — the
only signal we have to distinguish the 6 participants is their voice prints.

## 2. Goal & success criteria

1. Replace flat `.txt` output with structured Markdown, broken into
   per-speaker-turn sections, each attributed to an arbitrary, stable label
   (`Person 1`..`Person 6`) based on voice-print clustering — for **any**
   single recording (the general case for future one-off meetings).
2. For this specific pair of recordings of one meeting, go further: fuse the
   two sources (Teams video + phone audio) into a single, unified transcript
   where `Person N` means the same physical person across both sources, using
   whichever source is clearer at each moment.
3. `Person N` labels must be applied consistently as a literal, repeated
   token throughout the document, so the user can watch the first few minutes
   of the video, identify each person by name, and rename everyone via a
   simple find/replace pass over the Markdown.

Success =
- Single-file mode: `output/<stem>.md` with clearly separated, plausibly
  correct speaker turns for a 6-person recording.
- Fusion mode: one `output/<primary-stem>.md` combining both recordings, with
  consistent `Person N` identity across the whole document.
- Manual spot-check (by ear, a handful of speaker transitions + samples from
  start/middle/end) confirms turns are attributed to the right person more
  often than not, and speaker identity does not drift or renumber partway
  through the document.

## 3. Decision

**Adopt `pyannote.audio`** (pretrained `pyannote/speaker-diarization-3.1`
pipeline: VAD + embeddings + clustering) for diarization, and build a small
fusion layer on top for the two-source case.

Rejected alternatives:
- **Build our own embeddings + clustering** (e.g. SpeechBrain/Resemblyzer +
  custom agglomerative clustering) — more code, more tuning, and pyannote's
  pretrained pipeline is already tuned for exactly this task.
- **WhisperX** (bundles ASR + diarization) — rejected because it runs ASR via
  faster-whisper/CTranslate2, not MLX. Adopting it would reverse the recent
  GPU-migration decision ([2026-07-03 design](2026-07-03-mlx-whisper-gpu-transcription-design.md))
  to run ASR via `mlx-whisper` on Apple's Metal backend. We still borrow its
  core idea (word-level ASR + diarization alignment) without adopting the
  library.
- **Diarize both recordings independently, pick the "better" one by hand** —
  simpler and lower-risk, but explicitly rejected by the user in favor of true
  fusion (§ single source of truth is the actual goal, not a fallback plan).

## 4. Architecture

Keep `transcribe.py`'s existing single-file batch path intact and reusable;
add a new `src/fusion.py` module for the two-source case so the sync/matching
logic doesn't tangle with the simple per-file path.

**Shared primitives** (added to `transcribe.py`, used by both modes):

- `run_diarization(wav_path, *, num_speakers=None) -> (turns, embeddings)` —
  wraps `pyannote.audio.Pipeline`, called with `return_embeddings=True` so we
  get one representative embedding per detected speaker directly from the
  pipeline (no separate embedding-extraction pass). `turns` is a list of
  `(start, end, local_speaker_id)`.
- `run_whisper(..., word_timestamps=True)` — existing function, extended to
  request word-level timestamps (mlx-whisper supports this) instead of
  relying on segment-level timing.
- `align_words_to_speakers(words, turns) -> [(start, end, speaker_id, word)]`
  — assigns each *word* (not each multi-second segment) to whichever
  diarization turn overlaps it most. This is the fix for the
  segment-boundary misattribution problem raised during brainstorming:
  shrinking the attribution unit from a whole Whisper segment (~5-10s) to a
  single word (~0.3-1s) means a speaker change mid-segment no longer drags
  the whole segment to the wrong person.
- `group_into_turns(aligned_words) -> paragraphs` — merges consecutive
  same-speaker words into paragraphs, relabels speakers by first-appearance
  order into `Person 1..Person N`, and renders the Markdown.

**Fusion-only logic** (`src/fusion.py`):

- `find_offset(wav_a, wav_b) -> seconds` — cross-correlation (via `scipy`) on
  the audio envelope to find the constant time offset between the two
  recordings' clocks.
- `select_source_per_turn(turns_a, turns_b, words_a, words_b) -> merged turns`
  — for each diarization-turn window, compare average word-confidence between
  source A and source B in that window and keep the higher-confidence
  source's text for that turn.
- `match_speakers(embeddings_a, embeddings_b) -> {a_id: b_id}` — cosine
  similarity matrix between the two sources' 6 speaker-embedding centroids,
  resolved via `scipy.optimize.linear_sum_assignment` (Hungarian algorithm)
  for an optimal one-to-one match, so `Person 3` is the same physical person
  in both sources.

## 5. CLI surface

Extend the existing single-file argument rather than adding a new command:

```
uv run python src/transcribe.py "data/Meeting....mp4" --fuse "data/Tag5....m4a" --num-speakers 6
```

- `media` (existing positional arg) — the primary recording.
- `--fuse SECOND_FILE` (new) — triggers dual-source fusion mode. Invalid when
  `media` resolves to a directory (batch mode) rather than a single file.
- `--num-speakers N` (new) — hint passed to pyannote for both sources when
  known (e.g. `6` for these recordings); defaults to `None` (auto-detect) for
  the general case where the count isn't known ahead of time.

Existing flags (`--model`, `--language`, `--prompt`, `--output-dir`,
`--preprocess`, `--denoise`, `--audio-filter`, `--verbose`) are unchanged and
apply to both modes.

## 6. Data flow

**Single-file mode** (no `--fuse`):
```
media file
  └─ preprocess_audio() [now UNCONDITIONAL: always extract 16kHz mono WAV;
     the --preprocess/--denoise filter chain is applied on top only if passed]
      ├─ run_whisper(word_timestamps=True) → words w/ timestamps + confidence
      └─ run_diarization()                  → speaker turns + embeddings
      └─ align_words_to_speakers()
      └─ group_into_turns()                 → Person 1..N labeled paragraphs
      └─ write output/<stem>.md
```

**Fusion mode** (`--fuse SECOND_FILE`):
```
primary file, secondary file
  └─ preprocess_audio() on each → wav_a, wav_b
      ├─ run_whisper(word_timestamps=True) on each → words_a, words_b
      └─ run_diarization() on each                 → turns_a/emb_a, turns_b/emb_b
      └─ align_words_to_speakers() on each
      └─ find_offset(wav_a, wav_b)                 → shift b's timeline onto a's
      └─ match_speakers(emb_a, emb_b)               → unify local speaker ids
      └─ select_source_per_turn(...)                → merged turn list
      └─ group_into_turns()                         → Person 1..N labeled paragraphs
      └─ write output/<primary-stem>.md
```

## 7. Output format

Heading per speaker turn, with a timestamp (mm:ss or hh:mm:ss for long
recordings), relative to the start of the (primary, in fusion mode)
recording:

```markdown
## Person 1 — 00:03

Guidelines, but here you can see create client. I just confirm, and then
Michael will come up in here and we can search it.

## Person 3 — 01:42

Yep, we have both choices, it's up to you, and the same works with workers
as well.
```

`.txt` output is retired; Markdown is the sole output format going forward.

## 8. Dependencies & setup

`pyproject.toml` additions:
- `pyannote.audio` — diarization pipeline (pulls in more of the `torch` stack
  already present transitively via `mlx-whisper`).
- `python-dotenv` — load `HF_TOKEN` from the (gitignored, already-present)
  `.env` file automatically rather than requiring the user to `export` it
  each session.
- `scipy` — cross-correlation (`find_offset`) and the Hungarian algorithm
  (`match_speakers`).

One-time manual setup (not automatable): accept the
`pyannote/speaker-diarization-3.1` model terms on Hugging Face. The token is
already saved in `.env`.

Diarization will request the MPS (Apple GPU) device via torch where
available; see risks below re: MPS op coverage.

## 9. Error handling

- Preserve existing per-file failure handling (missing file, ffmpeg failure
  reported and skipped, batch continues, non-zero exit on any failure).
- `--fuse` with a directory `media` argument: fail fast with a clear error
  before doing any work.
- Missing/invalid `HF_TOKEN`: fail fast with a message pointing at the `.env`
  file and the model-terms acceptance step, rather than a raw pyannote
  auth-error traceback.
- If `--num-speakers` is given but pyannote's diarization produces a
  different number of embeddings than expected (shouldn't happen when the
  hint is honored, but defensively): surface a clear warning rather than
  silently truncating/padding speaker identities.

## 10. Risks & mitigations

- **MPS op coverage** — same class of issue hit during the MLX-GPU migration
  with `openai-whisper`/MPS: some `torch` ops used by pyannote may not have
  Metal kernels and silently/loudly fall back to CPU. Mitigation: try MPS,
  catch and fall back to CPU with a printed notice; not a hard blocker since
  correctness matters more than diarization speed here.
- **Clock drift between the two recordings** — `find_offset` assumes a single
  constant offset across the whole ~70-minute recording. Consumer-device
  drift is normally small relative to word-level timestamp granularity, but
  this is an assumption, not a guarantee. Mitigation: sanity-check alignment
  quality separately in the first and last few minutes of the recording
  during validation; if drift turns out to be material, escalate to a
  piecewise/windowed offset — explicitly out of scope for v1 unless the
  single-offset assumption demonstrably fails.
- **No automated ground truth** — the docx transcript cannot validate
  per-speaker accuracy (see §1). Mitigation: manual spot-check by ear is the
  acceptance test, not an automated metric.
- **Sparse-speech speaker embeddings** — a participant who barely spoke in
  one of the two recordings (e.g. too quiet for the phone mic) yields a
  noisier embedding centroid, risking a wrong match in `match_speakers`.
  Mitigation: none automated for v1; flagged for manual review during the
  spot-check.
- **Turn-level (not word-level) source selection in fusion mode** — splicing
  two independently-run ASR passes at word granularity risks garbled
  sentences where the two passes segment differently; selecting per
  diarization-turn window instead avoids this at the cost of occasionally
  keeping a slightly-worse phrase within an otherwise-better turn.

## 11. Testing & validation

Feasibility validation first (mirrors the incremental approach used in the
2026-07-03 design):
1. `uv add pyannote.audio python-dotenv scipy` resolves on the existing
   Python 3.13 venv.
2. `HF_TOKEN` loads from `.env` and authenticates against the gated pyannote
   model.
3. Single-file diarization end-to-end on a short clip: confirm turns +
   embeddings are returned, `Person N` labels render correctly in Markdown.
4. `word_timestamps=True` on `mlx_whisper.transcribe`: confirm word-level
   start/end/confidence are present in the result.
5. `find_offset` on the two real recordings: confirm a plausible offset
   (sanity-checked by ear against a recognizable moment in both sources,
   e.g. someone's name being said).
6. `match_speakers` on the two real recordings: confirm a plausible 1:1
   mapping (spot-check 2-3 matched pairs by ear).

Acceptance:
7. Full fusion run on both real recordings → `output/<primary-stem>.md`.
   Manual spot-check: several speaker transitions, and samples from
   start/middle/end of the ~70-minute meeting, confirm turns are attributed
   to the right person more often than not and speaker identity is stable
   throughout (no renumbering drift).

## 12. Out of scope (YAGNI)

- More than 2 fused sources.
- Automatic piecewise/windowed drift correction (only added if the
  single-offset assumption demonstrably fails during validation).
- A UI or tool for correcting individual misattributed turns beyond the
  planned manual find/replace of `Person N` → real names.
- Real name identification/recognition — labels remain arbitrary
  (`Person N`); renaming is an intentionally manual, out-of-tool step.
- Non-Markdown output formats.
