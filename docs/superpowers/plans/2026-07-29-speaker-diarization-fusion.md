# Speaker Diarization + Dual-Source Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Replace flat `.txt` transcription output with speaker-attributed Markdown (`Person 1`..`Person N` headings), and fuse the two real recordings of the same 6-person meeting (Teams `.mp4` + phone `.m4a`) into one unified transcript.

**Architecture:** Shared pure-logic primitives (word→speaker alignment, turn grouping/relabeling, Markdown rendering) live in `src/transcribe.py` alongside the existing single-file pipeline. A new `src/fusion.py` reuses those primitives and adds sync (cross-correlation), cross-source speaker matching (Hungarian algorithm on pyannote speaker embeddings), and confidence-based turn selection.

**Tech Stack:** `mlx-whisper` (ASR, unchanged engine), `pyannote.audio` (diarization), `scipy` (cross-correlation + Hungarian algorithm), `python-dotenv` (load `HF_TOKEN` from `.env`), `pytest` (new — this repo has no existing test suite).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-29-speaker-diarization-fusion-design.md` — every task below implements a section of it; do not deviate from the documented function names/behavior without updating the spec first.
- Python `>=3.12` (existing `pyproject.toml` constraint).
- MLX is the sole ASR engine — no fallback engine (per the 2026-07-03 design, unchanged).
- `.txt` output is retired. Markdown is the only output format.
- `word_timestamps=True` is the default for `run_whisper` — this is the fix for segment-boundary speaker misattribution; do not silently regress it to `False`.
- No automated ground truth exists for speaker-attribution correctness (the docx transcript only labels one speaker). Real-file validation steps in this plan are manual/by-ear, not automated assertions — do not fabricate a pass/fail test for something that requires human judgment.
- `HF_TOKEN` is already present in the gitignored `.env` at the project root — never print its value or commit it.
- `pyannote.audio` resolved (Task 1) to `4.0.7`, not the `3.x` line the model card examples are written against: `Pipeline.from_pretrained(...)` takes `token=`, not `use_auth_token=` (renamed upstream). Already corrected throughout this plan.
- Word dicts from `mlx_whisper` (when `word_timestamps=True`) have exactly these keys: `"word"` (str, includes leading punctuation/whitespace per Whisper's tokenizer — join words with `"".join(...)`, not `" ".join(...)`), `"start"` (float), `"end"` (float), `"probability"` (float). Confirmed by reading the installed `mlx_whisper` 0.4.x source (`timing.py`/`transcribe.py`).

---

### Task 1: Feasibility spike — dependencies, HF auth, pytest scaffold

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/__init__.py` (empty)
- Create: `tests/test_environment.py`

**Interfaces:**
- Produces: a working `pytest` invocation (`uv run pytest`) that later tasks build on. No production code interfaces from this task.

This repo currently has zero automated tests (confirmed: no `tests/` dir, no `pytest` in the venv). This task establishes the harness before any TDD work.

- [x] **Step 1: Add the new dependencies**

```bash
uv add pyannote.audio python-dotenv scipy
uv add --dev pytest
```

- [x] **Step 2: Write the environment smoke test**

Create `tests/test_environment.py`:

```python
"""Smoke tests confirming the new diarization/fusion dependencies are usable."""
import os

import scipy.signal  # noqa: F401
from dotenv import load_dotenv


def test_new_dependencies_import():
    import pyannote.audio  # noqa: F401


def test_hf_token_loads_from_dotenv():
    load_dotenv()
    assert os.environ.get("HF_TOKEN"), "HF_TOKEN not found after load_dotenv() -- check .env"
```

- [x] **Step 3: Run it, confirm it passes**

Run: `uv run pytest tests/test_environment.py -v`
Expected: both tests PASS. If `test_hf_token_loads_from_dotenv` fails, stop and check `.env` contains `HF_TOKEN=...` at the project root before continuing to any later task.

- [x] **Step 4: Confirm gated-model access manually**

Run:
```bash
uv run python -c "
from dotenv import load_dotenv
import os
load_dotenv()
from pyannote.audio import Pipeline
p = Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', token=os.environ['HF_TOKEN'])
print(type(p))
"
```
Expected: prints the pipeline's type with no authentication error. If it fails with a 403/gated-model error, the model terms haven't been accepted on huggingface.co yet — stop and accept them before continuing (this is a one-time manual step per the spec, not something to script around).

- [x] **Step 5: Confirm the `return_embeddings=True` output shape**

Run a short real clip (reuse one of the files already in `data/` — pick a ~30s slice isn't available, so just run on the full `Tag5_29_July_2026_10903_pm.m4a` once, it's fine for a one-off spike):
```bash
uv run python -c "
from dotenv import load_dotenv
import os
load_dotenv()
from pyannote.audio import Pipeline
p = Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', token=os.environ['HF_TOKEN'])
diarization, embeddings = p('data/Tag5_29_July_2026_10903_pm.m4a', num_speakers=6, return_embeddings=True)
labels = sorted(diarization.labels())
print('labels:', labels)
print('embeddings shape:', embeddings.shape)
"
```
Expected: `embeddings.shape == (len(labels), N)` for some embedding dimension `N`, with rows in the same order as `sorted(labels)`. **Record whether this holds** — Task 6's `run_diarization` assumes `embeddings[i]` corresponds to `sorted(diarization.labels())[i]`. If the actual order differs (e.g. insertion order instead of sorted), adjust Task 6's implementation to match before writing it.

- [x] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock tests/__init__.py tests/test_environment.py
git commit -m "test: add pytest harness, verify pyannote/scipy/dotenv deps"
```

---

### Task 2: `align_words_to_speakers` — word-level speaker attribution

**Files:**
- Modify: `src/transcribe.py`
- Test: `tests/test_transcribe.py` (new)

**Interfaces:**
- Produces: `align_words_to_speakers(words: list[dict], turns: list[dict]) -> list[dict]`
  - `words`: each `{"word": str, "start": float, "end": float, "probability": float}`
  - `turns`: each `{"start": float, "end": float, "speaker": str}` (diarization turns, any order)
  - returns: each input word dict plus `"speaker": str` (the assigned turn's speaker id), sorted by `start` ascending (same order as input `words`, which is assumed already chronological).

This is the fix for the segment-boundary misattribution problem: assigning speaker identity per-word (not per multi-second Whisper segment) so a mid-segment speaker change only misattributes the handful of words actually on the wrong side, not the whole segment.

- [x] **Step 1: Write the failing test**

Create `tests/test_transcribe.py`:

```python
"""Unit tests for the pure speaker-attribution/grouping/rendering logic in transcribe.py."""
from src.transcribe import align_words_to_speakers


def test_align_words_to_speakers_assigns_by_max_overlap():
    turns = [
        {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
        {"start": 5.0, "end": 10.0, "speaker": "SPEAKER_01"},
    ]
    words = [
        {"word": "Hello", "start": 0.5, "end": 1.0, "probability": 0.9},
        {"word": " world", "start": 6.0, "end": 6.5, "probability": 0.9},
        # straddles the turn boundary at 5.0s, but spends more time (0.6s) in
        # turn 2 than turn 1 (0.2s) -- must be assigned to SPEAKER_01, proving
        # word-level (not whole-segment) attribution.
        {"word": " boundary", "start": 4.8, "end": 5.6, "probability": 0.8},
    ]

    result = align_words_to_speakers(words, turns)

    assert [w["speaker"] for w in result] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_01"]
    # original word fields are preserved
    assert result[0]["word"] == "Hello"
    assert result[0]["probability"] == 0.9


def test_align_words_to_speakers_handles_gap_with_no_overlapping_turn():
    turns = [
        {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
        {"start": 4.0, "end": 6.0, "speaker": "SPEAKER_01"},
    ]
    # falls in the silent gap between the two turns, closer to turn 2
    words = [{"word": "gap", "start": 3.6, "end": 3.8, "probability": 0.7}]

    result = align_words_to_speakers(words, turns)

    assert result[0]["speaker"] == "SPEAKER_01"
```

- [x] **Step 2: Run it, confirm it fails**

Run: `uv run pytest tests/test_transcribe.py -v`
Expected: FAIL with `ImportError: cannot import name 'align_words_to_speakers'`.

- [x] **Step 3: Implement it**

Add to `src/transcribe.py` (near the other pure helpers, above `run_whisper`):

```python
def overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Seconds of overlap between two [start, end) intervals (0.0 if disjoint).

    Public (not underscore-prefixed): fusion.py's merge_turns (Task 10) imports
    this directly rather than duplicating the same interval-overlap logic.
    """
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def align_words_to_speakers(words: list[dict], turns: list[dict]) -> list[dict]:
    """Assign each word to the diarization turn it overlaps most.

    Attribution happens per word (not per multi-second Whisper segment) so a
    speaker change mid-segment only misattributes the words actually on the
    wrong side of the change, not the whole segment.
    """
    if not turns:
        raise ValueError("align_words_to_speakers: no diarization turns to align against")

    aligned = []
    for word in words:
        best_turn = max(
            turns,
            key=lambda t: overlap_seconds(word["start"], word["end"], t["start"], t["end"]),
        )
        if overlap_seconds(word["start"], word["end"], best_turn["start"], best_turn["end"]) <= 0.0:
            # word falls in a silent gap between turns; attribute to whichever
            # turn boundary is closest in time.
            best_turn = min(
                turns,
                key=lambda t: min(abs(word["start"] - t["end"]), abs(t["start"] - word["end"])),
            )
        aligned.append({**word, "speaker": best_turn["speaker"]})
    return aligned
```

- [x] **Step 4: Run it, confirm it passes**

Run: `uv run pytest tests/test_transcribe.py -v`
Expected: PASS (2 tests).

- [x] **Step 5: Commit**

```bash
git add src/transcribe.py tests/test_transcribe.py
git commit -m "feat: add word-level speaker attribution (align_words_to_speakers)"
```

---

### Task 3: Turn grouping, speaker relabeling, and Markdown rendering

**Files:**
- Modify: `src/transcribe.py`
- Test: `tests/test_transcribe.py`

**Interfaces:**
- Consumes: `align_words_to_speakers` output (list of word dicts with `"speaker"`).
- Produces:
  - `_group_consecutive(aligned_words: list[dict]) -> list[dict]` — each `{"speaker": str (original diarization id, NOT relabeled), "start": float, "end": float, "text": str, "confidence": float}`. Shared by both single-file grouping and fusion merging (Task 10 needs the un-relabeled form).
  - `relabel_speakers(turns: list[dict]) -> list[dict]` — same shape, `"speaker"` renamed to `"Person 1"`.."Person N"` by first-appearance order in the input list. A speaker id seen again later keeps its **original** number (no renumbering drift).
  - `group_into_turns(aligned_words: list[dict]) -> list[dict]` = `relabel_speakers(_group_consecutive(aligned_words))` — the single-file convenience entry point.
  - `render_markdown(turns: list[dict]) -> str` — turns must already be relabeled (i.e. `"speaker"` is `"Person N"`).

- [x] **Step 1: Write the failing tests**

Append to `tests/test_transcribe.py`:

```python
from src.transcribe import (
    _group_consecutive,
    group_into_turns,
    relabel_speakers,
    render_markdown,
)


def test_group_consecutive_merges_same_speaker_words_into_one_turn():
    aligned = [
        {"word": "Hello", "start": 0.0, "end": 0.5, "probability": 0.9, "speaker": "SPEAKER_00"},
        {"word": " there", "start": 0.5, "end": 1.0, "probability": 0.8, "speaker": "SPEAKER_00"},
        {"word": " hi", "start": 1.0, "end": 1.3, "probability": 0.7, "speaker": "SPEAKER_01"},
    ]

    turns = _group_consecutive(aligned)

    assert len(turns) == 2
    assert turns[0]["speaker"] == "SPEAKER_00"
    assert turns[0]["text"] == "Hello there"
    assert turns[0]["start"] == 0.0
    assert turns[0]["end"] == 1.0
    assert turns[0]["confidence"] == (0.9 + 0.8) / 2
    assert turns[1]["speaker"] == "SPEAKER_01"
    assert turns[1]["text"] == "hi"


def test_relabel_speakers_is_stable_and_does_not_renumber_on_return():
    turns = [
        {"speaker": "SPEAKER_05", "start": 0.0, "end": 1.0, "text": "a", "confidence": 0.9},
        {"speaker": "SPEAKER_02", "start": 1.0, "end": 2.0, "text": "b", "confidence": 0.9},
        # SPEAKER_05 talks again later -- must stay "Person 1", not become "Person 3"
        {"speaker": "SPEAKER_05", "start": 2.0, "end": 3.0, "text": "c", "confidence": 0.9},
    ]

    relabeled = relabel_speakers(turns)

    assert [t["speaker"] for t in relabeled] == ["Person 1", "Person 2", "Person 1"]


def test_group_into_turns_combines_grouping_and_relabeling():
    aligned = [
        {"word": "Hi", "start": 0.0, "end": 0.5, "probability": 0.9, "speaker": "SPEAKER_03"},
        {"word": " yo", "start": 0.5, "end": 1.0, "probability": 0.9, "speaker": "SPEAKER_01"},
    ]

    turns = group_into_turns(aligned)

    assert turns[0]["speaker"] == "Person 1"
    assert turns[1]["speaker"] == "Person 2"


def test_render_markdown_formats_heading_and_timestamp():
    turns = [
        {"speaker": "Person 1", "start": 3.0, "text": "Guidelines, but here you can see."},
        {"speaker": "Person 3", "start": 102.0, "text": "Yep, we have both choices."},
        # past the 1-hour mark -- must render hh:mm:ss
        {"speaker": "Person 1", "start": 3725.0, "text": "Back again."},
    ]

    markdown = render_markdown(turns)

    assert "## Person 1 — 00:03\n\nGuidelines, but here you can see.\n" in markdown
    assert "## Person 3 — 01:42\n\nYep, we have both choices.\n" in markdown
    assert "## Person 1 — 01:02:05\n\nBack again.\n" in markdown
```

- [x] **Step 2: Run it, confirm it fails**

Run: `uv run pytest tests/test_transcribe.py -v`
Expected: FAIL with `ImportError` for `_group_consecutive`, `group_into_turns`, `relabel_speakers`, `render_markdown`.

- [x] **Step 3: Implement it**

Add to `src/transcribe.py`, directly below `align_words_to_speakers`:

```python
def _join_words(words: list[str]) -> str:
    """Whisper word strings already carry leading spaces/punctuation; join without adding more."""
    return "".join(words).strip()


def _group_consecutive(aligned_words: list[dict]) -> list[dict]:
    """Merge consecutive same-speaker words into turns, keeping original speaker ids.

    Returned turns are NOT relabeled to "Person N" -- fusion (Task 10) needs the
    original diarization ids to merge two sources before a single final relabel.
    """
    if not aligned_words:
        return []

    turns: list[dict] = []
    current_speaker = aligned_words[0]["speaker"]
    current_words: list[str] = []
    current_probs: list[float] = []
    current_start = aligned_words[0]["start"]
    current_end = current_start

    def _flush() -> None:
        turns.append({
            "speaker": current_speaker,
            "start": current_start,
            "end": current_end,
            "text": _join_words(current_words),
            "confidence": sum(current_probs) / len(current_probs),
        })

    for word in aligned_words:
        if word["speaker"] != current_speaker:
            _flush()
            current_speaker = word["speaker"]
            current_words = []
            current_probs = []
            current_start = word["start"]
        current_words.append(word["word"])
        current_probs.append(word["probability"])
        current_end = word["end"]
    _flush()
    return turns


def relabel_speakers(turns: list[dict]) -> list[dict]:
    """Rename each turn's speaker id to "Person N" by first-appearance order.

    A speaker id that recurs later in the list keeps the number it was first
    assigned -- speaker identity must not drift/renumber partway through.
    """
    labels: dict[str, str] = {}

    def _label_for(speaker_id: str) -> str:
        if speaker_id not in labels:
            labels[speaker_id] = f"Person {len(labels) + 1}"
        return labels[speaker_id]

    return [{**turn, "speaker": _label_for(turn["speaker"])} for turn in turns]


def group_into_turns(aligned_words: list[dict]) -> list[dict]:
    """Single-file convenience entry point: group, then relabel to Person N."""
    return relabel_speakers(_group_consecutive(aligned_words))


def _format_timestamp(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def render_markdown(turns: list[dict]) -> str:
    """Render relabeled turns (speaker == "Person N") as heading-per-turn Markdown."""
    blocks = [
        f"## {turn['speaker']} — {_format_timestamp(turn['start'])}\n\n{turn['text']}\n"
        for turn in turns
    ]
    return "\n".join(blocks).rstrip() + "\n"
```

- [x] **Step 4: Run it, confirm it passes**

Run: `uv run pytest tests/test_transcribe.py -v`
Expected: PASS (6 tests total including Task 2's).

- [x] **Step 5: Commit**

```bash
git add src/transcribe.py tests/test_transcribe.py
git commit -m "feat: add turn grouping, stable speaker relabeling, Markdown rendering"
```

---

### Task 4: Unconditional audio extraction

**Files:**
- Modify: `src/transcribe.py:84-120` (`build_audio_filter`, `preprocess_audio`)
- Test: `tests/test_transcribe.py`

**Interfaces:**
- Produces: `build_ffmpeg_args(media_path: Path, cleaned_path: Path, audio_filter: str | None) -> list[str]` (pure, replaces the inline arg list previously built directly inside `preprocess_audio`).
- Modifies: `preprocess_audio(media_path: Path, tmp_dir: Path, audio_filter: str | None) -> Path` — `audio_filter` becomes `Optional`; when `None`, extraction still happens (plain resample to 16kHz mono, no `-af` filter chain).

Currently `main()` only calls `preprocess_audio()` when `--preprocess`/`--denoise`/`--audio-filter` is passed; otherwise raw files (including videos) go straight into `mlx_whisper`, which decodes them internally. Diarization needs an actual WAV file on disk regardless, and both `run_whisper` and `run_diarization` need to see the *same* audio so their timestamps share one time base. This task makes extraction unconditional; the `--preprocess`/`--denoise` flags now only control whether a filter chain is layered on top.

- [x] **Step 1: Write the failing test**

Append to `tests/test_transcribe.py`:

```python
from pathlib import Path

from src.transcribe import build_ffmpeg_args


def test_build_ffmpeg_args_without_filter():
    args = build_ffmpeg_args(Path("in.mp4"), Path("/tmp/out.wav"), None)

    assert args == [
        "ffmpeg", "-nostdin", "-y", "-i", "in.mp4",
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", "/tmp/out.wav",
    ]


def test_build_ffmpeg_args_with_filter():
    args = build_ffmpeg_args(Path("in.m4a"), Path("/tmp/out.wav"), "highpass=f=80,loudnorm=I=-16:TP=-1.5:LRA=11")

    assert "-af" in args
    assert args[args.index("-af") + 1] == "highpass=f=80,loudnorm=I=-16:TP=-1.5:LRA=11"
    assert args[-1] == "/tmp/out.wav"
```

- [x] **Step 2: Run it, confirm it fails**

Run: `uv run pytest tests/test_transcribe.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_ffmpeg_args'`.

- [x] **Step 3: Implement it**

Replace the body of `preprocess_audio` in `src/transcribe.py` (currently lines 99-120) with:

```python
def build_ffmpeg_args(media_path: Path, cleaned_path: Path, audio_filter: str | None) -> list[str]:
    """Build the ffmpeg argv that extracts media_path to a 16kHz mono WAV.

    audio_filter is applied via -af only when given; extraction to 16kHz mono
    happens unconditionally either way (both run_whisper and run_diarization
    need the same WAV on the same time base).
    """
    args = ["ffmpeg", "-nostdin", "-y", "-i", str(media_path)]
    if audio_filter:
        args += ["-af", audio_filter]
    args += ["-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(cleaned_path)]
    return args


def preprocess_audio(media_path: Path, tmp_dir: Path, audio_filter: str | None) -> Path:
    """Extract media_path to a 16 kHz mono WAV inside tmp_dir, for Whisper and diarization alike.

    Returns the path to the WAV. Whisper resamples to 16 kHz mono anyway, so
    emitting that here costs nothing and lets ffmpeg do any requested filtering
    in the same pass.
    """
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("ffmpeg not found on PATH; cannot extract audio.")

    cleaned = tmp_dir / (media_path.stem + ".clean.wav")
    subprocess.run(build_ffmpeg_args(media_path, cleaned, audio_filter), check=True, capture_output=True)
    return cleaned
```

Then in `main()` (currently around lines 229-253), change:

```python
    do_preprocess = args.preprocess or args.denoise or args.audio_filter is not None
    audio_filter = args.audio_filter or build_audio_filter(args.denoise)
```
to:
```python
    do_preprocess = args.preprocess or args.denoise or args.audio_filter is not None
    audio_filter = args.audio_filter or (build_audio_filter(args.denoise) if do_preprocess else None)
```

and change the per-file loop body from:
```python
                source = media_path
                if do_preprocess:
                    source = preprocess_audio(media_path, tmp_dir, audio_filter)
```
to:
```python
                source = preprocess_audio(media_path, tmp_dir, audio_filter)
```

Also update the `if do_preprocess:` print-guard above the loop — it should still only print the filter message when a filter chain is actually being applied, so leave that `if do_preprocess:` print block as-is; it's independent of extraction now being unconditional.

- [x] **Step 4: Run it, confirm it passes**

Run: `uv run pytest tests/test_transcribe.py -v`
Expected: PASS (8 tests total).

- [x] **Step 5: Commit**

```bash
git add src/transcribe.py tests/test_transcribe.py
git commit -m "refactor: make audio extraction unconditional (needed by diarization)"
```

---

### Task 5: `word_timestamps=True` in `run_whisper`, plus `extract_words`

**Files:**
- Modify: `src/transcribe.py:123-152` (`run_whisper`)
- Test: `tests/test_transcribe.py`

**Interfaces:**
- Modifies: `run_whisper(media_path, *, model_repo, language="en", initial_prompt=None, verbose=None, word_timestamps=True) -> dict` — new keyword-only param, defaults to `True`.
- Produces: `extract_words(result: dict) -> list[dict]` — flattens `result["segments"][*]["words"]` into one chronological list, for feeding into `align_words_to_speakers`.

- [x] **Step 1: Write the failing tests**

Append to `tests/test_transcribe.py`:

```python
from pathlib import Path
from unittest.mock import patch

from src.transcribe import extract_words, run_whisper


def test_run_whisper_requests_word_timestamps_by_default(tmp_path):
    media = tmp_path / "clip.wav"
    media.write_bytes(b"fake wav bytes")
    fake_result = {"text": "hi", "segments": [], "language": "en"}

    with patch("src.transcribe.mlx_whisper.transcribe", return_value=fake_result) as mock_transcribe:
        result = run_whisper(media, model_repo="mlx-community/whisper-large-v3-turbo")

    assert result == fake_result
    assert mock_transcribe.call_args.kwargs["word_timestamps"] is True


def test_run_whisper_word_timestamps_can_be_disabled(tmp_path):
    media = tmp_path / "clip.wav"
    media.write_bytes(b"fake wav bytes")

    with patch("src.transcribe.mlx_whisper.transcribe", return_value={}) as mock_transcribe:
        run_whisper(media, model_repo="turbo", word_timestamps=False)

    assert mock_transcribe.call_args.kwargs["word_timestamps"] is False


def test_extract_words_flattens_segments_in_order():
    result = {
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "Hi", "words": [
                {"word": "Hi", "start": 0.0, "end": 0.5, "probability": 0.9},
            ]},
            {"start": 1.0, "end": 2.0, "text": " there", "words": [
                {"word": " there", "start": 1.0, "end": 1.5, "probability": 0.8},
            ]},
        ],
    }

    words = extract_words(result)

    assert [w["word"] for w in words] == ["Hi", " there"]
```

- [x] **Step 2: Run it, confirm it fails**

Run: `uv run pytest tests/test_transcribe.py -v`
Expected: FAIL — `test_run_whisper_requests_word_timestamps_by_default` fails because `mock_transcribe.call_args.kwargs` has no `"word_timestamps"` key (current `run_whisper` doesn't pass it); `extract_words` fails to import.

- [x] **Step 3: Implement it**

Replace `run_whisper`'s signature and body in `src/transcribe.py`:

```python
def run_whisper(
    media_path: Path,
    *,
    model_repo: str,
    language: str | None = "en",
    initial_prompt: str | None = None,
    verbose: bool | None = None,
    word_timestamps: bool = True,
) -> dict:
    """Transcribe one audio/video file on the Apple GPU and return the result dict.

    word_timestamps defaults to True: word-level timing is what lets
    align_words_to_speakers attribute speaker identity per-word instead of
    per multi-second segment.
    """
    if not media_path.exists():
        raise FileNotFoundError(f"Input file not found: {media_path}")

    decode_options: dict = {}
    if language is not None:
        decode_options["language"] = language
    return mlx_whisper.transcribe(
        str(media_path),
        path_or_hf_repo=model_repo,
        initial_prompt=initial_prompt,
        verbose=verbose,
        word_timestamps=word_timestamps,
        **decode_options,
    )


def extract_words(result: dict) -> list[dict]:
    """Flatten a run_whisper() result's per-segment word lists into one chronological list."""
    return [word for segment in result["segments"] for word in segment["words"]]
```

- [x] **Step 4: Run it, confirm it passes**

Run: `uv run pytest tests/test_transcribe.py -v`
Expected: PASS (11 tests total).

- [x] **Step 5: Commit**

```bash
git add src/transcribe.py tests/test_transcribe.py
git commit -m "feat: request word-level timestamps by default, add extract_words"
```

---

### Task 6: `run_diarization` — pyannote wrapper with injectable pipeline

**Files:**
- Modify: `src/transcribe.py` (new imports + function)
- Test: `tests/test_transcribe.py`

**Interfaces:**
- Produces:
  - `load_diarization_pipeline(hf_token: str | None) -> Pipeline` — loads `pyannote/speaker-diarization-3.1`, attempts to move it to MPS, falls back to CPU silently on failure (per the spec's MPS-op-coverage risk).
  - `run_diarization(wav_path: Path, pipeline, *, num_speakers: int | None = None) -> tuple[list[dict], dict[str, "numpy.ndarray"]]` — `pipeline` is injected (any callable matching pyannote's `Pipeline.__call__` signature) so this is unit-testable without loading real model weights. Returns `(turns, embeddings)`:
    - `turns`: list of `{"start": float, "end": float, "speaker": str}` sorted by `start`.
    - `embeddings`: `{speaker_id: numpy.ndarray}`, one row per `sorted(pipeline_output.labels())`.

- [x] **Step 1: Write the failing test**

Append to `tests/test_transcribe.py`:

```python
import numpy as np
import pytest

from src.transcribe import load_diarization_pipeline, run_diarization


class _FakeTurn:
    def __init__(self, start, end):
        self.start = start
        self.end = end


class _FakeDiarization:
    def __init__(self, tracks):
        self._tracks = tracks  # list of (start, end, label)

    def itertracks(self, yield_label=True):
        for start, end, label in self._tracks:
            yield _FakeTurn(start, end), None, label

    def labels(self):
        return sorted({label for _, _, label in self._tracks})


def test_run_diarization_parses_pipeline_output_sorted_by_start():
    tracks = [(5.0, 9.0, "SPEAKER_00"), (0.0, 5.0, "SPEAKER_01")]
    embeddings_array = np.array([[1.0, 0.0], [0.0, 1.0]])  # row 0 -> SPEAKER_00, row 1 -> SPEAKER_01 (sorted order)

    def fake_pipeline(path, **kwargs):
        assert kwargs == {"return_embeddings": True, "num_speakers": 2}
        return _FakeDiarization(tracks), embeddings_array

    turns, embeddings = run_diarization(Path("fake.wav"), fake_pipeline, num_speakers=2)

    assert turns == [
        {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_01"},
        {"start": 5.0, "end": 9.0, "speaker": "SPEAKER_00"},
    ]
    assert set(embeddings.keys()) == {"SPEAKER_00", "SPEAKER_01"}
    assert np.array_equal(embeddings["SPEAKER_00"], np.array([1.0, 0.0]))
    assert np.array_equal(embeddings["SPEAKER_01"], np.array([0.0, 1.0]))


def test_run_diarization_omits_num_speakers_when_not_given():
    def fake_pipeline(path, **kwargs):
        assert kwargs == {"return_embeddings": True}
        return _FakeDiarization([(0.0, 1.0, "SPEAKER_00")]), np.array([[1.0]])

    run_diarization(Path("fake.wav"), fake_pipeline, num_speakers=None)


def test_run_diarization_warns_on_speaker_count_mismatch(capsys):
    def fake_pipeline(path, **kwargs):
        return _FakeDiarization([(0.0, 1.0, "SPEAKER_00")]), np.array([[1.0]])

    run_diarization(Path("fake.wav"), fake_pipeline, num_speakers=6)

    assert "warning" in capsys.readouterr().err.lower()


def test_load_diarization_pipeline_raises_clear_error_without_token():
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        load_diarization_pipeline(None)
```

**Note:** if Task 1 Step 5's real-world check found `embeddings` rows are in a *different* order than `sorted(labels())` (e.g. insertion order), update `run_diarization`'s implementation below accordingly before writing it, and adjust this test's `embeddings_array`/expectations to match reality rather than the assumption.

- [x] **Step 2: Run it, confirm it fails**

Run: `uv run pytest tests/test_transcribe.py -v`
Expected: FAIL with `ImportError: cannot import name 'run_diarization'`.

- [x] **Step 3: Implement it**

Add near the top of `src/transcribe.py`, with the other imports:

```python
import numpy as np
```

Add after `run_whisper`/`extract_words`:

```python
def load_diarization_pipeline(hf_token: str | None):
    """Load the pretrained pyannote diarization pipeline, preferring the Apple GPU (MPS)."""
    from pyannote.audio import Pipeline

    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN is not set. Add it to the project's .env file (HF_TOKEN=...) and "
            "make sure you've accepted the pyannote/speaker-diarization-3.1 model terms "
            "at https://huggingface.co/pyannote/speaker-diarization-3.1 -- see the design "
            "spec (docs/superpowers/specs/2026-07-29-speaker-diarization-fusion-design.md, "
            "section 8) for details."
        )
    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=hf_token)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load the pyannote diarization pipeline. This usually means either "
            "HF_TOKEN is invalid/expired, or the pyannote/speaker-diarization-3.1 model "
            "terms haven't been accepted yet at "
            "https://huggingface.co/pyannote/speaker-diarization-3.1. "
            f"Underlying error: {exc}"
        ) from exc
    try:
        import torch
        pipeline.to(torch.device("mps"))
    except Exception:
        print("warning: could not move diarization pipeline to MPS; falling back to CPU", file=sys.stderr)
    return pipeline


def run_diarization(
    wav_path: Path, pipeline, *, num_speakers: int | None = None
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Run a (possibly fake, for testing) pyannote-style pipeline and parse its output.

    Returns (turns, embeddings): turns sorted by start; embeddings keyed by
    speaker id, one row per sorted(diarization.labels()).
    """
    kwargs: dict = {"return_embeddings": True}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    diarization, embeddings_array = pipeline(str(wav_path), **kwargs)

    turns = [
        {"start": float(turn.start), "end": float(turn.end), "speaker": speaker}
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]
    turns.sort(key=lambda t: t["start"])

    speaker_labels = sorted(diarization.labels())
    embeddings = {label: embeddings_array[i] for i, label in enumerate(speaker_labels)}

    if num_speakers is not None and len(speaker_labels) != num_speakers:
        print(
            f"warning: requested num_speakers={num_speakers} but diarization found "
            f"{len(speaker_labels)} speaker(s) in '{wav_path.name}'",
            file=sys.stderr,
        )

    return turns, embeddings
```

- [x] **Step 4: Run it, confirm it passes**

Run: `uv run pytest tests/test_transcribe.py -v`
Expected: PASS (15 tests total).

- [x] **Step 5: Commit**

```bash
git add src/transcribe.py tests/test_transcribe.py
git commit -m "feat: add run_diarization (pyannote wrapper, injectable pipeline for tests)"
```

---

### Task 7: Wire the single-file pipeline end-to-end + validate on one real file

**Files:**
- Modify: `src/transcribe.py:166-278` (`main`)

**Interfaces:**
- Consumes: everything from Tasks 2-6 (`align_words_to_speakers`, `group_into_turns`, `render_markdown`, `run_diarization`, `load_diarization_pipeline`, `extract_words`).
- Produces: the updated CLI — `--num-speakers` flag, `.md` output instead of `.txt`.

No new unit test here — this task wires already-tested pure functions together and its correctness is validated by actually running it, per the spec's manual-validation approach (there's no automated ground truth for "is this the right speaker").

- [x] **Step 1: Add the `--num-speakers` argument**

In `main()`'s `argparse` setup, after the `--verbose` argument:

```python
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Exact speaker count for diarization, if known (improves clustering accuracy). "
             "Default: auto-detect.",
    )
```

- [x] **Step 2: Load the diarization pipeline once per run**

Right after `model_repo = resolve_model_repo(args.model)` in `main()`, add:

```python
    load_dotenv()
    diarization_pipeline = load_diarization_pipeline(os.environ.get("HF_TOKEN"))
```

Add the needed imports at the top of `src/transcribe.py`:
```python
import os
from dotenv import load_dotenv
```

- [x] **Step 3: Replace the per-file transcribe-and-write body**

Replace this block inside the `for media_path in file_iter:` loop:
```python
                print(f"Transcribing '{media_path.name}' ...")
                result = run_whisper(
                    source,
                    model_repo=model_repo,
                    language=None if args.language == "auto" else args.language,
                    initial_prompt=args.prompt,
                    verbose=True if args.verbose else None,
                )
```
```python
            except FileNotFoundError as exc:
                print(f"error: {exc}", file=sys.stderr)
                failures += 1
                continue
            except subprocess.CalledProcessError as exc:
                stderr = exc.stderr.decode(errors="replace") if exc.stderr else ""
                print(f"error: ffmpeg preprocessing failed for '{media_path.name}':\n{stderr}",
                      file=sys.stderr)
                failures += 1
                continue
            # Name the output after the original file, not the temp cleaned WAV.
            out_path = output_dir / f"{media_path.stem}.txt"
            out_path.write_text(result["text"].strip() + "\n", encoding="utf-8")
            print(f"  -> wrote {out_path}")
```

with:

```python
                print(f"Transcribing '{media_path.name}' ...")
                result = run_whisper(
                    source,
                    model_repo=model_repo,
                    language=None if args.language == "auto" else args.language,
                    initial_prompt=args.prompt,
                    verbose=True if args.verbose else None,
                )
                print(f"Diarizing '{media_path.name}' ...")
                turns, _embeddings = run_diarization(
                    source, diarization_pipeline, num_speakers=args.num_speakers
                )
                aligned_words = align_words_to_speakers(extract_words(result), turns)
                speaker_turns = group_into_turns(aligned_words)
            except FileNotFoundError as exc:
                print(f"error: {exc}", file=sys.stderr)
                failures += 1
                continue
            except subprocess.CalledProcessError as exc:
                stderr = exc.stderr.decode(errors="replace") if exc.stderr else ""
                print(f"error: ffmpeg preprocessing failed for '{media_path.name}':\n{stderr}",
                      file=sys.stderr)
                failures += 1
                continue
            # Name the output after the original file, not the temp cleaned WAV.
            out_path = output_dir / f"{media_path.stem}.md"
            out_path.write_text(render_markdown(speaker_turns), encoding="utf-8")
            print(f"  -> wrote {out_path}")
```

- [x] **Step 4: Run the full pytest suite to confirm nothing regressed**

Run: `uv run pytest -v`
Expected: all tests still PASS (nothing in this task touched tested pure functions, only `main()`'s orchestration).

- [x] **Step 5: Validate manually on one real file**

Run against the phone recording (smaller scope than the fusion pair, good first real-world check):
```bash
uv run python src/transcribe.py "data/Tag5_29_July_2026_10903_pm.m4a" --num-speakers 6
```
Expected: completes without error, writes `output/Tag5_29_July_2026_10903_pm.md`. Open it and manually confirm:
- Headings alternate between `Person 1`..`Person 6` (not just one or two speakers) — a full 6-way split, not a degenerate clustering.
- No speaker number appears that shouldn't (e.g. `Person 7` would indicate a bug in `relabel_speakers` or a miscount).
- Spot-check 3-4 transitions by ear against the actual recording — text should belong to who's speaking at that timestamp more often than not.

This is a judgment call, not a scripted assertion — record what you observe before moving on to fusion (Task 8+), since fusion's correctness partly depends on each source's diarization already being reasonable.

- [x] **Step 6: Commit**

```bash
git add src/transcribe.py
git commit -m "feat: wire single-file diarization pipeline end-to-end, write Markdown output"
```

---

### Task 8: `find_offset` — synchronize the two recordings

**Files:**
- Create: `src/fusion.py`
- Create: `tests/test_fusion.py`

**Interfaces:**
- Produces: `find_offset(wav_a: Path, wav_b: Path, *, window_seconds: float = 0.1) -> float` — seconds to **add** to source B's timestamps to align them onto source A's timeline (`a_time == b_time + offset`).

- [x] **Step 1: Write the failing test**

Create `tests/test_fusion.py`:

```python
"""Unit tests for src/fusion.py -- dual-source sync, speaker matching, turn merging."""
import numpy as np
from scipy.io import wavfile

from src.fusion import find_offset


def test_find_offset_recovers_known_delay(tmp_path):
    rate = 1000  # low sample rate keeps this test fast; the algorithm is rate-agnostic
    rng = np.random.default_rng(0)
    world = rng.normal(0, 0.01, 6000).astype(np.float32)
    world[2000:2100] += 5.0  # a distinctive "event" both sources should pick up

    delay_seconds = 1.5
    delay_samples = int(delay_seconds * rate)
    samples_a = world  # source A's clock starts at world-time 0
    samples_b = world[delay_samples:]  # source B started recording 1.5s later

    wav_a = tmp_path / "a.wav"
    wav_b = tmp_path / "b.wav"
    wavfile.write(wav_a, rate, samples_a)
    wavfile.write(wav_b, rate, samples_b)

    offset = find_offset(wav_a, wav_b)

    assert abs(offset - delay_seconds) < 0.15  # within one default window_seconds
```

- [x] **Step 2: Run it, confirm it fails**

Run: `uv run pytest tests/test_fusion.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.fusion'`.

- [x] **Step 3: Implement it**

Create `src/fusion.py`:

```python
"""Fuse two recordings of the same meeting into one speaker-attributed transcript.

Reuses transcribe.py's per-source pipeline (ASR + diarization + word-level
speaker alignment), then synchronizes the two sources' timelines, matches
their independently-clustered speaker identities to each other, and picks
the clearer source's text per overlapping turn.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile


def _rms_envelope(samples: np.ndarray, sample_rate: int, window_seconds: float) -> np.ndarray:
    """Windowed RMS energy envelope -- correlating on this is faster and more robust
    to speech-content differences between the two mics than correlating raw samples."""
    window = max(1, int(sample_rate * window_seconds))
    usable_length = len(samples) - (len(samples) % window)
    reshaped = samples[:usable_length].reshape(-1, window)
    return np.sqrt(np.mean(reshaped.astype(np.float64) ** 2, axis=1))


def find_offset(wav_a: Path, wav_b: Path, *, window_seconds: float = 0.1) -> float:
    """Seconds to ADD to source B's timestamps to align them onto source A's timeline."""
    rate_a, samples_a = wavfile.read(wav_a)
    rate_b, samples_b = wavfile.read(wav_b)
    if rate_a != rate_b:
        raise ValueError(f"sample rate mismatch between sources: {rate_a} vs {rate_b}")

    envelope_a = _rms_envelope(samples_a, rate_a, window_seconds)
    envelope_b = _rms_envelope(samples_b, rate_b, window_seconds)

    correlation = signal.correlate(envelope_a, envelope_b, mode="full", method="fft")
    lag_index = int(np.argmax(correlation)) - (len(envelope_b) - 1)
    return lag_index * window_seconds
```

- [x] **Step 4: Run it, confirm it passes**

Run: `uv run pytest tests/test_fusion.py -v`
Expected: PASS. If the sign is inverted (offset comes out as `-1.5`), that means the correlate/lag convention is flipped from what's assumed here — fix by negating `lag_index` in the return line, then re-run until it passes. This is the one piece of this task where empirically confirming the sign via the test *is* the implementation step, not a shortcut around it.

- [x] **Step 5: Commit**

```bash
git add src/fusion.py tests/test_fusion.py
git commit -m "feat: add find_offset for dual-source timeline synchronization"
```

---

### Task 9: `match_speakers` — cross-source speaker identity matching

**Files:**
- Modify: `src/fusion.py`
- Test: `tests/test_fusion.py`

**Interfaces:**
- Produces: `match_speakers(embeddings_a: dict[str, np.ndarray], embeddings_b: dict[str, np.ndarray]) -> dict[str, str]` — maps each of source A's local speaker ids to the corresponding source B local speaker id (the closest 1:1 assignment by cosine similarity, via the Hungarian algorithm).

- [x] **Step 1: Write the failing test**

Append to `tests/test_fusion.py`:

```python
from src.fusion import match_speakers


def _unit_vector(angle_degrees: float) -> np.ndarray:
    radians = np.radians(angle_degrees)
    return np.array([np.cos(radians), np.sin(radians)])


def test_match_speakers_recovers_permutation_across_relabeled_sources():
    embeddings_a = {
        "SPEAKER_00": _unit_vector(0),
        "SPEAKER_01": _unit_vector(90),
        "SPEAKER_02": _unit_vector(200),
    }
    # same 3 physical speakers, different local labels, slightly perturbed angles
    embeddings_b = {
        "SPK_B": _unit_vector(92),
        "SPK_A": _unit_vector(2),
        "SPK_C": _unit_vector(198),
    }

    mapping = match_speakers(embeddings_a, embeddings_b)

    assert mapping == {"SPEAKER_00": "SPK_A", "SPEAKER_01": "SPK_B", "SPEAKER_02": "SPK_C"}
```

- [x] **Step 2: Run it, confirm it fails**

Run: `uv run pytest tests/test_fusion.py -v`
Expected: FAIL with `ImportError: cannot import name 'match_speakers'`.

- [x] **Step 3: Implement it**

Add to `src/fusion.py`:

```python
from scipy.optimize import linear_sum_assignment


def match_speakers(embeddings_a: dict[str, np.ndarray], embeddings_b: dict[str, np.ndarray]) -> dict[str, str]:
    """Match source A's speaker embeddings to source B's via the Hungarian algorithm.

    Greedy nearest-match can be led astray when two voices are close together
    (assigning both of B's closest speakers to the same A speaker, then being
    forced into a bad leftover pairing); the Hungarian algorithm finds the
    globally optimal one-to-one assignment instead.
    """
    labels_a = list(embeddings_a.keys())
    labels_b = list(embeddings_b.keys())
    matrix_a = np.stack([embeddings_a[label] for label in labels_a])
    matrix_b = np.stack([embeddings_b[label] for label in labels_b])

    normalized_a = matrix_a / np.linalg.norm(matrix_a, axis=1, keepdims=True)
    normalized_b = matrix_b / np.linalg.norm(matrix_b, axis=1, keepdims=True)
    similarity = normalized_a @ normalized_b.T
    cost = 1.0 - similarity

    row_indices, col_indices = linear_sum_assignment(cost)
    return {labels_a[row]: labels_b[col] for row, col in zip(row_indices, col_indices)}
```

- [x] **Step 4: Run it, confirm it passes**

Run: `uv run pytest tests/test_fusion.py -v`
Expected: PASS (2 tests total).

- [x] **Step 5: Commit**

```bash
git add src/fusion.py tests/test_fusion.py
git commit -m "feat: add match_speakers (Hungarian-algorithm cross-source speaker matching)"
```

---

### Task 10: `merge_turns` — confidence-based turn selection across sources

**Files:**
- Modify: `src/fusion.py`
- Test: `tests/test_fusion.py`

**Interfaces:**
- Produces:
  - `_shift_and_remap(turns: list[dict], offset: float, speaker_map: dict[str, str]) -> list[dict]` — shifts `start`/`end` by `offset` and renames `"speaker"` via `speaker_map` (source B's local id → source A's local id namespace).
  - `merge_turns(turns_a: list[dict], turns_b_shifted: list[dict]) -> list[dict]` — both inputs are `_group_consecutive`-shaped (un-relabeled, with `"confidence"`), `turns_b_shifted` already shifted/remapped into A's namespace via `_shift_and_remap`. Source A's turns define the canonical turn boundaries; a turn is replaced by B's overlapping text only when B's confidence is strictly higher. Any B turn that doesn't overlap *any* A turn (A missed that speaker/moment entirely) is appended and the result re-sorted by `start`.

- [x] **Step 1: Write the failing test**

Append to `tests/test_fusion.py`:

```python
from src.fusion import _shift_and_remap, merge_turns


def test_shift_and_remap_applies_offset_and_speaker_map():
    turns = [{"speaker": "SPEAKER_00", "start": 1.0, "end": 2.0, "text": "hi", "confidence": 0.5}]

    result = _shift_and_remap(turns, offset=10.0, speaker_map={"SPEAKER_00": "SPEAKER_01"})

    assert result == [{"speaker": "SPEAKER_01", "start": 11.0, "end": 12.0, "text": "hi", "confidence": 0.5}]


def test_merge_turns_prefers_higher_confidence_source_and_appends_gaps():
    turns_a = [
        {"speaker": "A0", "start": 0.0, "end": 5.0, "text": "hello from a", "confidence": 0.5},
        {"speaker": "A1", "start": 5.0, "end": 9.0, "text": "garbled b word", "confidence": 0.3},
    ]
    turns_b_shifted = [
        # overlaps turn 2, HIGHER confidence -> should replace turn 2's text
        {"speaker": "A1", "start": 5.1, "end": 8.9, "text": "clear phone audio", "confidence": 0.9},
        # overlaps turn 1, LOWER confidence -> must NOT replace turn 1's text
        {"speaker": "A0", "start": 0.2, "end": 4.8, "text": "quieter mic", "confidence": 0.2},
        # doesn't overlap any A turn at all -> appended as a gap-fill
        {"speaker": "A2", "start": 9.0, "end": 11.0, "text": "only caught by phone", "confidence": 0.8},
    ]

    merged = merge_turns(turns_a, turns_b_shifted)

    assert merged == [
        {"speaker": "A0", "start": 0.0, "end": 5.0, "text": "hello from a", "confidence": 0.5},
        {"speaker": "A1", "start": 5.0, "end": 9.0, "text": "clear phone audio", "confidence": 0.9},
        {"speaker": "A2", "start": 9.0, "end": 11.0, "text": "only caught by phone", "confidence": 0.8},
    ]
```

- [x] **Step 2: Run it, confirm it fails**

Run: `uv run pytest tests/test_fusion.py -v`
Expected: FAIL with `ImportError` for `_shift_and_remap` and `merge_turns`.

- [x] **Step 3: Implement it**

Add to `src/fusion.py` (needs `overlap_seconds` — import it rather than duplicating the logic already written in `transcribe.py`):

```python
from src.transcribe import overlap_seconds


def _shift_and_remap(turns: list[dict], offset: float, speaker_map: dict[str, str]) -> list[dict]:
    """Move turns from source B's local clock/speaker-id namespace onto source A's."""
    return [
        {**turn, "start": turn["start"] + offset, "end": turn["end"] + offset, "speaker": speaker_map[turn["speaker"]]}
        for turn in turns
    ]


def merge_turns(turns_a: list[dict], turns_b_shifted: list[dict]) -> list[dict]:
    """Merge two sources' turns (already sharing a timeline + speaker-id namespace).

    Source A's turns define the canonical paragraph boundaries. A turn is
    replaced by B's overlapping text only when B's confidence is strictly
    higher (selection happens at turn granularity, not per-word -- splicing
    two independently-run ASR passes word-by-word risks garbled sentences
    where the two passes segment speech slightly differently).
    """
    merged = []
    for turn in turns_a:
        overlapping_b = [
            b for b in turns_b_shifted
            if b["speaker"] == turn["speaker"]
            and overlap_seconds(turn["start"], turn["end"], b["start"], b["end"]) > 0
        ]
        if overlapping_b:
            best_b = max(overlapping_b, key=lambda b: b["confidence"])
            if best_b["confidence"] > turn["confidence"]:
                merged.append({**turn, "text": best_b["text"], "confidence": best_b["confidence"]})
                continue
        merged.append(turn)

    for turn in turns_b_shifted:
        overlaps_any_a = any(
            overlap_seconds(turn["start"], turn["end"], a["start"], a["end"]) > 0 for a in turns_a
        )
        if not overlaps_any_a:
            merged.append(turn)

    merged.sort(key=lambda t: t["start"])
    return merged
```

- [x] **Step 4: Run it, confirm it passes**

Run: `uv run pytest -v`
Expected: all tests PASS, including the full suite from earlier tasks.

- [x] **Step 5: Commit**

```bash
git add src/transcribe.py src/fusion.py tests/test_transcribe.py tests/test_fusion.py
git commit -m "feat: add merge_turns for confidence-based cross-source turn selection"
```

---

### Task 11: `run_fusion` orchestrator + `--fuse` CLI wiring

**Files:**
- Modify: `src/fusion.py`
- Modify: `src/transcribe.py` (CLI wiring)

**Interfaces:**
- Produces: `run_fusion(primary_path: Path, secondary_path: Path, *, model_repo: str, language: str | None, initial_prompt: str | None, num_speakers: int | None, output_dir: Path, diarization_pipeline) -> Path` — runs the full pipeline on both sources and writes `output_dir / f"{primary_path.stem}.md"`, returning that path.

No new unit test — this task is pure orchestration of already-tested pieces (Tasks 2-3, 5-6, 8-10). Its correctness is checked via the real-recording validation in Tasks 12-13.

- [x] **Step 1: Implement `run_fusion` in `src/fusion.py`**

Add:

```python
from src.transcribe import (
    align_words_to_speakers,
    _group_consecutive,
    extract_words,
    preprocess_audio,
    relabel_speakers,
    render_markdown,
    run_diarization,
    run_whisper,
)


def _process_source(media_path: Path, tmp_dir: Path, *, model_repo, language, initial_prompt, num_speakers, diarization_pipeline):
    wav_path = preprocess_audio(media_path, tmp_dir, None)
    result = run_whisper(wav_path, model_repo=model_repo, language=language, initial_prompt=initial_prompt)
    words = extract_words(result)
    turns, embeddings = run_diarization(wav_path, diarization_pipeline, num_speakers=num_speakers)
    aligned = align_words_to_speakers(words, turns)
    return wav_path, _group_consecutive(aligned), embeddings


def run_fusion(
    primary_path: Path,
    secondary_path: Path,
    *,
    model_repo: str,
    language: str | None,
    initial_prompt: str | None,
    num_speakers: int | None,
    output_dir: Path,
    diarization_pipeline,
) -> Path:
    """Fuse two recordings of the same meeting into one Person-N Markdown transcript."""
    import tempfile

    with tempfile.TemporaryDirectory(prefix="whisper_fuse_") as tmp:
        tmp_dir = Path(tmp)
        wav_a, turns_a, embeddings_a = _process_source(
            primary_path, tmp_dir, model_repo=model_repo, language=language,
            initial_prompt=initial_prompt, num_speakers=num_speakers,
            diarization_pipeline=diarization_pipeline,
        )
        wav_b, turns_b, embeddings_b = _process_source(
            secondary_path, tmp_dir, model_repo=model_repo, language=language,
            initial_prompt=initial_prompt, num_speakers=num_speakers,
            diarization_pipeline=diarization_pipeline,
        )

        offset = find_offset(wav_a, wav_b)
        speaker_map_a_to_b = match_speakers(embeddings_a, embeddings_b)
        speaker_map_b_to_a = {b: a for a, b in speaker_map_a_to_b.items()}
        turns_b_shifted = _shift_and_remap(turns_b, offset, speaker_map_b_to_a)

        merged = merge_turns(turns_a, turns_b_shifted)
        speaker_turns = relabel_speakers(merged)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{primary_path.stem}.md"
    out_path.write_text(render_markdown(speaker_turns), encoding="utf-8")
    return out_path
```

- [x] **Step 2: Wire `--fuse` into `transcribe.py`'s CLI**

In `main()`'s `argparse` setup, after `--num-speakers`:

```python
    parser.add_argument(
        "--fuse",
        type=Path,
        default=None,
        help="A second recording of the SAME meeting (e.g. a phone recording from a "
             "different position). Triggers dual-source fusion: 'media' must be a "
             "single file, not a directory.",
    )
```

Right after `media_files = gather_media(args.media)` and its empty-check, add the fusion branch (before the normal batch loop):

```python
    if args.fuse is not None:
        if args.media is None or args.media.is_dir():
            print("error: --fuse requires 'media' to be a single file, not a directory/default batch", file=sys.stderr)
            return 1
        load_dotenv()
        diarization_pipeline = load_diarization_pipeline(os.environ.get("HF_TOKEN"))
        output_dir = (args.output_dir or DEFAULT_OUTPUT_DIR).resolve()
        out_path = run_fusion(
            args.media,
            args.fuse,
            model_repo=resolve_model_repo(args.model),
            language=None if args.language == "auto" else args.language,
            initial_prompt=args.prompt,
            num_speakers=args.num_speakers,
            output_dir=output_dir,
            diarization_pipeline=diarization_pipeline,
        )
        print(f"Fused transcript written to {out_path}")
        return 0
```

Add the import at the top of `src/transcribe.py`:
```python
from src.fusion import run_fusion
```

- [x] **Step 3: Run the full pytest suite**

Run: `uv run pytest -v`
Expected: all tests still PASS (this task only added orchestration and CLI wiring around already-tested pieces).

- [x] **Step 4: Commit**

```bash
git add src/fusion.py src/transcribe.py
git commit -m "feat: add run_fusion orchestrator and --fuse CLI flag"
```

---

### Task 12: Manual validation — offset and speaker matching on the real recordings

**Files:** none (validation only)

- [x] **Step 1: Check the sync offset is plausible**

```bash
uv run python -c "
from pathlib import Path
import tempfile
from src.transcribe import preprocess_audio
from src.fusion import find_offset

with tempfile.TemporaryDirectory() as tmp:
    tmp_dir = Path(tmp)
    wav_a = preprocess_audio(Path('data/Meeting with Michael Kingston-20260729_130839-Meeting Recording.mp4'), tmp_dir, None)
    wav_b = preprocess_audio(Path('data/Tag5_29_July_2026_10903_pm.m4a'), tmp_dir, None)
    print('offset (seconds):', find_offset(wav_a, wav_b))
"
```
Expected: a single plausible number (could be negative if the phone recording actually started *before* the Teams recording — that's fine, it just means source B's clock-zero comes earlier). Sanity-check it by ear: find a recognizable moment early in both recordings (e.g. someone saying a name) and confirm the reported offset roughly matches the real gap between when that moment occurs in each file.

**If this offset looks implausible** (e.g. wildly larger than the actual room setup would allow, or the sanity-check moment doesn't line up when shifted by this amount): the `_rms_envelope`-based correlation may be too coarse for these two mic positions/qualities. This is the clock-drift/sync risk flagged in the spec (§10) — don't force a plausible-looking number if it demonstrably isn't; note the discrepancy and treat single-offset sync as not viable for this file pair before proceeding further, rather than pushing ahead with fusion on a broken alignment.

- [x] **Step 2: Check the speaker matching is plausible**

```bash
uv run python -c "
from pathlib import Path
import os, tempfile
from dotenv import load_dotenv
load_dotenv()
from src.transcribe import preprocess_audio, run_diarization, load_diarization_pipeline
from src.fusion import match_speakers

pipeline = load_diarization_pipeline(os.environ.get('HF_TOKEN'))
with tempfile.TemporaryDirectory() as tmp:
    tmp_dir = Path(tmp)
    wav_a = preprocess_audio(Path('data/Meeting with Michael Kingston-20260729_130839-Meeting Recording.mp4'), tmp_dir, None)
    wav_b = preprocess_audio(Path('data/Tag5_29_July_2026_10903_pm.m4a'), tmp_dir, None)
    _, emb_a = run_diarization(wav_a, pipeline, num_speakers=6)
    _, emb_b = run_diarization(wav_b, pipeline, num_speakers=6)
    print(match_speakers(emb_a, emb_b))
"
```
Expected: a 1:1 mapping covering all 6 speakers from each source. Spot-check 2-3 of the matched pairs by ear: find a moment where that speaker talks in source A, and confirm the same voice is what's attributed to their matched label in source B at the corresponding (offset-shifted) time.

- [x] **Step 3: Record what you observed**

No commit for this task — it's a validation checkpoint. If either check looks wrong, stop here and reconsider before running the full fusion in Task 13 (garbage sync/matching will produce a garbled final transcript no amount of downstream logic can fix).

---

### Task 13: Full fusion run + acceptance spot-check

**Files:** none (validation only)

- [x] **Step 1: Run the full fusion pipeline on the real recordings**

```bash
uv run python src/transcribe.py "data/Meeting with Michael Kingston-20260729_130839-Meeting Recording.mp4" --fuse "data/Tag5_29_July_2026_10903_pm.m4a" --num-speakers 6
```
Expected: completes without error, writes `output/Meeting with Michael Kingston-20260729_130839-Meeting Recording.md`.

- [x] **Step 2: Spot-check against the spec's acceptance criteria**

Open the output and check, per the spec's §2 success criteria and §11 acceptance step:
- Several speaker transitions, checked by ear against the actual recording(s) — is the attributed speaker correct more often than not?
- Samples from the start, middle, and end of the ~70-minute meeting — does speaker identity stay stable (no `Person N` renumbering drift partway through)?
- All 6 people appear as distinct `Person N` labels somewhere in the document (not collapsed into fewer than 6, and not split into more than 6).

- [x] **Step 3: Record the outcome**

No commit for this task (no code changes) — this is the spec's final acceptance check. If it passes, the feature is functionally done and ready for `/review` → `/verify` per the outer workflow. If it doesn't hold up, decide with the user whether to iterate on `merge_turns`'s selection heuristic, the sync/matching steps from Task 12, or scale back scope — don't silently ship a transcript that doesn't meet the stated bar.

---

## Amendment log

**2026-07-29 — pyannote.audio API correction (pre-implementation).** Task 1's dependency
install resolved `pyannote.audio` to 4.0.7, not the 3.x line the model-card examples in this
plan were written against. `Pipeline.from_pretrained()` takes `token=`, not `use_auth_token=`.
All three occurrences in this plan were corrected before Task 6 built against them
(commit `3c1ac3c`).

**2026-07-30 — pyannote 4.x return-type correction (during live validation).** A second,
larger API break surfaced only once a real (non-mocked) pipeline ran: `return_embeddings=True`
is silently ignored in 4.0.7 (it emits a `UserWarning`), and the pipeline call returns a
`DiarizeOutput` dataclass (`.speaker_diarization`, `.speaker_embeddings`) rather than the
`(Annotation, embeddings)` tuple Task 6's plan code assumed. Every unit test had mocked the
old tuple contract, so this was invisible until Task 1's Step 5 ran against real audio.
`run_diarization` and its tests were corrected (commit `2853396`). This also retroactively
confirmed Task 6's carried-over open risk about embeddings-row ordering — pyannote's own
source documents the rows as sorted in `speaker_diarization.labels()` order, matching what
the code assumed.

**2026-07-30 — direct-script invocation fix (not in the original plan).** Task 11's
`from src.fusion import run_fusion` local import (added to break a `transcribe.py` ↔
`fusion.py` circular import) crashed with `ModuleNotFoundError` under the documented
invocation `uv run python src/transcribe.py ... --fuse ...`, because running a script
directly puts only `src/` on `sys.path`, not the project root. Fixed with a `__package__`-
guarded `sys.path` insert (commit `8c54428`); a subprocess-based regression test was added
later during `/review` (commit `13c7860`), since an in-process test would pass regardless of
the fix.

**2026-07-30 — `merge_turns` duplication fix (parked risk became real).** Task 10's review
flagged, and parked as hypothetical, that a single B turn spanning two same-speaker A turns
could have its text duplicated into both. Task 13's first real run reproduced it exactly.
Fixed with a used-B-turn set so each B turn replaces at most one A turn (commit `98b8fda`).

**2026-07-31 — `/review` findings (2 Critical, 4 Important) fixed before completion**
(commit `13c7860`). None were foreseen in this plan:
- `run_fusion` gave both sources the same temp directory; two inputs sharing a filename stem
  would silently overwrite each other's preprocessed WAV, corrupting `find_offset` into
  comparing a file against itself (a plausible-looking 0.0 offset). Now one subdirectory each.
- `match_speakers`'s speaker-count-mismatch `ValueError` — an expected outcome without
  `--num-speakers`, not a corner case — was unhandled in `--fuse`, surfacing as a raw
  traceback *after* both sources' full ASR + diarization passes. Now a clean error.
- `merge_turns`'s gap-fill check ignored speaker identity, silently dropping B turns that
  overlapped a *different* speaker's A turn. Fixing this recovered 160 turns of real speech
  on the target recordings (554 → 714 blocks) — the single highest-impact fix in the review.
- Negative sync offsets could render timestamps as `-1:59:59`; `--fuse` had no pre-flight
  file-existence check; `load_diarization_pipeline`'s `RuntimeError` was uncaught in both
  paths. All fixed.

**2026-07-31 — regression coverage added at FINISH.** The `--preprocess`/`--denoise`/
`--audio-filter` flag wiring (previously-working behaviour that Task 4 refactored when it
made extraction unconditional) had no test guarding it. Four tests added.

**Not done — deferred with reasoning.** See `bugs.md` for the residual `merge_turns`
whole-turn-granularity redundancy, the absent confidence check on the correlation peak, and
upstream Whisper hallucination artifacts.
