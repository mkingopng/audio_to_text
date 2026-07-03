# MLX Whisper GPU-accelerated transcription — design

**Date:** 2026-07-03
**Status:** Approved (design) — pending spec review
**Author:** Michael Kingston (with Claude)

## 1. Context & problem

`src/transcribe.py` currently uses `openai-whisper`. Its `load_model()` selects
`"cuda" if torch.cuda.is_available() else "cpu"` — there is **no MPS branch** — so
on this **Apple M4 (10 cores)** it silently runs on CPU even though Metal GPU
(`torch.backends.mps.is_available() == True`). The `FP16 not supported on CPU`
warning is the downstream symptom. CPU `large-v3` runs ~5× realtime (39 min →
~8 min), which is usable but leaves the GPU idle.

## 2. Goal & success criteria

Run transcription on the M4 **GPU** via Apple's MLX framework, keeping
accuracy on par with `large-v3`.

Success =
- Transcription uses the Apple GPU (Metal), not CPU.
- Wall-clock materially faster than the CPU baseline (target: 39-min file in
  ≲ 3 min vs ~8 min).
- Transcript quality is at least as good as the current `large-v3` baseline in
  `output/Crisis shield m1.txt` (spot-checked on the same file).
- Existing workflow preserved: no-arg run over `data/` → `output/`, `--preprocess`,
  `--prompt`, `--language`, progress bar.
- Output is **`.txt` only**.

## 3. Decision

**Adopt `mlx-whisper` as the sole engine (MLX-only).** No `openai-whisper`
fallback — this is a single-user tool on Apple Silicon, so a portability
abstraction is unwarranted (YAGNI).

Rejected alternatives (from brainstorming):
- **whisper.cpp** — marginally faster but heavy setup (native binaries, Core ML
  conversion).
- **faster-whisper** — does not use the Apple GPU (CTranslate2 has no Metal
  backend); would stay on CPU.
- **Force `device="mps"` on openai-whisper** — unreliable; some ops (FFT/mel) are
  not MPS-implemented and fall back to CPU.

## 4. Architecture

Keep the single-script design; introduce one thin **engine seam** so the CLI does
not embed MLX specifics.

- `run_whisper(media_path, *, model_repo, language, initial_prompt, verbose) -> dict`
  — wraps `mlx_whisper.transcribe(str(media_path), path_or_hf_repo=model_repo, ...)`
  and returns its result dict (`{"text", "segments", "language", ...}`).
  `mlx_whisper` caches the loaded model per repo, so a batch loads weights once.
- A `--model` friendly-name → HF-repo map, with pass-through for full repo ids:
  - `large-v3`        → `mlx-community/whisper-large-v3-mlx`   (default)
  - `turbo` / `large-v3-turbo` → `mlx-community/whisper-large-v3-turbo`
  - any string containing `/` → used verbatim as an HF repo id.

Unchanged units: `gather_media()`, `build_audio_filter()`, `preprocess_audio()`,
the batch loop, and the outer `tqdm` bar.

## 5. CLI surface

Unchanged: `media` (optional; default `data/`), `--model`, `--language`,
`--prompt`, `--output-dir`, `--preprocess`, `--denoise`, `--audio-filter`,
`--verbose`.

**Removed:** `--format` and all non-txt writers.

## 6. Data flow

```
media file(s)
  └─ (optional) preprocess_audio()  → ffmpeg → 16 kHz mono WAV (temp)
      └─ run_whisper()              → mlx_whisper.transcribe on M4 GPU → result dict
          └─ write .txt             → output/<original-stem>.txt  (result["text"])
```

Outputs are named after the **original** file, not the temp WAV.

## 7. Output

`.txt` only. Replace the `get_writer(...)` call (and its `# type: ignore`) with:

```python
out_path = output_dir / f"{media_path.stem}.txt"
out_path.write_text(result["text"].strip() + "\n", encoding="utf-8")
```

## 8. Dependencies

`pyproject.toml`:
- **Add** `mlx-whisper`.
- **Remove** `openai-whisper` (drops the `torch` stack — large install shrink).
- **Keep** `tqdm` (direct import for the batch bar).

`ffmpeg` (system) still required — unchanged.

## 9. Error handling

- **Platform guard:** on non-Apple-Silicon / non-macOS, fail fast with a clear
  message (`mlx-whisper requires Apple Silicon (arm64 macOS)`), since there is no
  fallback engine.
- Preserve existing per-file handling: missing file and ffmpeg-preprocess failure
  are reported and skipped, batch continues, non-zero exit if any file failed.
- Surface model-download/HF errors with a readable message (first run pulls
  weights from Hugging Face).

## 10. Testing & validation

First implementation step is **feasibility validation** before the full rewrite:
1. `uv add mlx-whisper` resolves and installs on this Python **3.13** venv.
2. `import mlx_whisper` succeeds; `help(mlx_whisper.transcribe)` confirms it
   accepts `initial_prompt`, `language`, `verbose` (parameter-name check).
3. Confirm the exact HF repo id for large-v3 (`mlx-community/whisper-large-v3-mlx`)
   resolves/downloads.
4. Transcribe a ~30 s clip end-to-end on GPU; confirm a `.txt` is written.

Acceptance:
5. Full `Crisis shield m1.m4a` run: wall-clock recorded, GPU confirmed in use,
   and the `.txt` spot-checked against the existing `large-v3` baseline for
   equal-or-better quality.

## 11. Risks & mitigations

- **`mlx-whisper` wheels for Python 3.13** — if unavailable, pin the project to
  3.12 (`uv python pin 3.12`). Caught at validation step 1.
- **Exact MLX model repo id** — confirmed at validation step 3; the `--model`
  pass-through allows overriding without code changes.
- **`verbose=None` progress-bar behavior** may differ from openai-whisper —
  verify the per-file progress display; fall back to the outer batch bar only if
  needed.
- **Accuracy parity** — validated by the step-5 baseline spot-check; if MLX
  large-v3 underperforms, `turbo` or the openai baseline remain reference points.

## 12. Out of scope (YAGNI)

- Multi-engine abstraction / `--engine` flag.
- Non-txt output formats (srt/vtt/tsv/json).
- Speaker diarization, word-level timestamp files.
- Silence-trim / VAD for the trailing-silence hallucination (tracked separately).

## 13. Cleanup

- Delete the stray `output/Crisis shield m1.{srt,vtt,tsv,json}` left by the earlier
  `--format all` run.
- Keep the `large-v3` CPU `.txt` as the accuracy baseline until MLX parity is
  confirmed, then it is simply overwritten by a normal run.