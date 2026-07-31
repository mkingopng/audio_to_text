# audio_to_text as a globally callable tool — design

**Date:** 2026-07-31
**Status:** approved (brainstorm), pending implementation plan

## Goal

Make the existing transcription/fusion pipeline usable from *any* project on this machine —
invoked by Claude Code in another repo — without copying the codebase and without duplicating
its 1.2 GB dependency tree.

Today the pipeline only runs from inside its own checkout. Four things pin it there:

| Blocker | Location | Effect |
|---|---|---|
| No `[build-system]`, no `[project.scripts]` | `pyproject.toml` | Not installable; `uv` treats it as a virtual project and never puts `audio_to_text` on the venv's import path |
| `src/` has no `__init__.py` | `src/` | Script-shaped, not package-shaped — forces the `sys.path` hack |
| Bare `load_dotenv()` | `src/transcribe.py:471,511` | Searches upward from **cwd**; from another project it finds that project's `.env` or nothing, then fails with "HF_TOKEN is not set" |
| Defaults `PROJECT_ROOT/data`, `PROJECT_ROOT/output` | `src/transcribe.py:45-46` | A no-arg run from another project would scan *this* repo's `data/` |

This is a packaging change. The pipeline itself is not touched.

## Approach

Three layers.

### 1. Package the project (this repo)

**Layout.** Move `src/transcribe.py` and `src/fusion.py` into `src/audio_to_text/`, add
`__init__.py`. Standard src-layout.

This deletes the `sys.path` manipulation at `src/transcribe.py:465-469`, which exists *only*
because `src` isn't a resolvable package when the file is run directly. Under `python -m
audio_to_text.transcribe`, `__package__` is `"audio_to_text"`, so the guard's condition is
already false — the block becomes dead code rather than merely unnecessary. The
`from src.fusion import run_fusion` on line 470 becomes `from audio_to_text.fusion import
run_fusion` and moves to the top of the module with the other imports.

**pyproject.** Add `[build-system]` (hatchling) and
`[project.scripts] audio-to-text = "audio_to_text.transcribe:main"`.

The `[build-system]` block is load-bearing, not decoration: without it `uv` treats this as a
virtual project and does not install it into `.venv`, so `python -m audio_to_text.transcribe`
would not resolve. `main(argv) -> int` already has the right signature for a console script —
its return value becomes the exit code.

`[project.scripts]` is *not* what the wrapper in layer 2 calls; the wrapper uses `python -m`.
It is included because it is free once `[build-system]` exists, and it makes `uv run
audio-to-text` work inside the repo during development. Only one entry path is load-bearing —
`python -m` — and the tests cover that one.

**Token resolution.** Replace bare `load_dotenv()` with an explicit chain, first hit wins:

1. `HF_TOKEN` already present in the environment
2. `./.env` in the current working directory — lets a calling project supply its own
3. `~/.config/audio-to-text/.env` — the user-global home for the token

The "not set" message at `src/transcribe.py:302` currently says *"Add it to the project's .env
file"*. That advice is wrong once the tool runs from elsewhere; it must name the global
location. Setup copies the existing `.env` to `~/.config/audio-to-text/.env`; the repo's own
`.env` keeps working, because during development cwd *is* the repo.

**Caller-relative defaults.**

- Output directory defaults to `./data/transcriptions/` relative to cwd, created if absent.
  `PROJECT_ROOT/output` is dropped. `data/transcriptions/` is the standard project structure
  here, so the transcript lands where it will ultimately live instead of being written to the
  project root and moved by hand.
- Input defaults to `./data/` relative to cwd and errors clearly when absent, though in
  practice the source media will usually be named explicitly on the command line.
- `--output-dir` still overrides.

Nesting the output directory inside the default input directory is safe: `gather_media` uses
`Path.iterdir()` with an `is_file()` filter (`src/transcribe.py:79-83`), so it is
non-recursive and a later batch run will not descend into `data/transcriptions/`. A test pins
this, because the safety is a property of `iterdir()` that a future switch to `rglob()` would
silently destroy.

`ensure_apple_silicon()` is unchanged. The constraint is real and the tool should keep failing
fast on it.

### 2. Put it on PATH — thin wrapper

A shell shim at `~/.local/bin/audio-to-text` (already on `PATH`):

```bash
#!/usr/bin/env bash
exec uv run --project "$HOME/Developer/GitHub/audio_to_text" \
  python -m audio_to_text.transcribe "$@"
```

Reuses the existing 1.2 GB `.venv`. Edits to the repo take effect immediately with no
reinstall step.

*Rejected:* `uv tool install --editable`, which is the more idiomatic route and matches how
`graphifyy` and `zotero-mcp` are already installed here. It builds a **separate** tool venv,
duplicating ~1.2 GB of torch/pyannote/mlx for a tool that only ever runs on this one machine
against this one checkout. The disk cost outweighs the idiom.

*Known limitation, accepted:* the wrapper hard-codes the repo path, so moving or deleting the
checkout breaks the command. This is inherent to "don't copy the codebase" and is the correct
trade.

### 3. The Claude-facing skill

`~/.claude/skills/transcribe-recording/SKILL.md`, mirrored into
`Templates/user/claude/skills/` so `install-user.sh` carries it to other machines.

The skill contains **no code** — only triggers and operating knowledge:

- **Triggers:** transcribe, transcript, "turn this recording into text", media extensions
  (`.m4a`, `.mp4`, `.wav`, `.mov`), and the fusion case — "two recordings of the same meeting",
  "I recorded it on my phone as well".
- **Preconditions**, stated so Claude fails fast with a useful message rather than retrying:
  Apple Silicon only, `ffmpeg` on `PATH`, `HF_TOKEN` resolvable, gated pyannote model terms
  accepted.
- **Both invocation modes**, single-file and `--fuse`.
- **Quality levers:** `--num-speakers` when the room size is known, `--prompt` to bias spelling
  of names and jargon.
- **Run it in the background.** A one-hour meeting takes many minutes; Bash's 120 s default
  timeout would kill it partway. This trap is the single most valuable thing the skill encodes.
- **Output location** — `./data/transcriptions/` in the calling project.

## Repository split

| Change | Repo |
|---|---|
| Package layout, `pyproject.toml`, token chain, cwd defaults, tests, README | `audio_to_text` (branch off `feature/mlx-whisper-gpu`) |
| `transcribe-recording` skill | `~/.claude/skills/` + `Templates/user/claude/skills/` |
| `~/.local/bin/audio-to-text` wrapper | Neither — machine-local, documented in the README |

## Error handling

Existing handling is already sound and stays as-is: per-file `DiarizationError` isolation so a
batch run does not abort, `ffmpeg` `CalledProcessError` surfacing, and the `--fuse` branch's
catch of `ValueError`/`RuntimeError` before a traceback escapes after both full ASR passes.

Two additions:

- Token resolution failure names all three lookup locations, not just "the project's .env".
- A missing default `./data/` directory produces a clear message telling the caller to pass an
  explicit path — not a bare "no media files found" that reads as though the tool is broken.

## Testing

- Existing tests import `src.transcribe` / `src.fusion`; update to `audio_to_text.*`. This is
  mechanical but must be verified green, not assumed.
- New: token resolution chain — env beats cwd `.env` beats `~/.config`, and the failure message
  when all three miss.
- New: output defaults to `./data/transcriptions/` relative to cwd, the directory is created
  when absent, and `--output-dir` still overrides it.
- New: default input resolves to `./data/` relative to cwd.
- New: `gather_media` on a `data/` containing a `transcriptions/` subdirectory returns only the
  media files — pins the non-recursive behaviour that keeps output nested inside input safe.
- New: `python -m audio_to_text.transcribe --help` exits 0 — this is the entry path the
  wrapper actually uses, and a `pyproject.toml` packaging mistake would break it in a way no
  import-level unit test would catch.
- Manual, and the real acceptance test: from a *different* repo, transcribe a file and fuse a
  pair, with pasted command output.

## Out of scope

- **Any coupling to `meeting-minutes`.** That skill is entirely separate work. This tool's
  deliverable is the transcript; what anything downstream does with it is not this tool's
  concern and the skill must not mention it.
- **An MCP server.** The operation runs for minutes, produces a file on disk rather than a chat
  payload, and is machine-local. MCP would re-encode a CLI across a protocol boundary for no
  gain, and its tool schemas consume context in every session whether used or not. A skill
  loads on demand.
- **Non-Apple-Silicon or cloud fallback.**
- **Naming speakers** — unchanged; voice prints cannot identify people.
- **Publishing to PyPI.**

## Done criteria

1. `audio-to-text --help` runs from an arbitrary directory.
2. From a different repo, a single named file transcribes and the `.md` lands in that repo's
   `data/transcriptions/`, which is created if it did not exist.
3. From a different repo, `--fuse` produces a fused transcript in the same place.
4. `HF_TOKEN` resolves from `~/.config/audio-to-text/.env` with no `.env` in cwd.
5. `uv run pytest` passes in `audio_to_text`.
6. The `transcribe-recording` skill triggers on a natural request in an unrelated project and
   runs the tool in the background.
7. The skill is mirrored into `Templates/user/claude/skills/` and `sync.sh` reports no drift.