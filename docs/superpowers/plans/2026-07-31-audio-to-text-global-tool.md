# audio_to_text as a Global Tool — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the transcription/fusion pipeline callable by Claude Code from any project on this machine, without copying the codebase or duplicating its 1.2 GB dependency tree.

**Architecture:** Three layers. (1) The repo becomes a real src-layout Python package with caller-relative defaults and a token lookup that works outside its own checkout. (2) A ~10-line shim on `PATH` delegates to the repo's existing venv via `uv run --project`. (3) A global Claude skill carries the operating knowledge — preconditions, invocation modes, and the requirement to run in the background.

**Tech Stack:** Python 3.12+, uv, hatchling, pytest, MLX Whisper, pyannote-audio, bash.

## Global Constraints

- Python `>=3.12`; the project is uv-managed — always `uv run`, never bare `python`.
- Apple Silicon only. `ensure_apple_silicon()` stays exactly as it is.
- Work on branch `feature/global-tool` in `audio_to_text` (already created, spec already committed).
- The pipeline's transcription/diarization/fusion logic is **not** modified. This is a packaging change. If a task seems to require touching the algorithms, stop and ask.
- Nothing in the skill or the docs may reference the `meeting-minutes` skill. It is separate work.
- No MCP server. No PyPI publishing. No non-Apple-Silicon fallback.
- The commit gate and checkbox gate are **not** installed in this repo (`.claude/hooks/` does not exist), so there is no `verify-mark.sh` / `plan-track.sh` to run here.
- Existing working-tree noise on this branch — modified `bugs.md`, untracked `docs/validate/` and `tools/` — is unrelated. Do not commit it. Always `git add` explicit paths, never `git add -A`.

---

### Task 1: Convert to an installable src-layout package

The single highest-risk task: it moves every source file and re-points all 40 existing tests. Do it as one commit so the repo is never half-moved.

**Files:**
- Create: `src/audio_to_text/__init__.py`
- Move: `src/transcribe.py` → `src/audio_to_text/transcribe.py`
- Move: `src/fusion.py` → `src/audio_to_text/fusion.py`
- Modify: `pyproject.toml` (add `[build-system]`, `[project.scripts]`, hatch wheel target)
- Modify: `src/audio_to_text/fusion.py:17` (import path)
- Modify: `src/audio_to_text/transcribe.py:465-470` (delete `sys.path` hack, fix import path)
- Modify: `tests/test_transcribe.py:6-14`, `tests/test_fusion.py:6` (import paths)
- Test: `tests/test_packaging.py` (new)

**Interfaces:**
- Consumes: nothing — this is the first task.
- Produces: the importable package `audio_to_text`, with `audio_to_text.transcribe` and `audio_to_text.fusion` as modules. Every later task imports from these paths. The runnable entry point is `python -m audio_to_text.transcribe`.

- [x] **Step 1: Write the failing test**

Create `tests/test_packaging.py`:

```python
"""Pins the packaging contract: the tool must be importable and runnable as a
module, because the PATH wrapper invokes `python -m audio_to_text.transcribe`.
A pyproject mistake breaks this in a way no import-level test would catch."""
import subprocess
import sys


def test_package_is_importable():
    import audio_to_text  # noqa: F401


def test_module_entry_point_runs():
    """`python -m audio_to_text.transcribe --help` must exit 0.

    This fails if [build-system] is missing, because uv then treats the project
    as virtual and never installs it onto the venv's import path.
    """
    result = subprocess.run(
        [sys.executable, "-m", "audio_to_text.transcribe", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--fuse" in result.stdout
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_packaging.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'audio_to_text'`

- [x] **Step 3: Move the sources into a package**

```bash
mkdir -p src/audio_to_text
git mv src/transcribe.py src/audio_to_text/transcribe.py
git mv src/fusion.py src/audio_to_text/fusion.py
```

Create `src/audio_to_text/__init__.py`:

```python
"""Speaker-attributed transcription of audio and video on Apple Silicon."""
```

- [x] **Step 4: Make the project installable**

Append to `pyproject.toml`:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project.scripts]
audio-to-text = "audio_to_text.transcribe:main"

[tool.hatch.build.targets.wheel]
packages = ["src/audio_to_text"]
```

`[build-system]` is load-bearing: without it uv treats this as a virtual project and never installs it onto the venv's import path, so `python -m audio_to_text.transcribe` cannot resolve. `[project.scripts]` is not what the PATH wrapper calls, but it is free here and makes `uv run audio-to-text` work during development.

- [x] **Step 5: Fix fusion.py's import**

In `src/audio_to_text/fusion.py:17`, change `from src.transcribe import (` to:

```python
from audio_to_text.transcribe import (
```

Leave the nine imported names unchanged.

- [x] **Step 6: Delete the sys.path hack, keep the deferred import**

In `src/audio_to_text/transcribe.py`, replace lines 465-470 — currently:

```python
        if __package__ in (None, ""):
            # Invoked directly (`python src/transcribe.py`, not `-m src.transcribe`):
            # only src/ itself is on sys.path, so `src` isn't a resolvable package
            # until the project root is added.
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from src.fusion import run_fusion
```

with:

```python
        # Deferred deliberately: fusion.py imports nine names from this module at
        # module level, so importing it at the top here would be a circular import
        # that fails at load time. Do not hoist this.
        from audio_to_text.fusion import run_fusion
```

**This import must stay inside `main()`.** The original comment blamed `sys.path` only; the cycle is the other, permanent reason. Hoisting it crashes on import.

- [x] **Step 7: Re-point the test imports and patch targets**

There are more references than a skim suggests: `tests/test_fusion.py` imports the module at lines 6, **92 (mid-file)** and **195 (inside a function)**, and `tests/test_transcribe.py` has five *string* patch targets that no import-statement search would find. Use sed so none are missed:

```bash
sed -i '' \
  -e 's/from src\.transcribe import/from audio_to_text.transcribe import/g' \
  -e 's/from src\.fusion import/from audio_to_text.fusion import/g' \
  -e 's/from src import fusion/from audio_to_text import fusion/g' \
  -e 's/"src\.transcribe\./"audio_to_text.transcribe./g' \
  -e 's/"src\.fusion\./"audio_to_text.fusion./g' \
  tests/test_transcribe.py tests/test_fusion.py
```

That covers, specifically:

| Location | Reference |
|---|---|
| `test_transcribe.py:6,7,13,14` | `from src.transcribe import ...` |
| `test_transcribe.py:144,155` | `patch("src.transcribe.mlx_whisper.transcribe", ...)` |
| `test_transcribe.py:409,436,464` | `patch("src.fusion.run_fusion", ...)` |
| `test_fusion.py:6,92` | `from src.fusion import ...` |
| `test_fusion.py:195` | `from src import fusion` |

**Found during execution — the sed above is not sufficient.** Two more things the plan missed:

1. A *third* import idiom, `import src.transcribe as t`, appears **8 times** in `test_transcribe.py` (lines 258, 295, 359, 377, 398, 421, 453, 478). Add:
   `-e 's/import src\.transcribe as t/import audio_to_text.transcribe as t/'`
2. `test_transcribe.py:498` — `test_fuse_direct_script_invocation_resolves_fusion_import` exists specifically to pin the `sys.path` hack that Step 6 deletes, and it invokes `src/transcribe.py` by path. Rewritten rather than deleted: the underlying risk (a deferred import failing at runtime) is still real, so it now runs `python -m audio_to_text.transcribe` and is renamed `test_fuse_module_invocation_resolves_fusion_import`. It also now runs from an empty cwd with `HOME` redirected, so that Task 2's `~/.config/audio-to-text/.env` fallback cannot accidentally satisfy the token lookup and defeat the assertion.

Also update the usage examples in `transcribe.py`'s own module docstring (lines 10-25) — they reference `uv run python src/transcribe.py`.

`patch("audio_to_text.fusion.run_fusion")` keeps working even though `main()` imports `run_fusion` inside the function body: the deferred import runs at call time and picks up the already-patched module attribute. Do not "fix" those tests to patch somewhere else.

Then update the two module docstrings by hand — `tests/test_fusion.py:1` says `src/fusion.py` and `tests/test_transcribe.py:1` says `transcribe.py`; make them `audio_to_text/fusion.py` and `audio_to_text/transcribe.py`.

- [x] **Step 8: Re-sync the venv so the package is installed**

Run: `uv sync`
Expected: uv installs the project itself into `.venv` (it did not before, because there was no build backend).

- [x] **Step 9: Run the full suite**

Run: `uv run pytest -v`
Expected: PASS — all 40 pre-existing tests plus the 2 new packaging tests. Any failure here is a mis-edited import, not a real regression; fix the import.

- [x] **Step 10: Verify no stale references remain**

Run: `grep -rn "src\.transcribe\|src\.fusion\|from src import\|src/transcribe\|src/fusion" src/ tests/`
Expected: no matches. This pattern deliberately catches bare `src.transcribe` inside quoted patch targets, not just `import` lines — the earlier, narrower pattern missed five of them.

`README.md` still holds stale `src/transcribe.py` paths at this point; Task 4 rewrites it.

- [x] **Step 11: Commit**

```bash
git add pyproject.toml src/audio_to_text tests/test_transcribe.py tests/test_fusion.py tests/test_packaging.py
git commit -m "refactor: make audio_to_text an installable src-layout package"
```

---

### Task 2: Resolve HF_TOKEN from outside the checkout

**Files:**
- Modify: `src/audio_to_text/transcribe.py` (add `CONFIG_DIR` + `resolve_hf_token()`; update `load_diarization_pipeline`'s error message; replace both call sites)
- Test: `tests/test_token.py` (new)

**Interfaces:**
- Consumes: the `audio_to_text.transcribe` module from Task 1.
- Produces: `resolve_hf_token() -> str | None` and `CONFIG_DIR: Path` (`~/.config/audio-to-text`). Task 4's README documents `CONFIG_DIR`.

- [x] **Step 1: Write the failing test**

Create `tests/test_token.py`:

```python
"""The token lookup is the difference between working and not working from
another project: a bare load_dotenv() searches upward from the caller's cwd,
finds that project's .env or nothing, and reports 'HF_TOKEN is not set'."""
import pytest

from audio_to_text.transcribe import resolve_hf_token


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """Every test starts with no ambient token and a cwd with no .env."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.chdir(tmp_path)


def test_environment_variable_wins(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("HF_TOKEN=from_cwd\n")
    monkeypatch.setenv("HF_TOKEN", "from_env")

    assert resolve_hf_token() == "from_env"


def test_cwd_dotenv_beats_global(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("HF_TOKEN=from_cwd\n")
    config = tmp_path / "config"
    config.mkdir()
    (config / ".env").write_text("HF_TOKEN=from_global\n")
    monkeypatch.setattr("audio_to_text.transcribe.CONFIG_DIR", config)

    assert resolve_hf_token() == "from_cwd"


def test_falls_back_to_global_config(monkeypatch, tmp_path):
    """The case that makes this whole feature work: called from a project that
    has no .env of its own."""
    config = tmp_path / "config"
    config.mkdir()
    (config / ".env").write_text("HF_TOKEN=from_global\n")
    monkeypatch.setattr("audio_to_text.transcribe.CONFIG_DIR", config)

    assert resolve_hf_token() == "from_global"


def test_returns_none_when_nowhere(monkeypatch, tmp_path):
    monkeypatch.setattr("audio_to_text.transcribe.CONFIG_DIR", tmp_path / "nonexistent")

    assert resolve_hf_token() is None


def test_does_not_mutate_process_environment(monkeypatch, tmp_path):
    """Reading a .env must not leak into os.environ -- that would make the
    lookup order depend on whatever ran earlier in the process."""
    import os

    (tmp_path / ".env").write_text("HF_TOKEN=from_cwd\n")
    monkeypatch.setattr("audio_to_text.transcribe.CONFIG_DIR", tmp_path / "nonexistent")

    resolve_hf_token()

    assert "HF_TOKEN" not in os.environ
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_token.py -v`
Expected: FAIL — `ImportError: cannot import name 'resolve_hf_token'`

- [x] **Step 3: Implement the resolution chain**

In `src/audio_to_text/transcribe.py`, change the dotenv import on line 41 from `from dotenv import load_dotenv` to:

```python
from dotenv import dotenv_values
```

`dotenv_values` reads a file into a dict without touching `os.environ`; `load_dotenv` mutates global state, which makes lookup order depend on call history.

Add near the other module constants (after the `DEFAULT_*` block):

```python
CONFIG_DIR = Path.home() / ".config" / "audio-to-text"
```

Add this function above `load_diarization_pipeline`:

```python
def resolve_hf_token() -> str | None:
    """Find the Hugging Face token, first hit wins.

    1. HF_TOKEN already in the environment
    2. ./.env  -- lets the calling project supply its own
    3. ~/.config/audio-to-text/.env  -- the global home for it

    Step 3 is what makes this tool usable from another project: a bare
    load_dotenv() only searches upward from the caller's cwd, so invoked from
    somewhere else it silently finds nothing.
    """
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    for candidate in (Path.cwd() / ".env", CONFIG_DIR / ".env"):
        if candidate.is_file():
            value = dotenv_values(candidate).get("HF_TOKEN")
            if value:
                return value
    return None
```

- [x] **Step 4: Update the error message**

In `load_diarization_pipeline`, replace the `raise RuntimeError(...)` at lines 301-307 with:

```python
        raise RuntimeError(
            "HF_TOKEN is not set. Looked in: the HF_TOKEN environment variable, "
            f"'{Path.cwd() / '.env'}', and '{CONFIG_DIR / '.env'}'. When calling this "
            "tool from another project, put it in the last of those. Also make sure "
            "you've accepted the pyannote/speaker-diarization-3.1 model terms at "
            "https://huggingface.co/pyannote/speaker-diarization-3.1"
        )
```

- [x] **Step 5: Replace both call sites**

There are two identical blocks, in the `--fuse` branch and the batch branch. In each, replace:

```python
        load_dotenv()
        try:
            diarization_pipeline = load_diarization_pipeline(os.environ.get("HF_TOKEN"))
```

with:

```python
        try:
            diarization_pipeline = load_diarization_pipeline(resolve_hf_token())
```

(The batch-branch copy is indented one level less — match the surrounding indentation.)

- [x] **Step 6: Confirm load_dotenv is fully gone from the module**

Run: `grep -n "load_dotenv" src/audio_to_text/transcribe.py`
Expected: no matches. `tests/test_environment.py` still imports `load_dotenv` from `dotenv` directly — that is correct and stays.

- [x] **Step 7: Run the tests**

Run: `uv run pytest tests/test_token.py -v && uv run pytest -q`
Expected: PASS, all green.

- [x] **Step 8: Install the token globally**

```bash
mkdir -p ~/.config/audio-to-text
cp .env ~/.config/audio-to-text/.env
chmod 600 ~/.config/audio-to-text/.env
```

`chmod 600` because this file holds a credential and `~/.config` is not restricted by default.

- [x] **Step 9: Commit**

```bash
git add src/audio_to_text/transcribe.py tests/test_token.py
git commit -m "feat: resolve HF_TOKEN from env, cwd .env, or ~/.config/audio-to-text"
```

---

### Task 3: Make input and output defaults caller-relative

**Files:**
- Modify: `src/audio_to_text/transcribe.py:44-46` (replace the three `PROJECT_ROOT` constants), `:70-84` (`gather_media`), `:449-453` (the not-found message), `:477` and `:518` (output resolution)
- Test: `tests/test_defaults.py` (new)

**Interfaces:**
- Consumes: `audio_to_text.transcribe` from Task 1.
- Produces: `default_input_dir() -> Path` and `resolve_output_dir(arg: Path | None) -> Path`. `resolve_output_dir` both resolves and creates the directory.

- [x] **Step 1: Write the failing test**

Create `tests/test_defaults.py`:

```python
"""Defaults must follow the caller, not this checkout. PROJECT_ROOT-based
defaults meant a no-arg run from another project scanned *this* repo's data/."""
from pathlib import Path

import pytest

from audio_to_text.transcribe import (
    default_input_dir,
    gather_media,
    resolve_output_dir,
)


def test_default_input_dir_follows_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    assert default_input_dir() == tmp_path / "data"


def test_default_output_is_data_transcriptions(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    assert resolve_output_dir(None) == (tmp_path / "data" / "transcriptions").resolve()


def test_default_output_dir_is_created(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    out = resolve_output_dir(None)

    assert out.is_dir()


def test_explicit_output_dir_overrides_and_is_created(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "somewhere" / "else"

    out = resolve_output_dir(target)

    assert out == target.resolve()
    assert out.is_dir()


def test_gather_media_defaults_to_cwd_data(monkeypatch, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "meeting.m4a").touch()
    monkeypatch.chdir(tmp_path)

    assert gather_media(None) == [data / "meeting.m4a"]


def test_gather_media_ignores_the_transcriptions_subdir(monkeypatch, tmp_path):
    """Output nests inside input, so a second batch run must not descend into
    it. Safe only because gather_media uses iterdir() + is_file(); a switch to
    rglob() would silently break this."""
    data = tmp_path / "data"
    (data / "transcriptions").mkdir(parents=True)
    (data / "meeting.m4a").touch()
    (data / "transcriptions" / "meeting.md").touch()
    (data / "transcriptions" / "stray.m4a").touch()
    monkeypatch.chdir(tmp_path)

    assert gather_media(None) == [data / "meeting.m4a"]


def test_gather_media_returns_nothing_when_no_data_dir(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    assert gather_media(None) == []
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_defaults.py -v`
Expected: FAIL — `ImportError: cannot import name 'default_input_dir'`

- [x] **Step 3: Replace the constants**

In `src/audio_to_text/transcribe.py`, delete lines 44-46:

```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
```

and put in their place:

```python
# Defaults follow the caller's working directory, not this checkout, so the tool
# behaves sensibly when invoked from another project. (PROJECT_ROOT is gone: after
# the package move, parent.parent would resolve to src/ and quietly point the
# defaults at the wrong place.)
DEFAULT_INPUT_SUBDIR = "data"
DEFAULT_OUTPUT_SUBDIR = Path("data") / "transcriptions"
```

- [x] **Step 4: Add the two resolver functions**

Insert directly above `gather_media`:

```python
def default_input_dir() -> Path:
    """The directory scanned when no media argument is given: ./data in the caller's cwd."""
    return Path.cwd() / DEFAULT_INPUT_SUBDIR


def resolve_output_dir(arg: Path | None) -> Path:
    """Resolve the output directory and make sure it exists.

    Defaults to ./data/transcriptions in the caller's cwd -- the standard project
    layout here -- so the transcript lands where it will live rather than in the
    project root.
    """
    out = (arg or Path.cwd() / DEFAULT_OUTPUT_SUBDIR).resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out
```

- [x] **Step 5: Point gather_media at the new default**

In `gather_media`, change line 78 from `target = DEFAULT_INPUT_DIR` to:

```python
        target = default_input_dir()
```

Then change the guard on line 79 from `if target.is_dir():` to handle a missing directory without raising:

```python
    if not target.exists():
        return [] if target == default_input_dir() else [target]
    if target.is_dir():
```

This keeps the existing contract for an explicitly-named file (returned as-is; existence is checked later in `run_whisper`) while letting a missing default `data/` produce the friendlier message in the next step.

Update the docstring's first bullet to match — it currently says "every media file in the default data/ folder":

```python
    - None            -> every media file in ./data relative to the caller's cwd,
                         or [] if there is no such directory
```

- [x] **Step 6: Improve the not-found message**

Replace lines 450-453:

```python
    if not media_files:
        where = args.media or DEFAULT_INPUT_DIR
        print(f"error: no media files found in '{where}'", file=sys.stderr)
        return 1
```

with:

```python
    if not media_files:
        if args.media is not None:
            print(f"error: no media files found in '{args.media}'", file=sys.stderr)
        elif not default_input_dir().is_dir():
            print(
                f"error: no '{DEFAULT_INPUT_SUBDIR}/' directory in {Path.cwd()}. "
                "Name the recording explicitly, e.g. "
                "'audio-to-text path/to/recording.m4a'.",
                file=sys.stderr,
            )
        else:
            print(f"error: no media files in '{default_input_dir()}'", file=sys.stderr)
        return 1
```

- [x] **Step 7: Use the resolver at both output sites**

At line 477 (the `--fuse` branch), replace:

```python
        output_dir = (args.output_dir or DEFAULT_OUTPUT_DIR).resolve()
```

with:

```python
        output_dir = resolve_output_dir(args.output_dir)
```

At line 518-519 (the batch branch), replace both lines:

```python
    output_dir = (args.output_dir or DEFAULT_OUTPUT_DIR).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
```

with:

```python
    output_dir = resolve_output_dir(args.output_dir)
```

The `--fuse` branch previously never created its output directory — it relied on `run_fusion`. Routing both through `resolve_output_dir` fixes that asymmetry.

- [x] **Step 8: Update the --output-dir help text**

In the `--output-dir` argument at line 404-409, the help string says `.txt outputs` and names the project's `output/` folder. Both are wrong — the tool writes `.md`. Replace the `help=` value with:

```python
        help="Directory to write .md transcripts to "
             "(default: ./data/transcriptions/, created if absent).",
```

- [x] **Step 9: Confirm the old constants are gone**

Run: `grep -n "PROJECT_ROOT\|DEFAULT_INPUT_DIR\|DEFAULT_OUTPUT_DIR" src/ tests/ -r`
Expected: no matches.

- [x] **Step 10: Run the tests**

Run: `uv run pytest -q`
Expected: PASS — 40 original + 2 packaging + 5 token + 7 defaults.

- [x] **Step 11: Commit**

```bash
git add src/audio_to_text/transcribe.py tests/test_defaults.py
git commit -m "feat: default input to ./data and output to ./data/transcriptions in cwd"
```

---

### Task 4: PATH wrapper and README

**Files:**
- Create: `~/.local/bin/audio-to-text` (machine-local, not in any repo)
- Modify: `README.md` (Usage, Requirements, Development sections)

**Interfaces:**
- Consumes: `python -m audio_to_text.transcribe` from Task 1; `CONFIG_DIR` from Task 2; the `./data/transcriptions/` default from Task 3.
- Produces: the `audio-to-text` command on `PATH`. Task 5's skill invokes exactly this name.

- [x] **Step 1: Write the wrapper**

Create `~/.local/bin/audio-to-text` (`~/.local/bin` is already on `PATH`):

```bash
#!/usr/bin/env bash
# audio-to-text -- run the audio_to_text pipeline from any directory.
#
# Delegates to the repo's own uv-managed venv rather than installing a second
# copy of its ~1.2 GB of ML dependencies. Edits to the repo take effect with no
# reinstall step.
#
# `uv run --project` sets the project without changing the working directory,
# which is what lets the tool's ./data/transcriptions default resolve against
# the CALLER's cwd. Do not switch this to --directory.
set -euo pipefail

REPO="${AUDIO_TO_TEXT_REPO:-$HOME/Developer/GitHub/audio_to_text}"

if [[ ! -d "$REPO" ]]; then
  echo "audio-to-text: repo not found at '$REPO'." >&2
  echo "Set AUDIO_TO_TEXT_REPO to point at your audio_to_text checkout." >&2
  exit 1
fi

exec uv run --project "$REPO" python -m audio_to_text.transcribe "$@"
```

- [x] **Step 2: Make it executable**

```bash
chmod +x ~/.local/bin/audio-to-text
```

- [x] **Step 3: Verify it resolves from an unrelated directory**

```bash
cd /tmp && audio-to-text --help
```

Expected: the argparse help, exit 0, including `--fuse` and the new `--output-dir` wording. If this prints a `ModuleNotFoundError`, Task 1's `[build-system]` did not take effect — run `uv sync` in the repo.

- [x] **Step 4: Verify cwd is the caller's, not the repo's**

This is the assumption the whole design rests on — that `--project` does not chdir. Prove it rather than trusting it:

```bash
mkdir -p /tmp/atx-check && cd /tmp/atx-check && audio-to-text
```

Expected: exit 1 with `error: no 'data/' directory in /tmp/atx-check`. The path in the message must be `/tmp/atx-check`, **not** the audio_to_text repo. If it names the repo, `--project` is chdir-ing and the wrapper needs rethinking — stop and report.

- [x] **Step 5: Verify token resolution from outside the repo**

```bash
cd /tmp/atx-check && env -u HF_TOKEN audio-to-text --help >/dev/null && echo "wrapper ok"
```

Expected: `wrapper ok`. (`--help` exits before the token is needed; the real end-to-end token check happens in the VERIFY phase with an actual recording.)

- [x] **Step 6: Update the README**

In `README.md`:

- Under **Requirements**, change the token bullet from "A Hugging Face token in `.env` as `HF_TOKEN=...`" to: "A Hugging Face token, in any of: the `HF_TOKEN` environment variable, `./.env` where you run the tool, or `~/.config/audio-to-text/.env` (use this one to call the tool from other projects)."
- Add a short **Install** section documenting the `~/.local/bin/audio-to-text` wrapper, that it reuses the repo venv, that `AUDIO_TO_TEXT_REPO` overrides the path, and that moving the checkout breaks the command until that variable is set.
- In **Usage**, replace the four `uv run python src/transcribe.py` invocations with `audio-to-text`, and state that transcripts default to `./data/transcriptions/` in the current directory.
- In **Development**, keep `uv sync` / `uv run pytest`, and note that `uv run audio-to-text` runs the console script inside the repo.

- [x] **Step 7: Confirm the README has no stale paths**

Run: `grep -n "src/transcribe.py\|output/\|the project's .env" README.md`
Expected: no matches.

- [x] **Step 8: Commit**

```bash
git add README.md
git commit -m "docs: document the audio-to-text wrapper and global token location"
```

The wrapper itself is machine-local and intentionally not tracked — the README explains how to recreate it.

---

### Task 5: The global `transcribe-recording` skill

**Files:**
- Create: `~/.claude/skills/transcribe-recording/SKILL.md`
- Create: `/Users/mkingomac.com/Developer/GitHub/Templates/user/claude/skills/transcribe-recording/SKILL.md` (identical mirror)

**Interfaces:**
- Consumes: the `audio-to-text` command from Task 4.
- Produces: nothing code-facing. This is the last task.

- [x] **Step 1: Write the skill**

Create `~/.claude/skills/transcribe-recording/SKILL.md`:

```markdown
---
name: transcribe-recording
description: "Turn an audio or video recording into a speaker-attributed Markdown transcript, and fuse two recordings of the same meeting (e.g. a Teams capture plus a phone left elsewhere in the room) into one transcript that takes the clearer audio from whichever source caught each moment better. Use whenever the user points at a media file (.m4a, .mp3, .wav, .mp4, .mov, .m4v, .webm) and wants the words out of it, says 'transcribe this', 'what was said in this recording', 'turn this call into text', or mentions having two recordings of the same meeting they want combined. Apple Silicon only."
---

# Transcribe a recording

Runs `audio-to-text`, a wrapper on PATH that delegates to a local MLX Whisper +
pyannote diarization pipeline. The code is NOT in this project — never look for it
here, never vendor it, just call the command.

## Preconditions — check before running, fail fast

- **Apple Silicon macOS.** The tool exits immediately on anything else; there is no
  CPU fallback. If `uname -m` is not `arm64`, say so and stop. Do not retry.
- **`ffmpeg` on PATH.**
- **A Hugging Face token**, from `$HF_TOKEN`, `./.env`, or `~/.config/audio-to-text/.env`.
- If `audio-to-text` is not found, the wrapper is missing. Point the user at the
  audio_to_text repo's README rather than trying to reconstruct it.

## Run it in the background

Transcription is slow — roughly real-time-ish per source, so a one-hour meeting takes
many minutes, and fusion runs the whole pipeline **twice**. The default Bash timeout
will kill it partway and look like a crash.

Always launch it with `run_in_background: true`, then poll. Do not raise the timeout
and block — the user gets no feedback that way.

## Single recording

    audio-to-text "path/to/recording.m4a"

## Two recordings of the same meeting

The first file is primary: its speaker turn boundaries are kept as canonical, and the
second only replaces text where it was more confident. Pass the better/more complete
recording first.

    audio-to-text "path/to/teams-capture.mp4" --fuse "path/to/phone.m4a"

## Quality levers — ask about these, they matter

- `--num-speakers N` when the number of people in the room is known. Speaker
  clustering is materially better with it. Worth asking the user.
- `--prompt "Crisis Shield, ZeroW, Margu"` biases spelling of names, products and
  jargon. Worth asking whether unusual terms come up.
- `--preprocess` (high-pass + loudness normalisation) or `--denoise` for poor audio.
  Off by default; only reach for these if a first pass reads badly.

## Output

Markdown at `./data/transcriptions/<input-name>.md`, relative to the directory the
command was run in, created if absent. Override with `--output-dir`.

Speakers are numbered (`Person 1`, `Person 2`), not named — voice prints cannot
identify who someone is. Numbering is stable across the document, so the user can
identify each person once from the first few minutes and find-and-replace throughout.
Offer that; don't guess at names yourself.
```

- [x] **Step 2: Verify the skill is well-formed**

Run: `head -4 ~/.claude/skills/transcribe-recording/SKILL.md`
Expected: valid YAML frontmatter with `name:` and `description:`, matching the layout of the sibling skills in that directory.

- [x] **Step 3: Mirror it into Templates**

```bash
mkdir -p ~/Developer/GitHub/Templates/user/claude/skills/transcribe-recording
cp ~/.claude/skills/transcribe-recording/SKILL.md \
   ~/Developer/GitHub/Templates/user/claude/skills/transcribe-recording/SKILL.md
```

- [x] **Step 4: Check for drift on this skill specifically**

```bash
diff ~/.claude/skills/transcribe-recording/SKILL.md \
     ~/Developer/GitHub/Templates/user/claude/skills/transcribe-recording/SKILL.md
```

Expected: no output.

`sync.sh` will still report pre-existing drift for `meeting-minutes` and
`meeting-minutes-workspace`. That is known, unrelated, and explicitly deferred to a
separate workflow — do not fix it here.

- [x] **Step 5: Commit in Templates**

```bash
cd ~/Developer/GitHub/Templates
git add user/claude/skills/transcribe-recording/SKILL.md
git commit -m "feat: add transcribe-recording skill to the user mirror"
```

`Templates` has its own unrelated uncommitted work (`core/claude/commands/scope-and-requirements.md`, `core/claude/skills/scope-and-requirements/`, `core/tests/`, modified `README.md`). Commit **only** the path above. This repo does have the commit gate — if it blocks, mark the commit `[no-plan-task]`.

---

## Amendment log

In-flight changes to the plan as written. All five tasks shipped; none dropped.

1. **Task 1, Step 7 — the import sweep was incomplete twice over.** The plan's sed covered
   three idioms; the code had a fourth, `import src.transcribe as t`, appearing 8 times. It
   also had `test_fuse_direct_script_invocation_resolves_fusion_import`, a test written
   specifically to pin the `sys.path` hack that Step 6 deletes. Rewritten rather than
   deleted — the risk it guards (a deferred import failing at runtime) outlives the hack —
   and re-pointed at `python -m`, with `HOME` redirected so Task 2's global token fallback
   could not later satisfy its assertion and hollow the test out.
2. **Task 1 — `transcribe.py`'s own module docstring** carried five `uv run python
   src/transcribe.py` examples. Not in the plan's file list; updated.
3. **Task 2 — six existing tests stubbed `t.load_dotenv`,** which the task deletes. Not
   anticipated. Re-pointed to stub `resolve_hf_token` with an explicit fake.
   *Corrected after review:* this was originally recorded here as "strictly stronger,
   because the old stub still let a real ambient `HF_TOKEN` reach the code." That was
   wrong — all six also stub `load_diarization_pipeline`, so the ambient token was already
   inert. The swap is **neutral** in effect, and it converted the token lookup into a
   stubbed seam that nothing verified (see amendment 7).
4. **Task 3, Step 5 — simplified.** The plan proposed a `not target.exists()` branch with a
   `target == default_input_dir()` comparison. Unnecessary: handling the missing directory
   inside the `target is None` branch is equivalent and clearer, and an explicitly-named
   path still falls through to `return [target]`.
5. **Task 4 — added `prog="audio-to-text"` to the ArgumentParser.** Not in the plan. Under
   `python -m`, argparse reported the tool as `transcribe.py` in its usage line — a name no
   user types.
6. **FINISH — added `test_main_writes_into_data_transcriptions_by_default`.** The plan's
   Task 3 tests covered `resolve_output_dir` in isolation but nothing proved `main()`
   routed through it; a `main()` still writing to a stale constant would have passed every
   other test. Mutation-checked: replacing the call with `Path.cwd()` fails it.

7. **REVIEW — the adversarial pass failed the change and three real coverage gaps were
   fixed.** A mutation-testing contrarian pass found that the suite was green under each of
   these mutations, i.e. the behaviour was entirely unpinned:
   - **Both `main()` call sites reverted to `os.environ.get("HF_TOKEN")`** — the headline
     feature deleted, 57/57 still passing. Cause: the six re-pointed stubs (amendment 3)
     stub `resolve_hf_token` and never assert it was called or that its value reaches
     `load_diarization_pipeline`. Fixed by `test_main_passes_the_resolved_token_to_the_pipeline`,
     which captures the pipeline's argument and asserts it equals the token from
     `CONFIG_DIR/.env`.
   - **The `--fuse` default output dir** — every `--fuse` test passes an explicit
     `--output-dir` and mocks `run_fusion` to raise, so the default never executed. Fixed by
     `test_main_fuse_defaults_to_data_transcriptions`. (Commit `2b36c5a`'s claim that this
     site "fixes an asymmetry" is also partly redundant: `fusion.py:211` already mkdirs.)
   - **The "no `data/` directory" guidance message** — a user-facing deliverable of this
     change, unpinned; `elif False:` left the suite green. Fixed by
     `test_main_reports_missing_data_dir_with_actionable_guidance`.

   Two minor fixes from the same pass: `test_module_entry_point_runs` now runs with
   `cwd=tmp_path` (`python -m` puts CWD on `sys.path`, so it could have resolved by
   directory accident), and `test_environment.py`'s bare `load_dotenv()` — which
   permanently injected a real `HF_TOKEN` into `os.environ` for the whole session, poisoning
   any later `main()`-level token test — now goes through `resolve_hf_token()`.

   All three mutations were re-run after the fix and now fail the suite. 60 tests green.

## Acceptance (VERIFY phase)

Runs after all five tasks, from a real project directory, with pasted output:

1. `cd /tmp && audio-to-text --help` → exit 0.
2. From a scratch directory with no `.env`: transcribe one real recording; the `.md`
   appears in `./data/transcriptions/` and the directory was created.
3. From the same directory: `--fuse` two recordings of one meeting; a fused transcript
   appears in the same place.
4. `uv run pytest` in `audio_to_text` → all green.
5. In an unrelated project, a natural request ("transcribe this recording") triggers
   the skill and it runs the command in the background.
