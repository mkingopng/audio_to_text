"""Pins the packaging contract: the tool must be importable and runnable as a
module, because the PATH wrapper invokes `python -m audio_to_text.transcribe`.
A pyproject mistake breaks this in a way no import-level test would catch."""
import subprocess
import sys


def test_package_is_importable():
    import audio_to_text  # noqa: F401


def test_module_entry_point_runs(tmp_path):
    """`python -m audio_to_text.transcribe --help` must exit 0.

    This fails if [build-system] is missing, because uv then treats the project
    as virtual and never installs it onto the venv's import path.

    Runs from tmp_path deliberately: `python -m` puts the CWD on sys.path, so
    inheriting pytest's cwd would let the module resolve by directory accident
    from inside the source tree rather than from the installed package.
    """
    result = subprocess.run(
        [sys.executable, "-m", "audio_to_text.transcribe", "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--fuse" in result.stdout
