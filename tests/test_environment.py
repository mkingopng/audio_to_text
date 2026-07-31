"""Smoke tests confirming the new diarization/fusion dependencies are usable."""
import pytest
import scipy.signal  # noqa: F401

from audio_to_text.transcribe import resolve_hf_token


def test_new_dependencies_import():
    import pyannote.audio  # noqa: F401


def test_hf_token_is_resolvable():
    """Environment precondition, not a property of the code: this project's .env
    is gitignored, so this legitimately fails on a fresh clone or CI with no local
    .env configured. Skip rather than fail in that case, so the rest of the suite
    (which doesn't need a real token) stays green off the author's machine.

    Goes through resolve_hf_token rather than a bare load_dotenv(): that call
    permanently injected a real HF_TOKEN into os.environ for the whole session,
    and since this module sorts before test_token.py it would silently poison any
    main()-level token test that forgot to clear the environment itself.
    """
    if not resolve_hf_token():
        pytest.skip("HF_TOKEN not set -- add it to a local .env to run this check")
