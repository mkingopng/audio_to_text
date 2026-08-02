"""Smoke tests confirming the new diarization/fusion dependencies are usable."""
import os

import pytest
import scipy.signal  # noqa: F401
from dotenv import load_dotenv


def test_new_dependencies_import():
    import pyannote.audio  # noqa: F401


def test_hf_token_loads_from_dotenv():
    """Environment precondition, not a property of the code: this project's .env
    is gitignored, so this legitimately fails on a fresh clone or CI with no local
    .env configured. Skip rather than fail in that case, so the rest of the suite
    (which doesn't need a real token) stays green off the author's machine."""
    load_dotenv()
    if not os.environ.get("HF_TOKEN"):
        pytest.skip("HF_TOKEN not set -- add it to a local .env to run this check")
