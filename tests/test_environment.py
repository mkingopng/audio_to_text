"""Smoke tests confirming the new diarization/fusion dependencies are usable."""
import os

import scipy.signal  # noqa: F401
from dotenv import load_dotenv


def test_new_dependencies_import():
    import pyannote.audio  # noqa: F401


def test_hf_token_loads_from_dotenv():
    load_dotenv()
    assert os.environ.get("HF_TOKEN"), "HF_TOKEN not found after load_dotenv() -- check .env"
