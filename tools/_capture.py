"""Where the persisted fusion intermediates live.

The analysis tools all read the capture that tools/capture_fusion_intermediates.py
writes. Default to data/fusion-capture/ inside the repo (the standard layout: inputs
in data/, derived artifacts under it), overridable so a capture can live on a
scratch disk -- it holds two ~135 MB WAVs.

    export FUSION_CAPTURE_DIR=/path/to/capture
"""
from __future__ import annotations

import os
from pathlib import Path

CAPTURE_DIR = Path(
    os.environ.get(
        "FUSION_CAPTURE_DIR",
        Path(__file__).resolve().parent.parent / "data" / "fusion-capture",
    )
)

CAPTURE_PKL = CAPTURE_DIR / "capture.pkl"


def require_capture() -> Path:
    """Return the capture pickle, with an actionable message if it isn't there."""
    if not CAPTURE_PKL.is_file():
        raise SystemExit(
            f"error: no capture at '{CAPTURE_PKL}'.\n"
            "Create one with:\n"
            "    uv run python tools/capture_fusion_intermediates.py\n"
            "(that re-runs ASR + diarization on both sources and takes a while), or "
            "point FUSION_CAPTURE_DIR at an existing capture."
        )
    return CAPTURE_PKL
