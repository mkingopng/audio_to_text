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
from scipy.optimize import linear_sum_assignment

from src.transcribe import (
    align_words_to_speakers,
    _group_consecutive,
    extract_words,
    overlap_seconds,
    preprocess_audio,
    relabel_speakers,
    render_markdown,
    run_diarization,
    run_whisper,
)


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


def match_speakers(embeddings_a: dict[str, np.ndarray], embeddings_b: dict[str, np.ndarray]) -> dict[str, str]:
    """Match source A's speaker embeddings to source B's via the Hungarian algorithm.

    Greedy nearest-match can be led astray when two voices are close together
    (assigning both of B's closest speakers to the same A speaker, then being
    forced into a bad leftover pairing); the Hungarian algorithm finds the
    globally optimal one-to-one assignment instead.

    Raises ValueError if the two embedding dicts have different sizes, since
    linear_sum_assignment on a rectangular matrix silently returns only
    min(len(A), len(B)) pairs, corrupting downstream fusion with no error.
    """
    if len(embeddings_a) != len(embeddings_b):
        raise ValueError(
            f"len(embeddings_a) != len(embeddings_b): {len(embeddings_a)} vs {len(embeddings_b)}"
        )

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

    A single B turn can span (temporally overlap) more than one A turn -- e.g.
    when A's diarization splits one continuous utterance into two turns that
    B's diarization kept as one. Each B turn is used as a replacement at most
    once (first-come, chronologically-earliest A turn wins it), so the same
    B text can't get duplicated into two separate merged turns.
    """
    merged = []
    used_b_ids: set[int] = set()
    for turn in turns_a:
        overlapping_b = [
            b for b in turns_b_shifted
            if id(b) not in used_b_ids
            and b["speaker"] == turn["speaker"]
            and overlap_seconds(turn["start"], turn["end"], b["start"], b["end"]) > 0
        ]
        if overlapping_b:
            best_b = max(overlapping_b, key=lambda b: b["confidence"])
            if best_b["confidence"] > turn["confidence"]:
                merged.append({**turn, "text": best_b["text"], "confidence": best_b["confidence"]})
                used_b_ids.add(id(best_b))
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
