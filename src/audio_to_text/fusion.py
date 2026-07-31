"""Fuse two recordings of the same meeting into one speaker-attributed transcript.

Reuses transcribe.py's per-source pipeline (ASR + diarization + word-level
speaker alignment), then synchronizes the two sources' timelines, matches
their independently-clustered speaker identities to each other, and picks
the clearer source's text per overlapping turn.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.optimize import linear_sum_assignment

from audio_to_text.transcribe import (
    align_words_to_speakers,
    _group_consecutive,
    extract_words,
    overlap_seconds,
    preprocess_audio,
    relabel_speakers,
    render_markdown,
    run_diarization,
    run_whisper,
    smooth_micro_turns,
    warn_on_repetition_loops,
)


def _rms_envelope(samples: np.ndarray, sample_rate: int, window_seconds: float) -> np.ndarray:
    """Windowed RMS energy envelope -- correlating on this is faster and more robust
    to speech-content differences between the two mics than correlating raw samples."""
    window = max(1, int(sample_rate * window_seconds))
    usable_length = len(samples) - (len(samples) % window)
    reshaped = samples[:usable_length].reshape(-1, window)
    return np.sqrt(np.mean(reshaped.astype(np.float64) ** 2, axis=1))


# Below this peak/best_rival ratio, the alignment is reported as untrustworthy.
# Measured on the real 70-minute pair against five negative controls built from
# those same two recordings (so any separation is attributable to alignment, not
# to recording character): true pair 1.5162, nulls 1.0002-1.0050. 1.2 clears the
# null ceiling with margin and sits below the one true observation.
#
# WARN, never gate: the null floor is well characterised but the true-pair
# distribution is n=1. peak/median (3.135 vs nulls to 2.098) and the z-score were
# also measured and rejected -- they separate by less than a factor of two.
OFFSET_CONFIDENCE_THRESHOLD = 1.2

# A rival peak must be at least this far from the argmax to count as a rival,
# rather than as the shoulder of the same peak.
_RIVAL_EXCLUSION_SECONDS = 5.0


def _correlate_envelopes(
    wav_a: Path, wav_b: Path, *, window_seconds: float = 0.1
) -> tuple[float, float]:
    """Return (offset_seconds, confidence) for aligning source B onto source A.

    confidence is peak / best_rival -- the correlation peak divided by the best
    peak more than _RIVAL_EXCLUSION_SECONDS away from it. A genuinely shared
    recording produces one dominant peak; two recordings with no shared acoustic
    content produce a field of near-equal peaks, and the argmax among them is
    meaningless even though it looks like a precise number.
    """
    rate_a, samples_a = wavfile.read(wav_a)
    rate_b, samples_b = wavfile.read(wav_b)
    if rate_a != rate_b:
        raise ValueError(f"sample rate mismatch between sources: {rate_a} vs {rate_b}")

    envelope_a = _rms_envelope(samples_a, rate_a, window_seconds)
    envelope_b = _rms_envelope(samples_b, rate_b, window_seconds)

    correlation = signal.correlate(envelope_a, envelope_b, mode="full", method="fft")
    peak_index = int(np.argmax(correlation))
    offset = (peak_index - (len(envelope_b) - 1)) * window_seconds

    exclusion = int(_RIVAL_EXCLUSION_SECONDS / window_seconds)
    rivals = correlation.copy()
    rivals[max(0, peak_index - exclusion):peak_index + exclusion + 1] = -np.inf
    best_rival = float(np.max(rivals))

    peak = float(correlation[peak_index])
    if not np.isfinite(best_rival) or best_rival <= 0.0:
        # Nothing survived the exclusion mask (both recordings are shorter than
        # the window), or every rival is silence. Either way there is no
        # independent peak to judge this one against, so the alignment is
        # UNMEASURABLE -- which must score as untrustworthy, not as perfect.
        # Returning infinity here would make two unrelated clips report a
        # flawless alignment and suppress the warning: the exact failure this
        # metric exists to catch, inverted.
        return offset, 0.0
    return offset, peak / best_rival


def find_offset(wav_a: Path, wav_b: Path, *, window_seconds: float = 0.1) -> float:
    """Seconds to ADD to source B's timestamps to align them onto source A's timeline."""
    offset, _confidence = _correlate_envelopes(wav_a, wav_b, window_seconds=window_seconds)
    return offset


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
    """Move turns from source B's local clock/speaker-id namespace onto source A's.

    A negative offset (B's recording started before A's) can shift a B turn to
    end at or before A's timeline zero -- that's speech with no valid position
    on the merged timeline, so it's dropped. A turn straddling zero is clipped
    to start at 0 rather than rendering a nonsensical negative timestamp.
    """
    shifted = []
    for turn in turns:
        start = turn["start"] + offset
        end = turn["end"] + offset
        if end <= 0.0:
            continue
        shifted.append({
            **turn,
            "start": max(start, 0.0),
            "end": end,
            "speaker": speaker_map[turn["speaker"]],
        })
    return shifted


# How far, in A-turn index terms, the containment guard looks for a sibling.
#
# Must be >= 2. _group_consecutive flushes a turn on EVERY speaker change, so two
# same-speaker A turns are never adjacent -- a radius-1 guard is structurally
# incapable of firing on the same-speaker case, and measuring it on the real pair
# confirms it: 0 same-speaker fires at radius 1, 13 at radius 2. Beyond 2 adds
# nothing at the length floor below (5 fires at radius 2, 3 and 4 alike).
_CONTAINMENT_RADIUS = 2

# Minimum normalized length of a sibling turn before it may be consumed.
#
# Without a floor the guard stops being a redundancy fix and quietly becomes a
# fragmentation smoother. Measured on the real pair, the fires are bimodal: a
# cluster at <= 8 normalized chars ("I", "for", "And", "as", "you know",
# "Daniel") which are micro-turns, and a cluster at >= 20 which is genuinely
# restated content. 20 sits in the empty gap between them. Absorbing micro-turns
# is separate, riskier work -- "Daniel" may be a real one-word answer.
_CONTAINMENT_MIN_CHARS = 20


def _normalize_for_containment(text: str) -> str:
    """Casefold and drop punctuation/whitespace differences, so containment is
    judged on words rather than on how two ASR passes punctuated them."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", text.lower())).strip()


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

    That leaves CONTAINMENT duplication, which the containment guard below
    removes: B's spanning text still contains what the sibling A turn says, and
    that sibling would otherwise be emitted again underneath it.
    """
    # Pass 1 -- decide which A turns B's text wins, without emitting anything yet.
    # The containment guard needs to know the whole replacement set before it can
    # tell a sibling from a turn that is itself about to be replaced.
    replacement: dict[int, dict] = {}
    used_b_ids: set[int] = set()
    for index, turn in enumerate(turns_a):
        overlapping_b = [
            b for b in turns_b_shifted
            if id(b) not in used_b_ids
            and b["speaker"] == turn["speaker"]
            and overlap_seconds(turn["start"], turn["end"], b["start"], b["end"]) > 0
        ]
        if overlapping_b:
            best_b = max(overlapping_b, key=lambda b: b["confidence"])
            if best_b["confidence"] > turn["confidence"]:
                replacement[index] = best_b
                used_b_ids.add(id(best_b))

    # Pass 2 -- containment guard. Only a turn that is NOT itself replaced can be
    # consumed: every duplication of this shape has an untouched A turn as its
    # other half, and consuming a replaced turn would discard B text instead.
    consumed: set[int] = set()
    for index, best_b in replacement.items():
        b_text = _normalize_for_containment(best_b["text"])
        speaker = turns_a[index]["speaker"]
        low = max(0, index - _CONTAINMENT_RADIUS)
        high = min(len(turns_a), index + _CONTAINMENT_RADIUS + 1)
        for sibling_index in range(low, high):
            if sibling_index == index or sibling_index in replacement:
                continue
            if turns_a[sibling_index]["speaker"] != speaker:
                continue
            # The sibling must be speech B's turn ACTUALLY COVERS. Index adjacency
            # alone is not enough: A-turn indices say nothing about elapsed time, so
            # a turn twenty minutes later can sit two indices away and would be
            # deleted, its words reappearing under a heading timestamped at the
            # start of B's span. That is the same defect the replacement branch was
            # fixed for -- a turn's span and text describing different speech --
            # coming back through a different door.
            if overlap_seconds(
                turns_a[sibling_index]["start"], turns_a[sibling_index]["end"],
                best_b["start"], best_b["end"],
            ) <= 0.0:
                continue
            sibling = _normalize_for_containment(turns_a[sibling_index]["text"])
            if len(sibling) >= _CONTAINMENT_MIN_CHARS and sibling in b_text:
                consumed.add(sibling_index)

    # Pass 3 -- emit.
    merged = []
    for index, turn in enumerate(turns_a):
        if index in consumed:
            continue
        best_b = replacement.get(index)
        if best_b is not None:
            # Take B's SPAN along with B's text. Keeping A's start/end here
            # desynchronizes the timestamps from the words: when B's turn covers
            # more speech than A's, the merged turn claims B's whole sentence was
            # spoken inside A's much shorter window (the shipped reference output
            # had a 93-word block with a 0.4s duration, and the rendered mm:ss
            # heading pointed at the wrong moment). A merged turn's span and text
            # must always describe the same speech.
            merged.append({
                **turn,
                "start": best_b["start"],
                "end": best_b["end"],
                "text": best_b["text"],
                "confidence": best_b["confidence"],
            })
            continue
        merged.append(turn)

    for turn in turns_b_shifted:
        # Same-speaker only: a B turn that merely overlaps in TIME with a
        # different speaker's A turn (e.g. cross-talk one mic caught and the
        # other didn't) is not "already represented" and must still be
        # appended, or that speech is silently lost.
        overlaps_same_speaker_a = any(
            a["speaker"] == turn["speaker"]
            and overlap_seconds(turn["start"], turn["end"], a["start"], a["end"]) > 0
            for a in turns_a
        )
        if not overlaps_same_speaker_a:
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
        # Separate subdirectories: preprocess_audio names its output after the
        # source file's stem, so two sources sharing a stem (e.g. a "meeting.mp4"
        # + "meeting.m4a" pair, or two files from different folders that happen
        # to share a name) would otherwise overwrite each other's WAV in a
        # shared tmp_dir -- silently corrupting find_offset into comparing a
        # file against itself.
        dir_a = tmp_dir / "a"
        dir_b = tmp_dir / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        wav_a, turns_a, embeddings_a = _process_source(
            primary_path, dir_a, model_repo=model_repo, language=language,
            initial_prompt=initial_prompt, num_speakers=num_speakers,
            diarization_pipeline=diarization_pipeline,
        )
        wav_b, turns_b, embeddings_b = _process_source(
            secondary_path, dir_b, model_repo=model_repo, language=language,
            initial_prompt=initial_prompt, num_speakers=num_speakers,
            diarization_pipeline=diarization_pipeline,
        )

        offset, confidence = _correlate_envelopes(wav_a, wav_b)
        # Surface both on every run. Without this the tool computes the offset,
        # uses it, and never shows it -- a misaligned pair fuses into a
        # plausible-looking transcript with no signal that anything went wrong.
        print(f"Offset: {offset:+.1f}s (alignment confidence {confidence:.2f})")
        if confidence < OFFSET_CONFIDENCE_THRESHOLD:
            print(
                f"warning: weak alignment (confidence {confidence:.2f} < "
                f"{OFFSET_CONFIDENCE_THRESHOLD}). The two recordings may not overlap, "
                "or may not be of the same meeting. Check the fused transcript before "
                "trusting it.",
                file=sys.stderr,
            )
        speaker_map_a_to_b = match_speakers(embeddings_a, embeddings_b)
        speaker_map_b_to_a = {b: a for a, b in speaker_map_a_to_b.items()}
        turns_b_shifted = _shift_and_remap(turns_b, offset, speaker_map_b_to_a)

        merged = merge_turns(turns_a, turns_b_shifted)
        speaker_turns = relabel_speakers(smooth_micro_turns(merged))
        # Both sources produce hallucination loops independently, so fusion can
        # carry one through from whichever source won the turn.
        warn_on_repetition_loops(speaker_turns)

    output_dir.mkdir(parents=True, exist_ok=True)
    # ".fused.md", not ".md": the single-file path also names its output after the
    # primary's stem, so transcribing teams.mp4 and then fusing it against a second
    # recording used to write teams.md twice -- the fused run silently replacing the
    # single-file transcript. The suffix keeps both and records which pipeline
    # produced which.
    out_path = output_dir / f"{primary_path.stem}.fused.md"
    out_path.write_text(render_markdown(speaker_turns), encoding="utf-8")
    return out_path
