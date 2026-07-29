"""
Transcribe audio or video files to text using Whisper on the Apple GPU (MLX).

Runs OpenAI's Whisper models via Apple's MLX framework, which executes natively
on Apple Silicon GPUs (Metal) -- far faster than the CPU-only path of the
reference `openai-whisper` package. ffmpeg (used by MLX) decodes the audio track
directly, so audio files (.m4a, .mp3, .wav, ...) and video files (.mp4, .mov, ...)
both work through the same path -- no separate audio-extraction step is required.

Run with no arguments (e.g. the IDE "Run" button) to transcribe every media
file in the project's data/ folder, writing a .txt per file to output/:

    uv run python src/transcribe.py

Or point it at a specific file or directory:

    uv run python src/transcribe.py "data/Crisis shield m1.m4a"
    uv run python src/transcribe.py "data/Crisis_shield_m2_3_July_2026.mov" --preprocess --denoise --prompt "Allan, Michael"

    uv run python src/transcribe.py meeting.mp4 --prompt "Crisis Shield, ZeroW, Margu"

Pass --preprocess to clean the audio with ffmpeg first (high-pass + loudness
normalization), and add --denoise for an extra FFT noise-reduction pass:

    uv run python src/transcribe.py --preprocess --denoise
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path

import mlx_whisper
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"

# Friendly model names -> Hugging Face repos of MLX-converted Whisper weights.
# Any value containing "/" is treated as a full HF repo id and used verbatim.
MODEL_REPOS = {
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "turbo": "mlx-community/whisper-large-v3-turbo",
}

# Media containers Whisper can read via ffmpeg (audio and video alike).
MEDIA_EXTENSIONS = {
    ".m4a", ".mp3", ".wav", ".flac", ".ogg", ".oga", ".opus", ".aac", ".m4b", ".wma",
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv",
}


def resolve_model_repo(name: str) -> str:
    """Map a friendly model name to an MLX Hugging Face repo (or pass a repo id through)."""
    if "/" in name:
        return name
    return MODEL_REPOS.get(name, name)


def gather_media(target: Path | None) -> list[Path]:
    """Resolve a CLI target into a concrete list of media files.

    - None            -> every media file in the default data/ folder
    - a directory     -> every media file directly inside it
    - a specific file -> just that file (existence is checked in run_whisper())
    """
    if target is None:
        target = DEFAULT_INPUT_DIR
    if target.is_dir():
        return sorted(
            p for p in target.iterdir()
            if p.is_file() and p.suffix.lower() in MEDIA_EXTENSIONS
        )
    return [target]


def build_audio_filter(denoise: bool) -> str:
    """Build an ffmpeg -af chain tuned for speech going into Whisper.

    - highpass  : remove sub-80 Hz rumble / handling noise / HVAC hum
    - afftdn     : optional gentle FFT noise reduction (opt-in; can add artifacts)
    - loudnorm   : EBU R128 loudness normalization so quiet/uneven levels are
                   brought to a consistent target Whisper handles better
    """
    parts = ["highpass=f=80"]
    if denoise:
        parts.append("afftdn=nf=-25")
    parts.append("loudnorm=I=-16:TP=-1.5:LRA=11")
    return ",".join(parts)


def build_ffmpeg_args(media_path: Path, cleaned_path: Path, audio_filter: str | None) -> list[str]:
    """Build the ffmpeg argv that extracts media_path to a 16kHz mono WAV.

    audio_filter is applied via -af only when given; extraction to 16kHz mono
    happens unconditionally either way (both run_whisper and run_diarization
    need the same WAV on the same time base).
    """
    args = ["ffmpeg", "-nostdin", "-y", "-i", str(media_path)]
    if audio_filter:
        args += ["-af", audio_filter]
    args += ["-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(cleaned_path)]
    return args


def preprocess_audio(media_path: Path, tmp_dir: Path, audio_filter: str | None) -> Path:
    """Extract media_path to a 16 kHz mono WAV inside tmp_dir, for Whisper and diarization alike.

    Returns the path to the WAV. Whisper resamples to 16 kHz mono anyway, so
    emitting that here costs nothing and lets ffmpeg do any requested filtering
    in the same pass.
    """
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("ffmpeg not found on PATH; cannot extract audio.")

    cleaned = tmp_dir / (media_path.stem + ".clean.wav")
    subprocess.run(build_ffmpeg_args(media_path, cleaned, audio_filter), check=True, capture_output=True)
    return cleaned


def overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Seconds of overlap between two [start, end) intervals (0.0 if disjoint).

    Public (not underscore-prefixed): fusion.py's merge_turns (Task 10) imports
    this directly rather than duplicating the same interval-overlap logic.
    """
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


class DiarizationError(RuntimeError):
    """Wraps any failure from diarizing/aligning/grouping one file.

    main()'s per-file loop already treats FileNotFoundError and
    subprocess.CalledProcessError as skip-this-file-and-continue; this lets it
    do the same for diarization-step failures (e.g. pyannote raising on one
    file's audio, or align_words_to_speakers's ValueError when zero speaker
    turns were detected) without having to catch bare Exception around the
    whole per-file try block.
    """


def align_words_to_speakers(words: list[dict], turns: list[dict]) -> list[dict]:
    """Assign each word to the diarization turn it overlaps most.

    Attribution happens per word (not per multi-second Whisper segment) so a
    speaker change mid-segment only misattributes the words actually on the
    wrong side of the change, not the whole segment.
    """
    if not turns:
        raise ValueError("align_words_to_speakers: no diarization turns to align against")

    aligned = []
    for word in words:
        best_turn = max(
            turns,
            key=lambda t: overlap_seconds(word["start"], word["end"], t["start"], t["end"]),
        )
        if overlap_seconds(word["start"], word["end"], best_turn["start"], best_turn["end"]) <= 0.0:
            # word falls in a silent gap between turns; attribute to whichever
            # turn boundary is closest in time.
            best_turn = min(
                turns,
                key=lambda t: min(abs(word["start"] - t["end"]), abs(t["start"] - word["end"])),
            )
        aligned.append({**word, "speaker": best_turn["speaker"]})
    return aligned


def _join_words(words: list[str]) -> str:
    """Whisper word strings already carry leading spaces/punctuation; join without adding more."""
    return "".join(words).strip()


def _group_consecutive(aligned_words: list[dict]) -> list[dict]:
    """Merge consecutive same-speaker words into turns, keeping original speaker ids.

    Returned turns are NOT relabeled to "Person N" -- fusion (Task 10) needs the
    original diarization ids to merge two sources before a single final relabel.
    """
    if not aligned_words:
        return []

    turns: list[dict] = []
    current_speaker = aligned_words[0]["speaker"]
    current_words: list[str] = []
    current_probs: list[float] = []
    current_start = aligned_words[0]["start"]
    current_end = current_start

    def _flush() -> None:
        turns.append({
            "speaker": current_speaker,
            "start": current_start,
            "end": current_end,
            "text": _join_words(current_words),
            "confidence": sum(current_probs) / len(current_probs),
        })

    for word in aligned_words:
        if word["speaker"] != current_speaker:
            _flush()
            current_speaker = word["speaker"]
            current_words = []
            current_probs = []
            current_start = word["start"]
        current_words.append(word["word"])
        current_probs.append(word["probability"])
        current_end = word["end"]
    _flush()
    return turns


def relabel_speakers(turns: list[dict]) -> list[dict]:
    """Rename each turn's speaker id to "Person N" by first-appearance order.

    A speaker id that recurs later in the list keeps the number it was first
    assigned -- speaker identity must not drift/renumber partway through.
    """
    labels: dict[str, str] = {}

    def _label_for(speaker_id: str) -> str:
        if speaker_id not in labels:
            labels[speaker_id] = f"Person {len(labels) + 1}"
        return labels[speaker_id]

    return [{**turn, "speaker": _label_for(turn["speaker"])} for turn in turns]


def group_into_turns(aligned_words: list[dict]) -> list[dict]:
    """Single-file convenience entry point: group, then relabel to Person N."""
    return relabel_speakers(_group_consecutive(aligned_words))


def _format_timestamp(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def render_markdown(turns: list[dict]) -> str:
    """Render relabeled turns (speaker == "Person N") as heading-per-turn Markdown."""
    blocks = [
        f"## {turn['speaker']} — {_format_timestamp(turn['start'])}\n\n{turn['text']}\n"
        for turn in turns
    ]
    return "\n".join(blocks).rstrip() + "\n"


def run_whisper(
    media_path: Path,
    *,
    model_repo: str,
    language: str | None = "en",
    initial_prompt: str | None = None,
    verbose: bool | None = None,
    word_timestamps: bool = True,
) -> dict:
    """Transcribe one audio/video file on the Apple GPU and return the result dict.

    word_timestamps defaults to True: word-level timing is what lets
    align_words_to_speakers attribute speaker identity per-word instead of
    per multi-second segment.
    """
    if not media_path.exists():
        raise FileNotFoundError(f"Input file not found: {media_path}")

    decode_options: dict = {}
    if language is not None:
        decode_options["language"] = language
    return mlx_whisper.transcribe(
        str(media_path),
        path_or_hf_repo=model_repo,
        initial_prompt=initial_prompt,
        verbose=verbose,
        word_timestamps=word_timestamps,
        **decode_options,
    )


def extract_words(result: dict) -> list[dict]:
    """Flatten a run_whisper() result's per-segment word lists into one chronological list."""
    return [word for segment in result["segments"] for word in segment["words"]]


def load_diarization_pipeline(hf_token: str | None):
    """Load the pretrained pyannote diarization pipeline, preferring the Apple GPU (MPS)."""
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN is not set. Add it to the project's .env file (HF_TOKEN=...) and "
            "make sure you've accepted the pyannote/speaker-diarization-3.1 model terms "
            "at https://huggingface.co/pyannote/speaker-diarization-3.1 -- see the design "
            "spec (docs/superpowers/specs/2026-07-29-speaker-diarization-fusion-design.md, "
            "section 8) for details."
        )
    from pyannote.audio import Pipeline
    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=hf_token)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load the pyannote diarization pipeline. This usually means either "
            "HF_TOKEN is invalid/expired, or the pyannote/speaker-diarization-3.1 model "
            "terms haven't been accepted yet at "
            "https://huggingface.co/pyannote/speaker-diarization-3.1. "
            f"Underlying error: {exc}"
        ) from exc
    try:
        import torch
        pipeline.to(torch.device("mps"))
    except Exception:
        print("warning: could not move diarization pipeline to MPS; falling back to CPU", file=sys.stderr)
    return pipeline


def run_diarization(
    wav_path: Path, pipeline, *, num_speakers: int | None = None
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Run a (possibly fake, for testing) pyannote-style pipeline and parse its output.

    Returns (turns, embeddings): turns sorted by start; embeddings keyed by
    speaker id, one row per sorted(diarization.labels()).
    """
    kwargs: dict = {"return_embeddings": True}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    diarization, embeddings_array = pipeline(str(wav_path), **kwargs)

    turns = [
        {"start": float(turn.start), "end": float(turn.end), "speaker": speaker}
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]
    turns.sort(key=lambda t: t["start"])

    speaker_labels = sorted(diarization.labels())
    embeddings = {label: embeddings_array[i] for i, label in enumerate(speaker_labels)}

    if num_speakers is not None and len(speaker_labels) != num_speakers:
        print(
            f"warning: requested num_speakers={num_speakers} but diarization found "
            f"{len(speaker_labels)} speaker(s) in '{wav_path.name}'",
            file=sys.stderr,
        )

    return turns, embeddings


def ensure_apple_silicon() -> None:
    """Fail fast on non-Apple-Silicon hosts -- MLX is the only engine, no fallback."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        print(
            "error: mlx-whisper requires Apple Silicon (arm64 macOS); "
            f"this host is {platform.system()}/{platform.machine()}.",
            file=sys.stderr,
        )
        raise SystemExit(2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe audio or video files to text using Whisper on the Apple GPU (MLX).",
    )
    parser.add_argument(
        "media",
        type=Path,
        nargs="?",
        default=None,
        help="Audio/video file or directory (default: the project's data/ folder).",
    )
    parser.add_argument(
        "--model",
        default="turbo",
        help="Model name (turbo, large-v3) or a full MLX Hugging Face repo id "
             "(default: turbo -- best speed on the GPU, near-large-v3 accuracy).",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language code, or 'auto' to let Whisper detect it (default: en).",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="initial_prompt to bias the spelling of names/jargon.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write .txt outputs to (default: the project's output/ folder).",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Clean audio with ffmpeg first (high-pass + loudness normalization).",
    )
    parser.add_argument(
        "--denoise",
        action="store_true",
        help="Add an FFT noise-reduction pass to preprocessing (implies --preprocess).",
    )
    parser.add_argument(
        "--audio-filter",
        default=None,
        help="Override the ffmpeg -af filter chain used when preprocessing.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Also stream each segment to the console as it is decoded.",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Exact speaker count for diarization, if known (improves clustering accuracy). "
             "Default: auto-detect.",
    )
    parser.add_argument(
        "--fuse",
        type=Path,
        default=None,
        help="A second recording of the SAME meeting (e.g. a phone recording from a "
             "different position). Triggers dual-source fusion: 'media' must be a "
             "single file, not a directory.",
    )
    args = parser.parse_args(argv)

    ensure_apple_silicon()

    media_files = gather_media(args.media)
    if not media_files:
        where = args.media or DEFAULT_INPUT_DIR
        print(f"error: no media files found in '{where}'", file=sys.stderr)
        return 1

    if args.fuse is not None:
        if args.media is None or args.media.is_dir():
            print("error: --fuse requires 'media' to be a single file, not a directory/default batch", file=sys.stderr)
            return 1
        load_dotenv()
        diarization_pipeline = load_diarization_pipeline(os.environ.get("HF_TOKEN"))
        output_dir = (args.output_dir or DEFAULT_OUTPUT_DIR).resolve()
        from src.fusion import run_fusion
        try:
            out_path = run_fusion(
                args.media,
                args.fuse,
                model_repo=resolve_model_repo(args.model),
                language=None if args.language == "auto" else args.language,
                initial_prompt=args.prompt,
                num_speakers=args.num_speakers,
                output_dir=output_dir,
                diarization_pipeline=diarization_pipeline,
            )
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode(errors="replace") if exc.stderr else ""
            print(f"error: ffmpeg preprocessing failed:\n{stderr}", file=sys.stderr)
            return 1
        print(f"Fused transcript written to {out_path}")
        return 0

    do_preprocess = args.preprocess or args.denoise or args.audio_filter is not None
    audio_filter = args.audio_filter or (build_audio_filter(args.denoise) if do_preprocess else None)
    model_repo = resolve_model_repo(args.model)

    load_dotenv()
    diarization_pipeline = load_diarization_pipeline(os.environ.get("HF_TOKEN"))

    output_dir = (args.output_dir or DEFAULT_OUTPUT_DIR).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {model_repo} (first run downloads the weights from Hugging Face)")
    if do_preprocess:
        print(f"Preprocessing audio with ffmpeg filter: {audio_filter}")

    # Outer progress bar across files (only shown for a batch); MLX draws its own
    # per-file bar for progress within each file.
    file_iter: Iterable[Path] = media_files
    if len(media_files) > 1:
        file_iter = tqdm(media_files, desc="Files", unit="file")

    failures = 0
    with tempfile.TemporaryDirectory(prefix="whisper_clean_") as tmp:
        tmp_dir = Path(tmp)
        for media_path in file_iter:
            try:
                source = preprocess_audio(media_path, tmp_dir, audio_filter)
                print(f"Transcribing '{media_path.name}' ...")
                result = run_whisper(
                    source,
                    model_repo=model_repo,
                    language=None if args.language == "auto" else args.language,
                    initial_prompt=args.prompt,
                    verbose=True if args.verbose else None,
                )
                print(f"Diarizing '{media_path.name}' ...")
                try:
                    turns, _embeddings = run_diarization(
                        source, diarization_pipeline, num_speakers=args.num_speakers
                    )
                    aligned_words = align_words_to_speakers(extract_words(result), turns)
                    speaker_turns = group_into_turns(aligned_words)
                except Exception as exc:
                    # Narrowly scoped to the diarization/alignment/grouping calls only
                    # (not the whole per-file try block) -- covers both
                    # align_words_to_speakers's ValueError on zero detected turns and
                    # arbitrary errors the pyannote pipeline itself can raise for one
                    # file's audio, without masking bugs elsewhere in the loop body.
                    raise DiarizationError(
                        f"diarization failed for '{media_path.name}': {exc}"
                    ) from exc
            except FileNotFoundError as exc:
                print(f"error: {exc}", file=sys.stderr)
                failures += 1
                continue
            except subprocess.CalledProcessError as exc:
                stderr = exc.stderr.decode(errors="replace") if exc.stderr else ""
                print(f"error: ffmpeg preprocessing failed for '{media_path.name}':\n{stderr}",
                      file=sys.stderr)
                failures += 1
                continue
            except DiarizationError as exc:
                print(f"error: {exc}", file=sys.stderr)
                failures += 1
                continue
            # Name the output after the original file, not the temp cleaned WAV.
            out_path = output_dir / f"{media_path.stem}.md"
            out_path.write_text(render_markdown(speaker_turns), encoding="utf-8")
            print(f"  -> wrote {out_path}")

    print(f"\nDone. {len(media_files) - failures}/{len(media_files)} file(s) transcribed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())