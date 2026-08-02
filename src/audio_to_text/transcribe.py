"""
Transcribe audio or video files to text using Whisper on the Apple GPU (MLX).

Runs OpenAI's Whisper models via Apple's MLX framework, which executes natively
on Apple Silicon GPUs (Metal) -- far faster than the CPU-only path of the
reference `openai-whisper` package. ffmpeg (used by MLX) decodes the audio track
directly, so audio files (.m4a, .mp3, .wav, ...) and video files (.mp4, .mov, ...)
both work through the same path. Audio is always extracted to a 16kHz mono WAV
first, since diarization needs one.

Installed as `audio-to-text` on PATH, so it can be called from any project. Run
with no arguments to transcribe every media file in ./data/, writing one .md per
file to ./data/transcriptions/ -- both relative to the current directory:

    audio-to-text

Or point it at a specific file or directory:

    audio-to-text "data/Crisis shield m1.m4a"
    audio-to-text "data/Crisis_shield_m2_3_July_2026.mov" --preprocess --denoise --prompt "Allan, Michael"

    audio-to-text meeting.mp4 --prompt "Crisis Shield, ZeroW, Margu"

Pass --preprocess to clean the audio with ffmpeg first (high-pass + loudness
normalization), and add --denoise for an extra FFT noise-reduction pass:

    audio-to-text --preprocess --denoise

Inside this repo, `uv run audio-to-text ...` runs the same entry point.
"""
from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from difflib import SequenceMatcher
from pathlib import Path

import mlx_whisper
import numpy as np
from dotenv import dotenv_values
from tqdm import tqdm

# Defaults follow the caller's working directory, not this checkout, so the tool
# behaves sensibly when invoked from another project. (PROJECT_ROOT is gone: after
# the package move, parent.parent would resolve to src/ and quietly point the
# defaults at the wrong place.)
DEFAULT_INPUT_SUBDIR = "data"
DEFAULT_OUTPUT_SUBDIR = Path("data") / "transcriptions"

# Where the Hugging Face token lives when this tool is called from another
# project, which has no .env of its own.
CONFIG_DIR = Path.home() / ".config" / "audio-to-text"

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
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        # e.g. --output-dir pointing at an existing FILE. Uncaught this exits with
        # a traceback rather than the "error: ..." line every other path produces,
        # which matters more now the tool is on PATH and run from anywhere.
        raise RuntimeError(f"cannot use '{out}' as the output directory: {exc}") from exc
    return out


def gather_media(target: Path | None) -> list[Path]:
    """Resolve a CLI target into a concrete list of media files.

    - None            -> every media file in ./data relative to the caller's cwd,
                         or [] if there is no such directory
    - a directory     -> every media file directly inside it
    - a specific file -> just that file (existence is checked in run_whisper())

    Deliberately non-recursive: the default output directory, ./data/transcriptions,
    nests inside the default input directory, and iterdir() + is_file() is what keeps
    a later batch run from descending into it.
    """
    if target is None:
        target = default_input_dir()
        if not target.is_dir():
            return []
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


# A Whisper repetition loop is a run of one token repeated far past anything
# speech produces. Calibrated against the 13,454-token reference transcript, whose
# longest legitimate consecutive run is 4 ("okay", "yeah", "easier" -- natural
# emphasis); an observed hallucination ran to 183. 10 clears the former with margin
# and catches the latter easily.
REPETITION_LOOP_THRESHOLD = 10


def detect_repetition_loops(
    turns: list[dict], *, threshold: int = REPETITION_LOOP_THRESHOLD
) -> list[dict]:
    """Find Whisper repetition loops in a transcript's token stream.

    Deliberately scanned DOC-WIDE, across turn boundaries, rather than within each
    turn. Diarization jitter shreds a loop across dozens of one-word turns (~31% of
    blocks in the reference output hold a single word), so a within-turn scan can
    only see loops that fragmentation happened not to scatter -- it reported
    "negligible" on a transcript that in fact carried a 183-token loop.

    Returns one entry per run, with the token, its length, and where it starts, so
    the caller can point at the span rather than just assert something is wrong.
    Upstream in Whisper and not fixable here; the point is that a run which
    hallucinates says so instead of looking clean.
    """
    tokens: list[tuple[str, float]] = [
        (word.lower().strip(".,?!\"'"), turn["start"])
        for turn in turns
        for word in turn["text"].split()
    ]

    loops = []
    i = 0
    while i < len(tokens):
        j = i
        while j + 1 < len(tokens) and tokens[j + 1][0] == tokens[i][0]:
            j += 1
        length = j - i + 1
        if length >= threshold and tokens[i][0]:
            loops.append({
                "token": tokens[i][0],
                "count": length,
                "start": tokens[i][1],
                "end": tokens[j][1],
            })
        i = j + 1
    return loops


# A shared span must be at least this long, and cover at least this share of the
# shorter block, before two blocks count as near-duplicates. Matches the criterion
# tools/analyze_transcript.py measures with, so the warning and the analysis agree.
_DUPLICATE_MIN_SHARED_CHARS = 40
_DUPLICATE_MIN_SHARE = 0.5
# How far apart, in blocks, to look. Measured on the real pair: every cross-speaker
# near-duplicate sits within 5.
_DUPLICATE_MAX_DISTANCE = 5


def detect_cross_speaker_duplicates(turns: list[dict]) -> list[dict]:
    """Find near-identical text emitted under two DIFFERENT speaker headings.

    One of the two headings is necessarily wrong: measured on the real pair, every
    such pair sits at a gap of 0.0 s between its spans -- one stretch of speech
    rendered twice -- and two speakers cannot both produce the same 40+ characters
    simultaneously. The cause is the two sources' diarizations disagreeing about
    who spoke, which no arbiter here can settle: the Hungarian embedding match is
    already the best signal available and is precisely what disagreed.

    So this reports rather than resolves. Silently picking one copy would turn a
    visibly odd artifact -- the same sentence under two names, which a reader
    notices -- into a confident-looking misattribution, which they do not. Deleting
    the correctly attributed copy is a real risk and the guard in merge_turns
    deliberately declines to take it.

    Blocks inside a repetition loop are skipped: degenerate text trivially matches
    itself, and on the real pair that accounted for 5 of 16 apparent pairs.
    """
    loops = detect_repetition_loops(turns)
    def in_loop(turn: dict) -> bool:
        return any(loop["start"] <= turn["start"] <= loop["end"] for loop in loops)

    def normalize(text: str) -> str:
        return re.sub(r"[^a-z0-9 ]", "", text.lower())

    found = []
    for distance in range(1, _DUPLICATE_MAX_DISTANCE + 1):
        for index in range(len(turns) - distance):
            first, second = turns[index], turns[index + distance]
            if first["speaker"] == second["speaker"]:
                continue
            if in_loop(first) or in_loop(second):
                continue
            a, b = normalize(first["text"]), normalize(second["text"])
            if not a or not b:
                continue
            shared = SequenceMatcher(None, a, b, autojunk=False).find_longest_match(
                0, len(a), 0, len(b)
            ).size
            if shared > _DUPLICATE_MIN_SHARED_CHARS and shared > _DUPLICATE_MIN_SHARE * min(len(a), len(b)):
                found.append({
                    "speakers": (first["speaker"], second["speaker"]),
                    "start": first["start"],
                    "shared_chars": shared,
                })
    return found


def warn_on_cross_speaker_duplicates(turns: list[dict]) -> None:
    """Tell the user which timestamps carry an uncertain speaker attribution."""
    duplicates = detect_cross_speaker_duplicates(turns)
    if not duplicates:
        return
    print(
        f"warning: {len(duplicates)} block pair(s) carry near-identical text under "
        "two different speakers, so one heading in each pair is wrong. The two "
        "recordings' diarizations disagreed about who spoke; this tool cannot tell "
        "which is right, so both copies are kept rather than silently choosing. "
        "Check the speaker labels at:",
        file=sys.stderr,
    )
    for duplicate in sorted(duplicates, key=lambda d: d["start"])[:10]:
        first, second = duplicate["speakers"]
        print(
            f"  {_format_timestamp(duplicate['start'])}  {first} vs {second} "
            f"({duplicate['shared_chars']} shared characters)",
            file=sys.stderr,
        )
    if len(duplicates) > 10:
        print(f"  ... and {len(duplicates) - 10} more", file=sys.stderr)


def warn_on_repetition_loops(turns: list[dict]) -> None:
    """Print a stderr warning for each repetition loop found (never raises/gates)."""
    for loop in detect_repetition_loops(turns):
        print(
            f"warning: possible Whisper repetition loop: {loop['token']!r} x{loop['count']} "
            f"({_format_timestamp(loop['start'])}-{_format_timestamp(loop['end'])}). "
            "That span is unlikely to be real speech -- review it before trusting it.",
            file=sys.stderr,
        )


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


# Agreement tokens that are GENUINE one-word turns in a meeting, as opposed to
# function words stolen out of a neighbour's sentence by diarization jitter. Of
# the <=2-word blocks in the reference output, 27% are these -- so a length
# threshold alone cannot separate the two populations.
BACKCHANNEL_TOKENS = frozenset({
    "yeah", "yep", "yes", "yup", "mm-hmm", "-hmm", "mhm", "uh-huh", "ok", "okay",
    "no", "nope", "right", "sure", "correct", "exactly", "great", "cool", "thanks",
})


def _is_backchannel(text: str) -> bool:
    """True when EVERY word is an agreement token.

    Checked per word, not over the whole string: the rule admits turns of up to two
    words, and squashing "Yeah yeah" into one key ("yeahyeah") put the commonest
    real two-word backchannels -- "yeah yeah", "no no", "yeah okay" -- outside the
    set they exist to protect. The hyphen is deliberately kept, or "mm-hmm" and
    "uh-huh" fall out of it too.
    """
    words = [re.sub(r"[^a-z-]", "", word.lower()) for word in text.split()]
    words = [word for word in words if word]
    return bool(words) and all(word in BACKCHANNEL_TOKENS for word in words)


def smooth_micro_turns(turns: list[dict]) -> list[dict]:
    """Re-attribute one- and two-word jitter fragments back into the sentence they
    were taken from.

    pyannote emits very short turns (42 of 96 under 0.5 s on a sampled slice, p10
    = 0.02 s); align_words_to_speakers assigns words to them and _group_consecutive
    faithfully starts a new block at each switch. The result is that roughly a
    quarter of blocks hold a single word, and half the document's headings
    introduce fragments like "So", "the", "it?".

    A fragment is absorbed only when ALL of these hold:

      sandwiched   -- the same speaker on both sides, different from the fragment.
                      Otherwise absorbing would move words between two people.
      <= 2 words   -- longer blocks are not jitter.
      zero duration - the actual jitter signal. Shortness alone is not: a short
                      turn that occupies real time is someone speaking. (Duration
                      < 0.5 s was measured and rejected -- it absorbs genuine
                      turns, including a 93-word one.)
      not backchannel - "yeah"/"mm-hmm" are real turns, see BACKCHANNEL_TOKENS.

    RE-ATTRIBUTION, NOT DELETION: the fragment rejoins its sentence and no word is
    ever lost. The discriminator is imperfect, so the failure mode must be a
    misattributed word rather than a missing one. Every one of the 19 candidates
    on the reference pair was hand-checked and all were jitter.
    """
    if not turns:
        return []

    smoothed = [dict(turns[0])]
    index = 1
    while index < len(turns):
        turn = turns[index]
        following = turns[index + 1] if index + 1 < len(turns) else None
        preceding = smoothed[-1]
        if (
            following is not None
            and preceding["speaker"] == following["speaker"] != turn["speaker"]
            and len(turn["text"].split()) <= 2
            and turn["end"] - turn["start"] == 0.0
            and not _is_backchannel(turn["text"])
        ):
            parts = [preceding["text"], turn["text"], following["text"]]
            preceding["text"] = " ".join(part for part in parts if part)
            preceding["end"] = following["end"]
            preceding["confidence"] = min(
                preceding["confidence"], turn["confidence"], following["confidence"]
            )
            index += 2
            continue
        smoothed.append(dict(turn))
        index += 1
    return smoothed


def group_into_turns(aligned_words: list[dict], *, smooth: bool = False) -> list[dict]:
    """Single-file convenience entry point: group, optionally smooth jitter, relabel.

    smooth defaults to False because smooth_micro_turns RE-ATTRIBUTES words from
    one speaker to another. Every other step here either preserves diarization's
    attribution or improves it against measured evidence; this one overrides it
    on a discriminator (zero duration + <=2 words + not backchannel) that was
    hand-checked on one meeting. That earns an opt-in, not a default -- a wrong
    absorption puts words in someone else's mouth, silently.
    """
    grouped = _group_consecutive(aligned_words)
    if smooth:
        grouped = smooth_micro_turns(grouped)
    return relabel_speakers(grouped)


def _format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
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


def resolve_hf_token() -> str | None:
    """Find the Hugging Face token, first hit wins.

    1. HF_TOKEN already in the environment
    2. ./.env  -- lets the calling project supply its own
    3. ~/.config/audio-to-text/.env  -- the global home for it

    Step 3 is what makes this tool usable from another project: a bare
    load_dotenv() only searches upward from the caller's cwd, so invoked from
    somewhere else it silently finds nothing.

    Uses dotenv_values rather than load_dotenv so reading a file never mutates
    os.environ -- otherwise the lookup order would depend on call history.
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


def load_diarization_pipeline(hf_token: str | None):
    """Load the pretrained pyannote diarization pipeline, preferring the Apple GPU (MPS)."""
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN is not set. Looked in: the HF_TOKEN environment variable, "
            f"'{Path.cwd() / '.env'}', and '{CONFIG_DIR / '.env'}'. When calling this "
            "tool from another project, put it in the last of those. Also make sure "
            "you've accepted the pyannote/speaker-diarization-3.1 model terms at "
            "https://huggingface.co/pyannote/speaker-diarization-3.1"
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

    pyannote 4.x's pipeline call always computes embeddings and returns a
    DiarizeOutput object (`.speaker_diarization`, `.speaker_embeddings`) rather
    than the `(Annotation, embeddings)` tuple `return_embeddings=True` produced
    on 3.x -- there is no longer a kwarg to request/suppress embeddings.
    """
    kwargs: dict = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    output = pipeline(str(wav_path), **kwargs)
    diarization = output.speaker_diarization
    embeddings_array = output.speaker_embeddings

    turns = [
        {"start": float(turn.start), "end": float(turn.end), "speaker": speaker}
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]
    turns.sort(key=lambda t: t["start"])

    # labels() verbatim, NOT sorted(labels()). pyannote reorders the embedding
    # rows to match this exact sequence, and it sorts with key=str -- re-sorting
    # with the default comparator can only preserve that correspondence or break
    # it. The two agree for "SPEAKER_NN" strings and disagree for the integer
    # labels pyannote also documents, where every embedding would silently
    # attach to the wrong speaker.
    speaker_labels = diarization.labels()
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
        # Without this, `python -m audio_to_text.transcribe` reports itself as
        # "transcribe.py" -- a name the user never types.
        prog="audio-to-text",
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
        help="Directory to write .md transcripts to "
             "(default: ./data/transcriptions/, created if absent).",
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
    parser.add_argument(
        "--smooth",
        action="store_true",
        help="Re-attribute one- and two-word zero-duration fragments into the "
             "surrounding sentence. Cuts the ~quarter of blocks holding a single "
             "word, at the risk of putting a genuine short turn under the wrong "
             "speaker. Off by default: it is the only option here that can change "
             "who a word is attributed to.",
    )
    args = parser.parse_args(argv)

    ensure_apple_silicon()

    # --fuse is validated FIRST: gather_media's "no media files found in X"
    # otherwise pre-empts it, so `--fuse b.m4a somedir/` reported an input-scan
    # failure instead of the actual mistake.
    if args.fuse is not None and (args.media is None or args.media.is_dir()):
        print("error: --fuse requires 'media' to be a single file, not a directory/default batch", file=sys.stderr)
        return 1

    media_files = gather_media(args.media)
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

    if args.fuse is not None:
        if args.media is None or args.media.is_dir():
            print("error: --fuse requires 'media' to be a single file, not a directory/default batch", file=sys.stderr)
            return 1
        if not args.media.exists():
            print(f"error: no such file: '{args.media}'", file=sys.stderr)
            return 1
        if not args.fuse.exists():
            print(f"error: no such file: '{args.fuse}'", file=sys.stderr)
            return 1
        # Deferred deliberately: fusion.py imports nine names from this module at
        # module level, so importing it at the top here would be a circular import
        # that fails at load time. Do not hoist this.
        from audio_to_text.fusion import run_fusion
        # Computed here rather than after this branch: --denoise/--preprocess/
        # --audio-filter used to be resolved below the early return, so the fusion
        # path silently ignored every one of them.
        fuse_filter = args.audio_filter or (
            build_audio_filter(args.denoise)
            if (args.preprocess or args.denoise or args.audio_filter is not None)
            else None
        )
        if fuse_filter:
            print(f"Preprocessing audio with ffmpeg filter: {fuse_filter}")
        try:
            diarization_pipeline = load_diarization_pipeline(resolve_hf_token())
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        try:
            output_dir = resolve_output_dir(args.output_dir)
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
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
                smooth=args.smooth,
                audio_filter=fuse_filter,
            )
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode(errors="replace") if exc.stderr else ""
            print(f"error: ffmpeg preprocessing failed:\n{stderr}", file=sys.stderr)
            return 1
        except (ValueError, RuntimeError) as exc:
            # Covers match_speakers's speaker-count-mismatch ValueError,
            # align_words_to_speakers's zero-turns ValueError, and any other
            # RuntimeError-shaped failure from the fusion pipeline -- without
            # this, these surface as a raw traceback after both sources'
            # full ASR + diarization passes have already run.
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"Fused transcript written to {out_path}")
        return 0

    do_preprocess = args.preprocess or args.denoise or args.audio_filter is not None
    audio_filter = args.audio_filter or (build_audio_filter(args.denoise) if do_preprocess else None)
    model_repo = resolve_model_repo(args.model)
    try:
        diarization_pipeline = load_diarization_pipeline(resolve_hf_token())
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        output_dir = resolve_output_dir(args.output_dir)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

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
            source: Path | None = None
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
                    speaker_turns = group_into_turns(aligned_words, smooth=args.smooth)
                except Exception as exc:
                    # Narrowly scoped to the diarization/alignment/grouping calls only
                    # (not the whole per-file try block) -- covers both
                    # align_words_to_speakers's ValueError on zero detected turns and
                    # arbitrary errors the pyannote pipeline itself can raise for one
                    # file's audio, without masking bugs elsewhere in the loop body.
                    raise DiarizationError(
                        f"diarization failed for '{media_path.name}': {exc}"
                    ) from exc
                # Outside the diarization try on purpose: this only prints a
                # warning, and it must never be the reason a file is reported as
                # "diarization failed" and skipped.
                warn_on_repetition_loops(speaker_turns)
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
            finally:
                # Free each cleaned WAV as soon as its file is done. The
                # TemporaryDirectory spans the whole batch, and preprocess_audio
                # writes a fresh 16kHz mono WAV per iteration (~115 MB per hour of
                # audio), so a folder of long recordings accumulated gigabytes
                # before anything was released -- and ffmpeg failed partway
                # through the batch on a full disk.
                # ONLY files we created inside tmp_dir. preprocess_audio is
                # expected to return a temp WAV, but this deletes things, and
                # "delete whatever that returned" would erase the user's original
                # recording the moment that ever stopped holding.
                if source is not None and source.parent == tmp_dir:
                    source.unlink(missing_ok=True)
            if not speaker_turns:
                # render_markdown([]) is a single newline. Writing it, announcing
                # success and exiting 0 reports a transcript that contains nothing.
                print(f"error: no speech found in '{media_path.name}'", file=sys.stderr)
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