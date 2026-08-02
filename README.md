# audio_to_text

Turns recorded meetings into speaker-attributed Markdown transcripts, running Whisper
on the Apple Silicon GPU via MLX.

Two modes:

- **Single file** — transcribe one recording, split into paragraphs by who's speaking.
- **Fusion** — combine *two recordings of the same meeting* (e.g. a Teams capture and a
  phone sitting elsewhere in the room) into one transcript that takes the clearer audio
  from whichever source caught each moment better.

## Requirements

- Apple Silicon Mac (MLX has no CPU fallback here — the tool exits if it isn't arm64 macOS)
- `ffmpeg` on `PATH`
- A Hugging Face token, found in the first of these that has one:
  1. the `HF_TOKEN` environment variable
  2. `./.env` in the directory you run from — lets a project supply its own
  3. `~/.config/audio-to-text/.env` — **use this one** to call the tool from other projects
- Accepted model terms on huggingface.co for **all three** of these gated pyannote models —
  the first one pulls in the other two, so accepting only the first still fails:
  - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
  - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
  - [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)

## Install

The tool is callable from any project via a small wrapper on `PATH`. It delegates to this
repo's own venv rather than installing a second copy of the ~1.2 GB of ML dependencies, so
edits here take effect immediately with no reinstall step.

Create `~/.local/bin/audio-to-text`:

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO="${AUDIO_TO_TEXT_REPO:-$HOME/Developer/GitHub/audio_to_text}"

if [[ ! -d "$REPO" ]]; then
  echo "audio-to-text: repo not found at '$REPO'." >&2
  echo "Set AUDIO_TO_TEXT_REPO to point at your audio_to_text checkout." >&2
  exit 1
fi

exec uv run --project "$REPO" python -m audio_to_text.transcribe "$@"
```

Then `chmod +x ~/.local/bin/audio-to-text`, and put the token where any project can find it:

```bash
mkdir -p ~/.config/audio-to-text
cp .env ~/.config/audio-to-text/.env
chmod 600 ~/.config/audio-to-text/.env
```

`uv run --project` deliberately does *not* change the working directory — that is what lets
the `./data/transcriptions/` default resolve against the directory you called from. Don't
swap it for `--directory`.

The wrapper hard-codes this checkout's path, so moving or deleting the repo breaks the
command until you set `AUDIO_TO_TEXT_REPO`.

## Usage

Paths and outputs are relative to wherever you run the command.

Every media file in `./data/`, writing one `.md` per file to `./data/transcriptions/`:

```bash
audio-to-text
```

A specific file, when you know how many people were in the room:

```bash
audio-to-text "data/meeting.m4a" --num-speakers 6
```

Fuse two recordings of the same meeting:

```bash
audio-to-text "data/meeting-video.mp4" \
  --fuse "data/phone-recording.m4a" \
  --num-speakers 6
```

A fused run writes `<primary-stem>.fused.md`, so it sits alongside — rather than
overwrites — the `<primary-stem>.md` a single-file run of the same recording produces.

Fold one- and two-word jitter fragments back into the surrounding sentence:

```bash
audio-to-text "data/meeting.m4a" --smooth
```

Off by default. Diarization scatters single words (`"So"`, `"the"`) into their own
blocks under the wrong speaker, and `--smooth` re-attributes them to the speaker
either side — moving words, never deleting them. It is the only option here that
changes who a word is attributed to, so it stays opt-in: a genuine short turn
absorbed by mistake becomes a misattributed one, silently. On the reference pair
it removed 6.8% of blocks with zero words lost.

Optional audio cleanup before transcription (off by default):

```bash
audio-to-text --preprocess          # highpass + loudness normalization
audio-to-text --denoise             # adds FFT noise reduction
audio-to-text --audio-filter "..."  # your own ffmpeg -af chain
```

These apply to fused runs too — `--fuse other.m4a --denoise` cleans both sources.

Transcripts go to `./data/transcriptions/`, created if it doesn't exist. Override with
`--output-dir`.

Bias spelling of names and jargon with `--prompt "Crisis Shield, ZeroW, Margu"`.

## What the tool tells you about its own output

Speech recognition and speaker diarization both fail in ways that produce
confident-looking, entirely wrong text. Rather than hide that, a run reports what it
knows it may have got wrong. None of these stop the run — they tell you where to look.

**Alignment confidence** (fused runs). Every fused run prints the offset it found between
the two recordings and how trustworthy that offset is:

```
Offset: +26.1s (alignment confidence 1.52)
```

Above ~1.2 the two recordings clearly share acoustic content. Near 1.00 they do not, and
you get a warning — the two files may not be the same meeting, or may barely overlap.
Measured on a real pair: a true match scored 1.52, unrelated audio 1.000–1.005.

Treat the 1.2 line as approximate. Those figures were measured before the correlation was
mean-subtracted, so they describe a slightly different computation than the one now
running, and they have not been re-measured against a real pair — see `bugs.md`. The
threshold only ever warns, so a mis-set value costs you a spurious warning or a missing
one, never a different transcript.

**Repetition loops.** Whisper sometimes gets stuck on quiet or ambiguous audio and emits
one word hundreds of times. It is an upstream flaw and this tool cannot fix it, but it
will tell you when it happened:

```
warning: possible Whisper repetition loop: 'paul' x200 (17:57-18:00).
```

**Uncertain speaker attribution** (fused runs). When both recordings' diarizations
disagree about who was speaking, the same sentence can appear under two names — one of
which is wrong. The tool cannot tell which, so it keeps both and points at them:

```
warning: 11 block pair(s) carry near-identical text under two different speakers ...
  14:12  Person 4 vs Person 1 (101 shared characters)
```

Choosing a copy silently would turn something you would notice while reading into a
confident-looking misattribution you would not.

See `bugs.md` for the measured incidence of each and what is still open.

## Output

Markdown, one block per speaker turn:

```markdown
## Person 1 — 00:02

but here you can see create client, I'll just go confirm and then...

## Person 2 — 00:33

So the feedback I got from Justin is that you would probably prefer the chat...
```

Speakers are numbered, not named — voice prints can't tell you who someone *is*. The
numbering is stable: watch the first few minutes, work out who `Person 1`–`Person 6` are,
then find-and-replace each label with a real name across the whole document. A speaker who
stops talking and comes back later keeps their original number.

## How it works

**Word-level attribution.** Whisper returns segments that can run several seconds and span a
speaker change. Attributing whole segments would misfile every word on the wrong side of that
change, so the pipeline requests per-word timestamps and assigns each *word* to whichever
diarization turn it overlaps most, then groups consecutive same-speaker words back into
paragraphs.

**Fusion — three problems, three solutions.** Two recordings of one meeting start at
different times, cluster speakers independently, and disagree about who said what:

1. *Different clocks.* The two files are cross-correlated on their RMS energy envelopes to
   recover the offset between them — correlating energy rather than raw samples is faster and
   survives the two microphones hearing quite different things.
2. *Different speaker labels.* Each source's diarization invents its own `SPEAKER_00`…`_05`.
   Their voice embeddings are matched with the Hungarian algorithm, which finds the globally
   optimal pairing. Greedy nearest-match was rejected: with two similar voices it can spend
   the obvious match early and get forced into a bad leftover pairing.
3. *Two versions of the same sentence.* The primary source's turn boundaries are kept as
   canonical, and the secondary's text replaces a turn only where its average word confidence
   is strictly higher. Selection happens per *turn*, not per word — splicing two independent
   ASR passes word-by-word garbles sentences wherever the passes segment speech differently.

Speech only one microphone caught is appended rather than dropped, which on the reference
recordings recovered about 160 turns that a naive time-overlap check discarded. "Already
covered" is judged by how much of a turn the other source overlaps, not by whether it
overlaps at all — a single shared instant used to suppress a whole turn, so a source that
kept recording after the other cut out lost everything it caught alone.

## Development

```bash
uv sync
uv run pytest
```

Inside the repo, `uv run audio-to-text ...` runs the same entry point as the wrapper.

See `bugs.md` for known limitations, `docs/superpowers/specs/` for design rationale.
