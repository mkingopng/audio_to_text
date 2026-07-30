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
- A Hugging Face token in `.env` as `HF_TOKEN=...`
- Accepted model terms on huggingface.co for **all three** of these gated pyannote models —
  the first one pulls in the other two, so accepting only the first still fails:
  - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
  - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
  - [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)

## Usage

Every media file in `data/`, writing one `.md` per file to `output/`:

```bash
uv run python src/transcribe.py
```

A specific file, when you know how many people were in the room:

```bash
uv run python src/transcribe.py "data/meeting.m4a" --num-speakers 6
```

Fuse two recordings of the same meeting:

```bash
uv run python src/transcribe.py "data/meeting-video.mp4" \
  --fuse "data/phone-recording.m4a" \
  --num-speakers 6
```

Optional audio cleanup before transcription (off by default):

```bash
uv run python src/transcribe.py --preprocess          # highpass + loudness normalization
uv run python src/transcribe.py --denoise             # adds FFT noise reduction
uv run python src/transcribe.py --audio-filter "..."  # your own ffmpeg -af chain
```

Bias spelling of names and jargon with `--prompt "Crisis Shield, ZeroW, Margu"`.

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
recordings recovered about 160 turns that a naive time-overlap check discarded.

## Development

```bash
uv sync
uv run pytest
```

See `bugs.md` for known limitations, `docs/superpowers/specs/` for design rationale.
