# VALIDATE — "the pipeline is CPU-bound; can MLX accelerate it?"

**Date:** 2026-07-31
**Branch:** feature/mlx-whisper-gpu
**Question posed:** the latest version is slow; is it CPU-bound, and can MLX fix it?
**Verdict: the premise is FALSE.** The pipeline is GPU-bound and already saturates the
GPU. MLX is already the ASR engine, and the diarizer already runs on the same GPU via
MPS. There is no CPU-bound work of any consequence to move.

---

## 1. Hardware and workload under test

| | |
|---|---|
| Machine | Apple M4 (base), 10 CPU cores (4P/6E), **10-core GPU**, 24 GB unified memory |
| ASR | `mlx-whisper` 0.4.3, `mlx-community/whisper-large-v3-turbo`, `word_timestamps=True` |
| Diarization | `pyannote.audio` 4.0.7, `pyannote/speaker-diarization-3.1`, torch 2.12.1 on MPS |
| Real inputs | two ~70 min recordings (4239 s and 4198 s) |
| Probe slice | first 300 s of the meeting recording, 16 kHz mono WAV |

## 2. Evidence — per-stage wall time vs process CPU time

`cpu/wall` is the discriminator. A stage pegging cores shows a ratio ≥ 1; a stage
*waiting on an accelerator* shows a ratio well below 1.

`scratchpad/profile_stages.py 300`:

```
  [ffmpeg_extract]  wall=   0.11s cpu=   0.13s ratio= 1.17
  [whisper_mlx]     wall=  34.36s cpu=   5.39s ratio= 0.16     <- 819 words
  [pyannote_load]   wall=   4.63s cpu=   2.08s ratio= 0.45
  [diarize_mps]     wall=  24.45s cpu=   5.48s ratio= 0.22     <- 70 turns
  [diarize_cpu]     wall= 165.52s cpu= 545.58s ratio= 3.30     <- forced-CPU probe
  [align_words]     wall=   0.01s cpu=   0.01s ratio= 1.00
  [group_turns]     wall=   0.00s cpu=   0.00s ratio= 1.00

pipeline wall total (excl. cpu-probe): 63.57s for 300s audio -> 4.72x realtime
```

Share of pipeline wall time: **whisper 54.0%, diarization 38.5%, pipeline load 7.3%,
everything else 0.2%.**

Three things follow immediately:

1. **The two heavy stages are not CPU-bound.** Ratios of 0.16 and 0.22 mean the process
   sat idle waiting on the GPU for ~80% of its wall time.
2. **MPS is genuinely working, not silently falling back to CPU.** This was the specific
   risk in `load_diarization_pipeline` (`transcribe.py:319-324`), whose `.to("mps")` is
   wrapped in a bare `except` that would mask a failure. Forcing the same pipeline onto
   CPU takes **165.52 s vs 24.45 s — 6.8× slower**, with a cpu/wall ratio of 3.30
   (3.3 cores busy). The GPU path is real and is already earning its keep.
3. **The pure-Python glue is free.** `align_words_to_speakers` is O(words × turns) and I
   expected it to be a hot spot; it measured 0.01 s. Extrapolated to the full 70 min
   (~17 k words × ~1.7 k turns, ~196× the slice's work) it is still ~2 s against a
   ~15 min run. Not worth touching.

## 3. Evidence — the GPU is *saturated*, so there is no headroom to exploit

Low CPU usage alone doesn't prove the GPU is busy — it could be poorly occupied, in which
case overlapping stages would be a free win. Sampling macOS `Device Utilization %` via
`ioreg` during each stage (`scratchpad/probe_gpu_concurrency.py`):

```
idle baseline    gpu busy%: mean= 16.1 median= 17.0 max= 29.0
whisper  34.40s  gpu busy%: mean= 94.9 median= 98.0 p90=100.0 max=100.0
diarize  24.83s  gpu busy%: mean= 96.6 median= 98.0 p90=100.0 max=100.0
```

Both stages hold the GPU at **95–97% mean occupancy**. The M4's 10-core GPU is the
binding constraint, and it is already pinned.

The concurrency test confirms it — running Whisper and pyannote in two threads:

```
sequential total = 59.24s
concurrent total = 54.32s   gpu busy%: mean= 99.1
  (whisper 34.40 -> 54.32s, diarize 24.83 -> 40.34s: both stretch, nothing is free)
```

That nominal 8.3% is **inside measurement noise** (see §5) and should be treated as zero.
Each stage slowed by roughly the amount the other gained: work was time-sliced on a
saturated device, not overlapped.

## 4. Evidence — the obvious "reduce GPU work" levers are already optimal or negative

Since the GPU is the constraint, only doing *less GPU work* can help. Both cheap levers
were measured and both are losses (`probe_reduce_work.py`, `probe_pyannote_tunables.py`):

| Lever | Result | Verdict |
|---|---|---|
| Whisper 4-bit weights (`...turbo-q4`) | 43.85 s vs 39.86 s fp16 (both cached, back-to-back) | **Slower** — dequantization costs more than the bandwidth it saves at this size |
| ↳ its output fidelity | text similarity **0.525** vs fp16 baseline | Disqualifying on its own — a different transcript, not a faster one |
| Whisper 8-bit (`...turbo-q8`) | 404 — repo does not exist | N/A |
| `segmentation/embedding_batch_size` 32→64 | 29.28 s → 34.77 s | **Slower** |
| ↳ 32→128 | 29.28 s → 46.38 s | **Much slower** — bigger batches thrash a saturated GPU |

The shipped configuration (fp16 turbo, batch size 32) is already the best of the
measured options.

## 5. Measurement variance — stated so §3 and §4 aren't over-read

Repeated identical measurements across the session, in chronological order:

- Whisper fp16, 300 s slice: **34.36, 34.40, 34.80, 39.86 s** (spread ~16%)
- Diarization MPS, 300 s slice: **24.45, 24.83, 26.90, 29.28 s** (spread ~20%)

Both drift *monotonically upward* over a session of sustained GPU load — consistent with
thermal throttling on a passively-constrained M4. **Consequences:** (a) any claimed win
under ~20% is unproven without repeated interleaved trials; the 8.3% concurrency result
does not clear that bar. (b) The §4 batch-size sweep ran in ascending order, so part of
that degradation is drift — but 29.28→46.38 s is far outside the noise band, so the
direction of that conclusion holds.

## 6. Root cause of the slowness

Not a defect, and not a misplaced workload. **The work is genuinely large and the machine
is small for it:**

- Sustained throughput is **4.72× realtime** for one source, on a saturated 10-core GPU.
- A 70-minute recording therefore costs **~15 minutes** per source.
- `run_fusion` (`fusion.py:192-201`) processes both sources **sequentially and in full** —
  two ASR passes and two diarization passes — so the fusion path costs **~30 minutes**.

That is the number the user is feeling. It is the product of the design (dual-source
fusion doubles everything; `word_timestamps=True` is required for the word-level
attribution the whole transcript quality rests on) meeting a base-model M4 GPU.

## 7. Answering the question as asked

**"Can we use MLX to accelerate this?" — No, not meaningfully.**

1. **Whisper is already on MLX.** 54% of the runtime is `mlx_whisper.transcribe`
   (`transcribe.py:283`) already executing on Metal at ~95% GPU occupancy.
2. **The diarizer is already on the GPU**, via torch-MPS, verified 6.8× faster than its
   CPU path (§2). Porting it to MLX changes *which Metal API* issues the work, not
   whether the work runs on the GPU.
3. **Amdahl caps the prize.** Diarization is 38.5% of wall time. Even an infinitely fast
   diarizer only buys 38.5%. A realistic MLX-vs-MPS efficiency gain on an
   already-saturated GPU (~1.2–1.5×) yields **~8–13% end-to-end** — the same order as the
   thermal noise in §5.
4. **The cost is high and the risk is real.** Research (§8) found MLX *weights* for both
   models pyannote 3.1 uses, but **no packaged Python MLX diarization pipeline**. The
   glue — powerset decoding, sliding-window aggregation, embedding extraction,
   agglomerative clustering — would have to be reimplemented, against a diarization
   quality baseline this repo has spent days stabilising (`bugs.md`,
   `docs/validate/2026-07-31-bugs-md-triage.md`).

Spending days to reimplement the risky part of the pipeline for a gain the measurement
noise can hide is a bad trade.

## 8. What the web research actually found

- **MLX-native diarization models exist; a Python pipeline does not.**
  [`mlx-community/pyannote-segmentation-3.0-mlx`](https://huggingface.co/mlx-community/pyannote-segmentation-3.0-mlx)
  is **segmentation only** — its own card says full diarization "isn't included" — and it
  ships no pip package, only a clone-this-repo snippet.
  [`mlx-community/wespeaker-voxceleb-resnet34-LM`](https://huggingface.co/mlx-community/wespeaker-voxceleb-resnet34-LM)
  supplies the embedding half. The one toolkit that assembles them,
  [soniqo/speech-swift](https://github.com/soniqo/speech-swift), is **Swift**, not Python.
- **[Senko](https://github.com/narcotic-sh/senko)** is the fastest option found (claims
  1 h in 7.7 s on M3) — but it is **CoreML/ANE, not MLX**, its documented API exposes
  only merged segments and **not the per-speaker embeddings** that `match_speakers`
  (`fusion.py:54`) requires for Hungarian matching, and it targets Python 3.13. Adopting
  it would mean rebuilding speaker matching, i.e. the fusion feature's core.
- **pyannote 3.1 is the fast branch already.** It removed the onnxruntime dependency and
  runs segmentation and embedding in pure PyTorch; the documented MPS caveat is
  unimplemented-operator fallback, which our §2 CPU probe shows is not biting here.

## 9. Falsification — what would prove this diagnosis wrong, and what looking found

The diagnosis dies if **the GPU turns out not to be the binding constraint**. Three
independent checks, each of which could have killed it:

| Falsifier | Predicted if I'm WRONG (CPU-bound) | Measured |
|---|---|---|
| cpu/wall on the heavy stages | ≥ 1.0 | **0.16 and 0.22** — refuted |
| Forcing diarization to CPU | similar time (MPS not really used) | **6.8× slower** (165.52 s vs 24.45 s) — refuted |
| GPU occupancy during the heavy stages | low, with headroom to overlap | **94.9% / 96.6% mean**, and concurrency returned nothing outside noise — refuted |

I also looked for the CPU-bound hot spot the premise predicts, in the one place it was
most plausible — the O(words × turns) Python loop in `align_words_to_speakers`. It
measured **0.01 s, 0.02% of wall time**. There is no CPU-bound stage to find.

**The diagnosis remains standing but is bounded**, and honesty requires naming what was
*not* tested: the M4's GPU is small (10 cores). These ratios are hardware-specific. On a
machine with a much larger GPU the balance could shift and the CPU-side glue could start
to matter — but that is not the machine in question.

## 10. Highest-impact fix, justified by the evidence

Ranked by measured impact per unit of risk. **None of these is "port to MLX".**

1. **Don't re-run the pipeline you've already run.** The heaviest waste is not in a stage,
   it's in repetition: every re-run of a 70-min file redoes ~15 min of saturated-GPU work
   that produced identical intermediates. Caching ASR + diarization output per
   (file content hash, model, params) makes iteration ~free and is **zero risk to
   transcript quality** — it changes nothing about what is computed, only how often.
   This is the only change that can plausibly deliver an order of magnitude.
2. **Give the user progress and a cheap preview.** The 30-minute fusion run is currently
   near-silent per source. A `--limit-seconds`-style probe run makes a bad `--num-speakers`
   guess cost 1 minute instead of 30. Cost-of-being-wrong is a real component of the
   felt slowness.
3. **Only then, and only if the user will trade accuracy for speed:** `segmentation_step`
   is 0.1, i.e. a 10 s window advancing 1 s — roughly 10× redundant segmentation compute.
   Raising it is the single largest remaining GPU-work reduction, but it *will* move
   diarization output, and this repo has just spent days stabilising that. Needs its own
   measured accuracy/speed curve before anyone touches it.

**Explicitly rejected, with evidence:** MLX port of diarization (§7, ≤13% for days of
work), quantized Whisper weights (§4, slower *and* similarity 0.525), larger batch sizes
(§4, monotonically slower), stage concurrency (§3, within noise).

---

## Reproduction

```bash
uv run python scratchpad/profile_stages.py 300        # §2 stage table
uv run python scratchpad/probe_gpu_concurrency.py     # §3 GPU occupancy + concurrency
uv run python scratchpad/probe_reduce_work.py         # §4 quantization + fidelity
uv run python scratchpad/probe_pyannote_tunables.py   # §4 batch sweep, §5 repeat timings
```

Probe scripts live in the session scratchpad
(`/private/tmp/claude-501/.../scratchpad/`) and are not committed.
