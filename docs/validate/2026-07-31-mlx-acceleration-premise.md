# VALIDATE — "the pipeline is CPU-bound; can MLX accelerate it?"

**Date:** 2026-07-31
**Branch:** feature/mlx-whisper-gpu (repo refactored to `src/audio_to_text/` mid-investigation)
**Question:** the latest version is slow; is it CPU-bound, and can MLX fix it?

**Answer: No, on both counts.** The pipeline is not CPU-bound. It is GPU-bound on a
**fanless MacBook Air**, running at the thermal ceiling of the machine. MLX is already the
ASR engine, and the diarizer already runs on the same GPU via MPS. There is no CPU-bound
work of consequence to move, and no meaningful headroom for a different Metal API to
recover.

> **Status of this document.** Version 1 failed its contrarian review. The central defect:
> it claimed the GPU was "saturated" on the strength of `ioreg` *Device Utilization %*,
> which measures **time-residency** (is any command buffer in flight), not **occupancy**
> (are the ALUs doing work) — and those diverge precisely for bandwidth-bound
> autoregressive decode. It also dismissed a concurrency result using a thermal bias that
> pointed the opposite way. This version replaces that evidence with `powermetrics` data
> and an interleaved multi-process experiment, both of which happen to reach the same
> conclusion by sounder means. §11 lists what remains unmeasured.

---

## 1. Hardware and workload

| | |
|---|---|
| Machine | **MacBook Air (Mac16,12)** — Apple M4, 4P+6E CPU, **10-core GPU**, 24 GB, **fanless** |
| Power | AC power, Low Power Mode off (verified — no free win there) |
| ASR | `mlx-whisper` 0.4.3, `mlx-community/whisper-large-v3-turbo`, `word_timestamps=True` |
| Diarization | `pyannote.audio` 4.0.7, `speaker-diarization-3.1`, torch 2.12.1 on MPS |
| Real inputs | two ~70 min recordings (4239 s and 4198 s) |
| Probe slice | first 300 s of the meeting recording, 16 kHz mono WAV |

The fanless part is not incidental — see §4.

## 2. Evidence — per-stage wall time vs process CPU time

`cpu/wall` discriminates: a stage pegging cores shows ≥ 1; a stage waiting on an
accelerator shows well below 1. `RUSAGE_CHILDREN` is included, so ffmpeg's CPU is counted.

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

Share of pipeline wall: **whisper 54.0%, diarization 38.5%, pipeline load 7.3%, rest 0.2%.**

1. **The heavy stages are not CPU-bound.** Ratios of 0.16 and 0.22 mean the process sat
   idle awaiting the GPU ~80% of its wall time.
2. **MPS is genuinely working, not silently falling back.** This was the real risk:
   `load_diarization_pipeline` wraps `.to("mps")` in a bare `except` that would mask a
   failure. Forcing the same pipeline to CPU takes **165.52 s vs 24.45 s — 6.8× slower**,
   at ratio 3.30 (3.3 cores busy). The GPU path is live and earning its keep.
3. **The pure-Python glue is free.** `align_words_to_speakers` is O(words × turns) and was
   the most plausible CPU hot spot. It measured **0.01 s**. Scaled to the full 70 min
   (819→~17 k words is 20.8×; 70→~1.7 k turns is 24.3×; product ≈ **505×**) it is ~5 s
   against a ~15 min run. Not worth touching.

## 3. Evidence — GPU state under load (`powermetrics`, sudo, user-supplied)

`ioreg` could not answer whether the GPU was working or merely occupied. `powermetrics`
can. Five consecutive 1-second samples taken while the pipeline was running:

```
GPU HW active residency:  98.13% 100.00% 100.00% 100.00% 100.00%
GPU HW active frequency:    1399    1402    1325    1216    1225  MHz
residency in top 1470MHz bin: 63%     67%     38%    1.5%    7.9%
GPU SW requested state:   P9 94%  P9 100% P9 100% P9 100% P9 100%
GPU idle residency:        1.87%   0.00%   0.00%   0.00%   0.00%
GPU Power:                 11404   11085   11549   11856   11816  mW
```

Read together these say something the residency figure alone cannot:

- The GPU is **active essentially 100% of the time**, idling ~0%.
- Software is requesting the **maximum P-state (P9) continuously**.
- Yet the **achieved clock falls 1402 → 1216 MHz within five seconds**, and time in the
  top 1470 MHz bin collapses from 67% to 1.5%.
- Sustained draw is **~11.1–11.9 W**.

That is the signature of **power/thermal limiting**, not of a stalled or under-occupied
device. A GPU idling on memory stalls would neither draw ~11.5 W nor be forced to
downclock against a standing max-P-state request. The chip is delivering what its thermal
envelope allows, and on a **fanless** chassis that envelope is the binding constraint.

**Caveat:** these samples were taken during a running experiment without per-stage
attribution, so they characterise the pipeline in aggregate, not Whisper vs pyannote
separately.

## 4. Evidence — concurrency does not help (interleaved, multi-process)

The first attempt at this used two threads in one interpreter, ran once, in one order, and
appeared to show an 8.3% win. That result was worthless: the threads shared a GIL with a
sampler thread, and the concurrent condition ran *after* the sequential one, i.e. hotter
and slower — a directional bias, not symmetric noise.

Redone as two OS processes, six trials, order interleaved (SEQ/CON/CON/SEQ/SEQ/CON), 45 s
cooldowns:

```
trial 1 [seq]  63.44s      sequential median = 65.47s  (min 63.44, max 69.72)
trial 2 [con]  73.40s      concurrent median = 73.40s  (min 60.92, max 86.67)
trial 3 [con]  86.67s
trial 4 [seq]  69.72s      >>> saving = -7.93s (-12.1%)
trial 5 [seq]  65.47s
trial 6 [con]  60.92s
```

**Concurrency is 12% slower**, and its spread (60.9–86.7 s) is far wider than the effect
being chased. Under a fixed power budget, overlapping two GPU workloads cannot create
throughput — it only adds contention. This is consistent with §3 and inconsistent with the
"under-occupied GPU" hypothesis.

Note the honest ordering: the *biased* experiment favoured my conclusion's opposite, and
the *corrected* experiment favours it. Both were run; both are reported.

## 5. Evidence — reducing GPU work: what was tried

| Lever | Result | Verdict |
|---|---|---|
| Whisper 4-bit (`turbo-q4`), cached, back-to-back vs fp16 | **43.85 s vs 39.86 s** | Slower |
| ↳ fidelity | similarity **0.525** vs fp16, 824 words vs 819 | Disqualifying |
| Whisper 8-bit (`turbo-q8`) | 404, repo absent | N/A |
| `segmentation/embedding_batch_size` 32→64 | 29.28 → 34.77 s | Slower |
| ↳ 32→128 | 29.28 → 46.38 s | Much slower |
| pyannote fp16 (`half()` the models) | `pipeline._models` is empty on 4.0.7 — no handle found | Not tested |

On q4: word count is 824 vs 819, so the slower time is *not* an artifact of degenerate
looping — the model decoded a comparable amount of text, differently and more slowly. The
batch sweep ran in ascending order so drift inflates part of it, but 29.28 → 46.38 s is far
outside the noise band, so its direction holds.

The shipped configuration (fp16 turbo, batch 32) is the best of the measured options.

## 6. Measurement variance — stated so nothing above is over-read

Repeated identical measurements, chronological:

- Whisper fp16, 300 s slice: **34.36, 34.40, 34.80, 39.86 s** (~16% spread)
- Diarization MPS, 300 s slice: **24.45, 24.83, 26.90, 29.28 s** (~20% spread)

Both drift monotonically upward across a session — the same throttling §3 shows directly.
**Any claimed win under ~20% is unproven without repeated interleaved trials.** That
standard is applied uniformly here: it is why the 8.3% thread result was discarded, why
the −12.1% figure is quoted from medians over six interleaved trials, and why the Amdahl
estimate in §7 is given as a range rather than a point.

## 7. Why MLX cannot be the answer

1. **Whisper is already on MLX** — 54% of runtime is `mlx_whisper.transcribe` on Metal.
2. **The diarizer is already on the GPU** via torch-MPS, verified 6.8× faster than CPU
   (§2). Porting it to MLX changes which Metal API issues the work, not whether the GPU
   does it.
3. **Amdahl caps the prize.** Diarization is 38.5% of slice wall time. On a full-length
   run the fixed `pyannote_load` cost (4.63 s) shrinks from 7.3% to ~0.55%, pushing
   diarization's real share to ~**41%**. A realistic MLX-vs-MPS efficiency gain of 1.2–1.5×
   on an already power-capped GPU yields **6.9–13.8% end-to-end** — the same order as the
   thermal drift in §6.
4. **No Python pipeline exists.** MLX *weights* exist for both models pyannote 3.1 uses,
   but the glue does not (§8). It would have to be reimplemented against a diarization
   quality baseline this repo has spent days stabilising.

Days of work on the risky half of the pipeline, for a gain the noise floor can hide.

## 8. What the research found

- **MLX-native diarization models exist; a Python pipeline does not.**
  [`mlx-community/pyannote-segmentation-3.0-mlx`](https://huggingface.co/mlx-community/pyannote-segmentation-3.0-mlx)
  is **segmentation only** — its card states full diarization "isn't included" — and ships
  no pip package.
  [`mlx-community/wespeaker-voxceleb-resnet34-LM`](https://huggingface.co/mlx-community/wespeaker-voxceleb-resnet34-LM)
  supplies embeddings. The toolkit that assembles them,
  [soniqo/speech-swift](https://github.com/soniqo/speech-swift), is **Swift**.
- **[Senko](https://github.com/narcotic-sh/senko)** is the fastest option found (1 h in
  7.7 s on M3) but is **CoreML/ANE, not MLX**; its documented API exposes only merged
  segments, **not the per-speaker embeddings** `match_speakers` needs for Hungarian
  matching. Adopting it means rebuilding the core of fusion.
- **pyannote 3.1 is already the fast branch** — it dropped onnxruntime for pure PyTorch.
  Its documented MPS caveat is operator fallback, which §2's CPU probe shows is not biting.

## 9. Root cause of the slowness

Not a defect and not a misplaced workload. **The work is large and the machine is a
fanless laptop:**

- Sustained throughput is **4.72× realtime** for one source at the thermal ceiling.
- A 70-minute recording therefore costs **~15 minutes** per source.
- `run_fusion` processes both sources **sequentially and in full** — two ASR passes and two
  diarization passes — so fusion costs **~30 minutes**.

`word_timestamps=True` is required for the word-level attribution the transcript quality
rests on, and dual-source fusion doubles everything by design. Both are deliberate.

## 10. Options considered and rejected

**Rejected on measurement:** MLX port of diarization (§7); quantized weights (§5, slower
*and* similarity 0.525); larger batches (§5); stage concurrency (§4, 12% slower).

**Rejected on user decision (2026-07-31):** smaller or distilled Whisper checkpoints, and
raising `segmentation_step` from its default 0.1. Maximum accuracy is a hard requirement
for this project, and both trade output quality for speed. *Neither was measured* — the
sweeps were cancelled once the requirement was stated.

**Rejected on analysis — selective transcription of source B.** Fusion transcribes both
recordings in full, though B's text is only used where it beats A's confidence on an
overlapping same-speaker turn, or where B caught speech A missed. Gating B's ASR looked
attractive until costed properly:

- Whisper bills in **30-second windows**, not speech seconds. B-only speech (~160 recovered
  turns) is scattered thinly across 70 minutes, so scattered spans defeat window packing.
- A zero-loss confidence threshold is set by the **worst case, not the typical one**: a
  single high-confidence A turn that B beat forces the threshold high enough to transcribe
  most of the meeting anyway.
- Recovering packing requires concatenating non-adjacent audio, which risks Whisper
  bridging sentences across artificial joins — unacceptable under the accuracy requirement.

Expected payoff: single-digit percent, for a large change to the most delicate part of the
pipeline. Not built.

**Environmental levers:** already on AC with Low Power Mode off. Improving cooling would
genuinely help a thermally-capped fanless machine, but is impractical here.

## 11. Falsification, and what remains unmeasured

The diagnosis dies if the GPU is not the binding constraint. Three checks, each able to
kill it:

| Falsifier | Predicted if CPU-bound | Measured |
|---|---|---|
| cpu/wall on heavy stages | ≥ 1.0 | **0.16 / 0.22** — refuted |
| Forcing diarization to CPU | similar time | **6.8× slower** — refuted |
| GPU state under load | idle/under-occupied, headroom to overlap | **~100% active, max P-state, downclocking, ~11.5 W**; concurrency **12% slower** — refuted |

I also looked for the CPU hot spot the premise predicts, in the most plausible place — the
O(words × turns) loop. It was **0.01 s, 0.02% of wall time**.

**Not measured — stated so this document is not read as more than it is:**

1. **No full-length run was completed.** Every timing here comes from a 300 s slice.
   A full-length profile was started twice and killed both times (once to fix the analysis,
   once when the work was called off). Linear extrapolation to 70 minutes is therefore
   *unvalidated*, and two effects would make the real figure **worse**, not better:
   sustained throttling beyond what a 35 s burst shows, and pyannote's agglomerative
   clustering, which is superlinear in embedding count.
2. **Fusion-only stages were never profiled** — `find_offset` (cross-correlating two ~67 M
   sample envelopes), full-file `preprocess_audio`, `match_speakers`, `merge_turns`. The
   "fusion = 2 × pipeline" claim in §9 is an assumption, not a measurement.
3. **`segmentation_step` and alternative checkpoints were never swept** (§10).
4. **`powermetrics` was not attributed per stage** (§3).
5. **Source-A ‖ source-B concurrency** (as opposed to whisper ‖ diarize) was never run.
   §4's result and §3's power ceiling make a positive outcome unlikely, but it is untested.

These do not threaten the headline finding — items 1–2 would only make the pipeline look
slower, and 3–5 concern rejected options — but they bound what may be cited from here.

## 12. Recommendation

Accept current performance on this hardware. The pipeline is correctly engineered for the
machine it is on; the machine is a fanless laptop being asked to run ~2.3 hours of neural
inference per fused meeting. **The effective fix is different hardware** — the user's own
plan to move these runs to a cluster/Spark is the right answer, and no code change here
competes with it.

---

## Reproduction

```bash
uv run python scratchpad/profile_stages.py 300        # §2
uv run python scratchpad/conc_driver.py               # §4 (interleaved, multi-process)
uv run python scratchpad/probe_reduce_work.py         # §5 quantization + fidelity
uv run python scratchpad/probe_pyannote_tunables.py   # §5 batch sweep, §6 repeat timings
sudo powermetrics --samplers gpu_power -n 5 -i 1000   # §3 (requires sudo)
```

Probe scripts lived in the session scratchpad and are not committed. They import
`audio_to_text.transcribe`; note the package moved from `src/` to `src/audio_to_text/`
partway through this investigation.
