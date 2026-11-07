# Gemma 4 31B 4-bit conversion: technical reference

This is the detail behind the [simple guide](gemma4-31b-conversion.md). Read the simple guide first: it has the
preset table, the one-command build and the recommended settings. This document has every measurement, every
converter flag, the manual step-by-step commands, how to reproduce the numbers, and the known limits.

## Contents

1. [Scope and the size basis used here](#1-scope-and-the-size-basis-used-here)
2. [A16 versus W4A8, once](#2-a16-versus-w4a8-once)
3. [Requirements](#3-requirements)
4. [Results](#4-results)
   - [4.1 The instrument](#41-the-instrument)
   - [4.2 All arms, W4A8 (what you serve)](#42-all-arms-w4a8-what-you-serve)
   - [4.3 All arms, A16 (weights only)](#43-all-arms-a16-weights-only)
   - [4.4 Paired differences](#44-paired-differences)
   - [4.5 How the five presets map onto the arms](#45-how-the-five-presets-map-onto-the-arms)
   - [4.6 Reading the results](#46-reading-the-results)
   - [4.7 MTP, speed and long context](#47-mtp-speed-and-long-context)
   - [4.8 MTP assistant storage](#48-mtp-assistant-storage)
   - [4.9 Caveats](#49-caveats)
5. [Manual conversion, step by step](#5-manual-conversion-step-by-step)
   - [5.1 Convert without calibration](#51-convert-without-calibration)
   - [5.2 Convert with calibration (RTN)](#52-convert-with-calibration-rtn)
   - [5.3 Converter flags](#53-converter-flags)
   - [5.4 Per-region group sizes](#54-per-region-group-sizes)
   - [5.5 Two-level scales (`--scale-bits`)](#55-two-level-scales---scale-bits)
   - [5.6 GPTQ import (`--gptq-source`)](#56-gptq-import---gptq-source)
6. [Check the quality of your own conversion](#6-check-the-quality-of-your-own-conversion)
7. [Serving](#7-serving)
8. [When something goes wrong](#8-when-something-goes-wrong)
9. [Known limitations](#9-known-limitations)
10. [How this relates to the rest of the repository](#10-how-this-relates-to-the-rest-of-the-repository)

## 1. Scope and the size basis used here

Everything here is for a Gemma 4 31B dense checkpoint in BF16 converted to the group-wise 4-bit family of
`.ninfer` weights (`q4_g32_fp16`, `q4_g128_fp16`, `q4_g256_fp16`, `q4_g32_s6_fp16`), with the vocabulary table in FP8.
26B-A4B (MoE) is not supported by this route.

Sizes are decimal gigabytes (10^9 bytes) of the whole `.ninfer` file. Two bases appear, and each table says which:

- **Text only** means no vision tower and no MTP assistant (`--text-only --no-mtp`). The quality arms were scored on
  text-only, no-MTP files, because the scorer does not load vision or speculative components.
- **Text + MTP** means text only plus the MTP assistant stored with the default `--mtp-weights fp8`. Measured on the
  calibrated g256 file the assistant adds about 0.48 GB (16.32 GB text only, 16.80 GB with FP8 MTP). A BF16-stored
  assistant adds 0.18 GB more (section 4.8); older numbers in the author's plan use that basis (for example 18.59 GB for
  the former shipping g32 file). Every "text + MTP" figure below is the FP8 basis.

The vision tower is left out in every build here (`--text-only`); leave that flag off to keep it.

## 2. A16 versus W4A8, once

A 4-bit file can be run two ways, and the recipe decides which:

- **W4A8** (`--weights q4_g32`, the default): weights are 4-bit, and prefill (reading the prompt) rounds
  activations to FP8 so it can use the FP8 tensor cores. Decode (one token at a time) uses 16-bit activations
  either way. This is what you serve.
- **A16** (`--weights q4_g32_a16`): the same 4-bit weights with no FP8 permission anywhere. It exists to measure
  what the weights alone cost, and as a fallback if W4A8 misbehaves.

Use W4A8. On the same 4-bit weights it costs 0.002 to 0.007 KL more than A16 (resolved at g32, +0.0066 with interval
(+0.0034, +0.0097); not resolved at g128, +0.0037 (-0.0006, +0.0072), or g256, +0.0018 (-0.0024, +0.0054), whose
intervals include zero) and prefills about seven times faster: the quality-scoring run (context 512, 128-token
chunks) went at 440 to 490 tokens/s with W4A8 against 62 to 71 tokens/s with A16. That gap is why a 100,000-token
prompt is practical at all. Every W4A8 quality number below is measured with `NINFER_Q4_EXACT=1` (section 6); without
it W4A8 scoring is not reproducible. The A16 numbers are there only to show the A8 cost and to match the GPTQ spike,
which was first scored A16.

## 3. Requirements

- Windows with an NVIDIA RTX 5090 (the fork builds for sm_120a only), a build of this fork that includes
  `q4_g128_fp16`/`q4_g256_fp16` (fork commit `4cff979` or later), and the built `ninfer-spo-serve`,
  `ninfer-spo` and `ninfer-spo-perplexity` (see [building.md](building.md)). Older binaries cannot load g128/g256 files.
- To store `--scale-bits 6 --native-groups` (the `q4_g32_s6_fp16` format), the binaries must be built from the
  fork commit that adds `q4_g32_s6_fp16`; older binaries cannot load those files.
- Python 3.11 with PyTorch and NumPy (the converter's own requirement), plus `pip install gguf` **only** if you
  pass `--imatrix`.
- The BF16 checkpoint of your 31B model (Hugging Face layout, safetensors), and for MTP the
  `google/gemma-4-31B-it-assistant` directory (`hf download google/gemma-4-31B-it-assistant --local-dir <DIR>`).
- For RTN calibration: a llama.cpp build with `convert_hf_to_gguf.py` and `llama-imatrix`. About 35 GB of disk for a
  Q8_0 GGUF and about 12 minutes of GPU time.
- For GPTQ: a separate Python environment (`tools/gptq/requirements.txt`), a GPU with about 12 GiB free, 64 GiB RAM and
  about 58 GiB of disk for the text-only tree plus 16 GiB per checkpoint. See [`tools/gptq/README.md`](../tools/gptq/README.md).
- Free disk: the output is 16–19 GB, intermediates are not kept.

Timings on an RTX 5090: conversion 5–7 minutes (15 with MTP and hashing), Q8_0 GGUF 7 minutes, statistics
collection 12 minutes, GPTQ 54 to 55 minutes, one quality measurement 4 minutes (A16) or about 30 seconds (W4A8).

## 4. Results

### 4.1 The instrument

All figures were measured on one model: ReadyArt's Wicked-Stuff-31B-Final with a small neuron edit (a
fine-tune of Gemma 4 31B), text only, on an RTX 5090, with FP8 global KV. The instrument is the mean KL
divergence from a near-lossless reference (`fp8_row`, an FP8 row-scaled artifact within 0.034 nats of BF16 on
every domain) over 13,728 English positions at context 512, with the chat prefix the fork's perplexity tool
uses. Lower is better. The reference scores PPL 11.277 on the same positions. Every row was scored on the
identical windows (47 windows), so differences between rows are paired. The tool is described in
[perplexity.md](perplexity.md#exact-paired-full-vocabulary-kl).

Column meanings:

- **KL**: mean `D_KL(reference || candidate)` over the whole vocabulary, per position, in nats. 0 means identical to
  the reference. This is the selection metric.
- **Top-1 agreement**: 1 minus the report's `top1_disagreement`: the share of positions where the candidate's most
  likely next token equals the reference's. Computed from `report.json` field `kl.statistics.overall.top1_disagreement`.
- **ΔNLL**: mean change in the log-loss of the token that actually came next (candidate minus reference).
- **PPL**: perplexity of the candidate on the same positions (the exponential of its mean NLL).

### 4.2 All arms, W4A8 (what you serve)

Sizes: text-only file / text + FP8-MTP file (section 1). "n.m." means not measured: the text-only size is a real
file, and the MTP file would be about 0.5 GB larger.

| Arm | Flags (besides `--model/--out`) | File, text only / text + MTP | KL | Top-1 agreement | ΔNLL | PPL |
|---|---|---|---|---|---|---|
| Reference, never ship | `--weights fp8_row` | 30.7 GB / n.a. | 0 | 100% | 0 | 11.277 |
| RTN absmax g32 (the old default) | `--weights q4_g32` | 17.9 GB / 18.4 GB | 0.2654 | 85.7% | +0.149 | 13.087 |
| RTN calibrated g32 | `--scale-search weighted --imatrix F` | 17.92 GB / 18.4 GB | 0.1569 | 89.0% | +0.061 | 11.982 |
| RTN calibrated g32, 6-bit scales | `... --scale-bits 6 --native-groups` | 17.0 GB / 17.5 GB | 0.1543 | 89.3% | +0.060 | 11.969 |
| RTN calibrated g128 | `... --group-size 128 --native-groups` | 16.55 GB / about 17.0 GB | 0.2327 | 86.4% | +0.106 | 12.542 |
| RTN calibrated g256 | `... --group-size 256 --native-groups` | 16.32 GB / 16.8 GB | 0.2376 | 85.9% | +0.052 | 11.881 |
| GPTQ `imatrix_mse` g32 | `--gptq-source DIR` (group 32) | 17.92 GB / n.m. | 0.0845 | 91.6% | +0.029 | 11.605 |
| GPTQ `imatrix_mse` g128 | `--gptq-source DIR --native-groups` | 16.59 GB / n.m. | 0.1081 | 90.9% | +0.049 | 11.839 |
| GPTQ `imatrix_mse` g256 | `--gptq-source DIR --native-groups` | 16.37 GB / 16.85 GB | 0.1213 | 90.0% | +0.056 | 11.926 |

The KL, ΔNLL and PPL values of the first six rows are the same as in the earlier version of this guide (rounded
there to three decimals). The top-1 and GPTQ W4A8 ΔNLL and PPL values are read from the same reports and were not
shown before. Report sources are listed in section 6.

### 4.3 All arms, A16 (weights only)

| Arm | KL | Top-1 agreement | ΔNLL | PPL |
|---|---|---|---|---|
| RTN absmax g32 | 0.2584 | 85.9% | +0.141 | 12.979 |
| RTN calibrated g32 | 0.1504 | 89.3% | +0.061 | 11.986 |
| RTN calibrated g32, 6-bit scales (native; identical to the `--scale-bits 6` emulation) | 0.1490 | 89.3% | +0.062 | 12.000 |
| RTN calibrated g128 | 0.2290 | 86.7% | +0.105 | 12.527 |
| RTN calibrated g256 | 0.2358 | 86.1% | +0.050 | 11.854 |
| GPTQ `imatrix_mse` g32 | 0.0805 | 91.6% | +0.025 | 11.566 |
| GPTQ `imatrix_mse` g128 | 0.1033 | 91.2% | +0.042 | 11.763 |
| GPTQ `imatrix_mse` g256 | 0.1192 | 90.2% | +0.061 | 11.982 |

GPTQ scale rules at g32 (what the `imatrix_mse` observer buys; `--weights q4_g32_a16` / `q4_g32`):

| Scales chosen by the GPTQ run | KL (A16) | KL (W4A8) | ΔNLL (A16) | ΔNLL (W4A8) | PPL (A16) | Top-1 agreement (A16 / W4A8) |
|---|---|---|---|---|---|---|
| calibrated RTN, weighted search (reference point) | 0.1504 | 0.1569 | +0.061 | +0.061 | 11.986 | 89.3% / 89.0% |
| GPTQ, `memoryless_minmax` observer (the default, absmax/7.5) | 0.1697 | 0.1740 | +0.058 | +0.060 | 11.953 | 88.9% / 88.8% |
| GPTQ, `memoryless_mse` observer | 0.1591 | not run | +0.042 | not run | 11.760 | 89.2% / not run |
| **GPTQ, `imatrix_mse` observer** (importance-weighted scale search) | **0.0805** | **0.0845** | +0.025 | +0.029 | 11.566 | 91.6% / 91.6% |

Scale-bits emulation on ordinary g32 words, calibrated, A16 (measurement only, no size change): 8-bit multipliers
KL 0.1499, 6-bit 0.1490, 5-bit 0.1533, against FP16 scales 0.1504 (5-bit about 0.003 worse).

### 4.4 Paired differences

Paired on the same windows, 95% window-bootstrap intervals.

RTN family, W4A8:

| Comparison | ΔKL | ΔΔNLL |
|---|---|---|
| calibrated g32 vs absmax g32 | -0.108 (-0.129, -0.090) | -0.088 (-0.113, -0.064) |
| calibrated g32 with 6-bit scales vs calibrated g32 | -0.003 (-0.007, +0.002) | -0.001 (-0.008, +0.006) |
| calibrated g128 vs calibrated g32 | +0.076 (+0.059, +0.093) | +0.046 (+0.020, +0.072) |
| calibrated g256 vs calibrated g32 | +0.081 (+0.066, +0.095) | -0.008 (-0.035, +0.017) |
| calibrated g256 vs calibrated g128 | +0.005 (-0.009, +0.016) | -0.054 (-0.075, -0.036) |

Native 6-bit scales against the g32 file of the same recipe, W4A8: -0.0034, standard error 0.0028 over the 47
windows, not resolved (section 5.5).

W4A8 minus A16 on the same weights (from the author's plan, section 6.5): g32 +0.0066 (+0.0034, +0.0097), g128
+0.0037 (-0.0006, +0.0072), g256 +0.0018 (-0.0024, +0.0054).

GPTQ against RTN and against GPTQ, both execution routes:

| Comparison | ΔKL, A16 | ΔKL, W4A8 | ΔΔNLL, W4A8 |
|---|---|---|---|
| GPTQ g32 vs calibrated RTN g32 | -0.0698 (-0.0845, -0.0570) | -0.0724 (-0.0871, -0.0598) | -0.0320 (-0.0561, -0.0094) |
| GPTQ g256 vs calibrated RTN g32 (1.55 GB, 8.7% smaller) | -0.0311 (-0.0456, -0.0178) | -0.0356 (-0.0502, -0.0226) | -0.0047 (-0.0323, +0.0207) |
| GPTQ g256 vs calibrated RTN g256 | -0.1166 (-0.1364, -0.0992) | -0.1162 (-0.1359, -0.0991) | +0.0037 (-0.0193, +0.0257) |
| GPTQ g128 vs calibrated RTN g32 | -0.0470 (-0.0632, -0.0316) | -0.0489 (-0.0645, -0.0344) | -0.0120 (-0.0388, +0.0102) |
| GPTQ g128 vs calibrated RTN g128 | -0.1257 (-0.1517, -0.1026) | -0.1247 (-0.1490, -0.1023) | -0.0577 (-0.0852, -0.0340) |
| GPTQ g128 vs GPTQ g32 | +0.0228 (+0.0153, +0.0313) | +0.0235 (+0.0158, +0.0319) | +0.0199 (+0.0048, +0.0347) |
| GPTQ g256 vs GPTQ g32 | +0.0387 (+0.0303, +0.0484) | +0.0368 (+0.0280, +0.0472) | +0.0273 (+0.0104, +0.0452) |

GPTQ g32 against calibrated RTN g32 is better on 46 of 47 windows and
in every domain (A16, mean KL: long-form 0.077 against 0.154, reference text 0.104 against 0.187, code 0.063 against
0.114). With the default `memoryless_minmax` observer GPTQ is worse than calibrated RTN by +0.019 (+0.009, +0.032):
error compensation alone does not beat a good scale search. ΔNLL differences against calibrated RTN g32 are not
resolved at g128 or g256 (the intervals span zero).

### 4.5 How the five presets map onto the arms

| Preset | Arm | KL (W4A8) | Top-1 agreement | Text-only file | Text + MTP file |
|---|---|---|---|---|---|
| `best-medium` | GPTQ `imatrix_mse` g32 | 0.0845 | 91.6% | 17.92 GB | 18.4 GB |
| `best-small` | GPTQ `imatrix_mse` g256, `--native-groups` | 0.1213 | 90.0% | 16.37 GB | 16.85 GB (16,852,045,060 B, measured) |
| `fast-medium` | RTN calibrated g32, `--scale-bits 6 --native-groups` | 0.1543 | 89.3% | about 17.0 GB | 17.5 GB |
| `fast-small` | RTN calibrated g256, `--native-groups` | 0.2376 | 85.9% | 16.32 GB | 16.8 GB |
| `draft` | RTN absmax g32 | 0.2654 | 85.7% | 17.9 GB | 18.4 GB |

The `best-small` text + MTP file was built end to end (a second GPTQ run, the converter, the FP8 assistant) and
measured: 16,852,045,060 bytes. The second GPTQ run was bit-reproducible: the scored W4A8 KL, ΔNLL and PPL are
identical to the first run (0.1213, +0.056, 11.926; paired difference exactly 0.0000 on all 47 windows). On that file,
eight fixed greedy prompts x3 (FP8 KV, 131,072 context, MTP) gave draft acceptance 31.5% and 168 tok/s; a roleplay run
of about 9,000 generated tokens 22.6% (5,196 of 23,028 drafted); one 128,674-token prompt plus 2,048 generated 21.2%
(1,145 of 5,408). Both long runs completed with the full 131,072 KV. The same prompts on the RTN g256 file measured
30.0% and 162 tok/s (the box was not otherwise quiet; compare acceptance, not tok/s). One run per file, so a
one-point acceptance difference is within what these runs resolve.

### 4.6 Reading the results

- **Calibration is the big lever for RTN.** It cuts KL by about 40% at the same file size and needs no new kernels.
- **GPTQ roughly halves it again.** GPTQ g32 scores 0.0845 against calibrated RTN g32 at 0.1569, and GPTQ g256 (0.1213,
  16.37 GB) beats calibrated RTN g32 (17.92 GB) in a file 8.7% smaller. The importance-weighted observer is what makes
  GPTQ win: error feedback with a plain scale rule does not beat a good RTN scale search.
- **g128 is not worth choosing.** For RTN, g256 is smaller and indistinguishable from g128 on KL (the interval spans
  zero). For GPTQ, g128 sits between g32 and g256 on KL (0.1081) and on size (16.59 GB); it is a valid technical choice
  but not one of the presets.
- **Group size costs more for RTN than for GPTQ.** Going from g32 to g256 costs +0.081 KL for RTN and +0.037 for GPTQ.
- **KL and ΔNLL measure different things, and here they disagree.** KL compares the whole next-token
  distribution with the reference at every position. ΔNLL (and PPL, which is just its exponential) looks only at
  the probability given to the token that actually came next. Quantization noise can flatten an over-confident
  prediction, which raises the true token's probability while moving the distribution further away. That is
  what the RTN g256 row shows: its ΔNLL equals calibrated g32 (no resolved difference) and beats g128 (resolved),
  while its KL is 0.08 higher than g32. We select on KL because it cannot be improved by that effect; read
  ΔNLL and PPL as a sanity check, not as a ranking, and never compare PPL across different instruments.
- **Top-1 agreement is coarser than KL.** It counts only whether the single most likely token matches, so it separates
  builds less than KL does: the old default (KL 0.2654) and RTN g256 (KL 0.2376) agree with the reference on 85.7% and
  85.9% of positions respectively. Use it for intuition, use KL to choose.
- **Statistics must come from the model you are converting.** On this model, statistics collected on the
  converted model itself beat the published base-model and published fine-tune files by 0.016 to 0.017 KL at g32.
  Do not reuse another model's imatrix for a deliverable. GPTQ's `imatrix_mse` observer computes its own importance
  from the activations it sees, so GPTQ needs no llama.cpp.

### 4.7 MTP, speed and long context

Calibrated RTN files (g32, g32 with 6-bit scales, g256), each with the default FP8 assistant, FP8 KV, 131,072 context,
MTP on. Eight fixed greedy 400-token prompts, three rounds: draft acceptance 30.5% (g32), 31.6% (6-bit scales),
30.0% (g256); mean tok/s 157.1, 164.6 and 166.6. A roleplay run of about 9,000 tokens: acceptance 23.5%, 23.3% and
22.1%. One 128,674-token prompt plus 2,048 generated: 21.6%, 21.8% and 22.1%. All six long runs completed with the
full 131,072 KV. This is the author's one fine-tune; differences of about a point are within what these runs can
resolve, and the tok/s ordering (6-bit scales and g256 faster than g32) is single-run per file.

Native 6-bit scales, serving flags 131,072 context, FP8 KV, prefill chunk 128, no speculation: 70.6 tok/s against
68.2 for the g32 file (+3.4%), and a 3,000-token prefill at 3,049 tok/s against 3,046 (medians of six runs).

A GPTQ file uses the same storage formats as the RTN files, so no kernel changes; its speed, MTP acceptance and
long-context behaviour have not been separately measured here.

### 4.8 MTP assistant storage

The assistant's output head is always FP8 and its two outer projections stay BF16 (loaded as FP8). `--mtp-weights`
changes only the 20 layer projections (query, output, gate, up, down in 4 layers), which default to FP8 and were BF16
before. On the calibrated g256 RTN file, eight fixed greedy prompts, three rounds each, FP8 KV, 131,072 context:

| `--mtp-weights` | File | Draft acceptance | tok/s | Greedy text vs bf16 |
|---|---|---|---|---|
| `bf16` (old layout) | 16.99 GB | 30.0% | 165.1 | reference |
| **`fp8` (default)** | 16.80 GB (-0.18 GB) | 30.0% | 164.6 | 8 of 8 identical |
| `q4` | 16.72 GB (-0.27 GB) | 28.8% | 163.6 | 8 of 8 identical |

`fp8` is free on this test: identical acceptance and output, and the file is 184 MB smaller. `q4` saves another 81 MB
for 1.2 points of acceptance, so it is not recommended. Draft errors never change the target's output (the target
verifies every token); they can only lower acceptance. The speed column is flat because the draft step is not
limited by weight traffic. Eight prompts is a smoke test: re-measure acceptance on your own workload before
switching.

### 4.9 Caveats

- One fine-tune, English text, one corpus. A different model can move every number. Section 6 shows how to reproduce
  the tables for yours; do that before publishing your own numbers.
- The unpaired 95% intervals on one arm's KL are about ±0.035. Compare arms by the paired differences above.
- One run per arm. GPU scoring is not bit-reproducible (the `NINFER_Q4_EXACT=1` route fixes the W4A8 path), so the
  fourth decimal is noise.
- The GPTQ spike used one calibration set (400 chat-prefixed chunks of 512 tokens, disjoint from the evaluation
  corpus: none of the 13,596 evaluation 13-token shingles occurs in them) and one dampening/observer setting;
  sensitivity of dampening and observer at g128 and g256 is not measured.
- The 6-bit-scale format cannot be a GPTQ target: the stored GPTQ scales are one FP16 value per 32 weights.
- MTP acceptance, speed and long-context data for a GPTQ file are not in this document.
- KL against the near-lossless FP8-row reference ranks candidates only if the reference is closer to BF16 than every
  candidate; it was within 0.034 nats of BF16 on every domain here.

## 5. Manual conversion, step by step

The one-command builder runs these steps for you; use them directly when you want a flag the builder does not expose.
Every command runs from the repository root.

### 5.1 Convert without calibration

```bash
python -m tools.convert.gemma4_q4 convert \
  --model /path/to/gemma-4-31B-bf16 \
  --mtp-assistant /path/to/gemma-4-31B-it-assistant \
  --text-only \
  --weights q4_g32 \
  --out models/my-31b-q4_g32.ninfer
```

Use `--no-mtp` instead of `--mtp-assistant` for a file without speculative decoding. `--dry-run` prints the
recipe and the projected size without reading weights. The command refuses to overwrite, refuses an output
inside the source tree, and refuses a source that is not BF16 or not Gemma 4 31B dense. The fork checkout must
have no uncommitted tracked changes (the receipt records the revision).

Every conversion writes `<out>.build.json`: source hashes, recipe name, group layout, calibration file hash,
revision and the verified file size. Keep it with the artifact. The converter verifies storage when it finishes;
to re-check a file later (every object read and hashed, not a quality check):

```bash
python -m tools.convert.gemma4_q4 verify --artifact models/my-31b-q4_g32.ninfer
```

Leave out `--text-only` to keep the vision tower in the file. The quality numbers in this document are text only.

### 5.2 Convert with calibration (RTN)

Collect statistics once per model, then reuse the file for every group size.

```bash
# 1. A Q8_0 GGUF of YOUR model (llama.cpp). Q8_0 is only used to collect statistics.
python convert_hf_to_gguf.py /path/to/gemma-4-31B-bf16 --outtype q8_0 --outfile model-q8_0.gguf

# 2. Calibration text: public English text of your choice (we used 20 MB of the eaddario
#    combined_en_large corpus). Keep it disjoint from any text you evaluate on. This renders 400 chunks of
#    512 tokens that each start with Gemma's chat prefix, which matters: chat fine-tunes see that prefix at
#    run time and bare text gives worse statistics (about 0.007 KL at g32).
python -m tools.convert.gemma4_q4 calibration-text \
  --tokenizer /path/to/gemma-4-31B-bf16/tokenizer.json \
  --source calibration.txt \
  --prefix-ids 2,105,2364,107,34230,506,2354,236761,106,107,105,4368,107,100,45518,107,101 \
  --chunks 400 --out calibration-templated.txt

# 3. Statistics (llama.cpp). Offload as many layers as fit.
llama-imatrix -m model-q8_0.gguf -f calibration-templated.txt --parse-special \
  --ctx-size 512 --chunks 400 -ngl 99 -o model.imatrix.gguf

# 4. Convert.
python -m tools.convert.gemma4_q4 convert \
  --model /path/to/gemma-4-31B-bf16 \
  --mtp-assistant /path/to/gemma-4-31B-it-assistant \
  --text-only \
  --weights q4_g32 \
  --scale-search weighted --imatrix model.imatrix.gguf \
  --group-size 256 --native-groups \
  --out models/my-31b-q4_g256-cal.ninfer
```

Use `--group-size 32` and drop `--native-groups` for the calibrated g32 row. The prefix ids above are Gemma 4's
tokenizer for "user: Continue the document." followed by an empty thought; they are the same for every model
with the Gemma 4 tokenizer. For another template use `python -m tools.bench.kv_eval.derive_prefix_ids`.

### 5.3 Converter flags

| Flag | Meaning |
|---|---|
| `--weights q4_g32` | 4-bit projections; prefill uses FP8 activations (W4A8). `q4_g32_a16` is the weight-only variant used for measurement. |
| `--scale-search absmax\|mse\|weighted` | How each group's scale is chosen. `absmax` is the old rule. `mse` searches for the lowest squared error. `weighted` does the same weighted by how much each input channel matters, from the imatrix. |
| `--imatrix F` | The statistics from step 3 (needed for `weighted`, refused otherwise). The file's hash goes into the receipt. |
| `--group-size 32\|64\|128\|256` | Weights per scale. With `--native-groups` 128/256 are stored in real `q4_g128_fp16`/`q4_g256_fp16`; without it the scales are repeated so the file stays 4.5 bits per weight (only useful for measuring). 64 has no Gemma kernel. |
| `--region-group REGION=G` | Per-region group size, for example `mlp_gate_up=256`. First match wins. Regions: `mlp_gate_up`, `mlp_down`, `attn_qkv`, `attn_out`, `global_layers`, `first4_layers`, `last4_layers`. |
| `--text-only` | Leave out the vision tower. |
| `--mtp-weights fp8\|bf16\|q4` | How the MTP assistant's layer projections are stored (default `fp8` for the q4 weights). See section 4.8. |
| `--scale-bits 4..8` | Two-level scales. See section 5.5. |
| `--gptq-source DIR` | Copy the codes and scales of an external GPTQ checkpoint. See section 5.6. |

### 5.4 Per-region group sizes

`--region-group` is for experimenters. With uncalibrated scales, making a region coarse cost least in `mlp_gate_up`
and the first four layers, and most in `mlp_down`, the last four layers, the global-attention layers and
`attn_out`. A calibrated mixed allocation (`alloc-a`, A16, in the author's plan) scored KL 0.192, between g32 (0.150) and
uniform g256 (0.236). It was measured only weight-only, so treat any mix as unqualified until you score it
as in section 6; every row in section 4 is a uniform group size.

### 5.5 Two-level scales (`--scale-bits`)

`--scale-bits 4..8` stores two-level scales: a small integer multiplier per 32 weights plus one FP16 super scale per
256. **`--scale-bits 6` with `--native-groups` stores the real `q4_g32_s6_fp16` format**: the same 4-bit codes as g32,
but each row's scale plane is one byte per 32 weights instead of two (4.25 instead of 4.5 bits per weight), and it
works with both `--weights q4_g32` and `q4_g32_a16`. The receipt's `weight_bytes` is then the real size. Any other
value, or 6 without `--native-groups`, is **experimental, measurement only**: the scales are emulated on ordinary g32
words, so the file does not shrink and no kernel reads that layout. On the author's model, in that emulation, 8- and
6-bit multipliers scored the same KL as FP16 scales (0.150) and 5-bit about 0.003 worse (section 4.3). It needs
`--scale-search mse|weighted`, `--group-size 32` and no `--region-group`; the real format also needs every projection
input width, padded to 128, to be a multiple of 256.

```bash
python -m tools.convert.gemma4_q4 convert \
  --model /path/to/gemma-4-31B-bf16 --mtp-assistant /path/to/gemma-4-31B-it-assistant --text-only \
  --weights q4_g32 --scale-search weighted --imatrix model.imatrix.gguf \
  --scale-bits 6 --native-groups \
  --out models/my-31b-q4_g32_s6-cal.ninfer
```

The recipe name ends in `_native_scale6`. The converter derives each 256-column block's super scale and
multipliers from the searched scales and refuses to write a block whose decoded scales would differ from them by
even one bit; the file therefore decodes exactly like the `--scale-bits 6` emulation, in a smaller scale plane.

Measured on the author's calibrated 31B (diode s15 t005, weighted search, text-only): the real file is 915,210,240
bytes (0.85 GiB, 5.1%) smaller than the g32 file of the same recipe (16,971,047,664 against 17,886,257,904 bytes of
weights). Scored as in section 6 (A16), the native artifact gives the same 47 windows with `kl_sum` identical to the
bit, and a byte-identical `kl_positions.f32`, as the `--scale-bits 6` emulation: mean KL 0.14897 (g32 FP16 scales
0.1504). With `NINFER_Q4_EXACT=1` the W4A8 native artifact scores 0.1543 (g32: 0.1569; the paired difference,
-0.0034 with a standard error of 0.0028 over the 47 windows, is not resolved). Speed is in section 4.7.

### 5.6 GPTQ import (`--gptq-source`)

`--gptq-source DIR` copies the 4-bit codes and scales of an external LLM Compressor GPTQ checkpoint (compressed-tensors
`pack-quantized`, symmetric INT4, static contiguous groups, no activation order; group size 32, 128 or 256) without
re-rounding. With `--native-groups` a 128/256 checkpoint is stored in the real `q4_g128_fp16`/`q4_g256_fp16` formats.
Produce the checkpoint with [`tools/gptq/`](../tools/gptq/README.md). It is mutually exclusive with `--scale-search`,
`--imatrix`, `--group-size`, `--region-group` and `--scale-bits`. It is experimental.

Recipe of the spike. The checkpoint came from LLM Compressor 0.13.0 (compressed-tensors 0.18.0, transformers 5.14.1,
torch 2.13.0+cu130): `GPTQModifier` with 4-bit symmetric int weights, `strategy group`, `group_size 32`, `actorder
None`, `block_size 128`, `dampening_frac 0.01`, sequential targets `Gemma4TextDecoderLayer`, and the 400 chat-prefixed
512-token chunks used for the imatrix as calibration data. transformers 5.14.1 cannot read the checkpoint's
`per_layer_config`, so GPTQ ran on a text-only `Gemma4ForCausalLM` tree whose config carries
`global_head_dim`/`num_global_key_value_heads` (what `normalize_config` produces). The run took 54 minutes on one
RTX 5090 (10.3 GiB GPU, 39 GiB RAM with 32 GiB of weights kept on the CPU and the rest offloaded to disk). Group sizes
128 and 256 take 55 minutes each, with no observer or dampening change.

What the importer guarantees. It copies codes and scales word for word: every one of the 915,210,240 BF16 scales
converts to binary16 unchanged (it refuses the checkpoint otherwise), `weight_zero_point` and `weight_g_idx` are
refused, and the converter's global-layer permutations are applied to the encoded words (q/k rows, o columns). A
32-group of the Hugging Face column order maps onto a whole 32-group of the converter's order because the rotated blocks
`[0,64)` and `[256,320)` of a 512-wide head are multiples of 32; the importer refuses any permutation that would split
a group. All 240 weight objects of the written artifact decode to exactly the checkpoint's values.

Coarse groups. With `--native-groups` the g128/g256 files store the real `q4_g128_fp16` / `q4_g256_fp16` formats: one
binary16 scale per source group, codes unchanged, decoded values identical to the repeated-scale g32 emulation. The ten
global-layer `o` projections stay `q4_g32_fp16` (with the source scale repeated): the converter's column permutation
interleaves two source groups into one converter group, so those tensors cannot be stored natively. That costs 41 to
48 MB (0.3%). All 240 weight objects of the written artifacts decode to exactly the checkpoint's values.

One fine-tune and one calibration set. To reproduce the route end to end (environment, calibration tokens, text-only
tree, GPTQ run, conversion) follow [`tools/gptq/README.md`](../tools/gptq/README.md), then score as in section 6.

## 6. Check the quality of your own conversion

The numbers in section 4 come from the exact paired full-vocabulary KL in `ninfer-spo-perplexity` (see
[perplexity.md](perplexity.md#exact-paired-full-vocabulary-kl)). To reproduce them for your model:

```bash
# Reference artifact (near-lossless FP8; about 41 minutes, 31 GB; delete it when you are done).
python -m tools.convert.gemma4_q4 convert --model <bf16> --text-only --no-mtp \
  --weights fp8_row --out work/ref-fp8row.ninfer

# English subset of the bundled corpus (12 streams, 13.7k scored positions, about 14 GB of cache).
python -m tools.bench.kv_eval.kl_corpus --source eval/corpora/perplexity-1m/manifest.json \
  --out work/corpus --chars chinese_reference=0

P=2,105,2364,107,34230,506,2354,236761,106,107,105,4368,107,100,45518,107,101
ninfer-spo-perplexity work/ref-fp8row.ninfer --corpus work/corpus/manifest.json --context 512 --stride 256 \
  --prefill-chunk 128 --prefix-ids $P --headroom-mib 128 --kl-reference-write work/ref.nklref

# Each candidate (set NINFER_Q4_EXACT=1 for every W4A8 candidate; without it W4A8 scoring is not reproducible).
set NINFER_Q4_EXACT=1
ninfer-spo-perplexity models/my-31b-q4_g256-cal.ninfer --corpus work/corpus/manifest.json --context 512 \
  --stride 256 --prefill-chunk 128 --prefix-ids $P --kl-reference work/ref.nklref --output work/runs/g256
```

The candidate must be a no-MTP, text-only file for scoring. `report.json` has the mean KL with a 95%
bootstrap interval (`kl.statistics.overall.mean_kl`), ΔNLL (`mean_delta_nll`), top-1 disagreement
(`top1_disagreement`; agreement is one minus it), PPL (`overall.perplexity`) and per-domain results. Compare
candidates by the paired difference on the same windows
(`python -m tools.bench.kv_eval.kl_compare --baseline A/report.json --candidate B/report.json`); the unpaired
intervals (about ±0.035) are wider than most differences you will care about.

Calibration text must be disjoint from the evaluation text. The GPTQ calibration chunks in the spike were checked
against the evaluation corpus by 13-token shingles.

Where the numbers in section 4 came from (reports in the author's evidence camp, `report.json` under
`<model>/convert/`): absmax W4A8 `ninfer-spo/kl-reference-20261105/runs/q4exact-vs-fp8-en`, absmax A16
`.../runs/q4a16-vs-fp8-en`; RTN g32 W4A8 `.../group-matrix/runs/w4a8-weighted-tpl-g32`, A16 `.../weighted-tpl-g32`; RTN
g128 W4A8 `.../w4a8-native-weighted-tpl-g128`, A16 `.../native-weighted-tpl-g128`; RTN g256 W4A8
`.../w4a8-native-g256-exact`, A16 `.../native-weighted-tpl-g256`; 6-bit scales `ninfer-spo/s6-native-20261106/kl-s6-w4a8`
and `kl-s6-a16`; GPTQ `gptq-spike-20261106/runs/gptq-g{32,128,256}-imatrix-mse-{,native-}{a16,w4a8}`, observers
`gptq-g32-minmax-*` and `gptq-g32-mse-a16`; paired intervals `gptq-spike-20261106/paired2-*.json`. The MTP and
long-context data are under `ninfer-spo/s6-checks-20261106/`. These are author-side evidence, not needed to convert.

Two warnings:

- llama.cpp's `llama-perplexity` is not a valid instrument for a chat fine-tune without the prefix (it reported
  PPL 1,605 on Wikipedia text for a model that scores 12 with the prefix). Do not quote its numbers next to these.
- GGUF numbers (for example an IQ4_XS of the same weights, +0.038 ΔNLL from BF16 on a different scorer) are not
  comparable with the KL column here.

## 7. Serving

The recommended serving command, with the reason for each flag, is in the [simple guide](gemma4-31b-conversion.md#run-it);
`python -m tools.convert.gemma4_build` writes it as a launcher (`serve-<name>.cmd`, or the `launcher` subcommand for an
existing file). It runs:

```bash
ninfer-spo-serve models/my-31b.ninfer --host 127.0.0.1 --port 8080 --model-id 131k --model-match any \
  --max-context 131072 --kv-capacity auto --kv-headroom-mib 1024 --kv-dtype fp8 --max-concurrency 1 \
  --prefill-chunk 128 --spec mtp --draft-tokens auto --device-state-slots 0 --host-context-mib 8192
```

`--spec mtp --draft-tokens auto` is added only when the artifact has the MTP component, and `--vision` only when it has
the vision tower. `--model-match any` is added by default (`launcher --model-match exact` omits it): with `any`,
OpenAI-style requests may send any `model` name, `/v1/models` still reports the `--model-id` alias, and a missing
`model` is still HTTP 400. The server's own default is `exact`, which answers 404 `model_not_found` to any name other
than the alias. The flag needs a `ninfer-spo-serve` build that includes `--model-match`. No API key is required: the
server ignores any key a client sends unless `--api-key` is set.

The quality scores in section 4 were produced with `ninfer-spo-perplexity` (context 512, FP8 global KV, prefill
chunk 128). The MTP and long-context runs in section 4.7 used 131,072 context, FP8 KV and MTP; the no-speculation speed
comparison in section 4.7 also used prefill chunk 128.

`--kv-dtype fp8` stores global-layer KV in FP8 (sliding layers stay BF16) and is what the numbers above were
run with; `bf16` is the conservative default of the server. Check the startup log for the actual KV capacity: the model
list can overstate an auto-sized pool. See [serving.md](serving.md).

## 8. When something goes wrong

| Symptom | Cause |
|---|---|
| "ninfer.spo has tracked modifications" | Commit or stash changes in this checkout before converting. |
| `K padded to 128 ... multiple of its group size` or `... multiple of 256` | `q4_g256_fp16` and `q4_g32_s6_fp16` have this limit (for the latter, K padded to 128 must be a multiple of 256); every Gemma 4 31B trunk width satisfies it. |
| `--imatrix` refused for missing statistics | The imatrix was collected on a different architecture or without the attention tensors. Collect it on the model you are converting. |
| g128/g256 file will not load | The binary predates `4cff979`; rebuild `ninfer-spo-serve`. |
| `q4_g32_s6_fp16` file will not load | The binary predates the commit that adds `q4_g32_s6_fp16`; rebuild `ninfer-spo-serve`. |
| W4A8 KL changes between runs | Set `NINFER_Q4_EXACT=1` for evaluation. |
| `--gptq-source` refused | The checkpoint has zero points, `weight_g_idx`, scales that are not exactly representable in binary16, or is asymmetric or not 4-bit. Produce it with `tools/gptq/` and run `check_checkpoint.py` first. |
| GPTQ environment resolves the wrong `transformers` | Call the GPTQ environment's interpreter by absolute path; see [`tools/gptq/README.md`](../tools/gptq/README.md). |
| 26B-A4B (MoE) | Not supported by this route. |

## 9. Known limitations

- Measured on one model, English text, one corpus, one run per arm (section 4.9).
- The GPTQ route is experimental and needs a separate environment, a GPU with about 12 GiB free and 64 GiB of RAM.
- GPTQ g128/g256 files keep the ten global-layer `o` projections at g32 storage (41 to 48 MB).
- The two-level 6-bit scale format cannot hold GPTQ scales.
- A mixed per-region allocation is unqualified (section 5.4).
- Speed, MTP acceptance and long-context behaviour of GPTQ files are not measured here.
- The native 4-bit formats need a binary built from a fork commit that has them (sections 3 and 8).
- 26B-A4B (MoE) is not supported.

## 10. How this relates to the rest of the repository

- [weight-conversion.md](weight-conversion.md) describes the general recipe converter (`python -m tools.convert`)
  and the other Gemma 4 recipes (FP8, NVFP4). This document covers the group-wise 4-bit family only.
- The formats are defined in [maintainer/tensor-formats.md](maintainer/tensor-formats.md) and
  [maintainer/storage-layouts.md](maintainer/storage-layouts.md).
- The GPTQ scripts are in [`tools/gptq/`](../tools/gptq/README.md).
- Scale search and imatrix weighting are described in the design notes of the plan that produced this guide;
  the measurements behind every row are in the author's tooling repository (`llama-local-tooling`,
  `docs/plans/20261105-ninfer.spo_gemma4_weight_quantization.md`). That repository also holds the orchestration
  (study matrices, camp layout, publishing) and is not needed to convert, calibrate or evaluate.
