# Converting Gemma 4 31B to a 4-bit `.ninfer`

This guide is for someone who has a Gemma 4 31B dense checkpoint (Google's, or a fine-tune) in BF16 and
wants a `.ninfer` that runs on one RTX 5090 with 131,072 tokens of context. It lists what each conversion
option costs and gives, so you can choose before spending an afternoon on it, then the commands.

Everything here runs from this repository. Nothing needs a second checkout except two external steps that are
marked: collecting calibration statistics (llama.cpp) and, optionally, the quality check's reference artifact.

## 1. Which execution route: use W4A8

A 4-bit file can be run two ways, and the recipe decides which:

- **W4A8** (`--weights q4_g32`, the default): weights are 4-bit, and prefill (reading the prompt) rounds
  activations to FP8 so it can use the FP8 tensor cores. Decode (one token at a time) uses 16-bit activations
  either way. This is what you serve.
- **A16** (`--weights q4_g32_a16`): the same 4-bit weights with no FP8 permission anywhere. It exists to measure
  what the weights alone cost, and as a fallback if W4A8 misbehaves.

Use W4A8. On the same 4-bit weights it costs 0.002 to 0.007 KL more than A16 (resolved at g32, +0.007; not
resolved at g128 or g256, whose intervals include zero) and prefills about seven times faster: the quality-scoring run (context 512, 128-token chunks) went at
440 to 490 tokens/s with W4A8 against 62 to 71 tokens/s with A16. That gap is why a 100,000-token prompt
is practical at all. Every quality number below is therefore W4A8, measured with `NINFER_Q4_EXACT=1`
(section 6). The A16 column is there only to show the A8 cost.

## 2. What you get

All figures were measured on one model: ReadyArt's Wicked-Stuff-31B-Final with a small neuron edit (a
fine-tune of Gemma 4 31B), text only, on an RTX 5090, with FP8 global KV. The instrument is the mean KL
divergence from a near-lossless reference (`fp8_row`, an FP8 row-scaled artifact within 0.034 nats of BF16 on
every domain) over 13,728 English positions at context 512, with the chat prefix the fork's perplexity tool
uses. Lower is better. The reference scores PPL 11.277 on the same positions. Every row was scored on the
identical windows, so differences between rows are paired.

| Recipe | Flags (besides `--model/--out`) | File, text only / with MTP | KL (W4A8) | KL (A16) | ΔNLL (W4A8) | PPL (W4A8) |
|---|---|---|---|---|---|---|
| Reference, never ship | `--weights fp8_row` | 30.7 GB / n.a. | 0 | 0 | 0 | 11.277 |
| g32 absmax (the old default) | `--weights q4_g32` | 17.9 GB / 18.6 GB | 0.265 | 0.258 | +0.149 | 13.087 |
| **g32 calibrated** | `--scale-search weighted --imatrix F` | 17.9 GB / 18.6 GB | **0.157** | 0.150 | +0.061 | 11.982 |
| g128 calibrated | `... --group-size 128 --native-groups` | 16.5 GB / about 17.2 GB | 0.233 | 0.229 | +0.106 | 12.542 |
| **g256 calibrated** | `... --group-size 256 --native-groups` | **16.3 GB / 17.0 GB** | 0.238 | 0.236 | +0.052 | 11.881 |

Paired differences on the same windows, with 95% bootstrap intervals (W4A8):

| Comparison | ΔKL | ΔΔNLL |
|---|---|---|
| calibrated g32 vs absmax g32 | -0.108 (-0.129, -0.090) | -0.088 (-0.113, -0.064) |
| calibrated g128 vs calibrated g32 | +0.076 (+0.059, +0.093) | +0.046 (+0.020, +0.072) |
| calibrated g256 vs calibrated g32 | +0.081 (+0.066, +0.095) | -0.008 (-0.035, +0.017) |
| calibrated g256 vs calibrated g128 | +0.005 (-0.009, +0.016) | -0.054 (-0.075, -0.036) |

Notes on reading it:

- **Calibration is the big lever.** It cuts KL by about 40% at the same file size and needs no new kernels.
- **g128 is not worth choosing.** g256 is smaller and indistinguishable from g128 on KL (the interval spans
  zero). Pick g32 or g256.
- **KL and ΔNLL measure different things, and here they disagree.** KL compares the whole next-token
  distribution with the reference at every position. ΔNLL (and PPL, which is just its exponential) looks only at
  the probability given to the token that actually came next. Quantization noise can flatten an over-confident
  prediction, which raises the true token's probability while moving the distribution further away. That is
  what the g256 row shows: its ΔNLL equals calibrated g32 (no resolved difference) and beats g128 (resolved),
  while its KL is 0.08 higher than g32. We select on KL because it cannot be improved by that effect; read
  ΔNLL and PPL as a sanity check, not as a ranking, and never compare PPL across different instruments.
- **Statistics must come from the model you are converting.** On this model, statistics collected on the
  converted model itself beat the published base-model and published fine-tune files by 0.016 to 0.017 KL at g32.
  Do not reuse another model's imatrix for a deliverable.
- **Decode speed.** On the g256 file with MTP and FP8 KV, five fixed 400-token greedy prompts averaged 155 tok/s
  against 138 tok/s for the uncalibrated g32 file, and MTP draft acceptance was equal within noise (22% on
  sampled roleplay). That is a smoke test, not a benchmark.
- **Scope.** One fine-tune, English text, one corpus. A different model can move every number. Section 6 shows
  how to reproduce the table for yours; do that before publishing your own numbers.

### Which one should I pick?

- You want the best quality at roughly the old size: **g32 calibrated**.
- You want a 9% smaller file (1.6 GB) that is still clearly better than the uncalibrated g32 default (KL 0.238
  against 0.265, ΔNLL +0.052 against +0.149): **g256 calibrated**.
- Skip calibration only if you cannot run llama.cpp; it costs 0.1 KL.

## 3. Requirements

- Windows with an NVIDIA RTX 5090 (the fork builds for sm_120a only), a build of this fork that includes
  `q4_g128_fp16`/`q4_g256_fp16` (fork commit `4cff979` or later), and the built `ninfer-spo-serve`,
  `ninfer-spo` and `ninfer-spo-perplexity` (see [building.md](building.md)). Older binaries cannot load g128/g256 files.
- Python 3.11 with PyTorch and NumPy (the converter's own requirement), plus `pip install gguf` **only** if you
  pass `--imatrix`.
- The BF16 checkpoint of your 31B model (Hugging Face layout, safetensors), and for MTP the
  `google/gemma-4-31B-it-assistant` directory (`hf download google/gemma-4-31B-it-assistant --local-dir <DIR>`).
- For calibration: a llama.cpp build with `convert_hf_to_gguf.py` and `llama-imatrix`. About 35 GB of disk for a
  Q8_0 GGUF and about 12 minutes of GPU time.
- Free disk: the output is 17–19 GB, intermediates are not kept.

Timings on an RTX 5090: conversion 5–7 minutes (15 with MTP and hashing), Q8_0 GGUF 7 minutes, statistics
collection 12 minutes, one quality measurement 4 minutes (A16) or about 30 seconds (W4A8).

## 4. Convert without calibration (fastest)

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

Leave out `--text-only` to keep the vision tower in the file. The quality numbers in this guide are text only.

## 5. Convert with calibration (recommended)

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

What the options mean:

| Flag | Meaning |
|---|---|
| `--weights q4_g32` | 4-bit projections; prefill uses FP8 activations (W4A8). `q4_g32_a16` is the weight-only variant used for measurement. |
| `--scale-search absmax\|mse\|weighted` | How each group's scale is chosen. `absmax` is the old rule. `mse` searches for the lowest squared error. `weighted` does the same weighted by how much each input channel matters, from the imatrix. |
| `--imatrix F` | The statistics from step 3 (needed for `weighted`, refused otherwise). The file's hash goes into the receipt. |
| `--group-size 32\|64\|128\|256` | Weights per scale. With `--native-groups` 128/256 are stored in real `q4_g128_fp16`/`q4_g256_fp16`; without it the scales are repeated so the file stays 4.5 bits per weight (only useful for measuring). 64 has no Gemma kernel. |
| `--region-group REGION=G` | Per-region group size, for example `mlp_gate_up=256`. First match wins. Regions: `mlp_gate_up`, `mlp_down`, `attn_qkv`, `attn_out`, `global_layers`, `first4_layers`, `last4_layers`. |
| `--text-only` | Leave out the vision tower. |
| `--mtp-weights bf16\|fp8\|q4` | How the MTP assistant's layer projections are stored (default `bf16`, unchanged). See the MTP note below. |
| `--scale-bits 4..8` | **Experimental, measurement only.** Emulates two-level scales (a small integer multiplier per 32 weights plus one FP16 scale per 256) on ordinary g32 words, so the file does not shrink and no kernel reads that layout yet. On the author's model 8- and 6-bit multipliers scored the same KL as FP16 scales (0.150) and 5-bit about 0.003 worse, so a real format would save about 0.7 to 1.1 GB at g32 quality. Needs `--scale-search mse\|weighted` and `--group-size 32`. |

`--region-group` is for experimenters. With uncalibrated scales, making a region coarse cost least in `mlp_gate_up`
and the first four layers, and most in `mlp_down`, the last four layers, the global-attention layers and
`attn_out`. A calibrated mixed allocation (`alloc-a`, A16, in the author's plan) scored KL 0.192, between g32 (0.150) and
uniform g256 (0.236). It was measured only weight-only, so treat any mix as unqualified until you score it
as in section 6; every row in section 2 is a uniform group size.

**MTP assistant storage.** The assistant's output head is always FP8 and its two outer projections stay BF16 (loaded
as FP8). `--mtp-weights` changes only the 20 layer projections (query, output, gate, up, down in 4 layers), which
are BF16 by default. On the calibrated g256 file, eight fixed greedy prompts, three rounds each, FP8 KV, 131,072
context:

| `--mtp-weights` | File | Draft acceptance | tok/s | Greedy text vs bf16 |
|---|---|---|---|---|
| `bf16` (default) | 16.99 GB | 30.0% | 165.1 | reference |
| `fp8` | 16.80 GB (-0.18 GB) | 30.0% | 164.6 | 8 of 8 identical |
| `q4` | 16.72 GB (-0.27 GB) | 28.8% | 163.6 | 8 of 8 identical |

`fp8` is free on this test: identical acceptance and output, and the file is 184 MB smaller. `q4` saves another 81 MB
for 1.2 points of acceptance, so we do not recommend it. Draft errors never change the target's output (the target
verifies every token); they can only lower acceptance. The speed column is flat because the draft step is not
limited by weight traffic. Eight prompts is a smoke test: re-measure acceptance on your own workload before
switching.

## 6. Check the quality of your own conversion

The numbers in section 2 come from the exact paired full-vocabulary KL in `ninfer-spo-perplexity` (see
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
bootstrap interval, ΔNLL, PPL and per-domain results. Compare candidates by the paired difference on the same
windows; the unpaired intervals (about ±0.035) are wider than most differences you will care about.

Two warnings:

- llama.cpp's `llama-perplexity` is not a valid instrument for a chat fine-tune without the prefix (it reported
  PPL 1,605 on Wikipedia text for a model that scores 12 with the prefix). Do not quote its numbers next to these.
- GGUF numbers (for example an IQ4_XS of the same weights, +0.038 ΔNLL from BF16 on a different scorer) are not
  comparable with the KL column here.

## 7. Run it

```bash
ninfer-spo-serve models/my-31b-q4_g256-cal.ninfer --port 8001 --model-id 131k --max-context 131072 \
  --kv-capacity auto --kv-headroom-mib 1024 --kv-dtype fp8 --prefill-chunk 128 --spec mtp --draft-tokens auto
```

`--kv-dtype fp8` stores global-layer KV in FP8 (sliding layers stay BF16) and is what the numbers above were
run with; `bf16` is the conservative default. Check the startup log for the actual KV capacity: the model list
can overstate an auto-sized pool. See [serving.md](serving.md).

## 8. When something goes wrong

| Symptom | Cause |
|---|---|
| "ninfer.spo has tracked modifications" | Commit or stash changes in this checkout before converting. |
| `K padded to 128 ... multiple of its group size` | Only `q4_g256_fp16` has this limit; every Gemma 4 31B trunk width satisfies it. |
| `--imatrix` refused for missing statistics | The imatrix was collected on a different architecture or without the attention tensors. Collect it on the model you are converting. |
| g128/g256 file will not load | The binary predates `4cff979`; rebuild `ninfer-spo-serve`. |
| W4A8 KL changes between runs | Set `NINFER_Q4_EXACT=1` for evaluation. |
| 26B-A4B (MoE) | Not supported by this route. |

## 9. How this relates to the rest of the repository

- [weight-conversion.md](weight-conversion.md) describes the general recipe converter (`python -m tools.convert`)
  and the other Gemma 4 recipes (FP8, NVFP4). This guide covers the group-wise 4-bit family only.
- The formats are defined in [maintainer/tensor-formats.md](maintainer/tensor-formats.md) and
  [maintainer/storage-layouts.md](maintainer/storage-layouts.md).
- Scale search and imatrix weighting are described in the design notes of the plan that produced this guide;
  the measurements behind every row are in the author's tooling repository (`llama-local-tooling`,
  `docs/plans/20261105-ninfer.spo_gemma4_weight_quantization.md`). That repository also holds the orchestration
  (study matrices, camp layout, publishing) and is not needed to convert, calibrate or evaluate.
