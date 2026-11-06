# Converting Gemma 4 31B to a 4-bit `.ninfer`

This guide is for someone who has a Gemma 4 31B dense checkpoint (Google's, or a fine-tune) in BF16 and
wants a `.ninfer` that runs on one RTX 5090 with 131,072 tokens of context. It lists what each conversion
option costs and gives, so you can choose before spending an afternoon on it, then the commands.

Everything here runs from this repository. Nothing needs a second checkout except two external steps that are
marked: collecting calibration statistics (llama.cpp) and, optionally, the quality check's reference artifact.

## 1. What you get

All figures were measured on one model: ReadyArt's Wicked-Stuff-31B-Final with a small neuron edit (a
fine-tune of Gemma 4 31B), text only, on an RTX 5090. Quality is the mean KL divergence from a near-lossless
reference (`fp8_row`, an FP8 row-scaled artifact that is within 0.034 nats of BF16 on every domain) over 13,728
English positions at context 512, with the chat prefix the fork's perplexity tool uses. Lower is better. PPL is
perplexity on the same positions, and the reference itself scores 11.277. W4A8 is the serving route (prefill
uses FP8 activations); the A16 rows are weight-only.

| Recipe | Command flags (besides `--model/--out`) | File, text only / with MTP | KL | ΔNLL | PPL |
|---|---|---|---|---|---|
| Reference, never ship | `--weights fp8_row` | 30.7 GB / n.a. | 0 | 0 | 11.277 |
| g32 absmax (the old default) | `--weights q4_g32` | 17.9 GB / 18.6 GB | 0.265 | +0.149 | 13.087 |
| **g32 calibrated** | `--weights q4_g32 --scale-search weighted --imatrix F` | 17.9 GB / 18.6 GB | **0.157** | **+0.061** | **11.982** |
| g128 calibrated (A16 numbers) | `... --group-size 128 --native-groups` | 16.5 GB / about 17.2 GB | 0.229 | +0.105 | 12.527 |
| **g256 calibrated** | `... --group-size 256 --native-groups` | **16.3 GB / 17.0 GB** | 0.238 | +0.052 | 11.881 |

Notes on reading it:

- **Calibration is the big lever.** It cuts KL by about 40% at the same file size and needs no new kernels.
  Coarser groups trade quality for size: g256 is 1.6 GB (9%) smaller than g32 and costs about 0.08 KL against
  calibrated g32, which is still better than uncalibrated g32.
- **Statistics must come from the model you are converting.** On this model, statistics collected on the
  converted model itself beat the published base-model and published fine-tune files by 0.016–0.017 KL at g32.
  Do not reuse another model's imatrix for a deliverable.
- **ΔNLL and KL disagree sometimes.** KL is the full-distribution measure and is what we select on; ΔNLL
  looks only at the one correct token and is noisier (g256 has the lowest ΔNLL and the higher KL).
- **Decode speed.** On the g256 file with MTP and FP8 KV, five fixed 400-token greedy prompts averaged 155 tok/s
  against 138 tok/s for the uncalibrated g32 file, and MTP draft acceptance was equal within noise (22% on
  sampled roleplay). That is a smoke test, not a benchmark.
- **Scope.** One fine-tune, English text, one corpus. A different model can move every number. Section 5 shows
  how to reproduce the table for yours; do that before publishing your own numbers.

### Which one should I pick?

- You want the best quality at roughly today's size: **g32 calibrated**.
- You want the smallest file that is still better than the uncalibrated g32 default: **g256 calibrated**.
- g128 is a middle step with no measured W4A8 result yet (kernels are tested; the whole-model W4A8 number is
  missing). Prefer g256 or g32 unless you measure it.
- Skip calibration only if you cannot run llama.cpp; it costs 0.1 KL.

## 2. Requirements

- Windows with an NVIDIA RTX 5090 (the fork builds for sm_120a only), a build of this fork that includes
  `q4_g128_fp16`/`q4_g256_fp16` (fork commit `4cff979` or later), and the built `ninfer-spo-serve`,
  `ninfer-spo` and `ninfer-spo-perplexity` (see [building.md](building.md)). Older binaries cannot load g128/g256 files.
- Python 3.11 with PyTorch and NumPy (the converter's own requirement), plus `pip install gguf` **only** if you
  pass `--imatrix`.
- The BF16 checkpoint of your 31B model (Hugging Face layout, safetensors), and for MTP the
  `google/gemma-4-31B-it-assistant` directory.
- For calibration: a llama.cpp build with `convert_hf_to_gguf.py` and `llama-imatrix`. About 35 GB of disk for a
  Q8_0 GGUF and about 12 minutes of GPU time.
- Free disk: the output is 17–19 GB, intermediates are not kept.

Timings on an RTX 5090: conversion 5–7 minutes (15 with MTP and hashing), Q8_0 GGUF 7 minutes, statistics
collection 12 minutes, one quality measurement 4 minutes (A16) or about 30 seconds (W4A8).

## 3. Convert without calibration (fastest)

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
revision and the verified file size. Keep it with the artifact.

## 4. Convert with calibration (recommended)

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

## 5. Check the quality of your own conversion

The numbers in section 1 come from the exact paired full-vocabulary KL in `ninfer-spo-perplexity` (see
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

## 6. Run it

```bash
ninfer-spo-serve models/my-31b-q4_g256-cal.ninfer --port 8001 --model-id 131k --max-context 131072 \
  --kv-capacity auto --kv-headroom-mib 1024 --kv-dtype fp8 --prefill-chunk 128 --spec mtp --draft-tokens auto
```

`--kv-dtype fp8` stores global-layer KV in FP8 (sliding layers stay BF16) and is what the numbers above were
run with; `bf16` is the conservative default. Check the startup log for the actual KV capacity: the model list
can overstate an auto-sized pool. See [serving.md](serving.md).

## 7. When something goes wrong

| Symptom | Cause |
|---|---|
| "ninfer.spo has tracked modifications" | Commit or stash changes in this checkout before converting. |
| `K padded to 128 ... multiple of its group size` | Only `q4_g256_fp16` has this limit; every Gemma 4 31B trunk width satisfies it. |
| `--imatrix` refused for missing statistics | The imatrix was collected on a different architecture or without the attention tensors. Collect it on the model you are converting. |
| g128/g256 file will not load | The binary predates `4cff979`; rebuild `ninfer-spo-serve`. |
| W4A8 KL changes between runs | Set `NINFER_Q4_EXACT=1` for evaluation. |
| 26B-A4B (MoE) | Not supported by this route. |

## 8. How this relates to the rest of the repository

- [weight-conversion.md](weight-conversion.md) describes the general recipe converter (`python -m tools.convert`)
  and the other Gemma 4 recipes (FP8, NVFP4). This guide covers the group-wise 4-bit family only.
- The formats are defined in [maintainer/tensor-formats.md](maintainer/tensor-formats.md) and
  [maintainer/storage-layouts.md](maintainer/storage-layouts.md).
- Scale search and imatrix weighting are described in the design notes of the plan that produced this guide;
  the measurements behind every row are in the author's tooling repository (`llama-local-tooling`,
  `docs/plans/20261105-ninfer.spo_gemma4_weight_quantization.md`). That repository also holds the orchestration
  (study matrices, camp layout, publishing) and is not needed to convert, calibrate or evaluate.
