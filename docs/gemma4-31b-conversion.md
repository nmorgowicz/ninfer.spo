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

| Recipe | Flags (besides `--model/--out`) | File, text only / with MTP (fp8) | KL (W4A8) | KL (A16) | ΔNLL (W4A8) | PPL (W4A8) |
|---|---|---|---|---|---|---|
| Reference, never ship | `--weights fp8_row` | 30.7 GB / n.a. | 0 | 0 | 0 | 11.277 |
| g32 absmax (the old default) | `--weights q4_g32` | 17.9 GB / 18.4 GB | 0.265 | 0.258 | +0.149 | 13.087 |
| **g32 calibrated** | `--scale-search weighted --imatrix F` | 17.9 GB / 18.4 GB | **0.157** | 0.150 | +0.061 | 11.982 |
| **g32 calibrated, 6-bit scales** | `... --scale-bits 6 --native-groups` | 17.0 GB / 17.5 GB | **0.154** | 0.149 | +0.060 | 11.969 |
| g128 calibrated | `... --group-size 128 --native-groups` | 16.5 GB / about 17.0 GB | 0.233 | 0.229 | +0.106 | 12.542 |
| **g256 calibrated** | `... --group-size 256 --native-groups` | **16.3 GB / 16.8 GB** | 0.238 | 0.236 | +0.052 | 11.881 |

Paired differences on the same windows, with 95% bootstrap intervals (W4A8):

| Comparison | ΔKL | ΔΔNLL |
|---|---|---|
| calibrated g32 vs absmax g32 | -0.108 (-0.129, -0.090) | -0.088 (-0.113, -0.064) |
| calibrated g32 with 6-bit scales vs calibrated g32 | -0.003 (-0.007, +0.002) | -0.001 (-0.008, +0.006) |
| calibrated g128 vs calibrated g32 | +0.076 (+0.059, +0.093) | +0.046 (+0.020, +0.072) |
| calibrated g256 vs calibrated g32 | +0.081 (+0.066, +0.095) | -0.008 (-0.035, +0.017) |
| calibrated g256 vs calibrated g128 | +0.005 (-0.009, +0.016) | -0.054 (-0.075, -0.036) |

Notes on reading it:

- **Calibration is the big lever.** It cuts KL by about 40% at the same file size and needs no new kernels.
- **g128 is not worth choosing.** g256 is smaller and indistinguishable from g128 on KL (the interval spans
  zero). Pick g32, g32 with 6-bit scales, or g256.
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
- **MTP and long context.** Three calibrated files (g32, g32 with 6-bit scales, g256), each with the default FP8
  assistant, FP8 KV, 131,072 context, MTP on. Eight fixed greedy 400-token prompts, three rounds: draft acceptance
  30.5% (g32), 31.6% (6-bit scales), 30.0% (g256); mean tok/s 157.1, 164.6 and 166.6. A roleplay run of about
  9,000 tokens: acceptance 23.5%, 23.3% and 22.1%. One 128,674-token prompt plus 2,048 generated: 21.6%, 21.8% and
  22.1%. All six long runs completed with the full 131,072 KV. This is the author's one fine-tune; differences of
  about a point are within what these runs can resolve, and the tok/s ordering (6-bit scales and g256 faster than
  g32) is single-run per file.
- **Scope.** One fine-tune, English text, one corpus. A different model can move every number. Section 6 shows
  how to reproduce the table for yours; do that before publishing your own numbers.

### Which one should I pick?

- You want the best quality: **g32 calibrated with 6-bit scales** (`--scale-bits 6 --native-groups`). It scored the
  same KL as plain calibrated g32 (0.154 against 0.157, not a resolved difference), is 0.9 GB smaller, and its
  prefill and decode speed matched or beat g32 in our runs. Plain calibrated g32 is the fallback if you cannot use
  a build that has the new format.
- You want the smallest file that is still clearly better than the uncalibrated g32 default (a further 0.7 GB
  smaller, KL 0.238 against 0.265, ΔNLL +0.052 against +0.149): **g256 calibrated**.
- Skip calibration only if you cannot run llama.cpp; it costs 0.1 KL.

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
| `--mtp-weights fp8\|bf16\|q4` | How the MTP assistant's layer projections are stored (default `fp8` for the q4 weights). See the MTP note below. |
| `--scale-bits 4..8` | Two-level scales: a small integer multiplier per 32 weights plus one FP16 super scale per 256. **`--scale-bits 6` with `--native-groups` stores the real `q4_g32_s6_fp16` format**: the same 4-bit codes as g32, but each row's scale plane is one byte per 32 weights instead of two (4.25 instead of 4.5 bits per weight), and it works with both `--weights q4_g32` and `q4_g32_a16`. The receipt's `weight_bytes` is then the real size. Any other value, or 6 without `--native-groups`, is **experimental, measurement only**: the scales are emulated on ordinary g32 words, so the file does not shrink and no kernel reads that layout. On the author's model, in that emulation, 8- and 6-bit multipliers scored the same KL as FP16 scales (0.150) and 5-bit about 0.003 worse. Needs `--scale-search mse\|weighted`, `--group-size 32` and no `--region-group`; the real format also needs every projection input width, padded to 128, to be a multiple of 256. |
| `--gptq-source DIR` | Copy the 4-bit codes and scales of an external LLM Compressor GPTQ checkpoint (compressed-tensors `pack-quantized`, symmetric INT4, static contiguous groups, no activation order; group size 32, 128 or 256) without re-rounding. With `--native-groups` a 128/256 checkpoint is stored in the real `q4_g128_fp16`/`q4_g256_fp16` formats. Produce the checkpoint with [`tools/gptq/`](../tools/gptq/README.md); results are in "External GPTQ import" below. Mutually exclusive with `--scale-search`, `--imatrix`, `--group-size`, `--region-group` and `--scale-bits`. Experimental. |

`--region-group` is for experimenters. With uncalibrated scales, making a region coarse cost least in `mlp_gate_up`
and the first four layers, and most in `mlp_down`, the last four layers, the global-attention layers and
`attn_out`. A calibrated mixed allocation (`alloc-a`, A16, in the author's plan) scored KL 0.192, between g32 (0.150) and
uniform g256 (0.236). It was measured only weight-only, so treat any mix as unqualified until you score it
as in section 6; every row in section 2 is a uniform group size.

To store the two-level format, add `--scale-bits 6 --native-groups` to a g32 command (`--group-size 32`, which is the default):

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
-0.0034 with a standard error of 0.0028 over the 47 windows, is not resolved). On the same serving flags (131,072
context, FP8 KV, prefill chunk 128, no speculation) the native file decodes at 70.6 tok/s against 68.2 for the g32
file (+3.4%) and prefills a 3,000-token prompt at 3,049 tok/s against 3,046 (medians of six runs).

**External GPTQ import (`--gptq-source`, experimental).** One spike on the same model and instrument as section 2.
The checkpoint came from LLM Compressor 0.13.0 (compressed-tensors 0.18.0, transformers 5.14.1, torch 2.13.0+cu130):
`GPTQModifier` with 4-bit symmetric int weights, `strategy group`, `group_size 32`, `actorder None`,
`block_size 128`, `dampening_frac 0.01`, sequential targets `Gemma4TextDecoderLayer`, and the 400 chat-prefixed
512-token chunks used for the imatrix as calibration data (no 13-token sequence of the evaluation corpus occurs in them).
transformers 5.14.1 cannot read the checkpoint's `per_layer_config`, so GPTQ ran on a text-only `Gemma4ForCausalLM`
tree whose config carries `global_head_dim`/`num_global_key_value_heads` (what `normalize_config` produces). The run took
54 minutes on one RTX 5090 (10.3 GiB GPU, 39 GiB RAM with 32 GiB of weights kept on the CPU and the rest offloaded to disk).
The importer copies codes and scales word for word: every one of the 915,210,240 BF16 scales converts to binary16
unchanged (it refuses the checkpoint otherwise), `weight_zero_point` and `weight_g_idx` are refused, and the converter's
global-layer permutations are applied to the encoded words (q/k rows, o columns). A 32-group of the Hugging Face column
order maps onto a whole 32-group of the converter's order because the rotated blocks `[0,64)` and `[256,320)` of a
512-wide head are multiples of 32; the importer refuses any permutation that would split a group. All 240 weight objects
of the written artifact decode to exactly the checkpoint's values.

| Scales chosen by the GPTQ run (`--weights q4_g32_a16` / `q4_g32`) | KL (A16) | KL (W4A8) | ΔNLL (A16) | PPL (A16) |
|---|---|---|---|---|
| calibrated RTN, weighted search (section 2) | 0.1504 | 0.1569 | +0.061 | 11.986 |
| GPTQ, `memoryless_minmax` observer (the default, absmax/7.5) | 0.1697 | 0.1740 | +0.058 | 11.953 |
| GPTQ, `memoryless_mse` observer | 0.1591 | not run | +0.042 | 11.760 |
| **GPTQ, `imatrix_mse` observer** (importance-weighted scale search) | **0.0805** | **0.0845** | +0.025 | 11.566 |

Paired on the same 47 windows, GPTQ with `imatrix_mse` against calibrated RTN: ΔKL -0.0698 (-0.0845, -0.0570) A16 and
-0.0724 (-0.0871, -0.0598) W4A8, better on 46 of 47 windows and in every domain (long-form 0.077 against 0.154, reference
0.104 against 0.187, code 0.063 against 0.114, A16). With the default observer GPTQ is worse than calibrated RTN by +0.019
(+0.009, +0.032): error compensation alone does not beat a good scale search. That file is the ordinary g32 file (17.9 GB
of weights).

**Coarse groups.** The same run at `group_size` 128 and 256 (observer `imatrix_mse`, defaults otherwise, 55 minutes each),
converted with `--native-groups`, stores the real `q4_g128_fp16` / `q4_g256_fp16` formats: one binary16 scale per
source group, codes unchanged, decoded values identical to the repeated-scale g32 emulation. The ten global-layer `o`
projections stay `q4_g32_fp16` (with the source scale repeated): the converter's column permutation interleaves two
source groups into one converter group, so those tensors cannot be stored natively. That costs 41 to 48 MB
(0.3%). All 240 weight objects of the written artifacts decode to exactly the checkpoint's values.

| Recipe | File | KL (A16) | KL (W4A8) | ΔNLL (A16) | PPL (A16) |
|---|---|---|---|---|---|
| calibrated RTN g32 | 17.92 GB | 0.1504 | 0.1569 | +0.061 | 11.986 |
| calibrated RTN g128 | 16.55 GB | 0.2290 | 0.2327 | +0.105 | 12.527 |
| calibrated RTN g256 | 16.32 GB | 0.2358 | 0.2376 | +0.050 | 11.854 |
| **GPTQ `imatrix_mse` g32** | 17.92 GB | **0.0805** | **0.0845** | +0.025 | 11.566 |
| **GPTQ `imatrix_mse` g128** | 16.59 GB | **0.1033** | **0.1081** | +0.042 | 11.763 |
| **GPTQ `imatrix_mse` g256** | **16.37 GB** | **0.1192** | **0.1213** | +0.061 | 11.982 |

Paired on the same windows: GPTQ g256 against calibrated RTN g32 (1.55 GB, 8.7% smaller): ΔKL -0.0311 (-0.0456, -0.0178)
A16 and -0.0356 (-0.0502, -0.0226) W4A8; against calibrated RTN g256, -0.1166 (-0.1364, -0.0992) and -0.1162
(-0.1359, -0.0991). GPTQ g128 against RTN g32: -0.0470 (-0.0632, -0.0316) and -0.0489 (-0.0645, -0.0344); against RTN
g128, -0.1257 and -0.1247. Against GPTQ g32 the coarse groups cost +0.0228 (g128) and +0.0387 (g256) KL A16
(+0.0235 and +0.0368 W4A8). ΔNLL differences against RTN g32 are not resolved at g256 and g128 (the intervals span zero).

One fine-tune and one calibration set. To reproduce the route end to end (environment, calibration tokens, text-only
tree, GPTQ run, conversion) follow [`tools/gptq/README.md`](../tools/gptq/README.md), then score as in section 6.

**MTP assistant storage.** The assistant's output head is always FP8 and its two outer projections stay BF16 (loaded
as FP8). `--mtp-weights` changes only the 20 layer projections (query, output, gate, up, down in 4 layers), which
default to FP8 and were BF16 before. On the calibrated g256 file, eight fixed greedy prompts, three rounds each, FP8 KV, 131,072
context:

| `--mtp-weights` | File | Draft acceptance | tok/s | Greedy text vs bf16 |
|---|---|---|---|---|
| `bf16` (old layout) | 16.99 GB | 30.0% | 165.1 | reference |
| **`fp8` (default)** | 16.80 GB (-0.18 GB) | 30.0% | 164.6 | 8 of 8 identical |
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
| `K padded to 128 ... multiple of its group size` or `... multiple of 256` | `q4_g256_fp16` and `q4_g32_s6_fp16` have this limit (for the latter, K padded to 128 must be a multiple of 256); every Gemma 4 31B trunk width satisfies it. |
| `--imatrix` refused for missing statistics | The imatrix was collected on a different architecture or without the attention tensors. Collect it on the model you are converting. |
| g128/g256 file will not load | The binary predates `4cff979`; rebuild `ninfer-spo-serve`. |
| `q4_g32_s6_fp16` file will not load | The binary predates the commit that adds `q4_g32_s6_fp16`; rebuild `ninfer-spo-serve`. |
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
