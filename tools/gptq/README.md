# GPTQ route for Gemma 4 31B (`--gptq-source`)

These scripts produce the checkpoint that `python -m tools.convert.gemma4_q4 convert --gptq-source DIR` imports:
LLM Compressor GPTQ, symmetric INT4, static contiguous groups, no activation order. The converter copies the codes
and the (binary16-exact) scales word for word; nothing is re-rounded. Measured on the author's calibrated 31B this
route halves the KL of calibrated RTN at group 32 (see [the technical reference](../../docs/gemma4-31b-conversion-technical.md#4-results)),
so use the recommended `imatrix_mse` observer: the default `memoryless_minmax` observer is worse than calibrated RTN.

| Script | What it does |
|---|---|
| `make_calib.py` | Chat-prefixed calibration text (from `gemma4_q4 calibration-text`) -> `[chunks, 512]` token tensor |
| `make_text_tree.py` | BF16 multimodal checkpoint -> text-only `Gemma4ForCausalLM` tree that transformers 5.14 can load |
| `run_gptq.py` | GPTQ with LLM Compressor, layer by layer, writes the packed projections as it goes |
| `check_checkpoint.py` | Verifies the checkpoint is importable (4-bit codes, scales exact in binary16) before you convert |

This is a separate environment from the converter's. The converter itself needs only what
[weight-conversion.md](../../docs/weight-conversion.md) lists; only these scripts need `requirements.txt`.

## 1. Environment

Needs an NVIDIA GPU with about 12 GiB free (peak torch memory was 10.3 GiB on an RTX 5090), 64 GiB RAM (see the
CPU cap below), and about 58 GiB of disk for the text-only tree plus 16 GiB per checkpoint.

```bash
uv venv --python 3.13 .venv-gptq
uv pip install --python /absolute/path/to/.venv-gptq/bin/python \
  --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match -r tools/gptq/requirements.txt
```

(On Windows the interpreter is `.venv-gptq\Scripts\python.exe`.) Every version is pinned in `requirements.txt`; the
route was measured with llmcompressor 0.13.0, compressed-tensors 0.18.0, transformers 5.14.1, torch 2.13.0+cu130.
**Always call the environment's interpreter by absolute path.** A relative `.venv/Scripts/python.exe` resolved to a
different environment's interpreter in the shell the route was developed in (the symptom was transformers 5.15.1 with
its `per_layer_config` error instead of the pinned 5.14.1). Check with
`<python> -c "import transformers; print(transformers.__version__, transformers.__file__)"`.

## 2. Calibration tokens

Calibration text must be public text disjoint from whatever you evaluate on, rendered as chat-prefixed chunks exactly
as for the imatrix route (`gemma4_q4 calibration-text`, [section 5.2 of the technical reference](../../docs/gemma4-31b-conversion-technical.md#52-convert-with-calibration-rtn)); the same 400 x 512 file
serves both. A chat fine-tune sees the chat prefix at run time, so bare text gives worse statistics.

```bash
python tools/gptq/make_calib.py --text calibration-templated.txt \
  --tokenizer /path/to/gemma-4-31B-bf16/tokenizer.json --out work/calib.pt
```

Every chunk must encode to exactly 512 tokens and start with the 17 prefix tokens; anything else is an error.

## 3. Text-only tree

```bash
python tools/gptq/make_text_tree.py --model /path/to/gemma-4-31B-bf16 --out work/text60
```

About 90 seconds and 58 GiB. The checkpoint's `per_layer_config` (written by transformers 5.15) cannot be read by
transformers 5.14.1, so the tree's config carries `global_head_dim` and `num_global_key_value_heads` instead, exactly
what the converter's `normalize_config` produces. Weights are copied bit for bit with `model.language_model.*`
renamed to `model.*`. `--layers N` makes a truncated tree for smoke tests; N must end on a global layer (6, 12, ...),
because transformers rewrites a truncated model's last layer to full attention.

## 4. GPTQ

```bash
python tools/gptq/run_gptq.py --tree work/text60 --calib work/calib.pt --out work/gptq-g32 \
  --group-size 32 --max-cpu-gb 32 --offload-dir work/offload --peak-log work/gptq-g32.peak.jsonl
```

`--observer` defaults to `imatrix_mse` (importance-weighted scale search from the activations seen at each
layer). Expect about 54 minutes on an RTX 5090 for 400 chunks: 2,400 s of GPTQ solves (the 21504-wide down projection
15 s per layer, the others 4 to 7 s), the rest forward passes. Peak torch memory 10.3 GiB; host RSS peaked at 39 GiB
with the 32 GiB CPU cap. Group sizes 128 and 256 take the same time and memory (`--group-size`, defaults unchanged)
and needed no observer or dampening change: on the author's model, with `--native-groups` at conversion, group 256
scores KL 0.1192 (A16) / 0.1213 (W4A8) in a 16.37 GB file and group 128 scores 0.1033 / 0.1081 in 16.59 GB, against
0.1504 / 0.1569 for calibrated RTN at group 32 (17.92 GB); group 32 GPTQ scores 0.0805 / 0.0845.

Workarounds the script applies, each learned the hard way:

- **`processor=<tokenizer>`**: `oneshot` wants a processor when given a dataset, and the Gemma4 processor imports
  torchvision, which is not installed.
- **Per-layer dump instead of `save_pretrained`**: the model does not fit in 64 GiB of RAM, so weights beyond
  `--max-cpu-gb` are disk-offloaded, and transformers' `save_pretrained` then either dies silently while assembling a
  20 GB shard or raises "could not revert some weight conversions". Either loses the run. `run_gptq.py` instead calls
  compressed-tensors' own `PackedQuantizationCompressor.compress` on each layer right after its solve and writes
  `part-NNNN.safetensors`; on a 6-layer model that is byte-identical to the stock save (123 of 123 tensors).
- **`--max-cpu-gb 32`** keeps 32 GiB of weights in RAM and offloads the rest. `auto_offload` with all available RAM
  left too little for the save step; an extreme cap (4 GiB on a 6-layer model) failed at the last layer with
  "Cannot copy out of meta tensor", so do not cap far below the model size.
- **Load check**: a tree with wrong key names loads "successfully" with random weights. The script compares three
  loaded tensors with the files and refuses to start otherwise.
- **No activation order**: `actorder` is set to `None` explicitly (LLM Compressor's default is `static`); the importer
  refuses `weight_g_idx`.

Without `--max-cpu-gb` all weights stay on the CPU (fine for truncated trees; the full model needs about 60 GiB).

## 5. Check, convert, verify, score

```bash
python tools/gptq/check_checkpoint.py --gptq work/gptq-g32 --tree work/text60

python -m tools.convert.gemma4_q4 convert --model /path/to/gemma-4-31B-bf16 --text-only --no-mtp \
  --weights q4_g32_a16 --gptq-source work/gptq-g32 --out work/gptq-g32-a16.ninfer
python -m tools.convert.gemma4_q4 verify --artifact work/gptq-g32-a16.ninfer
```

Use `--weights q4_g32` for the W4A8 artifact and `--mtp-assistant DIR` instead of `--no-mtp` for MTP, as in the
[technical reference](../../docs/gemma4-31b-conversion-technical.md#51-convert-without-calibration). For a group-128 or group-256 checkpoint add `--native-groups` to store the real
`q4_g128_fp16`/`q4_g256_fp16` formats (without it the scales are repeated into g32 words: same values, no size win).
`--gptq-source` is mutually exclusive with `--scale-search`, `--imatrix`, `--group-size`, `--region-group` and
`--scale-bits`. The converter refuses a checkpoint it cannot store exactly: zero points, `weight_g_idx`, scales that
are not exactly representable in binary16, asymmetric or non-4-bit schemes. Score the artifact as in
[section 6 of the technical reference](../../docs/gemma4-31b-conversion-technical.md#6-check-the-quality-of-your-own-conversion)
(`NINFER_Q4_EXACT=1` for W4A8).

Global-layer permutations: the converter permutes global q/k rows and o columns (`prope_perm`). Row permutations never
touch groups. The o permutation maps every 32-group of the Hugging Face order onto a whole 32-group (the rotated blocks
`[0,64)` and `[256,320)` of a 512-wide head are 32-aligned), so group 32 imports every tensor exactly. For group 128
and 256 the permutation interleaves 64-wide runs of two source groups into one converter group, so the ten global-layer
o projections cannot be stored natively; with `--native-groups` they are stored as `q4_g32_fp16` with the repeated
source scales (still exact), and the receipt lists them under `kept_g32_storage`.
