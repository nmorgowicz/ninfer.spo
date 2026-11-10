# Gemma 4 31B in 4 bits: pick a build, make it, run it

This turns a Gemma 4 31B model (Google's, or a fine-tune) in BF16 into a 4-bit `.ninfer` file. The file runs on one
RTX 5090 with 131,072 tokens of context and fast speculative decoding (a small helper model, called MTP here, guesses
several tokens ahead and the big model checks them). You pick one of five builds, run one command, and get the file
plus a ready-made launcher.

Measurements, every flag and the manual steps are in the [technical reference](gemma4-31b-conversion-technical.md).
You do not need it to follow this guide.

## Pick a build

| Preset | For | Size | Quality (agreement) | KL | Wait | You need |
|---|---|---|---|---|---|---|
| `best-small` | Most people | 16.9 GB | Very close (90.0%) | 0.1213 | about 30 min | GPU with 12 GiB free, 64 GiB RAM, about 75 GB disk |
| `best-medium` | Highest quality | 18.4 GB | Closest (91.6%) | 0.0845 | about 30 min | Same as `best-small` |
| `lite-medium` | If your machine cannot run GPTQ (best of the two lite builds) | 17.5 GB | Close (89.3%) | 0.1543 | about 35 min | llama.cpp, about 35 GB disk |
| `lite-small` | If your machine cannot run GPTQ, and you want a smaller file | 16.8 GB | Noticeably lower (85.9%) | 0.2376 | about 35 min | llama.cpp, about 35 GB disk |
| `draft` | A quick test build only | 18.4 GB | Lowest (85.7%) | 0.2654 | about 15 min | Nothing extra |

Sizes are the whole file with the MTP helper. Waits are for an RTX 5090. Both are measured complete runs: the GPTQ
builds take 25.5 minutes end to end (see "What to expect" below), a full `lite-small` build took 33 minutes (GGUF
conversion 7, statistics 16, conversion 9), and `draft` was measured earlier on the same GPU.
"GPTQ" is a smarter way of rounding the weights to 4 bits. It needs its own Python environment, a GPU with 12 GiB free
and a lot of RAM. The two `lite` builds ask for less: they use a simpler rounding method (calibrated rounding) with
statistics from llama.cpp, no GPTQ environment and less RAM and VRAM. They give lower quality than the GPTQ builds at
the same size.

- **Most people: `best-small`.** Best quality for its size.
- Highest quality: `best-medium`. 1.5 GB bigger than `best-small`.
- If your machine cannot run GPTQ (under 12 GiB of free VRAM, under about 64 GiB of RAM, or you do not want the GPTQ environment): `lite-medium`, the best of the two, in about 35 minutes.
- The same, with a smaller file: `lite-small`. It is 0.7 GB smaller than `lite-medium` and 1.6 GB smaller than `draft`, but its quality is noticeably lower, and only a little better than `draft`.
- A quick test build: `draft`.

### What the quality numbers mean

Every build is compared with a near-lossless copy of the same model. Agreement is how often the build's top choice
for the next token is the same as that copy's: higher is better. KL is a distance between the two models'
whole next-token predictions: lower is better, and 0 would mean identical. KL tells builds apart better than
agreement does, so use KL to choose.

Read the scale from the table: the old default build (`draft`) scores 0.27 and our best (`best-medium`) scores 0.08,
about a third of that. `best-small` scores 0.12 in a file 1.5 GB smaller.

These numbers come from one model (a Gemma 4 31B fine-tune) and English text. Another model can score differently. To
measure your own build, see [section 6 of the technical reference](gemma4-31b-conversion-technical.md#6-check-the-quality-of-your-own-conversion),
or pass `--score` to the build command (about one more hour and 60 GB of scratch disk).

## Build it

You need:

- Windows with an RTX 5090, and this fork built (`ninfer-spo-serve` and friends; see [building.md](building.md)). Rebuild
  if the builder says your binary does not know the preset's format.
- The BF16 checkpoint of your model (Hugging Face folder), and Google's MTP helper once:
  `hf download google/gemma-4-31B-it-assistant --local-dir D:\models\gemma-4-31B-it-assistant`.
- Python 3.11 with PyTorch and NumPy, run from the root of this repository, with no uncommitted changes to tracked files.
- Calibration text: the build downloads a pinned public English corpus (needs `pip install pyarrow`), or pass your own
  with `--calibration-text FILE`. Keep it separate from any text you later test on.

Check everything first without writing anything: add `--dry-run` to see the checks, the exact steps and the time for
each, or `--check` to run only the checks.

### GPTQ builds: `best-small`, `best-medium`

One time only, make the GPTQ environment (needs [`uv`](https://docs.astral.sh/uv/); about 10 GB):

```powershell
python -m tools.convert.gemma4_build setup-gptq
```

Then build (use `--preset best-medium` for the other one):

```powershell
python -m tools.convert.gemma4_build --preset best-small `
  --model D:\models\my-31b-bf16 `
  --assistant D:\models\gemma-4-31B-it-assistant `
  --out-dir D:\ninfer\my-31b-best-small
```

What to expect: about 25 to 30 minutes in one run (measured: one complete `best-small` build, from the command to the
finished file and launcher, took 25.5 minutes on the RTX 5090: text tree 1.6, GPTQ 17.2, file check 0.9, convert 5.3,
calibration under 1; before the GPTQ solver was made faster the same build took 71 minutes). It needs a GPU with 12 GiB free,
64 GiB of RAM and at least 75 GB of disk (the builder says about 103 GB is comfortable; a 58 GB temporary copy of the
model is removed when the build succeeds). To be honest about the RAM: the GPTQ step itself used about 38 GiB while it
ran, and 64 GiB is still what we recommend. Put `--work-dir` on a big fast
drive to move the scratch files. No llama.cpp is needed.

### Lite builds: `lite-medium`, `lite-small`

These need a llama.cpp source checkout (the release zips do not include `convert_hf_to_gguf.py`) and a
`llama-imatrix` program. The build makes a temporary 35 GB file, collects statistics from it, then deletes it.

```powershell
python -m tools.convert.gemma4_build --preset lite-medium `
  --model D:\models\my-31b-bf16 `
  --assistant D:\models\gemma-4-31B-it-assistant `
  --out-dir D:\ninfer\my-31b-lite-medium `
  --llama-cpp-dir D:\tools\llama.cpp
```

`--llama-imatrix PATH` names the `llama-imatrix` program if it is not under `--llama-cpp-dir` or on your PATH. About
40 minutes in total. Use `--preset lite-small` for the smaller file.

### Quick test build: `draft`

```powershell
python -m tools.convert.gemma4_build --preset draft `
  --model D:\models\my-31b-bf16 --assistant D:\models\gemma-4-31B-it-assistant `
  --out-dir D:\ninfer\my-31b-draft
```

About 15 minutes, nothing extra to install.

### What you get

In `--out-dir`: `<name>.ninfer` (the model), `serve-<name>.cmd` (the launcher), `BUILD-SUMMARY.md` (what was built
and how), and two receipts (`.build.json` from the converter and `.gemma4-build.json` from the builder). Keep the
receipts with the model. A build that stops can be resumed: run the same command again. Use a fresh `--out-dir` for
each new build. Add `--no-mtp` instead of `--assistant` for a file without speculative decoding, or `--with-vision` to
keep the vision tower.

After a build, check the serving side too: `python -m tools.convert.gemma4_build serve-check --launcher
serve-<name>.cmd` starts the launcher's server, probes `/v1/models` and the chat endpoint (the launcher's model id,
another name under the launcher's `--model-match`, a missing `model` field, an arbitrary API key), stops the server
again and writes a serving receipt `serve-check-<utcstamp>.json` beside the launcher; it exits non-zero and names the
failing check if anything is off.

## Run it

Double-click `serve-<name>.cmd`, or run it in a terminal. It starts the server on `127.0.0.1:8080` with the model
name `131k`, and any model name in a request is accepted. To change the name, the address, the port or the name rule,
write the launcher again:

```powershell
python -m tools.convert.gemma4_build launcher --artifact D:\ninfer\my-31b-best-small\my-31b-best-small.ninfer `
  --out serve.cmd --alias 131k --host 0.0.0.0 --port 8001
```

`--host 0.0.0.0` lets other machines on your network connect; the server has no login, so restrict it with your
firewall. Two opt-in settings change what the launcher runs (both are explained in the settings table below):

```powershell
python -m tools.convert.gemma4_build launcher --artifact D:\ninfer\my-31b-best-small\my-31b-best-small.ninfer `
  --out serve-rp.cmd --gemma4-qk fp8 --prefill-chunk auto
```

The one-command build takes the same two options (`--gemma4-qk bf16|fp8`, `--prefill-chunk 128|<multiple of 128>|auto`)
and writes them into `serve-<name>.cmd` and the `BUILD-SUMMARY.md` serve command, with the same defaults and refusals;
without them the launcher is unchanged.

The launcher runs this:

```text
ninfer-spo-serve <file>.ninfer --host 127.0.0.1 --port 8080 --model-id 131k --model-match any
  --max-context 131072 --kv-capacity auto --kv-headroom-mib 1024 --kv-dtype fp8
  --max-concurrency 1 --prefill-chunk 128
  --spec mtp --draft-tokens 1 --device-state-slots 0 --host-context-mib 8192
```

`--spec mtp --draft-tokens 1` appears only when the file has the MTP helper, and `--vision` only when it has the
vision tower. The launcher command accepts `--draft-tokens auto|1..7`, default `1`. Explicit `auto` uses six
drafts for Gemma and can hurt long-context speed; one draft is the conservative roleplay default.

Qualified builds use the shared-KV global D512 decode kernel by default; no launcher flag is needed.
`NINFER_GEMMA4_DECODE_ATTN=old` selects the original path for comparison before process startup.
It changes reduction order, not KV/QK precision; measured gains and qualification limits are in
[configuration.md](configuration.md#default-shared-kv-gemma-decode-kernel).


After it starts, read the startup log: the line that begins `capacity | KV` shows how many tokens of context really fit.
The model list can show more than the pool actually holds.

### The model name and keys

You do not need an API key: the server asks for none unless you pass `--api-key`, and it ignores any key a client sends
(for example `sk-ryne`). The launcher passes `--model-match any`, so chat and responses requests may send any `model`
name (for example `160k` or `262k`); `/v1/models` still reports the alias (`131k` by default), and a request with no
`model` is still rejected with 400. This needs a server build that includes `--model-match`. Without the flag (the
server default is `exact`, and `--model-match exact` on the launcher command gives the same), requests must send
`model` equal to the alias or the server answers 404 `model_not_found`. Pick the alias you want clients to see with
`--alias`.

### The settings, and why

What each preset already sets when it converts:

| Conversion setting | What it does and why |
|---|---|
| W4A8 route | 4-bit weights, and prefill (reading the prompt) uses FP8 maths: about 7 times faster prefill, at a small quality cost |
| Text only | The vision tower is left out so more memory holds context (`--with-vision` keeps it) |
| MTP helper stored in FP8 | 0.18 GB smaller than BF16 with the same acceptance in our test |
| Calibration | 400 chunks of 512 tokens of public English text, each starting with Gemma's chat prefix, because chat models see that prefix at run time |
| The preset's method and group size | See the table above: GPTQ or calibrated rounding, group 32 or 256 |

What the launcher sets when it serves:

| Serving setting | Value | Why |
|---|---|---|
| Context | `--max-context 131072` | The target: 131,072 tokens on one 5090 |
| KV storage | `--kv-dtype fp8` | Stores the global layers' memory in FP8, half the size of BF16 (5.04 GiB instead of 10 GiB at 131,072 tokens); the setting all the long-context runs used and what lets 131,072 tokens fit with room to spare. `--kv-dtype bf16` is the engine's own default for a hand-written `ninfer-spo-serve` command and the launcher's explicit alternative; see [FP8 KV and smaller cards](#fp8-kv-and-smaller-cards) |
| Attention precision | not set (`--gemma4-qk bf16`) | The engine default, so the launcher emits no flag. `--gemma4-qk fp8` is optional: +13-22% prompt speed at 32k-100k, measurably less faithful (top-1 disagreement about 4% against 1% run-to-run noise). Reasonable for roleplay and prose; keep the default for code, tools and structured output. Add it with `launcher --gemma4-qk fp8` (it needs `--kv-dtype fp8`; the launcher refuses it with `bf16`); see [Choosing --gemma4-qk](configuration.md#choosing---gemma4-qk) |
| KV size | `--kv-capacity auto --kv-headroom-mib 1024` | Uses what fits and leaves 1 GiB free |
| Concurrency | `--max-concurrency 1` | One request at a time, so all spare memory goes to context |
| Prefill chunk | `--prefill-chunk 128` | The size all the measurements used. `launcher --prefill-chunk <multiple of 128 up to 16384>` or `auto` changes it: on one g256 artifact at 131,072 context, no speculation, BF16 QK, 32k/100k prompts processed at 2,221/1,280 tok/s at chunk 128 and 3,239/1,683 tok/s at chunk 2048 (serving medians, [configuration.md](configuration.md#gemma-4-automatic-prefill-width) and the changelog). Larger chunks need more memory; `auto` picks the largest of 2048/1024/512/256/128 that fits after the weights load and needs a server build that has it |
| Speculative decoding | `--spec mtp --draft-tokens 1` | Conservative default for growing roleplay context; explicit `auto` uses six drafts, not adaptive selection |
| Checkpoints | `--device-state-slots 0 --host-context-mib 8192` | No GPU checkpoint slots, 8 GiB of system memory for them |
| Address and name | `--host`, `--port`, `--model-id`, `--model-match any` | Defaults `127.0.0.1`, `8080`, `131k`, any request model name accepted; change them with the launcher command above |

### FP8 KV and smaller cards

`--kv-dtype` is the setting that decides whether the long context fits. The engine's default is `bf16` (for a
hand-written `ninfer-spo-serve` or `ninfer-spo-cli` command, for every model family); this launcher writes `fp8`.
At 131,072 tokens the global layers' KV is 5.04 GiB in FP8 and about double (10 GiB) in BF16; the sliding layers'
rings (0.88 GiB at chunk 128) are BF16 either way. On the 32 GiB RTX 5090 with one artifact, FP8 held all 131,072
tokens with more than 5 GiB free, while BF16 plus MTP reached only 124,224 on the run that sized automatically. FP8
costs prefill speed against BF16 (32k prompt with MTP: 1,245 against 1,722 tok/s in that older measurement) and
changes about 5% of top-1 token choices against the BF16 cache with no resolvable perplexity change; the numbers and
their limits are in [Experimental Gemma 4 FP8 KV](configuration.md#experimental-gemma-4-fp8-kv) and
[KV-format evaluation](kv-evaluation.md). Use `--kv-dtype bf16` only when the card holds the larger pool at the
context you need.

Cards below 32 GiB: a 24 GiB card cannot host 131,072 tokens of this model at group 256 by arithmetic (the weights
plus 5.04 GiB of FP8 KV, the ring, workspace and headroom exceed it; not measured on such a card). Generate the launcher
with a smaller `--context` (the global KV scales linearly with it, 40.3125 KiB per token in FP8) and read the
`capacity | KV` startup line to see what was reserved. Do not copy the 32 GiB figures above to another card.

More about serving is in [serving.md](serving.md).

## When something goes wrong

| Problem | Fix |
|---|---|
| The check says there are tracked modifications | Commit or stash your changes in this checkout; the converter records the exact revision |
| The check says the server does not know the format, or the server rejects `--model-match` | Rebuild `ninfer-spo-serve` from this checkout ([building.md](building.md)); the binary is older than the fork commit that has it |
| `setup-gptq` fails | It needs [`uv`](https://docs.astral.sh/uv/) on your PATH and about 10 GB of disk |
| The calibration download fails | `pip install pyarrow`, or pass your own text with `--calibration-text` |
| Out of disk during the build, or the build stopped half way | Move the scratch files with `--work-dir` to a bigger drive (`--dry-run` lists the needs); run the same command again and it resumes from the work folder |
| `convert_hf_to_gguf.py` not found | Pass a llama.cpp source checkout with `--llama-cpp-dir` (release zips lack it) |
| The context is smaller than you asked | Read the startup line about KV capacity (the model list can overstate it) and close other GPU programs |
| The client gets 404 `model_not_found` | The launcher was made with `--model-match exact`, or the server predates the flag: send the alias as `model`, or make a launcher with `--model-match any` |

More cases are in the [technical reference](gemma4-31b-conversion-technical.md#8-when-something-goes-wrong).

## Where to go next

- [Technical reference](gemma4-31b-conversion-technical.md): all measured builds, paired intervals, caveats, the manual
  commands, how to reproduce the numbers, known limits.
- [weight-conversion.md](weight-conversion.md): the general converter and the other Gemma 4 recipes.

## Experimental calibrated NVFP4 A4-MLP (not a preset)

This is a separate opt-in artifact experiment, **not** a change to the recommended GPTQ builds.
Start with an existing **all-A16Only NVFP4** Gemma artifact, not a GPTQ-g256 artifact: activation
ranges depend on the actual weights that produced them. The first measured absmax candidate failed
the held-out quality gate below; this is not a recommended preset or a qualified speed improvement.

Build the offline target explicitly in your existing configured CUDA build:

```powershell
cmake --build D:\AI\loaders\build-spo-splitk-cuda13.5 --config Release --target ninfer-spo-gemma4-calibrate -j 2
```

It is excluded from default builds. `gemma4_a4 collect` executes that GPU program; run it only
when the GPU is free. It uses the existing GPTQ calibration-text parser (including recovery of
the first chunk's omitted BOS), requires the artifact's exact tokenizer, and resets each row's
causal positions. Collection is eager, BF16 KV/BF16 QK, without generation, prefix reuse, MTP or
vision. The two boundaries in every layer are:

- `text/layers/L/ffn_input`: the represented BF16 pre-FFN norm output, after the attention
  residual. Gate and up consume the very same tensor, so packed gate/up receive identical
  FP32 divisor bytes.
- `text/layers/L/mlp/product`: the represented BF16 GEGLU product, before down projection.
  Neither the gate/up outputs nor the post-FFN residual is a substitute.

The native sink reduces actual absolute maxima on the GPU, refuses nonfinite values, and
records width, token and call coverage. A zero/nonpositive maximum is rejected by the handoff;
there is no invented scale fallback. The activation consumer computes
`E4M3(divisor * block_absmax / 6)` and `E2M1(input * divisor / block_scale)`, then corrects
the GEMM by `1/(weight_divisor * divisor)`. The default `--scale-rule absmax` uses
`FP32(448*6/absmax)`, matching the existing `gemma4_nvfp4_a4_mlp` override and the consumer's
`/6` rule, not the weight encoder's `/4` versus `/6` search. The optional MSE recipe is below.

Example (paths are operator choices; keep all new evidence under a new campaign directory):

```powershell
python -m tools.convert.gemma4_a4 collect --artifact D:\experiments\mixed-a16.ninfer `
  --collector D:\AI\loaders\build-spo-splitk-cuda13.5\apps\ninfer-spo-gemma4-calibrate.exe `
  --text D:\calibration\templated-400x512.txt --tokenizer D:\models\my-31b-bf16\tokenizer.json `
  --context 512 --rows 2 --out D:\experiments\a4-smoke
# Repeat without --rows for the full corpus and with a NEW output directory.
python -m tools.convert.gemma4_a4 candidate --artifact D:\experiments\mixed-a16.ninfer `
  --receipt D:\experiments\a4-full\receipt.json --output D:\experiments\mixed-a4-mlp.ninfer
```

Use `--prefix-ids` for a different explicit prefix; the default is the existing Gemma GPTQ
chat prefix. `--rows` selects an explicitly recorded leading-row smoke subset, never a hidden
calibration shortcut. `tokens.json`, `statistics.json`, `collector.log`, `invocation.json` and
`receipt.json` retain the exact input/coverage and source/binary/tokenizer/text hashes.
Do not qualify a two-row smoke candidate as a 400-row candidate.

`candidate` is a CPU-only, bounded `ArtifactWriter` copy of all original encoded objects:
no model weights are decoded, recalibrated, or re-rounded. It checks the receipt against the
source's full file hashes and native ID, and enables `AllowA4` **only** on NVFP4 text MLP uses.
FP8-sensitive MLP layers stay A16; attention, vocabulary, MTP and vision remain unchanged.
Logical widths, row slices, complete per-layer coverage and shared-parent input identity must
agree. A new artifact ID and `.a4.json` receipt identify the candidate; no existing output or
source is overwritten. The normal forward path has no collector, hence no collection buffers,
copies, synchronizations or new runtime option. Host-observing taps explicitly reject CUDA
graph capture rather than collecting only the graph's first execution.

Before choosing this artifact, compare it and an A16 repeat against the same held-out full-
vocabulary reference (identical corpus windows, prefix, KV/QK, context, stride and chunk).
Declare a quality margin before inspecting results; inspect per-domain/tail KL, top-1 and NLL,
then gate long-context retrieval/continuation and MTP separately. Only after quality should
interleaved no-speculation and MTP serving runs measure fresh/cached-tail PP, TG and memory.
Keep the source A16 artifact and deployed GPTQ model/launchers intact throughout.

### Error-aware scales and selective projections

New collections also retain eight stratified, contiguous K16 blocks per tap/call in a
deterministic1024-block reservoir per logical input. Full-input absmax and coverage remain
checked. Each retained block has native scale/code probes at divisors1,256 and FP32(1.2),
computed by the actual serving `quantize_nvfp4_k16` helper. The third probe exercises
divisor-dependent product rounding; powers of two cannot. The handoff reproduces coordinates and
requires exact probe agreement before writing a candidate; old absmax-only statistics cannot
supply MSE samples or invented percentiles.

`candidate --scale-rule mse` searches FP32 divisors `absmax_divisor * 2^(step/16)`, steps-16..48,
minimizing sampled activation reconstruction NMSE with FP32 arithmetic and RN-even/SATFINITE
E4M3 block scales/E2M1 codes. Without scale rounding/range limits the divisor cancels; this
search targets scale phase, underflow and saturation effects, not a weight-aware GPTQ objective.
Search grids, baseline/selected errors, sampled energy and saturation/zero-scale counts go
into provenance and the sidecar. Sampling does not exhaustively cover tails or downstream drift.

`--mlp full|gate-up|down` (default `full`) and `--layers all|0,2-5,...` (default `all`, zero-based)
restrict the changed NVFP4 uses. Gate/up remain an inseparable shared-input pair. Nonselected
layers and protected components stay A16; a parent shared with an unselected/protected use
is refused rather than silently granting it A4 permission.

```powershell
python -m tools.convert.gemma4_a4 candidate --artifact D:\experiments\mixed-a16.ninfer `
  --receipt D:\experiments\a4-full\receipt.json --scale-rule mse --mlp gate-up `
  --layers all --output D:\experiments\mixed-mse-gate-up.ninfer
```

Choose and record any calibration-only layer ranking before inspecting held-out scores.
Lower reconstruction error is not model-quality or speed qualification. Quality gates still
apply to each candidate; serving comparisons must include both its own A16 control and the
deployed GPTQ artifact, since different weight recipes are not isolated kernel comparisons.

### Measured MSE/selective matrix: no rollout

Real400×512 collection passed all three native-codec probes across120 logical inputs and
1024 retained blocks/input. All710 original encoded objects matched in each candidate.
163 Python tests passed, plus53 focused tests after direct-caller argument migration.
The lowest-NMSE gate/up quarter was14 eligible layers selected from calibration before scoring.

Held-out protocol: identical mixed-A16 control/repeat, FP8-row-A16 full-vocabulary reference,
47 windows/13728 positions, FP8 KV/BF16 QK, context512/stride256/chunk128. The repeat was identical.
Upper95CI limits1.01 KL ratio and+0.001nat/token NLL applied overall and every domain:

| Candidate | KL ratio [95%CI] | Delta NLL nat/token [95%CI] | Gate |
|---|---|---|---|
| MSE full | 1.17174 [1.10720,1.23969] | +0.001318 [-0.012665,+0.013366] | Fail |
| MSE gate/up | 1.18780 [1.14306,1.23776] | +0.004514 [-0.005687,+0.015174] | Fail |
| MSE down | 1.02022 [0.97977,1.06033] | -0.018997 [-0.032525,-0.007462] | Fail |
| MSE gate/up quarter | 1.03954 [1.01022,1.06570] | +0.005730 [+0.000010,+0.012133] | Fail |

Down-only improved held-out PPL1.88%, but did not establish required distributional fidelity;
likelihood and reference KL are different instruments. Sample reconstruction NMSE fell a median
1.88% per calibrated input, not a model-quality guarantee. No candidate proceeded to speed,
long-context/MTP or default qualification. Source/deployed weights and launchers remain unchanged.
Full per-domain/tail diagnostics and encoded identity:
`pp-bench-20261108/additive-next-20261110/a4-error-aware/`.



### First measured absmax result: rejected for rollout

On one mixed-NVFP4 Gemma fine-tune/RTX5090, the collector completed both a2×512 smoke and
400×512 real calibration, covering120 logical inputs. The encoded-copy candidate changed168 NVFP4
MLP uses with112 shared/independent FP32 divisors; protected components and encoded weights were retained.
The collector-only target built successfully and128 Python calibration/builder tests passed.
The existing released serving/perplexity executables were not rebuilt or replaced.

Matched held-out full-vocabulary comparison:47 windows/13,728 positions, FP8-row-A16 reference,
FP8 KV/BF16 QK, context512/stride256/chunk128. A16 repeat was identical. Mean KL rose
0.127818→0.165489 (ratio1.29473,95%CI[1.23124,1.35862]); delta NLL+0.019077nat/token,
95%CI[0.007051,0.032367], approximately+1.93% PPL.
This fails the predeclared upper-CI limits of1.01 KL ratio and+0.001nat/token NLL.
No long-context/MTP/default promotion or performance qualification followed that failure.
This rejects this absmax recipe, not all possible calibrated A4/clipping/Hessian-aware methods.
Camp evidence: `a4-mlp-collector-20261109/{full400x512,quality-verdict.json,paired-a4-mlp.json}`.

## Follow-on scheduling investigation: no bake-overlap rollout

A temporary Program-owned two-slot Q4 bake pipeline passed four targeted CTests, including
bitwise original/prepared projection outputs, reuse/cancellation/destruction and memory budgeting.
It was then measured on the same fixed-path candidate EXE in off/on/off/on order, FP8 KV/BF16 QK,
explicit chunk2048, no speculation:32 successful fresh requests, two sessions per arm and three
timed requests after warmup at each context, with identical prompt hashes and token counts.

| Context | Off prompt tok/s | On prompt tok/s | Change |
|---|---:|---:|---:|
| ~32k | 3865.51 | 3856.88 | -0.22% |
| ~100k | 1984.40 | 1983.10 | -0.07% |

Those small deltas do not establish a regression, but show no useful gain for the additional
462,594,048 device bytes (441.16MiB). Both arms retained131072 native KV; startup runtime
allocation increased8.96→9.39GiB. The implementation, switch and experimental tests were removed;
there is no bake-overlap setting to enable. No full-model quality or MTP/cached-tail rollout
qualification followed the failed performance gate.

The three fixed-path EXEs were restored byte-identically to release16ff8fc. Actual d1 HTTP chat
returned40 tokens with24 drafts/15 accepted, default shared-KV attention and131072 native KV.
Source weights and existing launchers were unchanged. Evidence in the workstation camp:
`pp-bench-20261108/owner-bake-20261109/{candidate.patch,tests.log,serving-summary.json,decision.json,restored-release-smoke.json}`.
Cell-boundary clock/throttle snapshots are not continuous monitoring. This rejects this design
on this workload/GPU, not every possible overlap design.

## Default shared host checkpoint pages

Gemma host checkpoint sharing is default-on after the2026-11-10 operator-approved cutover.
Unset or `NINFER_GEMMA4_HOST_DELTA=1` enables it; explicit0 retains the original full-copy path.
It shares immutable global-page payloads inside the existing bounded pinned-host arena.
Physical-page write identities—not matching tokens—determine reuse; allocation, partial writes
and COW invalidate identities, and host restore propagates the restored identity. Each
checkpoint retains its own complete ring snapshot and original coverage/LRU rules.
This changes neither weight/KV/QK precision nor which mathematical positions a snapshot covers.

No host arena, or no sliding ring, leaves sharing inactive.
There is no second pinned allocation or new device workspace; host payload blocks are reference
counted and returned to the same arena when their last checkpoint releases them. Cold global
capture batches consecutive physical/host pages with pitched per-region copies; reused pages
submit no new global DtoH copy. Cold captures and independent ring copies still cost time.

Parent CUDA13.5 builds and tiny-buffer ownership/budget/byte-restoration tests passed.
The real Engine checkpoint scenario passed with the option off and on at16k/BF16 KV/MTP5:
displaced conversations, changed memory, explicit boundaries, eviction and token outputs matching
fresh prefills. This is the existing fixture's explicit MTP5, not a launcher recommendation.

On one5090/g256, same-binary off/on/off/on, FP8 KV/BF16 QK/chunk2048/noMTP,
host8192MiB/device0, the fixed100k scripted-reply-divergence workload completed16 requests.
Both sessions per arm used identical prompt hashes and token/cache counts; each arm contributed
four cached tails. Pooled median tail prompt time fell1682.62→1378.29ms (-18.1%); complete tail
wall time3.170→2.856s (-9.9%). Fresh-ingest and decode medians stayed within1%; individual
cells varied and were not discarded. All starts retained131072 native KV,8.96GiB runtime,
and8GiB pinned host storage. Boundary memory-clock samples were13801MHz; no continuous
clock-stability claim or cross-session comparison with earlier14001MHz results.

A second ABAB with actual d1/FP8-QK flags passed16 matching requests and identical draft/
acceptance counts per repetition. Tail prompt1482.08→1201.59ms (-18.9%), complete tail
wall2.751→2.519s (-8.4%). Fresh prompt median+1.34%, ingest decode-2.58%, tail decode-1.50%;
these are bounded workload observations, not a decode-kernel speed claim.

Near-capacity d1 smoke ingested124824 prompt tokens, appended twice, displaced the conversation,
then reused125917 tokens on return and after both prefill/decode cancellations. Recorded recovery
content matched, and no native errors appeared in that server log. Both cancellation follow-ups
succeeded; this is not exhaustive CUDA-error or universal semantic qualification.

One exact-window Activity pair observed tail DtoH10,524,057,856→2,275,983,616B (-78.37%),
425.54→87.38ms summed copy time, with39950 model kernel calls in each arm. Initial ingestion
already captured the host frontier, allowing both subsequent captures to share its global prefix.
The first cold ingest still copied5.23GB in both modes; rings and changed pages still transfer.
Traces lack final end metadata after forced teardown: no completeness/no-drop guarantee, and
instrumented timings are not the performance benchmark.

The measurements above qualified the unchanged algorithm while it was opt-in. The subsequent
default-selection cutover passed4 native default/explicit-on/opt-out/capacity cases and live
d1/FP8-KV/FP8-QK cold/cached HTTP requests in all3 modes. All starts retained131072 native KV;
cached content, token/cache counts and draft/acceptance counts matched. This is selection and
compatibility smoke, not a new performance estimate or universal semantic qualification.
No environment setting is needed. To opt out before starting the existing server:

```powershell
$env:NINFER_GEMMA4_HOST_DELTA = '0'
# Start your existing server command with its existing precision/context/MTP flags.
```

Existing launchers, preset choices and the released shared-KV decode default are unchanged.
Evidence: `pp-bench-20261108/additive-next-20261110/checkpoint-decision.json` and
`checkpoint-{native,nomtp,d1,stress,attribution}/`. Default-selection evidence:
`pp-bench-20261108/host-default-20261110/{ctest.log,results.json}`.

