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
