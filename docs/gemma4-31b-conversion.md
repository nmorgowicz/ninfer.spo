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
| `fast-medium` | No GPTQ wait | 17.5 GB | Close (89.3%) | 0.1543 | about 40 min | llama.cpp, about 35 GB disk |
| `fast-small` | Smallest file, lower quality | 16.8 GB | Noticeably lower (85.9%) | 0.2376 | about 40 min | llama.cpp, about 35 GB disk |
| `draft` | A quick test build only | 18.4 GB | Lowest (85.7%) | 0.2654 | about 15 min | Nothing extra |

Sizes are the whole file with the MTP helper. Waits are for an RTX 5090. The GPTQ step of the two GPTQ builds was
measured at 17 minutes and the other steps come from an earlier full build (see "What to expect" below); the `fast` and
`draft` waits were measured earlier on the same GPU.
"GPTQ" is a smarter way of rounding the weights to 4 bits and needs its own Python environment; the two `fast` builds
use a simpler method that needs statistics from llama.cpp.

- **Most people: `best-small`.** Best quality for its size.
- Highest quality: `best-medium`. 1.6 GB bigger than `best-small`.
- Do not want to wait for GPTQ: `fast-medium`. The best quality you can get in about 40 minutes.
- Want a smaller file without GPTQ: `fast-small`. It is 0.7 GB smaller than `fast-medium` and 1.6 GB smaller than `draft`, but its quality is noticeably lower, and only a little better than `draft`.

### What the quality numbers mean

Every build is compared with a near-lossless copy of the same model. Agreement is how often the build's top choice
for the next token is the same as that copy's: higher is better. KL is a distance between the two models'
whole next-token predictions: lower is better, and 0 would mean identical. KL tells builds apart better than
agreement does, so use KL to choose.

Read the scale from the table: the old default build (`draft`) scores 0.27 and our best (`best-medium`) scores 0.08,
about a third of that. `best-small` scores 0.12 in a file 1.6 GB smaller.

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

What to expect: about 30 minutes in one run. The GPTQ step alone is about 17 minutes (measured: 17.4 minutes for
group 32 and 16.9 for group 256). The other steps are from the earlier end-to-end `best-small` build, which took 71
minutes with 62.5 in GPTQ before the solver was made faster: text tree 1.7, file check 0.8, convert 5.2, calibration
under 1. The 30 minutes is the sum of those measured steps, not one measured run. It needs a GPU with 12 GiB free,
64 GiB of RAM and at least 75 GB of disk (the builder says about 103 GB is comfortable; a 58 GB temporary copy of the
model is removed when the build succeeds). Put `--work-dir` on a big fast
drive to move the scratch files. No llama.cpp is needed.

### Fast builds: `fast-medium`, `fast-small`

These need a llama.cpp source checkout (the release zips do not include `convert_hf_to_gguf.py`) and a
`llama-imatrix` program. The build makes a temporary 35 GB file, collects statistics from it, then deletes it.

```powershell
python -m tools.convert.gemma4_build --preset fast-medium `
  --model D:\models\my-31b-bf16 `
  --assistant D:\models\gemma-4-31B-it-assistant `
  --out-dir D:\ninfer\my-31b-fast-medium `
  --llama-cpp-dir D:\tools\llama.cpp
```

`--llama-imatrix PATH` names the `llama-imatrix` program if it is not under `--llama-cpp-dir` or on your PATH. About
40 minutes in total. Use `--preset fast-small` for the smaller file.

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

## Run it

Double-click `serve-<name>.cmd`, or run it in a terminal. It starts the server on `127.0.0.1:8080` with the model
name `131k`, and any model name in a request is accepted. To change the name, the address, the port or the name rule,
write the launcher again:

```powershell
python -m tools.convert.gemma4_build launcher --artifact D:\ninfer\my-31b-best-small\my-31b-best-small.ninfer `
  --out serve.cmd --alias 131k --host 0.0.0.0 --port 8001
```

`--host 0.0.0.0` lets other machines on your network connect; the server has no login, so restrict it with your
firewall. The launcher runs this:

```text
ninfer-spo-serve <file>.ninfer --host 127.0.0.1 --port 8080 --model-id 131k --model-match any
  --max-context 131072 --kv-capacity auto --kv-headroom-mib 1024 --kv-dtype fp8
  --max-concurrency 1 --prefill-chunk 128
  --spec mtp --draft-tokens auto --device-state-slots 0 --host-context-mib 8192
```

`--spec mtp --draft-tokens auto` appears only when the file has the MTP helper, and `--vision` only when it has the
vision tower.

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
| KV storage | `--kv-dtype fp8` | Stores the global layers' memory in FP8, half the size of BF16; the setting all the long-context runs used |
| KV size | `--kv-capacity auto --kv-headroom-mib 1024` | Uses what fits and leaves 1 GiB free |
| Concurrency | `--max-concurrency 1` | One request at a time, so all spare memory goes to context |
| Prefill chunk | `--prefill-chunk 128` | The size all the measurements used |
| Speculative decoding | `--spec mtp --draft-tokens auto` | About 165 tokens/s against about 70 without it, in our test runs |
| Checkpoints | `--device-state-slots 0 --host-context-mib 8192` | No GPU checkpoint slots, 8 GiB of system memory for them |
| Address and name | `--host`, `--port`, `--model-id`, `--model-match any` | Defaults `127.0.0.1`, `8080`, `131k`, any request model name accepted; change them with the launcher command above |

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
