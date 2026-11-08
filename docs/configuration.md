# Configuration

Serve flags win over the environment for the switches in `src/core/runtime_options.h`. The engine reads `NINFER_*` in `src/core/runtime_options.cpp`. The variables scripts and gates read are listed at the end. Diagnostics have no flag.

## Environment

| Variable | Effect |
|---|---|
| `NINFER_CUDA_SYNC` | CUDA sync mode: `spin`, `blocking`, `yield`, or `auto`. Any other value fails startup. Unset means spin. |
| `NINFER_NUMA_NODE` | NUMA node, `-1` through `63`. Also the default for `--numa-node`. |
| `NINFER_PRIORITY` | `normal` lowers process priority. Anything else, including unset, stays high. Windows only. |
| `NINFER_THREAD_TUNING` | `0` turns thread tuning off. Windows only. |
| `NINFER_PIN_CEILING_GIB` | Cap on pinned host memory in GiB. `0` or unset means no cap. A sign or a value that does not fit is ignored with a warning. |
| `NINFER_DECODE_WAIT` | `spin`, `sleep`, `block`, or `stock`. Empty keeps the platform default: spin on Windows, sleep elsewhere. Any other value is ignored with a warning. |
| `NINFER_WAKE_MARGIN_US` | Wake margin, clamped to 0..5000. Default 250 µs, or 1000 µs on Windows in sleep mode. |
| `NINFER_FLASH_NEXT_PREFILL_GRAPH_MAX_TOKENS` | Largest prefill graph, clamped to 0..512. Default 64. |
| `NINFER_FLASH_NEXT_PREFILL_GRAPH_CACHE` | How many prefill graphs to keep, at least 1. Default 64. |
| `NINFER_FLASH_NEXT_PREFILL_GRAPHS` | `0` turns prefill graphs off. |
| `NINFER_Q4_EXACT` | Present means `q4_g32_fp16` trunk projections never round their codes to E4M3: 128 tokens and more take the exact W4A8 GEMM (one FP32 scale per group), shorter prefills the A16 routes. Lower KL, slower prefill. |
| `NINFER_GEMMA4_ATTN_SPLIT_K` | D512 tensor-core attention context splitting (query widths >=64). Enabled by default; `0` or `off` disables it for A/B measurements. Underfilled query/head grids split the visible context across CTAs and merge FP32 partials/maxima/sums. Short contexts keep one segment. This does not enable FP8 QK or change the KV format. |
| `NINFER_EXPERT_VRAM_GIB` | Flash-Next only. GiB of VRAM for the hottest routed experts. Decode computes a miss on the CPU. Prefill copies the misses a layer needs into a VRAM scratch, then runs the GPU kernels. `0` or unset keeps every expert in VRAM. The cold copy is pinned and counts toward `NINFER_PIN_CEILING_GIB`. |
| `NINFER_EXPERT_PROFILE` | Path of a JSON file whose integers are `[layer, expert]` pairs, hottest first. Used only with `NINFER_EXPERT_VRAM_GIB`. Missing or empty falls back to expert-id order. |
| `NINFER_EXPERT_CPU_THREADS` | Workers for the CPU experts. Default 8. |
| `NINFER_EXPERT_ADAPT` | `0` keeps the profile's hot set fixed. Otherwise, after each round, up to four experts are swapped so the recent routes stay in VRAM. |
| `NINFER_ROUTING_DUMP` | Diagnostic. Appends one line per MoE call, `call router_address tokens id id ...`, when CUDA graphs are off (`--no-cuda-graph`). The router address names the layer; `tools/bench/expert_tier_model.py` reads the file and writes the `NINFER_EXPERT_PROFILE` JSON. |
| `NINFER_DEBUG_PLANNER_TRACE` | Present means trace the planner. |
| `NINFER_DEBUG_LOGITS_DIR` | Directory for logit dumps. |
| `NINFER_DEBUG_STATE_DIR` | Directory for state dumps. Setting it also turns prefill graphs off. |
| `NINFER_DEBUG_STATE_FRONTIER` | Frontier filter for the state dump. |
| `NINFER_DEBUG_PREFILL_GRAPH_AUDIT` | `1` audits prefill graphs. |
| `NINFER_DEBUG_PREFILL_GRAPH_MEMLOG` | `1` logs prefill-graph memory. |
| `NINFER_DEBUG_ROUND_STATS` | How many rounds to print, clamped to 0..10000. |

A numeric variable that is not a whole number is ignored with a warning and keeps its default.

The matching serve flags are `--cuda-sync`, `--priority`, `--no-thread-tuning`, `--pin-ceiling-gib`, `--decode-wait`, `--wake-margin-us`, `--prefill-graph-max-tokens`, `--prefill-graph-cache`, `--no-prefill-graphs`, and `--expert-vram-gib`.

## Scripts, tests, and gates

The engine does not read these. The scripts, tests, and probes that do are named in each row.

| Variable | Read by | Effect |
|---|---|---|
| `NINFER_MODEL` | `serve.cmd`, `serve.sh`, perf probes, `ppl_ab.cmd` | The `.ninfer` artifact. |
| `NINFER_PORT` | `serve.cmd`, `serve.sh`, probes | Server port. 8003 when unset; `run.py perf` and `soak` default it to 8013. |
| `NINFER_BUILD` | `serve.cmd`, `serve_ctl.py` | Build folder, relative to the repo root or absolute. Default `..\build-spo-release`. |
| `NINFER_LM_HEAD_DRAFT` | `serve.cmd` | `0` leaves out `--lm-head-draft`. |
| `NINFER_ALLOW_DEBUG_HEAP` | `serve.cmd` | Start even when the exe has an Image File Execution Options entry. |
| `NINFER_PRESETS`, `NINFER_MODEL_ID` | `serve.sh` | Cost presets file and public model id. |
| `NINFER_BUILD_TEMP` | `build.cmd` | Compiler temp folder. Default `<build>\.tmp`. |
| `NINFER_PERFORMANCE_TRACE` | CMake option (`build.cmd trace`) | Compiles the performance trace in. |
| `NINFER_QWEN38_FLASH_NEXT_WEIGHTS` | Flash-Next real tests | Artifact for the real tests. Unset skips them. |
| `NINFER_GEMMA4_WEIGHTS` | Gemma 4 real tests | bf16 artifact. Unset skips them. |
| `NINFER_GEMMA4_HF` | Gemma 4 tokenizer, template, fixture, and France tests | Hugging Face model folder (`tokenizer.json`, `chat_template.jinja`). Unset skips them. |
| `NINFER_GEMMA4_PARITY` | Gemma 4 parity test, engine test greedy check | Folder of Hugging Face logits and greedy dumps. Each prompt needs `<name>.logits_bf16` and `<name>.greedy_bf16` (the bf16 references). Without them the test uses the looser fp32 floors. Unset skips them. |
| `NINFER_GEMMA4_FIXTURES` | Gemma 4 fixture test | Overrides `tests/models/gemma4/fixtures`. |
| `NINFER_GEMMA4_MTP_WEIGHTS` | Gemma 4 engine MTP equality | Artifact with the `mtp` component. Unset skips that check. |
| `NINFER_GEMMA4_VISION_WEIGHTS` | Gemma 4 vision encode and engine image chat | Artifact with the `vision` component. Unset skips those checks. |
| `NINFER_GEMMA4_VISION_DUMPS` | Gemma 4 vision encode cosine | Folder of `image{0,1,2}.{patches,pos,mask,tokens,json}`. Unset skips the encode test. |
| `NINFER_TEST_ARTIFACT` | qwen3_5 artifact tests | Artifact for those tests. Unset skips them. |
| `NINFER_GATE_OUT` | `run.py`, `serve_ctl.py` | Gate output folder. Default `<workspace>/tmp/ninfer`. |
| `NINFER_TOKENIZER` | `p8_probe.py`, `cross_path.py` | Tokenizer directory. |
| `NINFER_SRC` | `p8_probe.py` | Tree that has `eval/corpora/perplexity-1m`. Default this repo. |
| `NINFER_CORPUS` | `mtp_ab.cmd`, `prof_decode.cmd` | `common_mix.ids` for the benchmark runs. |
| `NINFER_PPL_CORPUS` | `ppl_ab.cmd` | Perplexity manifest. |
| `NINFER_NSYS` | `prof_decode.cmd` | `nsys.exe` path. |
| `NINFER_SOAK_MEDIA` | `soak.py` | Folder of images for the image turns. |

## Server

`ninfer-spo-serve` takes the artifact path first. Defaults are in `src/serve/serve_options.h`.

| Flag | Default | Meaning |
|---|---|---|
| `--host` | `127.0.0.1` | Bind address. |
| `--port` | `8080` | Bind port. `scripts/serve.cmd` passes `NINFER_PORT`, or 8003. |
| `--api-key` | empty | Empty means no authentication. The value is redacted in the startup record. |
| `--model-id` | artifact name | Public model id. |
| `--request-log-jsonl` | empty | Request log path. Empty disables it. |
| `--max-context` | 8192 | Per-request context ceiling. |
| `--kv-capacity` | `--max-context` | KV pool in tokens, or `auto`. It must be at least `--max-context`. |
| `--kv-headroom-mib` | | Device memory `auto` leaves free. Requires `--kv-capacity auto`. |
| `--max-concurrency` | 1 | Resident lanes. The engine caps this at 8. `scripts/serve.cmd` passes 3. |
| `--max-pending-requests` | 16 | Queue length. |
| `--pending-timeout-ms` | 30000 | How long a queued request waits. |
| `--prefill-chunk` | 1024 | Prefill unit size, a multiple of 128. `scripts/serve.cmd` passes 8192. |
| `--capture-path` | | Directory for activation captures. A request records only when its JSON sets `capture.record` to true. The record id comes back as `capture.record_id` on the chat completion, the responses object, the last streamed chat chunk, and the Anthropic message. Capture turns do not reuse another request's prefix. |
| `--capture-sites` | `prompt_last,completion_last` | Which positions to record. Each site is `prompt_last` or `completion_last`. |
| `--steering-pack` | | Steering pack to load at startup. A request may set `steering.strength` in `[0,1]`. |
| `--context-cost-presets` | | JSON cost presets. |
| `--log-stats-interval-ms` | 5000 | `0` disables the throughput log. |
| `--max-request-mib` | 384 | Request body cap. |
| `--media-cache-mib` | 1024 | Encoded media cache. |
| `--media-live-mib` | 2048 | Decoded media residency. |
| `--media-preprocess-threads` | 0 | `0` picks a bounded pool from the host. |
| `--device-state-slots` | `--max-concurrency` | Extra device state slots beyond the live lanes. Each cached checkpoint holds one. `scripts/serve.cmd` passes 16 (see [serving](serving.md#flash-next-checkpoint-slots)). |
| `--host-context-mib` | 8192 plus eight StateImages | Host context budget. |
| `--response-store-max-records` | 1024 | Stored responses. |
| `--response-store-max-mib` | 256 | Stored response bytes. |
| `--device` | 0 | CUDA device. |
| `--kv-dtype` | bf16 | KV storage: `bf16`, `int8`, `fp8`, `nvfp4`, or `k8v4`. Flash-Next accepts `bf16`, `fp8`, and `int8`; `scripts/serve.cmd` passes `int8`. Gemma 4 accepts `bf16`, experimental global-only `fp8`, and opt-in unqualified global-only `int8`; its sliding rings remain BF16. |
| `--gemma4-qk` | bf16 | Gemma 4 global attention QK: `bf16` keeps Phase 1 cache expansion; experimental `fp8` uses FP8 tensor cores for D512 MMA widths ≥64 and requires `--kv-dtype fp8`. Small-width SIMT and sliding D256 remain unchanged. Also accepted by CLI and perplexity. |
| `--spec` | | Speculative backend: `mtp`, `dflash`, or `dflash2`. Flash-Next accepts only `mtp`, which `scripts/serve.cmd` uses. |
| `--draft-tokens` | | Draft length: `auto` or digits. Flash-Next MTP takes `auto` or 1..7. Gemma 4 MTP takes the same; its `auto` is a fixed 6 drafts per round. |
| `--default-max-tokens` | 8192 | Used when the client omits `max_tokens`. |
| `--default-thinking-budget` | | Thinking token budget. |
| `--vision` | off | Enable vision. |
| `--no-cuda-graph` | graphs on | Disable CUDA graphs. |
| `--no-prefix-reuse` | reuse on | Disable prefix reuse. |
| `--lm-head-draft` | off | Draft from the language-model head. |
| `--chat-template` | artifact | Override the chat template. |
| `--no-thinking` | | Disable thinking. |
| `--preserve-thinking` | | Keep reasoning for the next turn. |
| `--cors` | off | Permissive CORS. |
| `--temperature` | | 0..2. |
| `--top-p` | | 0..1. |
| `--top-k` | | 0..64; 0 is the model's default. The loaded model's ceiling applies: 20 for Qwen3.5 and Flash-Next, 64 for Gemma 4. |
| `--min-p` | | 0..1. |
| `--presence-penalty` | | -2..2. |
| `--frequency-penalty` | | -2..2. |
| `--seed` | fresh per request | |
| `--greedy` | off | Temperature 0, exact argmax. |
| `--log-level` | info | |
| `--ple-residency` | resident on Windows, mapped elsewhere | `mapped` or `resident`. |
| `--ple-large-pages` | off | Falls back to locked normal pages without the lock right. |
| `--numa-node` | `NINFER_NUMA_NODE`, else the GPU's node | `N` or `off`. |
| `--host-ram-budget-gib` | 200 | `0` skips the host RAM check. |

`ninfer-spo`, `ninfer-spo-perplexity`, and `ninfer_bench` resolve the NUMA node the same way and tune the process before loading. `ninfer-spo` and `ninfer_bench` also take the four host flags above.

### Experimental Gemma 4 FP8 KV

`--kv-dtype fp8` changes runtime cache storage, not artifact weights. Only
global D512 layers use E4M3 codes with two FP16 scales per K/V vector (one
per 256 elements); sliding D256 rings and their snapshots remain BF16.
With the default `--gemma4-qk bf16`, target and MTP attention expand cache
values by an FP32 code/scale product rounded to BF16, retaining BF16 QK
products (Phase 1). Explicit `--gemma4-qk fp8` selects Phase 2 for global
D512 MMA calls with at least 64 query tokens: quantize Q once per tile with
two FP16-rounded row-half scales, multiply Q/K E4M3 codes on FP8 tensor
cores, independently correct each 256-element partial, then sum the scores.
K codes come directly from the cache. V still expands to BF16, and PV retains
the hi/lo BF16 probability products. Short decode/MTP widths below 64 keep
the existing SIMT arithmetic; sliding D256 is unchanged. No artifact
reconversion is needed. FP8 QK is rejected for non-Gemma models or non-FP8 KV.
BF16 KV and BF16 QK remain defaults. NVFP4/K8V4 remain unsupported for Gemma.

Global D512 tensor-core attention can split a long context across CTAs when
the query/head grid alone cannot fill the GPU. The combine keeps partial
outputs, maxima and sums in FP32; empty/masked partitions contribute zero.
The workspace capacity covers every context length for each query-width
bucket before graph capture, and replay reads the current positions and
page mappings. This changes floating-point reduction order, not QK precision;
bit-identical outputs or identical greedy tokens are not guaranteed.
`NINFER_GEMMA4_ATTN_SPLIT_K=0` restores the unsplit context walk for paired
measurements. FP8/INT8 staging uses asynchronous raw-code copies followed
by the same FP32-product/BF16 expansion and vectorized shared-memory stores.
BF16 QK remains the default; FP8 QK remains a separate explicit quality tradeoff.

The finite-row domain is supported. Any NaN or infinity in a final normalized
BF16 K/V row-half poisons that complete 256-element half with NaN codes and
unit scale; nonfinite inputs do not promise useful attention outputs.

For the 31B geometry (10 global layers, four KV heads), global payload is
41,280 B/token including scales, versus 81,920 B/token for BF16. At 131,072,
that is 5.04 GiB instead of 10 GiB. These figures exclude spare pool pages,
sliding rings, snapshots, workspace, MTP and CUDA graphs. Sliding storage is
unchanged: chunk 128 uses a 900 MiB ring and up to 900 MiB per device snapshot.
Host checkpoint storage is not active-context offload.

FP8 and FP8+QK were compared with BF16 and INT8 on one artifact and corpus (teacher-forced fidelity,
needle retrieval, speed). FP8 halved the global KV payload with a per-position fidelity cost far above the
run-to-run noise floor and no resolvable perplexity change; `--gemma4-qk fp8` recovered most of the long-context
prefill speed at a small extra divergence. The reproducible method, the instruments, the evidence a comparison
must carry and the results are in [KV-format evaluation](kv-evaluation.md).

### Opt-in Gemma 4 INT8 KV (unqualified)

`--kv-dtype int8` is an opt-in, global-layer-only runtime cache format. It is
**not quality-qualified**; BF16 remains the default and FP8 remains the
separate experimental format. Only global D512 layers change: signed INT8
codes with one FP16 scale per 64 elements (eight scales per K/V vector),
symmetric `absmax/127` scale rounded to FP16, round-to-nearest codes clamped to
±127, no rotation. Sliding D256 rings, their snapshots and vision profiles
remain BF16. Target and MTP attention expand cache values by an FP32
code/scale product rounded to BF16 and then use the existing BF16 attention
products (SIMT and tensor-core staging alike); there is no INT8 tensor-core
path. `--gemma4-qk fp8` combined with `--kv-dtype int8` is rejected.

Per global layer a page holds 131,072 B of codes plus 4,096 B of scales per
K/V plane, so the 31B geometry (10 global layers, four KV heads) stores
42,240 B/token including 1,280 B/token of scales, versus 41,280 B/token for
FP8 and 81,920 B/token for BF16 (5.16 GiB at 131,072 tokens). INT8 is a
quality fallback hypothesis, not a capacity lever over FP8. Measured on one
artifact and corpus (teacher-forced, 4096 and 65536/32768 contexts, plus a
120-case needle grid), INT8 was indistinguishable from FP8 in fidelity and
retrieval and slower in long-context decode (MTP at 100k: 18.3 vs 26.5 tok/s), so
it showed no advantage; it remains unqualified (method, definitions, noise floor and
numbers: [KV-format evaluation](kv-evaluation.md)). Any NaN, infinity
or FP16-overflowing scale in a 64-element group poisons that group (NaN scale,
zero codes). Checkpoints, copy-on-write and host restore cover all four
global code/scale planes.

Startup reports requested/global/sliding formats, QK mode/scope and actual payload bytes.
Use resolved capacity, not the requested ceiling, as fit evidence. Explicit
capacities round up to 64-token pages before fit validation. Requests above
the artifact's native position limit fail before weight loading. Auto-sizing
headroom also covers lazy CUDA graph allocations; measure memory after graph
capture. Functional fit does not establish model quality or vision support.

## CLI

`ninfer-spo` takes the artifact path first. Flags that also exist on the server mean the same thing. The CLI-only flags are `--prompt`, `--messages`, `--max-new` (default 128), `--grammar-file`, `--json-object`, `--json-schema-file`, `--regex`, `--choice`, `--raw-output`, `--print-token-ids`, `--thinking-budget`, `--reasoning-effort`, `--stop`, `--reasoning-stop`, and `--stop-token-id`. Its context and KV defaults are 2048, and its prefill chunk default is 1024. The definitions are in `apps/cli/options.h`.
