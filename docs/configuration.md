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
| `NINFER_GEMMA4_ATTN_SPLIT_K` | D512 tensor-core attention context splitting (query widths >=64). Enabled by default; `0` or `off` disables it for A/B measurements. The automatic policy minimizes CTA waves per segment over counts 1..8, preferring fewer on ties, then clamps to the descriptor's context/window bound. FP32 partials/maxima/sums are merged without changing the KV format or enabling FP8 QK. |
| `NINFER_GEMMA4_ATTN_SEGMENTS` | Diagnostic D512 segment override: `1`..`32` pins `grid.z` and bypasses the automatic context/window clamp; `1` means unsplit. Unset, empty, zero, or negative selects automatic policy. Valid integers above 32 clamp to 32; malformed/out-of-int-range values warn and select automatic policy. Ignored when split-K is disabled or query width is below 64. Set before process startup/capture, not during graph replay. |
| `NINFER_GEMMA4_ATTN_RAW_V` | `1` opts into compact FP8 V staging in global D512 MMA attention (widths >=64). Converts code × FP16 scale in FP32 and rounds to BF16 directly in the consumer registers, preserving the existing V representation and FP32 PV arithmetic while removing the expanded V shared tile (8 KiB less dynamic shared memory). Default off; BF16/INT8 KV, short SIMT and sliding D256 are unchanged. Does not enable FP8 QK. Set before graph capture. Resource diagnostics report CUDA occupancy limits, not guaranteed resident CTAs. |
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
| `--prefill-chunk` | 1024 | Positive multiple of 128, or explicit Gemma-only `auto`. An integer bypasses automatic selection. `scripts/serve.cmd` passes 8192. |
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
| `--gemma4-qk` | bf16 | Gemma 4 global attention QK: `bf16` keeps Phase 1 cache expansion; experimental `fp8` uses FP8 tensor cores for D512 MMA widths ≥64 and requires `--kv-dtype fp8`. Small-width SIMT and sliding D256 remain unchanged. Also accepted by CLI and perplexity. `fp8` is faster on long prompts and measurably less faithful than `bf16`; see [Choosing --gemma4-qk](#choosing---gemma4-qk). |
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

### Gemma 4 automatic prefill width

CLI and serving accept explicit `--prefill-chunk auto`; the default remains
1024. After loading weights, Gemma chooses the largest fitting candidate from
2048, 1024, 512, 256 and 128. Selection uses the Program's actual fixed
workspace/ring/device-snapshot reservation at each width, the selected KV
format's per-token bytes, and the existing minimum of CUDA/NVML free-memory
probes. It retains the current headroom contract: 3 GiB when the policy has
no override; the existing automatic-KV reservation (including its 1 GiB
default) or an explicit scoring reservation takes precedence.

Every candidate must fit the full page-rounded requested `--max-context`.
Even with `--kv-capacity auto`, this mode fails rather than reducing context
to obtain a wider chunk. Precision, QK, speculative decoding, checkpoint
slots, host cache and graph settings remain the caller's. Spare checkpoint
pages use the remaining budget under the existing cap. Startup logs the
selected width, after-weights free bytes, fixed bytes, headroom, KV bytes per
token, unchanged context and spare tokens. This is a memory-fit policy,
not a throughput autotuner or model-quality qualification.

An explicit integer retains the previous integer/KV sizing behavior.
Non-Gemma models reject the new `auto` request; their existing integer paths
and defaults are unchanged. Engine callers may request `kAutoPrefillChunk`;
after successful construction `Engine::options().prefill_chunk` contains the
resolved positive integer. CLI literal `0` is not an alias for `auto`.

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
K codes come directly from the cache. V expands to BF16 (in shared memory by
default, in consumer registers with `NINFER_GEMMA4_ATTN_RAW_V=1`), and PV retains
the hi/lo BF16 probability products. Short decode/MTP widths below 64 keep
the existing SIMT arithmetic; sliding D256 is unchanged. No artifact
reconversion is needed. FP8 QK is rejected for non-Gemma models or non-FP8 KV.

The defaults are set per tool, not per model family. The engine default is `--kv-dtype bf16` in `ninfer-spo-serve`
and `ninfer-spo-cli` for every family, Gemma 4 included (`src/serve/serve_options.h`, `apps/cli/options.h`); only
`ninfer-spo-perplexity` defaults to `fp8` (`apps/perplexity/main.cpp`). So a hand-typed Gemma 4 `ninfer-spo-serve` or
`ninfer-spo-cli` command without `--kv-dtype` runs BF16 KV, while a launcher made by `python -m
tools.convert.gemma4_build` writes `--kv-dtype fp8` explicitly. `--gemma4-qk` defaults to `bf16` everywhere.
NVFP4/K8V4 remain unsupported for Gemma.

**Which KV format for which card.** FP8 KV is what makes the long Gemma context fit on smaller cards: at 131,072 tokens
the global KV is 5.04 GiB in FP8 against 10 GiB in BF16, and the sliding rings (0.88 GiB at chunk 128) are BF16 either
way. Measured on one RTX 5090 (32 GiB) and one artifact ([KV-format evaluation](kv-evaluation.md), `--kv-capacity
auto`, chunk 128): the BF16 KV payload was 10.88 GiB without MTP, but with MTP automatic sizing reached only 124,224 of
the 131,072 requested tokens (10.36 GiB), and the BF16 capacity varied between starts with the desktop's idle memory.
FP8 held the full 131,072 in 5.92 GiB with more than 5 GiB free, with or without MTP. The cost of FP8 KV on that
instrument was about 5.4% of top-1 choices differing from BF16 KV (BF16 against its own repeat: 0.003%), no resolvable
perplexity change, and slower prefill (32k with MTP: 1244.6 against 1722.0 tok/s; chunk 128, before the later prefill
work). BF16 KV costs twice the global memory and buys prefill speed and the exact BF16 cache; use it when the card holds
the pool at the context you need. A 24 GiB card is below the measured range: by arithmetic (not measured on such a card)
the g256 weights plus 5.04 GiB of FP8 global KV, the 0.88 GiB ring, workspace and headroom exceed 24 GiB, so such a card
needs a shorter `--max-context` (with FP8 KV). The global layers cost 40.3125 KiB per token in FP8 and 80 KiB in BF16.
Check the startup line `capacity | KV` for what was really reserved.

Global D512 tensor-core attention splits the context across CTAs using a
wave-aware policy: minimize `ceil(query_tiles * query_heads * segments / SMs) / segments`
over segment counts 1..8, choosing fewer segments on ties. On a 170-SM GPU with
32 query heads, widths 128/512/640/1024 choose 5/7/1/7 before the context clamp.
Automatic splitting allows at most one segment per 256 keys in the descriptor's
logical-page bound (also limited by a nonzero window), with at least one segment.
This bound is not the live sequence length. Short live sequences can therefore
launch empty segments when the descriptor covers a much larger context.
`NINFER_GEMMA4_ATTN_SEGMENTS=1..32` pins the count without that clamp;
`NINFER_GEMMA4_ATTN_SPLIT_K=0` restores the unsplit walk even with a pin set.

The combine keeps partial outputs, maxima and sums in FP32; empty/masked
partitions contribute zero. Workspace capacity scans every query width in the
bucket using the context-unclamped segment bound, covering future context
lengths before graph capture. Launch geometry depends on host descriptor
metadata and the startup switches, never on the live device positions.
A captured graph retains its grid while replay reads current positions and
page mappings; it does not shrink or enlarge the grid as the live context grows.

Paired output-half warps compute disjoint 256-element QK contractions and exchange
their FP32 partial scores instead of duplicating the full contraction.
FP8/INT8 staging still uses asynchronous raw-code copies followed by the same
FP32-product/BF16 expansion and vectorized shared-memory stores; PV retains
hi/lo BF16 probabilities. These changes alter floating-point reduction order,
not the selected QK precision; bit-identical outputs or identical greedy tokens
are not guaranteed. BF16 KV and BF16 QK remain defaults, and FP8 QK remains a
separate explicit quality tradeoff.

The subsequent opt-in raw-V experiment passed the same six split-on/off
attention/KV suite invocations at unchanged tolerances, including direct
raw-on/off exact-output checks, page/tail cases and graph replay. On RTX 5090,
CUDA resource APIs report 255 registers/thread, zero local bytes/thread and
one maximum active 256-thread CTA/SM for raw V, even with FP8 QK's 49,152-byte
dynamic shared tile; saving shared memory does not establish two-CTA occupancy.
Synthetic paired medians at width 128/history 131072 were 16.0286 ms with
expanded V and 15.4036 ms with raw V (BF16 QK), or 11.8188/11.3282 ms
(FP8 QK). The full 128/512/1024/2048 by 8192/32768/131072 matrix includes
regressions, so raw V remains opt-in. Sampled FP64 checks cover two query/head
rows and four output columns across all keys, not full-output or model fidelity.
Receipts: `d512dev/next-B-microbench-resources-oracle.log`,
`next-B-long-errors.json`, `next-B-summary.json`, and
`next-validation-receipts.json` under the workstation evidence root above.

On 2026-11-08, RTX 5090/CUDA 13.5 development snapshots of wave splitting
(`dbd4dc8`, A) and QK deduplication (`b160d71`, B) each passed the BF16-attention,
FP8-KV and INT8-KV suites with split-K both enabled and disabled, without
changing tolerances. The suites include eager/graph replay and empty partitions.
Synthetic eager CUDA-event microbenchmarks covered widths 128/512/640/1024
at histories 8192/32768/131072, three cache/QK modes, and pinned segment sweeps
1..13. With FP8 KV and BF16 QK at width 128/history 131072, median attention
time was 18.1222 ms for A and 15.5802 ms for B; the earlier `7080990` receipt
was 30.1703 ms (a historical, not interleaved comparison). B's FP8-KV/BF16-QK
medians improved over A in all 12 matrix cells; FP8-QK timings were mixed,
including small regressions, so this is not an across-mode performance guarantee.
Receipts are in the workstation's
`D:/AI/CONVERSION/ReadyArt__Wicked-Stuff-31B-Final/convert/pp-bench-20261108/d512dev/RESULTS.md`
and `validation-summary-recovery.json`. Those kernel measurements are distinct
from the subsequent fixed-path serving and matched-quality checks below.

The subsequent canonical rebuild of
`D:/AI/loaders/build-spo-splitk-cuda13.5` succeeded; all six attention/KV
split-on/off suite invocations on the new fixed-path executables exited zero.
On the same artifact with speculation disabled, controlled serving medians were:

| Prefill chunk | Prompt tokens | Old PP tok/s | New PP tok/s | Old TG tok/s | New TG tok/s |
|---:|---:|---:|---:|---:|---:|
| 128 | 32000 | 1699.284 | 2221.480 | 48.454 | 48.710 |
| 128 | 100000 | 829.291 | 1279.966 | 31.822 | 31.859 |
| 2048 | 32000 | 2897.654 | 3238.863 | 48.702 | 48.821 |
| 2048 | 100000 | 1324.312 | 1682.564 | 31.893 | 31.645 |

These are served prompt-processing (PP) and token-generation (TG) rates, not
the attention-event timings above. Existing G launchers and precision defaults
were not changed. The comparison establishes improvement over the measured
ninfer baseline for these cells, not superiority to llama.cpp.

A matched context-512/chunk-128 quality comparison (47 windows, 13,728 positions)
gave baseline/candidate mean KL 0.127134968/0.126281209. The candidate/baseline
KL ratio's 95% CI was [0.96614161, 1.01888304]; candidate-minus-baseline NLL was
-0.00225588, with 95% CI [-0.00992962, 0.00530373]. Neither resolves degradation
in this matched check; neither establishes long-context or chunk-2048 quality.
Receipts in the same `pp-bench-20261108` evidence root are
`recovery-serving-summary.json`, `quality-recovery-paired.json`, and
`fixed-final-tests.json`. Validation included the six kernel suites and the
runtime-options CPU regression check, not a new full 544-test run.

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
must carry and the results are in [KV-format evaluation](kv-evaluation.md). When to enable `--gemma4-qk fp8`, with the
speed and fidelity numbers for that flag alone, is in [Choosing --gemma4-qk](#choosing---gemma4-qk).

### Choosing --gemma4-qk

The engine default is `--gemma4-qk bf16` and stays that way. FP8 QK is a precision-for-speed trade with a measured
fidelity cost above the run-to-run noise floor, and its fidelity has not been measured on generated text, at deep
context, or with MTP. The flag is chosen per launch, so pick it per workload.

**What it changes.** Only the Q·K^T matmul of the global D512 layers, in calls with at least 64 query tokens (prefill,
wide verify): FP8-quantized Q times the cached FP8 K codes on FP8 tensor cores, with scales. It does not change KV
storage (`--kv-dtype fp8`), the Q4 weights, W4A8 prefill activations, V, the softmax probabilities P, the sliding D256
layers, or decode and MTP calls below 64 tokens. It needs Gemma with global FP8 KV and is rejected otherwise (and with
`--kv-dtype int8`). Decode speed does not change: no-speculation TG stayed at 47.5-48.8 tok/s (32k) and 31.4-31.5
tok/s (100k) across the BF16-QK and FP8-QK arms.

**Speed.** Prompt-processing (PP) tok/s on the g256 GPTQ artifact, FP8 KV, prefill chunk 2048, no speculation, one
session, three timed reps. Compare only within one session; absolute PP varies about 3% between sessions.

| Prompt | BF16 QK | FP8 QK | Change |
|---|---:|---:|---:|
| ~32k fresh | 3,215-3,239 | 3,640 | +13% |
| ~100k fresh | 1,664-1,683 | 2,036 | +22% |
| ~100k cached tail | 709 | 819-827 | +16% |

Evidence: `pp-bench-20261108/wide-chunk-new/` and `rollout-arms-summary.json` under the workstation evidence root
named above. Speed was measured at long prompts only (~32k and ~100k); short prompts were not measured.

**Fidelity.** Teacher-forced, so no text is sampled. Reference: the same artifact and `--kv-dtype fp8`, with
`--gemma4-qk bf16`; the only difference between arms is the QK precision. The noise floor is a same-flags repeat of the
reference. 13,728 scored positions over three domains (English long-form, English reference text, code); KL is the
truncated top-64 diagnostic, not full-vocabulary KL.

| Metric | Noise, ctx 512 | FP8 QK, ctx 512 | Noise, ctx 4096 | FP8 QK, ctx 4096 |
|---|---:|---:|---:|---:|
| dPPL vs reference | +0.03% | +0.77% | -0.01% | +0.51% |
| Top-1 agreement | 99.0% | 95.7% | 98.3% | 95.8% |
| Truncated KL, mean | 3.4e-3 | 2.25e-2 | 7.2e-3 | 2.04e-2 |
| Top-1 flips | 139 | 591 | 228 | 574 |
| Flips at reference p1 >= 0.9 (of about 6.6k) | 1 | 12 | 6 | 12 |

Reading it:

- About half of the excess flips are near-ties (reference top-1/top-2 margin under 0.25 nats) and about three quarters
  are under 0.5 nats. The reference's token is FP8 QK's second choice in 83% of flips.
- A small share is not a near-tie: roughly 0.1-0.2% of confident positions (reference p1 >= 0.9) flip, against 1 and 6
  for noise.
- Code carries the most: 10 of the 12 confident flips at ctx 512 and the largest ratio over noise (145 flips against
  14, about 10x).
- The dPPL shift is small: the paired window t is +2.11 (ctx 512) and +1.29 (ctx 4096), so the per-position disagreement
  is the clearer signal.
- Six needle-retrieval cases at 32k and 100k (5/50/95% depth) were 6/6, and the answers were byte-identical to BF16 QK.
  That is a smoke test, not a task-quality suite.

Evidence: `pp-bench-20261108/fp8qk-gate/ctx512/`, `ctx4096/` (`analysis/tables.md`) and
`margin-analysis/margin-analysis.md` under the same root.

**Not shown.** Generated text (a flipped top-1 does not imply a different sampled continuation, nor that errors
compound); fidelity at 32k-100k context (the corpus streams are about 1k tokens, so the ctx-4096 run is not deeper than
that); other domains and languages; temperature and sampling effects; noise variance beyond one repeat run; the
interaction with MTP.

**Guidance.** This is a recommendation from the numbers above, not a quality qualification.

| Workload | `--gemma4-qk` |
|---|---|
| Roleplay, creative prose, long chat history | `fp8` is reasonable: long prompts process 13-22% faster and the extra disagreement is mostly between near-equal tokens |
| Coding, tool calling, agentic loops, JSON or other structured output | Keep `bf16`: the code domain showed the largest disagreement over noise and a confident token can flip |
| Anything not measured here, or where an exact token matters | Keep `bf16` |

Enable it by adding `--gemma4-qk fp8` next to `--kv-dtype fp8` on the server, CLI or perplexity command; startup
reports the requested and resolved mode and scope. For a launcher, `python -m tools.convert.gemma4_build launcher
--artifact F.ninfer --out serve.cmd --gemma4-qk fp8` appends it to the `--kv-dtype fp8` line and adds a `rem` note
(the default emits no flag; with `--kv-dtype bf16` generation is refused, as the server rejects FP8 QK without FP8 KV).

**Measure it on your own model and text.** The [KV evaluation kit](kv-evaluation.md#running-it) isolates the QK change
when `fp8` (`--kv-dtype fp8 --gemma4-qk bf16`) is the reference and `fp8-qk` is the compared arm; both are in
[`arms.example.json`](../tools/bench/kv_eval/arms.example.json). Copy it to `<out-dir>/arms.json`, make `fp8` the
reference, and add a same-flags repeat as the noise arm:

```json
{"arms": [
  {"name": "fp8",        "role": "reference", "flags": ["--kv-dtype", "fp8", "--gemma4-qk", "bf16"]},
  {"name": "fp8-repeat", "role": "noise",     "flags": ["--kv-dtype", "fp8", "--gemma4-qk", "bf16"]},
  {"name": "fp8-qk",     "role": "quant",     "flags": ["--kv-dtype", "fp8", "--gemma4-qk", "fp8"]}
]}
```

Run `run_ppl` once per arm (the perplexity tool writes the `--dump-topk` file), then `analyze`, as in
[Running it](kv-evaluation.md#running-it); `--context 512 --stride 256` and `--context 4096 --stride 2048` gave the
tables above. Judge the result against the noise arm, not against zero.

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
