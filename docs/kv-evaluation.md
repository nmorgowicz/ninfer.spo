# KV-format evaluation

How to compare a KV-cache or quantization setting (BF16, FP8, FP8 with FP8 QK, INT8, ...) against a reference
with the same artifact, and how to report the result so that every number names the instrument that produced it.
The kit lives in [`tools/bench/kv_eval/`](../tools/bench/kv_eval/README.md); this page defines the metrics, the
protocols and the evidence a published comparison must carry, and records the Gemma 4 31B results that established
the method. It was written for Gemma 4 but nothing in the kit is tied to one model or one machine; the last section
lists what a new family must supply.

A comparison uses **three separate instruments**. Never put their numbers in one table or let one stand in for another:

| instrument | tool | what a number measures | what it does not measure |
| --- | --- | --- | --- |
| Teacher-forced fidelity | `ninfer-spo-perplexity --dump-topk`, `tools/bench/kv_eval/analyze.py` | How far each arm's next-token distribution moves from the reference arm's on one artifact and one corpus | Task quality, generation behaviour, speed |
| Needle retrieval | `tools/bench/kv_eval/needle.py` over HTTP | Whether one planted six-digit fact is retrieved by greedy decoding at a given length and depth | Fidelity, PPL, anything beyond single-fact exact match |
| Speed | `tools/bench/kv_eval/speed_client.py` over HTTP | Wall-clock prefill/decode throughput of one server build on one GPU | Quality of any kind |

## Definitions

Everything below is teacher-forced: the model is fed the corpus text and scored on the next token; nothing is sampled.
The reference arm is the model's BF16-KV run. All logarithms are natural. Source of truth for the arithmetic is
`tools/bench/kv_eval/metrics.py`; `tests/bench/kv_eval/test_analysis_core.py` checks it against hand-computed values.

### Windows, NLL and PPL

A report from `ninfer-spo-perplexity` ([metric](perplexity.md#metric)) holds, for every window `w` of every stream, its
scored target count `n_w` and `NLL_w`, the sum of `-log p(x_i | window context)` over the window's targets. The
full-vocabulary softmax is used. For a scope `S` (overall, one domain, or one stream):

```text
PPL(S)  = exp( sum_{w in S} NLL_w / sum_{w in S} n_w )
dPPL%   = 100 * ( PPL_arm / PPL_ref - 1 )
```

`PPL` is token-weighted over windows, equal to the report's `overall` and per-domain values. For small differences
`dPPL% ~ 100 * dNLL`, where `dNLL` is the change of mean NLL in nats per token.

### Top-k dump, top-1 agreement, bit identity

`--dump-topk` writes for every scored position the 64 most likely ids and their full-softmax log-probabilities
([format](perplexity.md#top-k-dump-for-truncated-distribution-comparison)). With `id_arm[i,0]` the rank-0 id of position `i`:

```text
top1_agree = mean_i [ id_arm[i,0] == id_ref[i,0] ]        over all scored positions
top1_disagree = 1 - top1_agree
bit_identical = mean_i [ all 64 ids equal and all 64 float32 logprobs equal ]
```

Ids are stored most likely first, so rank 0 is the first entry of a row.

### Target-token dlogprob

The dump does not store the target. The analysis re-derives targets by tokenizing each corpus stream with the
artifact's own `tokenizer.json` and slicing each window's target range. Three checks make that trustworthy and any
failure stops the run: the token count of every stream equals the report's `input_tokens`; the target lies inside the
reference top-64 on more than half of the positions of every domain (misaligned targets almost never do); and the
one-sided window bound holds. For a position whose target is not in the reference top-64 its true logprob is at most
the reference's 64th logprob `lp64_ref[i]`, so

```text
sum_{i in w, target in top64} -lp_ref(target_i)  +  sum_{i in w, target not in top64} -lp64_ref[i]   <=   NLL_w   (tolerance 1e-4 relative)
```

must hold for every window or the dump does not belong to the report. The statistic is

```text
dlogprob_i = lp_arm(target_i) - lp_ref(target_i)          only where target_i is in BOTH arms' top-64
```

and the tables report `mean |dlogprob|` (also p99 and max in `analysis.json`). This subset excludes the positions the
model finds hardest (about 10 % on 31B), so it is biased low; a maximum comes from one position and is not robust.

### Truncated top-K KL (a diagnostic, not full-vocabulary KL)

For position `i` let `S_i` be the 64 ids of the reference's dump and `lp_ref(s)` their logprobs. For an arm define on
the same set

```text
lq(s) = lp_arm(s)       if s is in the arm's dumped top-64 at position i
lq(s) = lp_arm[i, 63]   otherwise  (the arm's 64th = smallest dumped logprob)

P(s) = exp(lp_ref(s)) / sum_{t in S_i} exp(lp_ref(t))
Q(s) = exp(lq(s))     / sum_{t in S_i} exp(lq(t))

KL_i = sum_{s in S_i} P(s) * ( ln P(s) - ln Q(s) )
```

Reported: mean over positions, the 99th percentile, the maximum, and the fraction of positions where at least one
reference id is missing from the arm's dump (`ref_top64_ids_missing_from_arm_frac`). How the script treats missing mass:

* Probability mass outside `S_i`, in the reference **and** in the arm, is ignored; both distributions are renormalised
  over `S_i` (the sums above). The result is the KL between two conditional distributions on the reference's top-64.
* A reference id absent from the arm's dump gets the arm's 64th probability, which is an upper bound on its true
  probability (the arm ranked it below its own 64th). For that id the term is therefore optimistic. No other
  correction is applied.
* The quantity depends on the reference's top-64 set, is not symmetric, and **is not full-vocabulary KL**, nor a bound
  on it: the dump cannot give one because the tail is not stored. Every table labels it "truncated". Use it to rank arms
  against the same reference and to compare with the noise floor, never as a divergence in absolute terms.

### Paired window-level statistics

A **window** is one scored window of one stream (identified by stream id and target range). Targets of different windows
are disjoint but their contexts overlap. The analysis asserts that every arm has identical windows, so a **pair** is
"the same window in two arms". Three paired quantities are formed; each yields one number per window:

| quantity | per-window value `d_w` | unit |
| --- | --- | --- |
| PPL change vs the reference | `(NLL_arm,w - NLL_ref,w) / n_w` | nats/token |
| Divergence contrast between arms `x`, `y` | `mean_{i in w} KL_x,i  -  mean_{i in w} KL_y,i` (both vs the reference) | nats/position |
| Disagreement contrast between arms `x`, `y` | `mean_{i in w} [top1 differs]_x  -  mean_{i in w} [top1 differs]_y` | fraction of positions |

With `W` windows, the sample standard deviation `s_d` (ddof = 1):

```text
mean = (1/W) sum_w d_w        SE = s_d / sqrt(W)        t = mean / SE
```

`t` is undefined (reported `n/a`) when `W < 2` or every `d_w` is identical. No p-value is derived from `t`, windows are
treated as independent although their contexts overlap, and nothing is corrected for multiple comparisons; read
`|t| >= 4` as clearly non-zero and `|t| ~ 1` as nothing. A negative contrast means the first arm is closer to the
reference. The window count fixes the resolution: the short profile has 526 windows, the long 25 (see below).

### McNemar and Wilson for the needle grid

Every arm answers the same prompts, so arms are compared as paired binary outcomes. For arms A and B let `b` be the
prompts A answered correctly and B incorrectly, `c` the reverse, `n = b + c`. Two-sided exact McNemar:

```text
p = min( 1,  2 * sum_{k=0}^{min(b,c)} C(n,k) / 2^n )         p = 1.0 by convention when n = 0
```

**Degenerate case:** when `b = c = 0` (both arms right everywhere, as at a ceiling) the test has no discordant pair, `p`
is 1 and the result says nothing about equivalence; it only says no retrieval was lost in this task. Overall accuracy is
reported with a Wilson 95 % interval (`z = 1.96`); 120/120 gives 96.9 %-100 %, an upper bound near 3 % failure rate.

### Position bins

`position = stream token index - window input_begin`; the chat prefix ids are not counted. Window 0 of a stream scores
positions `[0, context)`, every later window only `[stride, context)`, so bins below `stride` come from window 0 alone
and bins above mix window 0 with later windows. Default edges are 0, 4096, 16384, 32768, 49152, 65536 (`--bins`).
Per bin the analysis reports top-1 agreement, truncated KL (mean and SE over the per-window bin means) and
`mean |dlogprob|` on the both-in-top-64 subset. **Per-bin dPPL does not exist**: the dump holds no per-position NLL.
Bins contain different text, so a trend across bins mixes distance with content and cannot be called error accumulation.

### Noise floor

The engine is not bit-deterministic, so each campaign includes a **same-flags repeat** of the reference arm as a separate
process, analysed exactly like a quantized arm against the first run. Its dPPL, top-1 agreement, KL and `t` are the floor
below which a difference cannot be attributed to a format. Compare the floor with each effect, not with zero.

## Protocols

### Corpus and its identity

The corpus is a manifest (`corpus_id`, `streams[]` with `id`, `domain`, `path`; `modes.quick` and `modes.full`) and the
stream files it names. The manifest hash alone does **not** cover the data files, so a published comparison records both:
`analyze.py` writes the manifest sha256, one sha256 per stream file and a combined stream hash into
`analysis.json.provenance.corpus`, and `run_ppl.py` records the manifest sha256 in each arm's receipt. Use the **full**
corpus (`--quick` is one stream per domain, only for smoke runs); a noise-floor arm run with `--quick` is compared on the
streams both arms contain, but quantized arms must be full for the arm-vs-arm contrasts. The 16-stream `ninfer-ppl-1m-v1`
corpus (English reference, English long-form, Chinese reference, NInfer C++/CUDA code; four streams each) has 1,092,299 input
and 1,092,283 scored tokens on the 31B artifact's tokenizer; counts are runtime results of the artifact's tokenizer.

### Profiles

| profile | `--context` / `--stride` | windows (31B, full corpus) | resolution of the paired ΔNLL SE | use |
| --- | --- | --- | --- | --- |
| short | 4096 / 2048 | 526 | about 7-8e-4 nats/token (about 0.07-0.08 % PPL) | PPL-level differences, many windows |
| long | 65536 / 32768 | 25 | about 2.5-5e-3 nats/token (about 0.25-0.5 % PPL) | per-position behaviour at long range, position bins |

The profiles share neither windows nor context, so **their PPL and fidelity numbers are not comparable**; only the shape
of arm contrasts is. The long profile has too few windows to resolve small aggregate PPL differences and is the better
probe of per-position behaviour. Run both when a new format is introduced.

### Chat-template prefix

Instruction-tuned models score raw prose far from their training distribution (Gemma 4: near PPL 1,000 without a
prefix). `--prefix-ids` prepends the ids of a rendered chat preamble that opens a model turn; they are context only, the
targets and token counts do not change, the report records them, and absolute PPL is conditional on them (identical for
all arms). Derive them from the artifact itself:

```bash
python -m tools.bench.kv_eval.derive_prefix_ids --artifact <artifact> --out <out-dir>/prefix.json
```

The script reads the artifact's embedded `resource/text/{tokenizer.json,tokenizer_config.json,chat_template.jinja}`,
renders one user message ("Continue the document.") through `apply_chat_template(add_generation_prompt=True)` with
the model's default settings (for Gemma 4 thinking is off), tokenizes the text with `tokenizer.json` without adding special
tokens, requires the `tokenizers` and `transformers` tokenizations to agree, and records the rendered text, ids and the sha256
of the three resource files. For the 31B artifact it yields
`2,105,2364,107,34230,506,2354,236761,106,107,105,4368,107,100,45518,107,101` (17 ids), identical to the ids first
derived from the source files; the embedded resources hash to the same sha256 values as those source files.

### Needle design

15 cells of 8 needles = 120 prompts per arm: nominal lengths 8k, 32k and 100k tokens, depths 5/25/50/75/95 % of the
text. The haystack is a natural-text window of the corpus's English pool (`corpus_pool.py`: the English streams, joined by
blank lines). One fact, "The access code for vault `<Adjective Noun>` is `<6 digits>`.", is inserted on its own line at the
first line break at or after the target depth, and "What is the access code for vault `<Adjective Noun>`? Answer with only the
6-digit code." is appended. Vault names and codes are distinct across all prompts; window offsets are seeded; prompts are
byte-identical across arms (`prompt_sha256` per row, checked by `summarize`). Request: `temperature 0`, `top_k 20`, `seed 1234`,
`max_tokens 32`, no streaming. Score: the reply contains exactly one distinct six-digit string and it equals the planted code.
The result is a 3x5 grid per arm, Wilson intervals and paired exact McNemar per pair and length. Use the server's `prompt_tokens`
as the true length (re-tokenizing inside the chat template differs from the nominal count by a few tokens). A single-fact task
saturates for strong models; if every arm scores 100 % the grid cannot separate formats.

### Speed method

The client posts to `/v1/chat/completions` against one resident server per (arm, speculative mode), nothing else running on
the GPU. Per nominal length (1k, 8k, 32k, 100k; no-MTP controls at 1k and 32k): one **warmup** request (recorded, excluded)
plus five timed repetitions. Every request carries a unique prompt so prefix reuse cannot cover more than the chat-template
head: a different corpus window per repetition (61,000-token stride) and a unique `Reference id <hash>` on the first line;
the summary flags any timed request with `cache_n` above 8. Sampling is pinned: `temperature 0`, `top_k 20`, `seed 1234`,
`max_tokens 256` (a request that does not produce 256 tokens is flagged). Per request the server's own `timings` give
prefill tok/s (`prompt_n / prompt_ms`), decode tok/s (`predicted_per_second`) and MTP acceptance (`draft_n_accepted / draft_n`,
accepted over drafted, not mean accepted length); cells report the **median (min-max)** of the five repetitions.
MTP decode tok/s carries content variance (arms generate different continuations after KV-induced divergence); the no-MTP
decode rows are the clean per-format comparison. `--spec none` is not accepted by the 31B serve build used: the no-MTP server omits
`--spec` and `--draft-tokens`. The server's `--request-log-jsonl` stream supplies resolved KV capacity, payload and memory.

## Running it

Placeholders: `<artifact>` the `.ninfer` file, `<corpus-manifest>` the corpus `manifest.json`, `<out-dir>` a fresh directory,
`<arm flags>` the arm's flags from `<out-dir>/arms.json` (the same flags go to the perplexity tool and to the server),
`<ppl-exe>`/`<serve-exe>` the binaries under test. All commands run from the repository root; every tool accepts `--help`.
Copy [`arms.example.json`](../tools/bench/kv_eval/arms.example.json) to `<out-dir>/arms.json` and adapt it.

```bash
# 1. prefix ids from the artifact's own template and tokenizer
python -m tools.bench.kv_eval.derive_prefix_ids --artifact <artifact> --out <out-dir>/prefix.json

# 2. one perplexity arm per call, sequentially, reference first; then the same-flags repeat of the reference
python -m tools.bench.kv_eval.run_ppl --exe <ppl-exe> --artifact <artifact> --corpus <corpus-manifest> \
    --arms-json <out-dir>/arms.json --arm bf16 --out-dir <out-dir>/ppl --profile long \
    --prefix-ids-file <out-dir>/prefix.json --dry-run          # prints the command; drop --dry-run to run
# repeat for bf16-repeat, fp8, fp8-qk, int8 (and with --profile short for the short protocol, in its own out-dir)

# 3. analysis (no GPU, no model): reports + dumps -> analysis.json and tables.md
python -m tools.bench.kv_eval.analyze --dir <out-dir>/ppl --arms-json <out-dir>/arms.json --artifact <artifact>

# 4. needle: start <serve-exe> <artifact> <arm flags> --model-id kv-eval ... then, per arm
python -m tools.bench.kv_eval.needle run --label <arm> --out <out-dir>/needle/<arm>.jsonl --manifest <corpus-manifest> --artifact <artifact> --dry-run
python -m tools.bench.kv_eval.needle run --label <arm> --out <out-dir>/needle/<arm>.jsonl --manifest <corpus-manifest> --artifact <artifact>
python -m tools.bench.kv_eval.needle summarize --dir <out-dir>/needle --arms bf16,fp8,int8

# 5. speed: one server per (arm, mode); stop it before the next
python -m tools.bench.kv_eval.speed_client --label <arm>-<mode> --out <out-dir>/speed/<arm>-<mode>.cells.jsonl \
    --manifest <corpus-manifest> --artifact <artifact>
python -m tools.bench.kv_eval.speed_summary --dir <out-dir>/speed --arms bf16,fp8,fp8-qk,int8
```

Server command used for the published speed and needle runs (flags other than the KV flags were identical for all arms):

```text
<serve-exe> <artifact> --host 127.0.0.1 --port 8001 --model-id kv-eval --max-context 131072 --kv-capacity auto
    --kv-headroom-mib 1024 <arm flags> --max-concurrency 1 --prefill-chunk 128 --device-state-slots 0
    --host-context-mib 8192 [--spec mtp --draft-tokens auto] --no-thinking --request-log-jsonl <out-dir>/speed/<arm>-<mode>.serve.jsonl
```

`--dry-run` of `run_ppl.py` prints the exact argv and output paths; of `needle.py` and `speed_client.py` it builds the prompts and prints
their hashes without a server, which also checks the corpus and tokenizer wiring. Each perplexity arm needs a fresh output directory
(the tool refuses an existing one). `gpu_state.py <label> --log <file>` records an idle-state snapshot; `run_ppl.py` calls it
before and after every arm and refuses to start while another ninfer process is running.

## Evidence checklist

A comparison is publishable when its receipts carry, per instrument:

* **Binary identity:** sha256 of each executable used (perplexity for fidelity, serve for speed and needle) and the source
  commit it was built from, taken from the build, not from memory.
* **Artifact identity:** sha256 of the artifact file (`run_ppl.py` computes it once per out-dir, or accepts `--artifact-sha256`) and the
  resource hashes of the tokenizer and template used for the prefix ids and for target re-derivation.
* **Corpus identity:** manifest sha256 and the per-stream hashes (`analysis.json.provenance.corpus`).
* **Exact argv** of every arm (`logs/ppl-<arm>.argv.json`, `receipt.json`; the server's `server_start` event for HTTP runs).
* **Report and dump hashes** per arm (`receipt.json`, `analysis.json.provenance.inputs`), and the analysis script's own
  version (the commit of this kit).
* **Token-count equality assertions**, enforced by the tools and failing the analysis otherwise: dump positions equal the report's
  scored tokens; every arm has the same schema, corpus id, context, stride, prefix ids, prefill chunk, score tile, device, artifact name,
  path, prefill signature and weight formats, scored-token total, per-stream counts and identical window input/target ranges;
  tokenizer input counts equal the report's for every stream; target coverage and the one-sided NLL bound hold. KV and QK fields
  must differ only as intended (`analysis.json.comparability` lists them per arm, including requested/resolved QK mode).
* **GPU idle state** before and after every model process: `nvidia-smi` memory/temperature/clock/power/utilization/driver version, no `ninfer`
  process, no System Error window on Windows (`logs/gpu-states.jsonl`); servers stopped between arms.
* **Noise floor:** a same-flags repeat of the reference, analysed with the same tool.
* **Sampling parameters and prompts** of HTTP runs: recorded per row (`request_params`, `prompt_sha256`), equal across arms.
* **Exit codes, wall time, UTC start/end** per process (`logs/ppl-<arm>.times.txt`).

## Published results: Gemma 4 31B, 2026-11-04

One artifact (`diode-s15-t005`: Gemma 4 31B-it, Q4 group-32 weights, text only with MTP; sha256
`f29e1353cfb1d0c0246e5208b32c0288a8add8e91f7f0f09d8d9c3853fabef4a`), one corpus (`ninfer-ppl-1m-v1`, manifest sha256
`81ce9a3b9ccdcbd80f403a7b4dfab8f2391a7269cf947111a125eb0dffe6300f`), one RTX 5090 (32,607 MiB), one run per quantized arm.
Arms: **BF16** (`--kv-dtype bf16`, the reference), **FP8** (`--kv-dtype fp8`, `--gemma4-qk bf16`), **FP8+QK**
(`--kv-dtype fp8 --gemma4-qk fp8`), **INT8** (`--kv-dtype int8`, unqualified). Sliding-window rings stay BF16 in every arm; only
the global layers' KV is quantized.

Instruments (build commits and hashes as supplied with the task; the build directories were not re-verified):

| instrument | binary | sha256 | fork commit |
| --- | --- | --- | --- |
| Fidelity (both profiles) | `ninfer-spo-perplexity.exe` | `09f653b927b1da0d3cda31944d0223584ebfa51c664d2613ce5c25f1c2404df2` | `1edba11` |
| Speed and needle | `ninfer-spo-serve.exe` | `6d1117d786b09cb9448f340c25800e1468c7c30af81516bac98ebd6b69c562ba` | `3df9567` |

### Fidelity, short profile (4096 / 2048, 526 windows, 1,092,283 scored tokens)

Instrument: teacher-forced fidelity, perplexity binary above, reference `bf16` run 1, chat-template prefix of 17 ids.
KL is the truncated top-64 diagnostic. `mean |dlogprob|` is on the both-in-top-64 subset (about 90.4 % of positions).

| arm | PPL | dPPL vs BF16 | paired window dNLL/token ± SE (t) | top-1 agree | mean \|dlogprob\| | truncated KL mean / p99 | bit-identical top-64 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| BF16 | 10.32755 | - | - | - | - | - | - |
| BF16 repeat (**noise floor**) | 10.32741 | -0.0014 % | -3.38e-05 ± 2.66e-05 (-1.27) | 99.973 % | 5.3e-4 | 7.4e-5 / 0 | 99.29 % |
| FP8 | 10.32212 | -0.0526 % | -4.54e-04 ± 7.58e-04 (-0.60) | 94.671 % | 0.1206 | 2.76e-2 / 0.364 | 0 % |
| FP8+QK | 10.32211 | -0.0527 % | -2.38e-04 ± 7.19e-04 (-0.33) | 94.571 % | 0.1233 | 2.89e-2 / 0.381 | 0 % |
| INT8 | 10.32091 | -0.0643 % | -7.49e-04 ± 7.69e-04 (-0.97) | 94.673 % | 0.1197 | 2.73e-2 / 0.363 | 0 % |

Pair contrasts (short): FP8+QK vs FP8 isolates query quantization: ΔPPL -0.0002 %, truncated KL +1.30e-3 ± 2.2e-4 (t = 5.9),
top-1 disagreement +9.9e-4 ± 2.5e-4 (t = 4.0). INT8 vs FP8: ΔPPL -0.0118 %, KL difference -2.6e-4 ± 2.2e-4 (t = -1.2), top-1 disagreement
t = 0.2. INT8 vs FP8+QK: KL difference -1.56e-3 ± 2.1e-4 (t = -7.4).

Reading: every quantized arm is far from BF16 per position (about 5.3 % top-1 flips against 0.03 % at the noise floor; KL about
400 times the floor) yet the aggregate PPL moves by -0.05 to -0.06 %, which is inside the paired-window uncertainty (SE about
7-8e-4 nats/token, |t| < 1). The sign is not evidence of improvement. INT8 is indistinguishable from FP8 on every metric; the quantized
arms disagree with each other (top-1 94.56-94.66 %) about as much as with BF16, so the perturbations are not shared between formats.

### Fidelity, long profile (65536 / 32768, 25 windows, 1,092,283 scored tokens)

Instrument: same binary and corpus, `--context 65536 --stride 32768`; all runs are full-corpus. Not comparable with the short table.

| arm | PPL | dPPL vs BF16 | paired window dNLL/token ± SE (t, 25 windows) | top-1 agree | truncated KL mean / p99 | mean \|dlogprob\| | bit-identical top-64 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| BF16 | 10.58914 | - | - | - | - | - | - |
| BF16 repeat (**noise floor**) | 10.58914 | +0.00002 % | +1.9e-05 ± 1.2e-05 (+1.58) | 99.997 % | 6.7e-06 / 0 | 0.0000 | 99.95 % |
| FP8 | 10.58573 | -0.032 % | -1.71e-03 ± 2.52e-03 (-0.68) | 94.570 % | 2.935e-02 / 0.373 | 0.1176 | 0 % |
| FP8+QK | 10.57393 | -0.144 % | -1.89e-03 ± 2.72e-03 (-0.69) | 94.466 % | 3.067e-02 / 0.394 | 0.1202 | 0 % |
| INT8 | 10.59365 | +0.043 % | +8.64e-04 ± 3.37e-03 (+0.26) | 94.639 % | 2.836e-02 / 0.359 | 0.1158 | 0 % |

Pair contrasts of divergence from BF16 (window-paired, 25 windows, negative = first arm closer): FP8 minus FP8+QK, truncated KL
-3.45e-03 ± 9.7e-04 (t = -3.57); FP8+QK minus INT8, +3.71e-03 ± 7.3e-04 (t = +5.06), top-1 disagreement +3.25e-03 ± 7.2e-04 (t = +4.53); FP8 minus
INT8, KL +2.6e-04 ± 5.2e-04 (t = +0.49), top-1 disagreement t = +1.84 (not significant). Pairwise PPL: INT8 vs FP8 +0.075 % (t = +0.50); FP8+QK vs FP8 -0.111 %
(t = -0.04). FP8+QK is therefore the most divergent arm from BF16 (small but detectable extra divergence from Q quantization); FP8 and INT8 are
separated by at most about 3 % of the KL.

Position bins (distance from the window start; top-1 agreement % / truncated KL mean ± SE over windows; instrument as above; bins hold different text):

| arm | 0-4k | 4-16k | 16-32k | 32-48k | 48-64k |
| --- | --- | --- | --- | --- | --- |
| FP8 | 94.06 / 3.28e-02 ± 3.6e-03 | 95.21 / 2.43e-02 ± 1.5e-03 | 94.89 / 2.45e-02 ± 1.6e-03 | 94.43 / 2.77e-02 ± 2.1e-03 | 94.12 / 3.74e-02 ± 3.2e-03 |
| FP8+QK | 93.95 / 3.37e-02 ± 3.1e-03 | 95.11 / 2.55e-02 ± 1.6e-03 | 94.82 / 2.54e-02 ± 1.5e-03 | 94.32 / 2.87e-02 ± 2.1e-03 | 93.98 / 3.96e-02 ± 3.6e-03 |
| INT8 | 94.01 / 3.09e-02 ± 3.0e-03 | 95.23 / 2.40e-02 ± 1.6e-03 | 94.93 / 2.42e-02 ± 1.5e-03 | 94.51 / 2.66e-02 ± 2.1e-03 | 94.25 / 3.57e-02 ± 3.2e-03 |
| BF16 repeat (floor) | 100.00 / 0 | 100.00 / 0 | 100.00 / 0 | 100.00 / 0 | 99.99 / 2.4e-05 ± 9e-06 |

Divergence does not grow cleanly with distance: a trough at 4-32k and higher values at both ends, the same shape for all three arms. The
noise floor is exactly identical through 48k and only becomes non-identical in the last bin ([INFERENCE] some reduction at the longest contexts is
not run-to-run deterministic).

### Needle retrieval (task level; a separate instrument)

Instrument: needle harness against the speed/needle serve binary, MTP on, `temperature 0`, `top_k 20`, `seed 1234`, `max_tokens 32`, 120 prompts per arm
at 8k/32k/100k (server `prompt_tokens` 7,898-7,902 / 31,897-31,902 / 99,898-99,902).

| arm | 8k | 32k | 100k | overall (Wilson 95 %) | request errors |
| --- | ---: | ---: | ---: | --- | ---: |
| BF16 | 40/40 | 40/40 | 40/40 | 120/120 (96.9-100 %) | 0 |
| FP8 | 40/40 | 40/40 | 40/40 | 120/120 (96.9-100 %) | 0 |
| INT8 | 40/40 | 40/40 | 40/40 | 120/120 (96.9-100 %) | 0 |

Every cell is 8/8 for all three arms. Pairwise exact McNemar has both-right = 120, b = c = 0, p = 1.000 overall and per length: the degenerate case. The test is at the
ceiling and cannot separate formats. FP8+QK was not run.

### Speed (a separate instrument; build `3df9567`, idle GPU)

Instrument: HTTP client above, 1 warmup + 5 timed requests per cell, median (min-max). Prompts are unique English corpus windows. GPU: NVIDIA GeForce RTX 5090, 32,607 MiB, the server's
`server_start` records CUDA driver/runtime 13.5 for kernels compiled with 13.3; the Windows display driver was 621.01 (recorded in the fidelity receipt;
the speed receipts do not store it). Clocks and temperature were not pinned. Prefill / decode tok/s with MTP (`--spec mtp --draft-tokens auto`):

| nominal prompt | BF16 | FP8 | FP8+QK | INT8 |
| --- | --- | --- | --- | --- |
| 1k (986 tok) | 2984.8 / 133.9 | 2781.7 / 129.3 | 2898.1 / 125.0 | 2859.1 / 126.9 |
| 8k (7,987 tok) | 2860.4 / 107.6 | 2370.0 / 106.9 | 2779.5 / 102.0 | 2414.9 / 95.0 |
| 32k (31,986 tok) | 1769.0 / 66.5 | 1214.6 / 56.6 | 1608.8 / 57.6 | 1233.3 / 50.7 |
| 100k (99,984 tok) | 832.4 / 28.6 | 511.6 / 26.5 | 750.1 / 26.0 | 506.8 / 18.3 |

Without speculative decoding (clean decode comparison), prefill / decode tok/s: 1k BF16 2900.5 / 66.6, FP8 2849.8 / 66.1, FP8+QK 2789.5 / 64.7, INT8 2824.6 / 66.2;
32k BF16 1722.0 / 53.1, FP8 1244.6 / 52.4, FP8+QK 1675.2 / 52.7, INT8 1231.2 / 49.0. Quantized-KV prefill is slower than BF16 at every length from 8k up
(cache-expansion cost); FP8+QK recovers most of it (0.91x BF16 at 32k with MTP, against 0.69x for FP8) at the small extra divergence above. INT8 has no speed or capacity
advantage over FP8: equal prefill, slower long-context decode (MTP at 100k 18.3 vs 26.5 tok/s).

Server memory and capacity at start (from each run's `server_start` event):

| label | KV (global/sliding) | KV capacity (tokens) | KV payload GiB (global + sliding) |
| --- | --- | ---: | --- |
| bf16, MTP | bf16 | 124,224 (clamped from 131,072) | 10.36 (9.48 + 0.88) |
| bf16, no MTP | bf16 | 131,072 | 10.88 (10.00 + 0.88) |
| fp8 (QK bf16 or fp8), MTP and no MTP | fp8-e4m3-row256 | 131,072 | 5.92 (5.04 + 0.88) |
| int8, MTP and no MTP | int8-group64 | 131,072 | 6.04 (5.16 + 0.88) |

BF16 `--kv-capacity auto` is not stable between starts with identical flags (124,224, 131,072 and 127,744 tokens were seen) because auto sizing consumes whatever memory
the desktop leaves free. Quantized arms always reached 131,072 with more than 5 GiB free.

### Caveats

* One artifact, one corpus (four domains, 16 streams), one machine, **one run per quantized arm** (a repeat only for BF16). Speed cells span about two hours of one session at unpinned clocks;
  the desktop's small background load (4-16 % utilization at idle readings) cannot be excluded.
* Nondeterminism: BF16 repeated against itself is not bit-identical (99.29 % of positions bit-identical in the short profile, 99.95 % in the long). Effects below about 0.01 % PPL or above
  99.9 % top-1 agreement cannot be attributed to a format.
* The needle test saturated at 100 %; it says no format lost single-fact retrieval, nothing more. A harder probe (near-duplicate distractors, multi-hop) was not run.
* Short and long profiles (4096/2048 vs 65536/32768) differ in windows and context; their absolute PPL and fidelity numbers are not comparable. The long profile resolves no aggregate PPL
  difference below about 0.5-1 %.
* Truncated KL and target dlogprob are diagnostics over the dumped top-64 (see Definitions); they are not full-vocabulary quantities and say nothing about generation quality.
* The chat-template prefix was applied to raw prose; absolute PPL is conditional on it.
* The `kv-capacity auto` variation makes the BF16 capacity a property of the machine's free memory on that start, not a constant of the format.
* Speed numbers belong to this GPU, driver, build and prompt text (English prose); code or Chinese prompts may differ. MTP decode rates depend on drafted content.
* INT8 Gemma KV is documented as unqualified; only global layers are quantized. Vision, sampling behaviour beyond `top_k 20` and task quality were not measured.
* Window `t` values assume independent windows (contexts overlap) and are uncorrected for the number of comparisons.
* Reports and dumps were produced before this kit existed; the kit's analysis reproduces the original analysis (below), but the original runs used the camp's scripts. The fidelity and speed binaries are different
  builds (see the instrument table); this is why the three instruments are never merged.

### Instrument check

The kit's `analyze.py` was run on the original reports and dumps of both profiles and compared with the originally published `analysis.json`: of 683 compared values for the short profile, 589 are bit-identical and the rest agree to a relative 1.1e-9 at worst (a rounding of
a near-zero ΔPPL%); of 1,593 compared values for the long profile, 1,592 are bit-identical and the remaining one is a descriptive text note. The kit's needle and speed harnesses rebuilt byte-identical prompts (all 360 needle prompt hashes over three arms and all 84 speed prompt hashes
over four label files match the recorded rows), the summarizers reproduced the needle summary and the speed cell statistics and server-start fields exactly, and `derive_prefix_ids.py` on the artifact reproduced the published prefix ids and source hashes.

## Adding a new model

Before the first quantized arm of a new family or size (for example Gemma 4 26B-A4B), check and parameterize exactly the following. The kit itself contains no layer, head or machine constants; everything model-specific
lives in the arms file, the artifact and the receipts.

1. **Arm flag sets.** Write `arms.json` from the flags the new family accepts ([configuration](configuration.md): `--kv-dtype` values per family, `--gemma4-qk` only for Gemma 4 with FP8 KV, `--gemma4-qk fp8` rejected with INT8). The reference arm is the model's BF16 KV run; add the same-flags repeat with role `noise`.
   Per-arm flags go to both the perplexity tool and the server. Do not copy the 31B arms blindly: a format the family does not support must fail closed, and that failure is a result.
2. **Dump support.** `--dump-topk` is Gemma 4 only today; other families reject it. The analysis needs the dump (top-1, KL, targets). Without it a family can only be compared on report PPL, which this kit does not
   analyse. Check that the dump header is `NTOPK001` with k = 64 (the loader refuses anything else).
3. **Prefix ids.** Derive them with `derive_prefix_ids.py --artifact` from the new artifact's own template and tokenizer; pass `--template-kwargs` for template switches (thinking, system prompt) that the family needs, and record them.
   Check the absolute PPL with and without the prefix once; raw-prose PPL near 1,000 on an instruction-tuned model is the sign a prefix is required.
4. **Tokenizer and corpus.** The analysis tokenizes the corpus with the artifact's embedded `resource/text/tokenizer.json`. Token counts must equal the report's, and the target-in-reference-top-64 coverage must stay above 50 % per domain
   (90.8 % overall on the 31B). A family with a different vocabulary must pass those checks, not skip them. Record manifest and per-stream hashes.
5. **KV bytes/token accounting.** Recompute payload per token from the new geometry instead of reusing the 31B numbers (41,280 / 42,240 / 81,920 B/token for FP8 / INT8 / BF16). For a family with global layers storing K and V planes:
   `L_global * 2 * H_kv * (D * code_bytes + (D / group) * 2)` bytes per token for a scaled 1-byte format (for the 31B, 10 global layers, 4 KV heads, D = 512: FP8 group 256 gives 41,280, INT8 group 64 gives 42,240, BF16 `D * 2` gives 81,920). The 31B FP8
   check at capacity 131,072 is 41,280 * 131,072 = 5,410,652,160 bytes, equal to the server's `global_payload_bytes`. Compare your figure with the `server_start` memory fields; sliding rings and snapshots are separate. State which layers the format changes
   (on Gemma 4 only global layers; sliding rings stay BF16).
6. **Capability gates and limits.** Context plus prefix must fit the artifact's native position limit; the long profile and the 100k needle need `--max-context` and a resolved KV capacity that fit (use the resolved capacity from `server_start`, not the requested one). Check whether the serve build accepts `--spec none` (the 31B build did not): if not, omit the speculative flags
   for the no-MTP control, and only run the MTP modes the artifact carries. Pinned `top_k 20` is accepted by every family (Gemma 4 allows up to 64, the others 20), so the pinned sampling works unchanged.
7. **Pool domains and lengths.** `--pool-domains` names the corpus domains used as haystack; `--lengths`, `--depths` and `--needles-per-cell` set the needle grid; `--lengths` for speed must fit the server's capacity. Keep the master seed unless the needle prompts need to change, and record any change.
8. **Before quantized arms.** Run the BF16 reference and its repeat first, read the noise floor and the target-coverage check, then run the arms. Re-run the unit tests of the kit (`python -m pytest tests/bench/kv_eval`) after touching the analysis.

## Tests

`tests/bench/kv_eval/test_analysis_core.py` checks the instrument with hand-computed values: per-window PPL and ΔPPL, top-1 agreement, truncated KL (including a missing reference id), target-in-both subset, paired `t`, exact McNemar (including the degenerate case), Wilson, position-bin means, and
an end-to-end `analyze()` on a synthetic run, plus the guards that reject a dump from another run, mismatched settings and a wrong corpus or tokenizer. It needs no GPU:

```bash
python -m pytest tests/bench/kv_eval
```
