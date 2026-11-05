# Perplexity evaluation

`ninfer-spo-perplexity` measures the causal perplexity produced by a v3 `.ninfer` artifact. `scripts/build.cmd dev` writes it to `build-spo-dev/apps`.
It uses the artifact's tokenizer, Text model, selected Main KV representation, final normalization,
and main output head. It is an offline evaluator, not a serving endpoint or a logits-export API;
the optional top-k dump below exists only to compare artifacts.
Only Text weights and resources are loaded; Vision and speculative components are not required.

## Run the fixed corpus

The repository includes `ninfer-ppl-1m-v1`, a fixed set of 16 independent UTF-8 streams covering
English reference text, English long-form text, Chinese reference text, and NInfer C++/CUDA code.
`full` selects all streams; `--quick` selects one stream from each domain.

```bash
ninfer-spo-perplexity <artifact> \
  --corpus eval/corpora/perplexity-1m/manifest.json \
  --quick \
  --kv-dtype fp8
```

The default evaluation uses a 4,096-token context and a 2,048-token stride. Use `--context` and
`--stride` to change that protocol, or score one UTF-8 file with `--text FILE`. The available Main
KV representations are `bf16`, `int8`, `fp8`, `nvfp4`, and `k8v4`.

```bash
ninfer-spo-perplexity <artifact> \
  --text notes.txt \
  --context 16384 --stride 8192 \
  --kv-dtype int8
```

Run `ninfer-spo-perplexity --help` for the complete command surface. `<artifact>` is a text artifact from the [README](../README.md). The evaluator loads
the model once, reads and tokenizes every selected stream before scoring, and writes readable
startup, corpus, scoring, and per-stream summaries to stderr. Interactive weight loading and
scoring use one transient progress line; redirected scoring emits persistent progress every ten
seconds. `--log-level debug` exposes internal startup and stream-begin detail. The final
domain/overall table remains product output on stdout; the independent full-precision machine
report is `report.json` under `profiles/perplexity/` unless `--output` supplies an empty directory.

For KV-format comparisons, the recommended long-context profile is the full corpus with
`--context 65536 --stride 32768` and without `--quick`. The default evaluator
KV format is FP8. Gemma 4 accepts BF16, experimental global-only FP8 and opt-in
unqualified global-only INT8; its sliding rings remain BF16. Use `--kv-dtype bf16` for a BF16 baseline.
Context plus prefix length must fit the artifact's native position limit.

For Gemma 4, `--gemma4-qk bf16` (default) retains Phase 1 BF16 QK over
expanded FP8 cache values. `--kv-dtype fp8 --gemma4-qk fp8` selects experimental
FP8 tensor-core QK for global D512 query widths ≥64; sliding attention and
short-width SIMT are unchanged. V expands to BF16 and PV keeps its hi/lo BF16
products. The report records requested/resolved QK mode and scope under
`execution`. Compare Phase 2 against Phase 1 with the same FP8 cache and
identical histories to isolate query quantization, and against BF16 KV
separately. Top-k dumps do not establish full-distribution KL or model quality.

[KV-format evaluation](kv-evaluation.md) defines the full comparison protocol built on this tool: the short
(4096/2048) and long (65536/32768) profiles, a same-flags noise-floor repeat, the metrics computed from the report
and the top-k dump (paired window statistics, top-1 agreement, truncated KL, target-token log-probability change,
position bins), the evidence a result must carry, and the measured Gemma 4 31B BF16, FP8, FP8+QK and INT8 results. Its
kit lives in `tools/bench/kv_eval/`.

`--prefix-ids 2,105,...` prepends the same token ids to every window. They are context only: the
scored targets and token counts do not change. Use it for models that are far from their training
distribution on bare text. Gemma 4 instruction-tuned models score raw prose near perplexity 1,000
but behave normally after a chat preamble that opens a model turn (the ids of the rendered
template up to the generation prompt). The report records the ids under `execution.prefix_ids`.

## Metric

For a stream `x[0..N)`, every token after `x[0]` is scored exactly once. A window `[b,e)` with target
suffix `[s,e)` contributes:

```text
log p(x[i] | x[b], ..., x[i-1])  for i in [s,e)
```

Each window starts from empty State and Main KV, so history before `b` is deliberately excluded.
The reported metric is therefore fixed-window, truncated-context causal perplexity:

```text
mean_nll = -sum(logprob) / scored_tokens
perplexity = exp(mean_nll)
```

The first window scores `[1,min(context,N))`. Each later window advances by `stride` targets while
retaining up to `context-stride` preceding tokens as local context. Streams never share history.

## Comparing runs

For a numerical comparison, keep the corpus, context, stride, and execution settings fixed except
the variable being measured. Compare KV formats with the same artifact and weight formats with the
same KV format.

The corpus name is a workload scale, not an exact token count. Exact input and scored-token counts
are runtime results from the current artifact tokenizer and are recorded in each report. Reports
contain unrounded NLL/PPL values for every window, stream, domain, and the token-weighted overall
aggregate.

The schema-v2 report identifies the artifact's architecture, public name, actual weight formats
and prefill signature alongside the workload and numerical results.

## Top-k dump for truncated-distribution comparison

`--dump-topk FILE` writes, for every scored position, the 64 most likely
next-token ids and their full-softmax log-probabilities. Matched dumps can
support top-1 agreement and explicitly defined truncated-distribution
diagnostics; they cannot establish full-vocabulary KL divergence because
the omitted tail is unavailable. Only Gemma 4 artifacts support this dump
today; other families reject the option.

The file is little-endian: the 8 bytes `NTOPK001`, a `u32` k (64), a `u32` 0, then per scored
position k `i32` ids (most likely first) followed by k `f32` log-probabilities. Positions follow
the stream, window and target order of the scoring itself.

## Exact paired full-vocabulary KL

`--kl-reference-write CACHE` and `--kl-reference CACHE` measure `D_KL(reference || candidate)` over the
whole vocabulary at every scored position. Gemma 4 artifacts only; other families reject the option. They
are mutually exclusive and cannot be combined with `--dump-topk`.

1. **Reference run** (`--kl-reference-write`): scores the reference artifact once and writes the F32
   log-softmax row of every scored position to `CACHE` (never raw logits, never BF16 rows). The cache is
   `positions x vocab x 4` bytes, about 1 MiB per position at vocab 262144: 16k positions is about 16 GiB and
   64k positions about 64 GiB. The full 1M-token corpus would need about 1 TB, so use a head-truncated subset
   (`python -m tools.bench.kv_eval.kl_corpus`). It refuses to overwrite an existing cache.
2. **Candidate run** (`--kl-reference`): scores the candidate and streams the cache sequentially, never
   holding more than one score tile of rows. Per position it computes in F64
   `sum_v exp(r_v) (r_v - c_v)` from the two F32 rows, unclamped, so same-arm noise stays visible as small
   signed values.

The cache header binds a **protocol**: corpus id and mode, per-stream ids/domains/token counts, an FNV-64
hash of every token and window boundary, context, stride, prefix ids, KV dtype, resolved Gemma QK mode,
scored-position count and vocab. A candidate run with any difference is refused with the differing field
named; a truncated, never-finished or foreign file is refused. A filename is not identity. The
reference artifact's path, file size, formats and prefill signature are stored as informational provenance
and copied into candidate reports (`kl.reference_artifact`); hash the artifact separately for a published
result.

The report gains a `kl` object: `protocol`, `reference_cache`, `reference_artifact`, `candidate_artifact`,
`target_logprob_max_abs_diff` (host log-softmax of the BF16 logits against the engine's own target
log-probability; a gap above 0.05 aborts the run as a column misalignment), and `statistics`. `statistics`
holds, overall and per domain: positions, windows, mean KL with a 95% percentile interval from a bootstrap
that resamples **whole windows** (`--kl-resamples`, default 1000; `--kl-seed`, default 20261105), median,
p95, p99 and p99.9 KL, mean change in target NLL (candidate minus reference) with its interval, and top-1
disagreement, plus per-window sums for pairing two arms. `kl_positions.f32` in the output directory holds
four F32 values per position (KL, reference target log-probability, candidate target log-probability,
top-1 agreement).

Interpretation rules: a candidate's KL is its distance from the *reference arm*, so it ranks candidates only
when the reference is closer to the unquantized model than every candidate (see the
[weight-quantization plan](https://github.com/nmorgowicz-org/llama-local-tooling/blob/research/tooling-reorganization/docs/plans/20261105-ninfer.spo_gemma4_weight_quantization.md)).
Run the same arm twice first: GPU scoring is not bit-reproducible, so the same-arm KL is a noise floor and
its tail (a few positions with large KL) bounds how small a difference can be resolved.
`python -m tools.bench.kv_eval.kl_compare --baseline A/report.json --candidate B/report.json` pairs two
candidate reports scored against one cache (same windows) and reports the mean-KL ratio and the paired
change in NLL with window-bootstrap intervals; `--delta` applies an operator-set acceptance factor and has no
default.
