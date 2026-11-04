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
