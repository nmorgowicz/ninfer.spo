# tools/bench/kv_eval

Reproducible, model-agnostic comparison of KV-cache and quantization arms against a reference arm on one artifact:
teacher-forced fidelity (perplexity reports plus top-k dumps), needle retrieval and speed. The definitions, protocols,
evidence checklist, the published Gemma 4 31B results and the checklist for porting the kit to a new model are in
[docs/kv-evaluation.md](../../../docs/kv-evaluation.md). Nothing here loads a model except the two executables you point
it at (`ninfer-spo-perplexity`, `ninfer-spo-serve`); the analysis needs no GPU.

Run every command from the repository root. Install the dependencies into the active Python environment
(Python 3.11 or newer):

```bash
pip install -r tools/bench/kv_eval/requirements.txt
```

`numpy` and `tokenizers` are enough for everything except `derive_prefix_ids.py`, which also needs `transformers` for
Jinja chat-template rendering. All text output is UTF-8 regardless of the console code page.

## Arms file

A JSON list (or an object with an `arms` list) of arms. Each arm has a `name`, `flags` (one argv token per element, passed to both
the perplexity tool and the server) and an optional `role`: `reference` (exactly one; the first arm by default), `noise` (a
same-flags repeat of the reference; default for names containing `repeat`) or `quant`. [`arms.example.json`](arms.example.json)
is the Gemma 4 31B set:

| arm | flags |
| --- | --- |
| `bf16` (reference) | `--kv-dtype bf16` |
| `bf16-repeat` (noise floor) | `--kv-dtype bf16` |
| `fp8` | `--kv-dtype fp8 --gemma4-qk bf16` |
| `fp8-qk` | `--kv-dtype fp8 --gemma4-qk fp8` |
| `int8` | `--kv-dtype int8` |

## Tools

| module | does |
| --- | --- |
| `derive_prefix_ids` | Renders the chat-template preamble from an artifact's embedded template/tokenizer (or a source directory) into `prefix.json` |
| `run_ppl` | Runs one perplexity arm (`--dry-run` prints the command): argv, stdout/stderr, times, GPU idle snapshots, binary/artifact/corpus hashes, dump-vs-report token-count check |
| `analyze` | Reports + dumps of all arms to `analysis.json` and `tables.md`: PPL/dPPL, top-1, truncated KL, target dlogprob, paired window `t`, domains, position bins, arm-vs-arm contrasts, provenance hashes |
| `needle` | `run` (HTTP, resumable, `--dry-run` builds prompts offline) and `summarize` (grids, Wilson, exact McNemar) |
| `speed_client` / `speed_summary` | Warmup + timed requests with unique prompts and pinned sampling; median (min-max) tables and server memory/capacity |
| `gpu_state` | One GPU/process idle-state row for the receipts |
| `metrics` | The analysis core (pure numpy functions); the instrument definition |
| `corpus_pool`, `common` | Haystack text pool; shared helpers |

Every module is run as `python -m tools.bench.kv_eval.<module>` and has `--help`.

## Example campaign (placeholders)

`<artifact>` the `.ninfer` file, `<corpus-manifest>` the corpus `manifest.json`, `<out-dir>` a fresh directory.

```bash
cp tools/bench/kv_eval/arms.example.json <out-dir>/arms.json
python -m tools.bench.kv_eval.derive_prefix_ids --artifact <artifact> --out <out-dir>/prefix.json

# One call per arm, sequential, idle GPU. Repeat for bf16-repeat, fp8, fp8-qk, int8. --profile short = 4096/2048, long = 65536/32768.
python -m tools.bench.kv_eval.run_ppl --exe <ppl-exe> --artifact <artifact> --corpus <corpus-manifest> \
    --arms-json <out-dir>/arms.json --arm bf16 --out-dir <out-dir>/ppl-long --profile long \
    --prefix-ids-file <out-dir>/prefix.json [--path-prepend <runtime-dll-dir>] [--dry-run]

python -m tools.bench.kv_eval.analyze --dir <out-dir>/ppl-long --arms-json <out-dir>/arms.json --artifact <artifact>
```

`<out-dir>/ppl-long` then holds `<arm>/report.json`, `dumps/<arm>.topk`, `logs/` (argv, stdout/stderr, times, receipt, GPU states),
and after the analysis `analysis.json`, `tables.md`, `artifact-resources/` (the tokenizer extracted from the artifact). Arms can be
analysed as a subset with `--arms name1,name2`; `--corpus-manifest` overrides the corpus path recorded in the reports (needed when the
reports were written on another machine).

Needle and speed need a running `ninfer-spo-serve` started with the arm's flags and `--model-id kv-eval` (the clients default
to that id and to `http://127.0.0.1:8001/v1/chat/completions`; override with `--model` and `--url`). The full server command
and the arm-by-arm procedure are in [docs/kv-evaluation.md](../../../docs/kv-evaluation.md#running-it).

```bash
python -m tools.bench.kv_eval.needle run --label fp8 --out <out-dir>/needle/fp8.jsonl \
    --manifest <corpus-manifest> --artifact <artifact> [--dry-run]
python -m tools.bench.kv_eval.needle summarize --dir <out-dir>/needle --arms bf16,fp8,int8

python -m tools.bench.kv_eval.speed_client --label fp8-mtp --out <out-dir>/speed/fp8-mtp.cells.jsonl \
    --manifest <corpus-manifest> --artifact <artifact> [--dry-run]
python -m tools.bench.kv_eval.speed_summary --dir <out-dir>/speed --arms bf16,fp8,fp8-qk,int8
```

Needle and speed labels are matched by file name: `needle/<arm>.jsonl`, `speed/<arm>-<mode>.cells.jsonl` and, for server memory,
`speed/<arm>-<mode>.serve.jsonl` (the server's `--request-log-jsonl`).

## Failure behaviour

Every tool fails with `error: <what is wrong>. <next action>` and exit status 2 for operator mistakes (missing file, existing arm
directory, unreachable server, tokenizer or corpus that does not match a report, arms that differ in anything but the intended
flags). The analysis raises on any violated comparability assertion instead of printing numbers.

## Tests

```bash
python -m pytest tests/bench/kv_eval
```

Hand-computed checks of the analysis core on synthetic reports and dumps; no GPU and no model.
