#!/usr/bin/env python3
"""Run ONE ninfer-spo-perplexity arm with a top-k dump and keep the evidence for it.

  python -m tools.bench.kv_eval.run_ppl --exe <perplexity-exe> --artifact <artifact> --corpus <corpus-manifest> \
      --arms-json <arms.json> --arm <arm name> --out-dir <out-dir> --prefix-ids-file <prefix.json> [--profile short|long] [--dry-run]

Runs   <exe> <artifact> --corpus <manifest> [--quick] --context C --stride S <arm flags>
              --prefix-ids <ids> --dump-topk <out-dir>/dumps/<arm>.topk --output <out-dir>/<arm>

Profiles: short = context 4096 / stride 2048, long = context 65536 / stride 32768 (override with --context/--stride).
The arm's output directory and dump must not exist yet (one fresh directory per arm); run each arm sequentially on an
otherwise idle GPU. Evidence written under <out-dir>/logs/: ppl-<arm>.{argv.json,stdout.txt,stderr.txt,times.txt,
receipt.json}, gpu-states.jsonl (snapshot before/after) and artifact.sha256.txt (computed once, unless supplied).
After a successful run the dump length is checked against the report's scored-token count.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.bench.kv_eval import gpu_state
from tools.bench.kv_eval.common import (
    fail, find_arm, load_arms, read_json, sha256_file, use_utf8_stdio, utc_now,
)
from tools.bench.kv_eval.metrics import HEADER_BYTES, REC

PROFILES = {"short": (4096, 2048), "long": (65536, 32768)}


def load_prefix_ids(file: Path | None, csv: str | None, none: bool) -> str | None:
    if none:
        return None
    if csv:
        return csv
    if file is None:
        fail("pass --prefix-ids-file <prefix.json> (from derive_prefix_ids.py), --prefix-ids 2,105,... or --no-prefix. "
             "Chat-tuned models score raw prose near PPL 1000 without a template prefix, so the choice must be explicit.")
    text = file.read_text(encoding="utf-8").strip() if file.is_file() else fail(f"prefix file not found: {file}")
    if text.startswith("{"):
        doc = json.loads(text)
        return ",".join(str(i) for i in doc["ids"]) if "ids" in doc else doc["ids_csv"]
    return text


def artifact_sha256(artifact: Path, logs: Path, supplied: str | None) -> dict:
    cache = logs / "artifact.sha256.txt"
    if supplied:
        return {"sha256": supplied.lower(), "how": "supplied by --artifact-sha256 (not recomputed)"}
    if cache.is_file():
        digest, _, path = cache.read_text(encoding="utf-8").strip().partition("  ")
        if Path(path).resolve() == artifact.resolve():
            return {"sha256": digest, "how": f"read from {cache.name} (computed by an earlier arm of this out-dir)"}
    print(f"hashing {artifact} (once per out-dir; pass --artifact-sha256 to skip)", file=sys.stderr, flush=True)
    digest = sha256_file(artifact)
    cache.write_text(f"{digest}  {artifact}\n", encoding="utf-8")
    return {"sha256": digest, "how": "computed now"}


def main() -> None:
    use_utf8_stdio()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", type=Path, required=True, help="ninfer-spo-perplexity executable")
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True, help="corpus manifest.json")
    ap.add_argument("--arms-json", type=Path, required=True)
    ap.add_argument("--arm", required=True, help="arm name from --arms-json")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--profile", choices=sorted(PROFILES), default="short")
    ap.add_argument("--context", type=int)
    ap.add_argument("--stride", type=int)
    ap.add_argument("--quick", action="store_true", help="one stream per domain instead of the full corpus")
    ap.add_argument("--prefix-ids-file", type=Path)
    ap.add_argument("--prefix-ids", help="comma-separated ids, instead of --prefix-ids-file")
    ap.add_argument("--no-prefix", action="store_true", help="score without a prefix (base models / raw text)")
    ap.add_argument("--artifact-sha256", help="precomputed artifact sha256 (hashing a large artifact takes minutes)")
    ap.add_argument("--path-prepend", type=Path, action="append", default=[], help="directory prepended to PATH for the run (runtime DLLs)")
    ap.add_argument("--allow-busy-gpu", action="store_true", help="do not refuse to start while another ninfer process is running")
    ap.add_argument("--dry-run", action="store_true", help="print the command and planned outputs, run nothing")
    args = ap.parse_args()

    arm = find_arm(load_arms(args.arms_json), args.arm, args.arms_json)
    context, stride = PROFILES[args.profile]
    context, stride = args.context or context, args.stride or stride
    exe, artifact, corpus, out_dir = (p.resolve() for p in (args.exe, args.artifact, args.corpus, args.out_dir))
    for what, path in (("perplexity executable", exe), ("artifact", artifact), ("corpus manifest", corpus)):
        if not path.is_file():
            fail(f"{what} not found: {path}")
    prefix = load_prefix_ids(args.prefix_ids_file, args.prefix_ids, args.no_prefix)
    arm_dir, dump = out_dir / arm.name, out_dir / "dumps" / f"{arm.name}.topk"
    argv = [str(exe), str(artifact), "--corpus", str(corpus)] + (["--quick"] if args.quick else []) + \
           ["--context", str(context), "--stride", str(stride)] + list(arm.flags) + \
           (["--prefix-ids", prefix] if prefix else []) + ["--dump-topk", str(dump), "--output", str(arm_dir)]
    if args.dry_run:
        print(json.dumps({"arm": arm.name, "argv": argv, "cwd": str(exe.parent), "profile": args.profile,
                          "output_dir": str(arm_dir), "dump": str(dump), "logs": str(out_dir / "logs"),
                          "output_dir_exists": arm_dir.exists(), "dump_exists": dump.exists()}, indent=2))
        return
    if arm_dir.exists() or dump.exists():
        fail(f"{arm_dir if arm_dir.exists() else dump} already exists; each arm needs a fresh output directory and dump. "
             "Choose another --out-dir or delete the failed attempt after keeping its logs.")
    logs = out_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    dump.parent.mkdir(parents=True, exist_ok=True)
    state_log = logs / "gpu-states.jsonl"
    before = gpu_state.snapshot(f"ppl-{arm.name}-before", state_log)
    if before["ninfer_processes"] and not args.allow_busy_gpu:
        fail(f"another ninfer process is running ({before['ninfer_processes']}); stop it or pass --allow-busy-gpu "
             "(a busy GPU invalidates speed numbers and risks out-of-memory)")
    identity = {"exe_sha256": sha256_file(exe), "artifact": artifact_sha256(artifact, logs, args.artifact_sha256),
                "corpus_manifest_sha256": sha256_file(corpus)}
    arm_dir.mkdir(parents=False, exist_ok=False)
    (logs / f"ppl-{arm.name}.argv.json").write_text(json.dumps(argv, indent=1), encoding="utf-8")
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([str(p.resolve()) for p in args.path_prepend] + [env.get("PATH", "")])
    start, t0 = utc_now(), time.monotonic()
    with open(logs / f"ppl-{arm.name}.stdout.txt", "wb") as so, open(logs / f"ppl-{arm.name}.stderr.txt", "wb") as se:
        rc = subprocess.run(argv, stdout=so, stderr=se, cwd=str(exe.parent), env=env).returncode
    end, wall = utc_now(), time.monotonic() - t0
    (logs / f"ppl-{arm.name}.times.txt").write_text(f"{start}\nexit={rc}\n{end}\nwall_seconds={wall:.1f}\n", encoding="utf-8")
    after = gpu_state.snapshot(f"ppl-{arm.name}-after", state_log)
    receipt = {"arm": arm.name, "flags": list(arm.flags), "argv": argv, "exit": rc, "start_utc": start, "end_utc": end,
               "wall_seconds": round(wall, 1), "profile": args.profile, "context": context, "stride": stride, "quick": args.quick,
               "prefix_ids": prefix, "identity": identity, "gpu_before": before, "gpu_after": after, "python": sys.version.split()[0]}
    if rc == 0:
        report_path = arm_dir / "report.json"
        if not report_path.is_file() or not dump.is_file():
            fail(f"exit 0 but {report_path if not report_path.is_file() else dump} is missing; see {logs / f'ppl-{arm.name}.stderr.txt'}")
        report = read_json(report_path, "report.json")
        scored = report["overall"]["scored_tokens"]
        positions = (dump.stat().st_size - HEADER_BYTES) / REC.itemsize
        receipt.update({"report_sha256": sha256_file(report_path), "dump_sha256": sha256_file(dump),
                        "scored_tokens": scored, "dump_positions": positions,
                        "input_tokens": sum(s["input_tokens"] for s in report["streams"]),
                        "windows": sum(len(s["windows"]) for s in report["streams"])})
        if positions != scored:
            (logs / f"ppl-{arm.name}.receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
            fail(f"dump holds {positions} positions but the report scored {scored}; the run is not usable")
    (logs / f"ppl-{arm.name}.receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"{arm.name}: exit={rc} wall={wall:.1f}s")
    sys.exit(rc)


if __name__ == "__main__":
    main()
