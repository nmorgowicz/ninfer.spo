"""Hand-computed checks of the KV-format evaluation instrument (tools/bench/kv_eval/metrics.py and analyze.py).

Every expected value is derived on paper from the metric definitions in docs/kv-evaluation.md, never from the code under
test. Dumps are synthetic: two meaningful candidate ids per position, 62 filler ids at logprob -40 (probability ~4e-18,
below every tolerance used here).
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from tools.bench.kv_eval import metrics
from tools.bench.kv_eval.analyze import analyze
from tools.bench.kv_eval.common import Arm, load_arms
from tools.bench.kv_eval.metrics import EvalError, K, REC

FILLER_IDS = list(range(100, 100 + K - 2))
FILLER_LP = -40.0


def rows(spec):
    """Dump records from [(ids, lps), ...] where each entry lists the two meaningful candidates."""
    rec = np.zeros(len(spec), REC)
    for i, (ids, lps) in enumerate(spec):
        rec["id"][i] = list(ids) + FILLER_IDS
        rec["lp"][i] = list(lps) + [FILLER_LP] * len(FILLER_IDS)
    return rec


def ln(x):
    return math.log(x)


# --------------------------------------------------------------------------------------------------------------------
# Per-position metrics
# --------------------------------------------------------------------------------------------------------------------

def four_position_case():
    ref = rows([
        ((10, 11), (ln(0.75), ln(0.25))),   # 0: identical in both arms
        ((10, 11), (ln(0.75), ln(0.25))),   # 1: same ids, different probabilities
        ((10, 11), (ln(0.60), ln(0.40))),   # 2: argmax flips
        ((10, 11), (ln(0.60), ln(0.30))),   # 3: id 11 missing from the arm; 0.1 of mass outside the pair
    ])
    arm = rows([
        ((10, 11), (ln(0.75), ln(0.25))),
        ((10, 11), (ln(0.50), ln(0.50))),
        ((11, 10), (ln(0.55), ln(0.45))),
        ((10, 12), (ln(0.50), ln(0.40))),
    ])
    targets = np.array([10, 10, 11, 11], np.int32)
    return arm, ref, targets


def test_truncated_kl_matches_hand_values():
    arm, ref, targets = four_position_case()
    pp = metrics.per_position(arm, ref, metrics.target_lp(arm, targets), metrics.target_lp(ref, targets))
    # pos 0: identical distributions.
    assert pp["kl"][0] == pytest.approx(0.0, abs=1e-12)
    # pos 1: KL((.75,.25) || (.5,.5)) = .75 ln 1.5 + .25 ln .5 = 0.130812036
    assert pp["kl"][1] == pytest.approx(0.130812036, abs=1e-6)
    # pos 2: KL((.6,.4) || (.45,.55) on ids (10,11)) = .6 ln(.6/.45) + .4 ln(.4/.55) = 0.045227746
    assert pp["kl"][2] == pytest.approx(0.045227746, abs=1e-6)
    # pos 3: ref renormalised over its set to (2/3, 1/3). The arm lacks id 11 and charges it its 64th logprob (-40), so
    # q = (.5, e^-40, ...) / Z with Z ~ .5, i.e. q10 ~ 1 and q11 ~ 2 e^-40:
    # KL ~ (2/3) ln(2/3) + (1/3) (ln(1/3) - ln 2 + 40) = 12.46577
    expected = (2 / 3) * ln(2 / 3) + (1 / 3) * (ln(1 / 3) - ln(2) + 40.0)
    assert expected == pytest.approx(12.46577011, abs=1e-7)
    assert pp["kl"][3] == pytest.approx(expected, abs=1e-6)
    assert list(pp["missing"]) == [0, 0, 0, 1]


def test_top1_exactness_and_target_subset():
    arm, ref, targets = four_position_case()
    pp = metrics.per_position(arm, ref, metrics.target_lp(arm, targets), metrics.target_lp(ref, targets))
    assert list(pp["top1"]) == [True, True, False, True]          # rank-0 ids 10/10, 10/10, 10 vs 11, 10/10
    assert list(pp["exact"]) == [True, False, False, False]       # bit-identical dump rows
    # Target 11 at position 3 is outside the arm's top-64: that position is excluded from the dlogprob subset.
    assert list(pp["both"]) == [True, True, True, False]
    d = pp["d"][pp["both"]]
    expected = [0.0, ln(0.50) - ln(0.75), ln(0.55) - ln(0.40)]    # arm minus reference target logprob
    assert d == pytest.approx(expected, abs=1e-6)                 # logprobs are stored as float32
    summary = metrics.agg(pp, np.ones(4, bool))
    assert summary["top1_agree"] == 0.75
    assert summary["bit_identical_topk_frac"] == 0.25
    assert summary["ref_top64_ids_missing_from_arm_frac"] == 0.25
    assert summary["target_in_both_top64_frac"] == 0.75
    assert summary["target_abs_dlogprob_mean_subset"] == pytest.approx((0.0 + abs(expected[1]) + expected[2]) / 3, abs=1e-6)
    assert summary["target_abs_dlogprob_max_subset"] == pytest.approx(max(abs(expected[1]), expected[2]), abs=1e-6)
    first_two = metrics.agg(pp, np.array([True, True, False, False]))
    assert first_two["top1_agree"] == 1.0
    assert first_two["kl_top64_truncated_mean"] == pytest.approx(0.130812036 / 2, abs=1e-6)


def test_window_bin_stats_and_window_means():
    win = np.array([0, 0, 1, 1])
    bins = np.array([0, 1, 0, -1])              # position 3 is outside every bin
    values = np.array([1.0, 3.0, 5.0, 100.0])
    got = metrics.window_bin_stats(win, bins, values, np.ones(4, bool), 2)
    assert got[0].tolist() == [1.0, 3.0]
    assert got[1][0] == 5.0 and math.isnan(got[1][1])
    assert metrics.window_means(win, values).tolist() == [2.0, 52.5]


# --------------------------------------------------------------------------------------------------------------------
# Report-level statistics
# --------------------------------------------------------------------------------------------------------------------

def test_paired_t_hand_values():
    stat = metrics.paired_t([0.01, 0.03, 0.02])          # mean .02, std(ddof=1) .01, se .01/sqrt(3), t = 2 sqrt(3)
    assert stat["mean"] == pytest.approx(0.02)
    assert stat["se"] == pytest.approx(0.01 / math.sqrt(3))
    assert stat["t"] == pytest.approx(2 * math.sqrt(3))
    assert metrics.paired_t([0.5, 0.5, 0.5])["t"] is None     # zero spread: t undefined, not infinite
    single = metrics.paired_t([0.5])
    assert single["se"] is None and single["t"] is None


def make_report(windows_by_stream):
    """Minimal report with {stream id: (domain, [(nll, tokens), ...])}."""
    streams = []
    for sid, (domain, wins) in windows_by_stream.items():
        streams.append({"id": sid, "domain": domain, "windows": [
            {"target_begin": 100 * i, "target_end": 100 * i + n, "scored_tokens": n, "total_nll": nll}
            for i, (nll, n) in enumerate(wins)]})
    return {"streams": streams}


def test_ppl_delta_hand_values():
    ref = make_report({"s1": ("d1", [(20.0, 10), (30.0, 10)]), "s2": ("d2", [(12.0, 4)])})
    arm = make_report({"s1": ("d1", [(21.0, 10), (33.0, 10)]), "s2": ("d2", [(12.4, 4)])})
    res = metrics.ppl_delta(arm, ref)
    assert res["windows"] == 3
    o = res["overall"]
    assert o["ppl_ref"] == pytest.approx(math.exp(62 / 24)) and o["ppl_arm"] == pytest.approx(math.exp(66.4 / 24))
    assert o["dppl_pct"] == pytest.approx(100 * (math.exp(4.4 / 24) - 1))
    # per-window dNLL/token = [0.1, 0.3, 0.1]: mean 1/6, std(ddof=1) sqrt(.013333), se = 1/15, t = 2.5
    assert o["mean_dnll_per_token_window_mean"] == pytest.approx(1 / 6)
    assert o["se"] == pytest.approx(1 / 15)
    assert o["t"] == pytest.approx(2.5)
    d1 = res["domains"]["d1"]
    assert d1["dppl_pct"] == pytest.approx(100 * (math.exp(54 / 20) / math.exp(50 / 20) - 1))


def test_ppl_delta_rejects_disjoint_reports():
    with pytest.raises(EvalError):
        metrics.ppl_delta(make_report({"a": ("d", [(1.0, 1)])}), make_report({"b": ("d", [(1.0, 1)])}))


def test_mcnemar_exact_hand_values_and_degenerate_case():
    assert metrics.mcnemar_exact(1, 5) == pytest.approx(2 * (1 + 6) / 64)        # n = 6, k = 1: 0.21875
    assert metrics.mcnemar_exact(0, 4) == pytest.approx(2 / 16)
    assert metrics.mcnemar_exact(0, 8) == pytest.approx(2 / 256)
    assert metrics.mcnemar_exact(3, 3) == 1.0                                    # 2 * 42 / 64 > 1 is capped at 1
    assert metrics.mcnemar_exact(0, 0) == 1.0                                    # no discordant pair: no information, p = 1


def test_wilson_interval_hand_values():
    lo, hi = metrics.wilson(120, 120)       # p = 1: lower = 1 / (1 + z^2/n), upper = 1
    assert lo == pytest.approx(1 / (1 + 1.96 ** 2 / 120), abs=1e-12)
    assert hi == pytest.approx(1.0, abs=1e-12)
    lo0, hi0 = metrics.wilson(0, 120)
    assert lo0 == pytest.approx(0.0, abs=1e-12) and hi0 == pytest.approx(1.96 ** 2 / 120 / (1 + 1.96 ** 2 / 120), abs=1e-12)


# --------------------------------------------------------------------------------------------------------------------
# End to end: analyze() on a synthetic run
# --------------------------------------------------------------------------------------------------------------------

TEXTS = {"s1": ("dom_a", "a b c d a b c d a b"), "s2": ("dom_b", "d c b a d c")}   # 10 and 6 tokens: 9 and 5 targets
IDS = {"s1": [1, 2, 3, 4, 1, 2, 3, 4, 1, 2], "s2": [4, 3, 2, 1, 4, 3]}
P_REF = 0.60
# arm probability given to the target token (the other candidate gets the rest), per stream
ARM_P = {"ref": {"s1": 0.60, "s2": 0.60}, "noise": {"s1": 0.60, "s2": 0.60},
         "q1": {"s1": 0.50, "s2": 0.55}, "q2": {"s1": 0.55, "s2": 0.55}}


def kl_vs_ref(p):
    """KL((.6,.4) || (p, 1-p)) on the two candidates."""
    return 0.6 * ln(0.6 / p) + 0.4 * ln(0.4 / (1 - p))


def write_dump(path, arm, header=None):
    spec = []
    for sid in ("s1", "s2"):
        for t in IDS[sid][1:]:
            other = 1 + (t % 4)
            p = ARM_P[arm][sid]
            spec.append(((t, other), (ln(p), ln(1 - p))))
    body = rows(spec).tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((header or (b"NTOPK001" + (64).to_bytes(4, "little") + (0).to_bytes(4, "little"))) + body)


def build_report(arm):
    streams, total = [], 0.0
    for sid in ("s1", "s2"):
        domain, _ = TEXTS[sid]
        n = len(IDS[sid]) - 1
        nll = -n * ln(ARM_P[arm][sid])
        total += nll
        streams.append({"id": sid, "domain": domain, "path": f"{sid}.txt", "input_tokens": len(IDS[sid]), "scored_tokens": n,
                        "windows": [{"input_begin": 0, "input_end": len(IDS[sid]), "target_begin": 1, "target_end": len(IDS[sid]),
                                     "scored_tokens": n, "total_nll": nll}]})
    n_total = 14
    return {"schema_version": 2, "corpus": {"id": "synthetic", "mode": "full", "source": "unused", "stream_count": 2},
            "execution": {"context_tokens": 4096, "stride_tokens": 2048, "prefix_ids": [2], "prefill_chunk_tokens": 1024,
                          "score_tile_tokens": 1024, "device": 0, "kv_dtype": "bf16"},
            "artifact": {"name": "synthetic", "prefill_signature": "sig", "path": "x.ninfer", "formats": ["bf16"]},
            "overall": {"scored_tokens": n_total, "total_nll": total, "mean_nll": total / n_total, "perplexity": math.exp(total / n_total)},
            "domains": [{"domain": TEXTS[s][0], "scored_tokens": len(IDS[s]) - 1, "mean_nll": -ln(ARM_P[arm][s]),
                         "perplexity": 1 / ARM_P[arm][s]} for s in ("s1", "s2")],
            "streams": streams}


@pytest.fixture
def run(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for sid, (_, text) in TEXTS.items():
        (corpus / f"{sid}.txt").write_text(text, encoding="utf-8")
    tok = Tokenizer(models.WordLevel({"[UNK]": 0, "a": 1, "b": 2, "c": 3, "d": 4}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(tmp_path / "tokenizer.json"))
    root = tmp_path / "ppl"
    for arm in ARM_P:
        (root / arm).mkdir(parents=True)
        (root / arm / "report.json").write_text(json.dumps(build_report(arm)), encoding="utf-8")
        write_dump(root / "dumps" / f"{arm}.topk", arm)
    arms = [Arm("ref", (), "reference"), Arm("noise", (), "noise"), Arm("q1", (), "quant"), Arm("q2", (), "quant")]
    return {"root": root, "corpus": corpus, "tokenizer": tmp_path / "tokenizer.json", "arms": arms}


def go(run):
    return analyze(run["root"], run["arms"], run["tokenizer"], run["corpus"])


def test_analyze_reproduces_hand_computed_statistics(run):
    out = go(run)
    assert out["target_derivation_check"]["target_in_ref_top64_frac"] == 1.0
    assert out["target_derivation_check"]["positions_by_bin"] == {"0-4k": 14, "4-16k": 0, "16-32k": 0, "32-48k": 0, "48-64k": 0}

    noise = out["pairs"]["noise floor: noise vs ref"]
    assert noise["metrics"]["overall"]["top1_agree"] == 1.0
    assert noise["metrics"]["overall"]["bit_identical_topk_frac"] == 1.0
    assert noise["metrics"]["overall"]["kl_top64_truncated_mean"] == pytest.approx(0.0, abs=1e-12)
    assert noise["ppl"]["overall"]["dppl_pct"] == pytest.approx(0.0, abs=1e-9)
    assert noise["ppl"]["overall"]["t"] is None            # identical windows: zero spread, t undefined

    q1 = out["pairs"]["q1 vs ref"]
    kl_expected = (9 * kl_vs_ref(0.50) + 5 * kl_vs_ref(0.55)) / 14
    assert q1["metrics"]["overall"]["kl_top64_truncated_mean"] == pytest.approx(kl_expected, abs=1e-6)
    assert q1["metrics"]["overall"]["top1_agree"] == 1.0   # the target stays the argmax in every arm
    assert q1["metrics"]["overall"]["bit_identical_topk_frac"] == 0.0
    assert q1["by_domain"]["dom_a"]["kl_top64_truncated_mean"] == pytest.approx(kl_vs_ref(0.50), abs=1e-6)
    assert q1["by_domain"]["dom_b"]["kl_top64_truncated_mean"] == pytest.approx(kl_vs_ref(0.55), abs=1e-6)
    ppl = q1["ppl"]["overall"]
    nll_arm, nll_ref = 9 * -ln(0.50) + 5 * -ln(0.55), 14 * -ln(0.60)
    assert ppl["dppl_pct"] == pytest.approx(100 * (math.exp((nll_arm - nll_ref) / 14) - 1), abs=1e-9)
    # two windows: d = (ln(.6/.5), ln(.6/.55)); for n = 2, t = (d1 + d2) / |d1 - d2|
    d1, d2 = ln(0.6 / 0.5), ln(0.6 / 0.55)
    assert ppl["windows"] == 2
    assert ppl["t"] == pytest.approx((d1 + d2) / abs(d1 - d2), abs=1e-9)
    assert ppl["se"] == pytest.approx(abs(d1 - d2) / 2, abs=1e-9)

    # arm vs arm: window means of KL vs the reference, q1 - q2 = (KL(.5) - KL(.55), KL(.55) - KL(.55)) = (x, 0): t = 1
    row = out["arm_vs_arm_paired"]["rows"]["q1 - q2"]["kl_top64_truncated"]
    x = kl_vs_ref(0.50) - kl_vs_ref(0.55)
    assert row["mean_diff"] == pytest.approx(x / 2, abs=1e-6)
    assert row["se"] == pytest.approx(x / 2, abs=1e-6)
    assert row["t"] == pytest.approx(1.0, abs=1e-6)
    assert out["arm_vs_arm_paired"]["rows"]["q1 - q2"]["top1_disagree"]["t"] is None   # both arms never disagree: zero spread


def test_analyze_rejects_dump_from_a_different_run(run):
    # A report whose NLL is below what the dump's target logprobs imply cannot belong to that dump.
    path = run["root"] / "ref" / "report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    for s in report["streams"]:
        for w in s["windows"]:
            w["total_nll"] *= 0.5
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(EvalError, match="one-sided NLL bound"):
        go(run)


def test_analyze_rejects_mismatched_settings(run):
    path = run["root"] / "q1" / "report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["execution"]["context_tokens"] = 65536
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(EvalError, match="context_tokens"):
        go(run)


def test_analyze_rejects_corpus_that_tokenizes_differently(run):
    (run["corpus"] / "s1.txt").write_text(TEXTS["s1"][1] + " c", encoding="utf-8")   # 11 tokens, the reports say 10
    with pytest.raises(EvalError, match="tokenizer mismatch for stream s1"):
        go(run)


def test_open_dump_rejects_bad_magic_and_truncation(run):
    bad = run["root"] / "dumps" / "q2.topk"
    bad.write_bytes(b"NOTADUMP" + bad.read_bytes()[8:])
    with pytest.raises(EvalError, match="not a top-64 dump"):
        go(run)
    write_dump(bad, "q2")
    bad.write_bytes(bad.read_bytes()[:-REC.itemsize])
    with pytest.raises(EvalError, match="dump holds 13 positions"):
        go(run)


def test_arms_file_roles(tmp_path):
    spec = tmp_path / "arms.json"
    spec.write_text(json.dumps([{"name": "bf16", "flags": ["--kv-dtype", "bf16"]}, {"name": "bf16-repeat", "flags": ["--kv-dtype", "bf16"]},
                                {"name": "fp8", "flags": ["--kv-dtype", "fp8"]}]), encoding="utf-8")
    assert [a.role for a in load_arms(spec)] == ["reference", "noise", "quant"]
    spec.write_text(json.dumps([{"name": "a", "role": "quant"}, {"name": "b", "role": "quant"}]), encoding="utf-8")
    with pytest.raises(SystemExit):
        load_arms(spec)
