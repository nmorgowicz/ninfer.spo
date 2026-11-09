"""`python -m tools.convert.gemma4_build`: preset commands, preflight, resume, launcher; runs nothing heavy."""

from __future__ import annotations

from functools import partial
import hashlib
import io
import json
import os
import socket
from pathlib import Path
import struct
import subprocess
import sys
import time

import pytest

from tools.convert import gemma4_build as build
from tools.convert import gemma4_build_checks as checks
from tools.convert import gemma4_build_cli as cli
from tools.convert import gemma4_build_launcher as launcher

PY = "PY"
PREFIX = "2,105,2364,107,34230,506,2354,236761,106,107,105,4368,107,100,45518,107,101"


# ---------------------------------------------------------------- fixtures


def write_safetensors(path: Path, dtype: str = "BF16") -> None:
    header = json.dumps({"w": {"dtype": dtype, "shape": [1], "data_offsets": [0, 2]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0")


def gemma_config(hidden: int = 5376, layers: int = 60) -> dict:
    types = (["sliding_attention"] * 5 + ["full_attention"]) * 10
    return {
        "architectures": ["Gemma4ForConditionalGeneration"],
        "model_type": "gemma4",
        "text_config": {
            "hidden_size": hidden, "vocab_size": 8, "num_hidden_layers": layers, "max_position_embeddings": 262144,
            "intermediate_size": 21504, "num_attention_heads": 32, "num_key_value_heads": 16,
            "num_global_key_value_heads": 4, "head_dim": 256, "global_head_dim": 512,
            "layer_types": types[:layers], "sliding_window": 1024, "hidden_activation": "gelu_pytorch_tanh",
            "rms_norm_eps": 1e-6, "final_logit_softcapping": 30.0, "tie_word_embeddings": True,
            "rope_parameters": {
                "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
                "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
            },
        },
    }


def make_model(path: Path, *, dtype: str = "BF16", config: dict | None = None) -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps(config or gemma_config()))
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.language_model.norm.weight": "model-00001-of-00001.safetensors"}})
    )
    write_safetensors(path / "model-00001-of-00001.safetensors", dtype)
    (path / "tokenizer.json").write_text("{}")
    return path


def make_assistant(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps({"architectures": ["Gemma4AssistantForCausalLM"],
                                                  "backbone_hidden_size": 5376}))
    (path / "model.safetensors").write_bytes(b"x")
    return path


class FakeProbes:
    """Machine probes with healthy defaults; tests override one fact at a time."""

    def __init__(self, **over):
        self.facts = {
            "gpu": checks.GpuInfo("NVIDIA GeForce RTX 5090", 32607, 30000, "12.0"),
            "ram": (64 * 2**30, 60 * 2**30),
            "disk": 500e9,
            "git": ("a" * 40, []),
            "gptq_versions": {"torch": "2.13.0+cu130", "transformers": "5.14.1", "llmcompressor": "0.13.0",
                              "compressed-tensors": "0.18.0"},
            "cuda": True,
            "formats": True,
            "dlls": [],
            "missing_packages": (),
        }
        self.facts.update(over)

    def gpu(self):
        return self.facts["gpu"]

    def ram(self):
        return self.facts["ram"]

    def disk_free(self, path):
        return self.facts["disk"]

    def device_of(self, path):
        return 1

    def python_info(self, python, packages, cuda=False):
        versions = {name: self.facts["gptq_versions"].get(name, "1.0") for name in packages}
        for name in self.facts["missing_packages"]:
            if name in versions:
                versions[name] = None
        return {"python": "3.13.1", "packages": versions, "cuda": self.facts["cuda"]}

    def git_state(self):
        return self.facts["git"]

    def tool_version(self, exe):
        return "version: test"

    def scan_binary(self, path, needles):
        return {needle: self.facts["formats"] for needle in needles}

    def missing_dlls(self, exe):
        return self.facts["dlls"]


class Env:
    """A tiny but valid set of inputs and fake tools under `tmp_path`."""

    def __init__(self, tmp_path: Path):
        self.root = tmp_path
        self.model = make_model(tmp_path / "model")
        self.assistant = make_assistant(tmp_path / "assistant")
        self.out = tmp_path / "out"
        self.server = tmp_path / "bin" / launcher.exe_name("ninfer-spo-serve")
        self.server.parent.mkdir()
        self.server.write_bytes(b"q4_g32_fp16 q4_g256_fp16 q4_g32_s6_fp16")
        self.perplexity = self.server.with_name(launcher.exe_name("ninfer-spo-perplexity"))
        self.perplexity.write_bytes(b"x")
        self.gptq_python = tmp_path / "gptq-venv" / "python.exe"
        self.gptq_python.parent.mkdir()
        self.gptq_python.write_bytes(b"x")
        self.llama = tmp_path / "llama.cpp"
        self.llama.mkdir()
        (self.llama / "convert_hf_to_gguf.py").write_text("")
        self.imatrix = tmp_path / "llama-imatrix.exe"
        self.imatrix.write_bytes(b"x")
        self.calibration = tmp_path / "calibration.txt"
        self.calibration.write_text("public text")

    def argv(self, preset: str, *extra: str, mtp: bool = True) -> list[str]:
        argv = ["--preset", preset, "--model", str(self.model), "--out-dir", str(self.out),
                "--server-exe", str(self.server), "--gptq-python", str(self.gptq_python),
                "--llama-cpp-dir", str(self.llama), "--llama-imatrix", str(self.imatrix),
                "--llama-python", PY, "--calibration-text", str(self.calibration)]
        argv += ["--assistant", str(self.assistant)] if mtp else ["--no-mtp"]
        return argv + list(extra)

    def config(self, preset: str, **over) -> build.BuildConfig:
        values = dict(
            preset=build.PRESETS[preset], model=self.model, assistant=self.assistant, with_vision=False,
            out_dir=self.out, work_dir=self.out / "work", name="m-" + preset, python=PY,
            gptq_python=self.gptq_python, llama_cpp_dir=self.llama, llama_imatrix=self.imatrix, llama_python=PY,
            calibration_text=self.calibration, perplexity_exe=self.perplexity,
        )
        values.update(over)
        return build.BuildConfig(**values)


@pytest.fixture
def env(tmp_path):
    return Env(tmp_path)


def serve_settings(env: Env, **over) -> build.ServeSettings:
    return build.ServeSettings(server_exe=env.server, **over)


def preflight(env: Env, cfg: build.BuildConfig, probes=None, **over) -> checks.Report:
    identity = build.build_identity(cfg)
    return checks.preflight(cfg, serve_settings(env, **over), env.server, probes or FakeProbes(),
                            build.fingerprint_of(identity), identity)


def finding(report: checks.Report, what: str) -> checks.Finding:
    matches = [item for item in report.findings if item.what == what]
    assert matches, f"no finding {what!r}: {[item.what for item in report.findings]}"
    return matches[0]


# ---------------------------------------------------------------- preset -> commands


def w(cfg: build.BuildConfig, *parts: str) -> str:
    return str(cfg.work(*parts))


def convert_expected(cfg: build.BuildConfig, *recipe: str, mtp: bool = True, text_only: bool = True) -> list[str]:
    argv = [PY, "-m", "tools.convert.gemma4_q4", "convert", "--model", str(cfg.model), "--weights", "q4_g32", *recipe]
    argv += ["--text-only"] if text_only else []
    argv += ["--mtp-assistant", str(cfg.assistant)] if mtp else ["--no-mtp"]
    return argv + ["--name", cfg.name, "--out", str(cfg.artifact)]


def plan_by_id(cfg: build.BuildConfig) -> dict[str, build.Step]:
    return {step.id: step for step in build.build_plan(cfg)}


def test_every_preset_has_the_documented_recipe_flags(env):
    expected = {
        "best-small": ("--gptq-source", "{w}/gptq", "--native-groups"),
        "best-medium": ("--gptq-source", "{w}/gptq"),
        "lite-medium": ("--scale-search", "weighted", "--imatrix", "{w}/model.imatrix.gguf",
                        "--scale-bits", "6", "--native-groups"),
        "lite-small": ("--scale-search", "weighted", "--imatrix", "{w}/model.imatrix.gguf",
                       "--group-size", "256", "--native-groups"),
        "draft": (),
    }
    assert set(expected) == set(build.PRESETS)
    for preset, flags in expected.items():
        cfg = env.config(preset, calibration_text=None)
        recipe = [flag.format(w=str(cfg.work_dir)).replace("/", "\\" if sys.platform == "win32" else "/")
                  for flag in flags]
        assert plan_by_id(cfg)["convert"].argv == convert_expected(cfg, *recipe), preset


def test_gptq_steps_are_the_documented_commands(env):
    for preset, group in (("best-small", "256"), ("best-medium", "32")):
        cfg = env.config(preset)
        steps = plan_by_id(cfg)
        gptq_dir = build.FORK_ROOT / "tools" / "gptq"
        gpy = str(env.gptq_python)
        tok = str(env.model / "tokenizer.json")
        assert list(steps) == ["calibration-render", "calibration-tokens", "text-tree", "gptq", "gptq-check",
                               "convert", "receipt-check", "finalize"]
        assert steps["calibration-render"].argv == [
            PY, "-m", "tools.convert.gemma4_q4", "calibration-text", "--tokenizer", tok, "--source",
            str(env.calibration), "--prefix-ids", PREFIX, "--chunks", "400", "--ctx-size", "512", "--out",
            w(cfg, "calibration", "templated.txt")]
        assert steps["calibration-tokens"].argv == [
            gpy, str(gptq_dir / "make_calib.py"), "--text", w(cfg, "calibration", "templated.txt"),
            "--tokenizer", tok, "--out", w(cfg, "calib.pt")]
        assert steps["text-tree"].argv == [
            gpy, str(gptq_dir / "make_text_tree.py"), "--model", str(env.model), "--out", w(cfg, "text60")]
        assert steps["gptq"].argv == [
            gpy, str(gptq_dir / "run_gptq.py"), "--tree", w(cfg, "text60"), "--calib", w(cfg, "calib.pt"),
            "--out", w(cfg, "gptq"), "--observer", "imatrix_mse", "--group-size", group, "--max-cpu-gb", "32",
            "--offload-dir", w(cfg, "offload"), "--peak-log", w(cfg, "gptq.peak.jsonl")]
        assert steps["gptq-check"].argv == [
            gpy, str(gptq_dir / "check_checkpoint.py"), "--gptq", w(cfg, "gptq"), "--tree", w(cfg, "text60")]
        assert steps["gptq"].prune_always == (cfg.work("offload"),)
        assert steps["gptq-check"].prune_if_short == (cfg.work("text60"),)


def test_rtn_steps_are_the_documented_commands_and_download_without_a_text_file(env):
    cfg = env.config("lite-medium", calibration_text=None)
    steps = plan_by_id(cfg)
    assert list(steps) == ["calibration-source", "calibration-render", "gguf", "imatrix", "convert",
                           "receipt-check", "finalize"]
    assert build.CALIBRATION_URL in steps["calibration-source"].note
    assert steps["gguf"].argv == [PY, str(env.llama / "convert_hf_to_gguf.py"), str(env.model), "--outtype", "q8_0",
                                  "--outfile", w(cfg, "model-q8_0.gguf")]
    assert steps["imatrix"].argv == [
        str(env.imatrix), "-m", w(cfg, "model-q8_0.gguf"), "-f", w(cfg, "calibration", "templated.txt"),
        "--parse-special", "--ctx-size", "512", "--chunks", "400", "--gpu-layers", "auto", "-o",
        w(cfg, "model.imatrix.gguf")]
    assert steps["imatrix"].prune_if_short == (cfg.work("model-q8_0.gguf"),)
    assert build.CALIBRATION_REVISION in build.CALIBRATION_URL


def test_draft_is_a_plain_convert_and_no_mtp_vision_variants(env):
    cfg = env.config("draft", assistant=None, with_vision=True)
    steps = plan_by_id(cfg)
    assert list(steps) == ["convert", "receipt-check", "finalize"]
    assert steps["convert"].argv == convert_expected(cfg, mtp=False, text_only=False)


def test_score_steps_follow_section_six_of_the_guide(env):
    cfg = env.config("best-small", score=True)
    steps = plan_by_id(cfg)
    assert list(steps)[-6:] == ["score-corpus", "score-reference", "score-reference-write", "score-candidate",
                                "score-run", "score-report"]
    root = cfg.work("score")
    common = ["--corpus", str(root / "corpus" / "manifest.json"), "--context", "512", "--stride", "256",
              "--prefill-chunk", "128", "--prefix-ids", PREFIX]
    assert steps["score-reference-write"].argv == [
        str(env.perplexity), str(root / "ref-fp8row.ninfer")] + common + [
        "--headroom-mib", "128", "--kl-reference-write", str(root / "ref.nklref")]
    assert steps["score-run"].argv == [str(env.perplexity), str(root / "candidate.ninfer")] + common + [
        "--kl-reference", str(root / "ref.nklref"), "--output", str(root / "run")]
    assert steps["score-run"].env == {"NINFER_Q4_EXACT": "1"}
    twin = steps["score-candidate"].argv
    assert "--no-mtp" in twin and "--text-only" in twin and "--mtp-assistant" not in twin
    assert steps["score-reference"].argv[-4:-2] == ["--weights", "fp8_row"]
    assert steps["gptq-check"].prune_if_short == (cfg.work("text60"),)


def test_estimates_and_table_match_the_decided_presets():
    table = {p.id: (p.file_gb, p.kl, p.wait) for p in build.PRESETS.values()}
    assert table == {
        "best-small": (16.9, 0.1213, "about 30 min"), "best-medium": (18.4, 0.0845, "about 30 min"),
        "lite-medium": (17.5, 0.1543, "about 35 min"), "lite-small": (16.8, 0.2376, "about 35 min"),
        "draft": (18.4, 0.2654, "about 15 min"),
    }
    for preset in build.PRESETS.values():
        limit = {"gptq": 45, "rtn": 45, "plain": 20}[preset.method]
        assert sum(preset.minutes.values()) < limit


# ---------------------------------------------------------------- dry run through the command line


@pytest.mark.parametrize("preset", list(build.PRESETS))
def test_dry_run_prints_the_plan_for_every_preset_and_writes_nothing(env, capsys, preset):
    before = sorted(path for path in env.root.rglob("*"))
    code = cli.main(env.argv(preset, "--dry-run", "--yes"), probes=FakeProbes())
    out = capsys.readouterr().out
    assert code == 0
    assert "Dry run: nothing was written." in out and "[FAIL]" not in out
    assert build.PRESETS[preset].summary in out and "tools.convert.gemma4_q4 convert" in out
    assert " min  " in out
    assert sorted(path for path in env.root.rglob("*")) == before
    assert not env.out.exists()


def test_dry_run_exits_zero_even_when_checks_would_stop_a_real_run(env, capsys):
    probes = FakeProbes(gpu=None)
    assert cli.main(env.argv("draft", "--dry-run"), probes=probes) == 0
    out = capsys.readouterr().out
    assert "[FAIL] GPU" in out and "would stop a real run" in out


def test_check_exit_code_follows_the_checks(env, capsys):
    assert cli.main(env.argv("best-small", "--check"), probes=FakeProbes()) == 0
    assert "All checks passed." in capsys.readouterr().out
    assert cli.main(env.argv("best-small", "--check"), probes=FakeProbes(gpu=None)) == 1
    assert "1 check(s) failed." in capsys.readouterr().out


def test_real_run_stops_on_failed_checks_without_writing(env, capsys):
    assert cli.main(env.argv("draft", "--yes"), probes=FakeProbes(ram=(1, 1), gpu=None)) == 1
    assert not env.out.exists()
    assert "Not starting" in capsys.readouterr().err


def test_confirmation_is_required_without_yes(env, capsys, monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    assert cli.main(env.argv("draft"), probes=FakeProbes()) == 1
    assert "pass --yes" in capsys.readouterr().err
    assert not env.out.exists()
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert cli.main(env.argv("draft"), probes=FakeProbes()) == 1
    assert not env.out.exists()


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--alias", "bad alias"], "--alias"),
        (["--port", "70000"], "--port"),
        (["--host", 'a"b'], "--host"),
    ],
)
def test_argument_mistakes_are_refused_before_any_probe(env, capsys, extra, message):
    assert cli.main(env.argv("draft", "--dry-run", *extra), probes=FakeProbes()) == 1
    assert message in capsys.readouterr().err


def test_mtp_choice_is_required_and_exclusive(env, capsys):
    argv = env.argv("draft", "--no-mtp")  # assistant and --no-mtp together
    assert cli.main(argv, probes=FakeProbes()) == 1
    assert "--assistant" in capsys.readouterr().err
    neither = [a for a in env.argv("draft", mtp=False) if a != "--no-mtp"]
    assert cli.main(neither, probes=FakeProbes()) == 1
    assert "explicitly --no-mtp" in capsys.readouterr().err


def test_help_lists_every_preset_and_flag(capsys):
    with pytest.raises(SystemExit) as stop:
        cli.main(["--help"])
    assert stop.value.code == 0
    text = capsys.readouterr().out
    for token in (*build.PRESETS, "--assistant", "--no-mtp", "--out-dir", "--work-dir", "--llama-cpp-dir",
                  "--calibration-text", "--gptq-python", "--score", "--keep-work", "--yes", "--dry-run", "--check",
                  "setup-gptq", "launcher"):
        assert token in text
    assert "fast-" not in text
    assert "lite-medium or\nlite-small only if you cannot run GPTQ" in text


def test_module_entry_point_runs_from_the_repository_root():
    result = subprocess.run(
        [sys.executable, "-m", "tools.convert.gemma4_build", "--list-presets"], cwd=build.FORK_ROOT,
        capture_output=True, text=True, check=True,
    )
    assert "best-small" in result.stdout and "lite-medium" in result.stdout and "fast-" not in result.stdout


# ---------------------------------------------------------------- preflight with fake probes


def test_healthy_machine_passes_every_preset(env):
    for preset in build.PRESETS:
        report = preflight(env, env.config(preset))
        assert not report.failed, [item.render() for item in report.failed]


@pytest.mark.parametrize(
    "preset, probes, what, cause, action",
    [
        ("best-small", FakeProbes(gpu=checks.GpuInfo("RTX", 32607, 8000, "12.0")), "GPU memory", "needs 12 GiB",
         "nvidia-smi"),
        ("best-small", FakeProbes(gpu=None), "GPU", "no NVIDIA GPU", "driver"),
        ("best-small", FakeProbes(ram=(64 * 2**30, 20 * 2**30)), "RAM", "20 GiB available", "close other programs"),
        ("best-small", FakeProbes(disk=30e9), "disk, work dir and output (same drive)", "30 GB free", "--work-dir"),
        ("best-small", FakeProbes(git=("b" * 40, [" M tools/x.py"])), "ninfer.spo checkout", "tracked file",
         "commit or stash"),
        ("best-small", FakeProbes(formats=False), "ninfer-spo-serve", "does not know the q4_g256_fp16", "rebuild"),
        ("best-small", FakeProbes(cuda=False), "converter Python", "CUDA", "CUDA build of torch"),
        ("best-small", FakeProbes(missing_packages=("tokenizers",)), "converter Python packages", "tokenizers",
         "pip install tokenizers"),
        ("best-small", FakeProbes(gptq_versions={"transformers": "5.15.1"}), "GPTQ environment",
         "transformers 5.15.1 (pinned 5.14.1)", "setup-gptq"),
    ],
)
def test_preflight_failures_name_the_cause_and_the_next_action(env, preset, probes, what, cause, action):
    item = finding(preflight(env, env.config(preset), probes), what)
    assert item.level == "fail" and cause in item.detail + item.what and action in item.action


def test_comfortable_disk_threshold_warns_instead_of_failing(env):
    item = finding(preflight(env, env.config("best-small"), FakeProbes(disk=90e9)), "disk, work dir and output (same drive)")
    assert item.level == "warn" and "103 GB" in item.detail


def test_missing_gptq_environment_points_at_setup_gptq(env, tmp_path):
    cfg = env.config("best-medium", gptq_python=tmp_path / "missing" / "python.exe")
    item = finding(preflight(env, cfg), "GPTQ environment")
    assert item.level == "fail" and "setup-gptq --venv" in item.action


def test_rtn_presets_name_missing_llama_cpp_pieces(env):
    cfg = env.config("lite-small", llama_cpp_dir=None, llama_imatrix=None)
    report = preflight(env, cfg)
    assert "--llama-cpp-dir" in finding(report, "llama.cpp convert script").action
    assert "--llama-imatrix" in finding(report, "llama-imatrix").action
    wrong = preflight(env, env.config("lite-small"), FakeProbes(missing_packages=("gguf",)))
    assert "pip install gguf" in finding(wrong, "converter Python packages").action


def test_model_checks_reuse_the_converter_and_name_the_fix(tmp_path):
    for label, kwargs in (
        ("fp16", dict(dtype="F16")),
        ("moe", dict(config=gemma_config(hidden=2816))),
        ("short", dict(config=gemma_config(layers=30))),
    ):
        folder = make_model(tmp_path / label, **kwargs)
        with pytest.raises(ValueError):
            checks.inspect_model(folder)
    folder = make_model(tmp_path / "ok")
    facts = checks.inspect_model(folder)
    assert facts["shards"] == 1 and facts["vision"] is False
    (folder / "model-00001-of-00001.safetensors").unlink()
    with pytest.raises(ValueError, match="missing weight files"):
        checks.inspect_model(folder)


def test_bad_model_and_assistant_report_actions(env):
    (env.model / "config.json").write_text(json.dumps({"architectures": ["LlamaForCausalLM"]}))
    (env.assistant / "config.json").write_text(json.dumps({"architectures": ["Other"]}))
    report = preflight(env, env.config("draft"))
    assert "Hugging Face directory" in finding(report, "model").action
    assert "--no-mtp" in finding(report, "MTP assistant").action


def test_server_missing_dlls_and_missing_binary(env):
    warn = finding(preflight(env, env.config("draft"), FakeProbes(dlls=["avcodec-*.dll"])), "server DLLs")
    assert warn.level == "warn" and "--cuda-bin" in warn.action
    report = checks.preflight(env.config("draft"), build.ServeSettings(), None, FakeProbes(), "x", {})
    assert "--server-exe" in finding(report, "ninfer-spo-serve").action
    skipped = checks.preflight(env.config("draft"), build.ServeSettings(launcher=False), None, FakeProbes(), "x", {})
    assert finding(skipped, "launcher").level == "ok"


def test_score_prerequisites_are_checked(env):
    env.perplexity.unlink()
    cfg = env.config("draft", score=True)
    item = finding(preflight(env, cfg), "--score prerequisites")
    assert item.level == "fail" and "ninfer-spo-perplexity" in item.detail and "drop --score" in item.action


def test_out_and_work_dirs_must_stay_outside_the_source(env):
    cfg = env.config("draft", out_dir=env.model / "out", work_dir=env.model / "work")
    report = preflight(env, cfg)
    assert finding(report, "--out-dir").level == "fail" and finding(report, "--work-dir").level == "fail"


# ---------------------------------------------------------------- runner, resume, receipts


def write_receipt(cfg: build.BuildConfig, status: str = "storage-verified; runtime-and-quality-unverified") -> None:
    data = cfg.artifact.read_bytes()
    receipt = {"status": status, "revision": "r" * 40, "recipe": "test", "seconds": 1.0,
               "artifact": {"file_bytes": len(data), "file_sha256": {cfg.artifact.name: hashlib.sha256(data).hexdigest()}}}
    Path(str(cfg.artifact) + ".build.json").write_text(json.dumps(receipt))


class FakeRunner:
    """Stands in for the child processes: records commands and creates each step's declared outputs."""

    def __init__(self, cfg: build.BuildConfig, steps: list[build.Step], fail: set[str] = frozenset()):
        self.cfg, self.fail, self.calls = cfg, set(fail), []
        self.by_argv = {tuple(step.argv): step for step in steps if step.argv}

    def __call__(self, argv, log_path, env, cwd, status):
        step = self.by_argv[tuple(argv)]
        self.calls.append(step.id)
        log_path.write_text(f"ran {step.id}\n")
        if step.id in self.fail:
            return 3
        for path in step.outputs:
            if path.suffix in (".ninfer",):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"artifact bytes")
                write_receipt(self.cfg)
            elif path.name.endswith((".json", ".jsonl")) or path.suffix in (".txt", ".pt", ".gguf", ".nklref"):
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    path.write_text("x")
            elif not str(path).endswith((".build.json", ".conversion.json")):
                path.mkdir(parents=True, exist_ok=True)
                (path / "blob").write_bytes(b"x" * 10)
        return 0


def make_builder(env, cfg, *, fail=(), disk_free=None, out=None):
    steps = build.build_plan(cfg)
    runner = FakeRunner(cfg, steps, set(fail))
    identity = build.build_identity(cfg)
    messages = [] if out is None else out
    builder = build.Builder(
        cfg, steps, identity, serve_settings(env), server_exe=env.server, runner=runner, out=messages.append,
        disk_free=disk_free or (lambda path: 10**13), artifact_info=launcher.ArtifactInfo(True, False, 262144),
    )
    return builder, runner, messages


def test_failure_names_the_log_and_a_rerun_skips_finished_steps(env):
    cfg = env.config("best-small")
    builder, runner, messages = make_builder(env, cfg, fail={"gptq"})
    with pytest.raises(build.BuildError) as error:
        builder.run()
    assert "step gptq failed (exit code 3)" in str(error.value)
    assert str(cfg.work("logs", "04-gptq.log")) in str(error.value) and "rerun the same command" in str(error.value)
    assert runner.calls == ["calibration-render", "calibration-tokens", "text-tree", "gptq"]
    state = json.loads(cfg.work("build-state.json").read_text())
    assert state["steps"]["gptq"]["status"] == "failed" and state["steps"]["text-tree"]["status"] == "done"

    again, second, messages = make_builder(env, cfg)
    again.run()
    assert second.calls == ["gptq", "gptq-check", "convert"]
    assert sum("already done, skipped" in line for line in messages) == 3
    assert not cfg.work("offload").exists()  # scratch is removed as soon as GPTQ is done
    assert cfg.work("text60").exists()  # the tree stays while disk is plentiful


def test_a_third_run_after_success_skips_everything(env):
    cfg = env.config("draft")
    first, runner, _ = make_builder(env, cfg)
    first.run()
    second, rerun, messages = make_builder(env, cfg)
    second.run()
    assert runner.calls == ["convert"] and rerun.calls == []
    assert sum("already done" in line for line in messages) == 3


def test_deliverables_summary_and_work_dir_cleanup(env):
    cfg = env.config("best-small")
    builder, _, _ = make_builder(env, cfg)
    builder.run()
    assert builder.cleanup() is True and not cfg.work_dir.exists()
    names = sorted(path.name for path in env.out.iterdir())
    assert names == sorted([cfg.artifact.name, cfg.artifact.name + ".build.json",
                            cfg.artifact.name + ".conversion.json", f"serve-{cfg.name}.cmd", "BUILD-SUMMARY.md",
                            f"{cfg.name}.gemma4-build.json"])
    summary = (env.out / "BUILD-SUMMARY.md").read_text()
    digest = hashlib.sha256(b"artifact bytes").hexdigest()
    assert digest in summary and "best-small" in summary and "--spec mtp --draft-tokens 1" in summary
    assert "`131k`" in summary and "mean KL 0.1213" in summary
    receipt = json.loads((env.out / f"{cfg.name}.gemma4-build.json").read_text())
    assert receipt["schema"] == build.BUILD_SCHEMA and receipt["sha256"] == digest and receipt["mtp"] is True
    launcher_text = (env.out / f"serve-{cfg.name}.cmd").read_text()
    assert f'"{env.server}"' in launcher_text and "--model-id 131k" in launcher_text


def test_keep_work_and_protected_directories(env):
    cfg = env.config("draft", keep_work=True)
    builder, _, _ = make_builder(env, cfg)
    builder.run()
    assert builder.cleanup() is False and cfg.work_dir.exists()
    for bad in (env.out, env.out.parent, env.model):
        guarded = env.config("draft", work_dir=bad)
        other, _, _ = make_builder(env, guarded)
        other.state = {}
        with pytest.raises(build.BuildError, match="refusing to delete"):
            other.cleanup()


def test_a_work_dir_made_with_other_arguments_is_refused(env):
    cfg = env.config("best-small")
    builder, _, _ = make_builder(env, cfg, fail={"gptq"})
    with pytest.raises(build.BuildError):
        builder.run()
    other = env.config("best-medium", name=cfg.name)
    again, _, _ = make_builder(env, other)
    with pytest.raises(build.BuildError, match=r"different arguments \(.*preset"):
        again.run()
    assert "different arguments" in finding(preflight(env, other), "work dir").detail


def test_calibration_file_change_changes_the_identity(env):
    cfg = env.config("best-small")
    before = build.fingerprint_of(build.build_identity(cfg))
    env.calibration.write_text("other text")
    assert build.fingerprint_of(build.build_identity(cfg)) != before


def test_a_foreign_non_empty_work_dir_is_refused(env):
    cfg = env.config("draft")
    cfg.work_dir.mkdir(parents=True)
    (cfg.work_dir / "precious.txt").write_text("mine")
    builder, runner, _ = make_builder(env, cfg)
    with pytest.raises(build.BuildError, match="not a gemma4_build work dir"):
        builder.run()
    assert runner.calls == [] and (cfg.work_dir / "precious.txt").exists()


def test_an_existing_output_is_never_overwritten(env):
    cfg = env.config("draft")
    cfg.out_dir.mkdir(parents=True)
    cfg.artifact.write_bytes(b"someone else's file")
    builder, runner, _ = make_builder(env, cfg)
    with pytest.raises(build.BuildError, match="Output already exists"):
        builder.run()
    assert runner.calls == [] and cfg.artifact.read_bytes() == b"someone else's file"
    assert finding(preflight(env, cfg), "output").level == "fail"


def test_big_intermediates_are_removed_only_when_disk_is_short(env):
    cfg = env.config("best-small")
    short, _, messages = make_builder(env, cfg, disk_free=lambda path: 1e9)
    short.run()
    assert not cfg.work("text60").exists() and any("disk is short" in line for line in messages)
    roomy = env.config("best-small", out_dir=env.root / "out2", work_dir=env.root / "out2" / "work", name="roomy")
    ample, _, _ = make_builder(env, roomy, disk_free=lambda path: 10**13)
    ample.run()
    assert roomy.work("text60").exists()
    # a pruned tree does not make the next run redo the step that wrote it
    rerun, runner, _ = make_builder(env, cfg, disk_free=lambda path: 1e9)
    rerun.run()
    assert runner.calls == []


def test_receipt_check_refuses_an_unverified_or_mismatched_receipt(env):
    cfg = env.config("draft")
    builder, _, _ = make_builder(env, cfg)
    cfg.out_dir.mkdir(parents=True)
    cfg.artifact.write_bytes(b"abc")
    builder.open_work_dir()
    builder._log_path = None
    with pytest.raises(RuntimeError, match="receipt missing"):
        build.check_receipt(builder)
    write_receipt(cfg, status="failed")
    with pytest.raises(RuntimeError, match="status is 'failed'"):
        build.check_receipt(builder)
    write_receipt(cfg)
    cfg.artifact.write_bytes(b"abcd")
    with pytest.raises(RuntimeError, match="4 bytes"):
        build.check_receipt(builder)


def test_main_builds_end_to_end_with_a_fake_runner(env, capsys, monkeypatch):
    cfg = env.config("lite-small", name="m-lite-small", work_dir=env.out / "work-m-lite-small", python=sys.executable)
    plan = build.build_plan(cfg)
    runner = FakeRunner(cfg, plan)
    monkeypatch.setattr(cli, "Builder", partial(build.Builder, runner=runner, disk_free=lambda path: 10**13))
    monkeypatch.setattr(launcher, "artifact_info", lambda path: launcher.ArtifactInfo(True, False, 262144))
    argv = env.argv("lite-small", "--name", "m-lite-small", "--yes")
    assert cli.main(argv, probes=FakeProbes()) == 0
    out = capsys.readouterr().out
    assert "Done. Artifact:" in out and "Work dir removed." in out
    assert runner.calls == ["calibration-render", "gguf", "imatrix", "convert"]
    assert (env.out / "BUILD-SUMMARY.md").is_file() and not (env.out / "work-m-lite-small").exists()


# ---------------------------------------------------------------- child processes


def test_run_child_streams_to_the_log_and_returns_the_exit_code(tmp_path):
    log = tmp_path / "child.log"
    seen = []
    code = build.run_child([sys.executable, "-c", "print('hello'); import sys; print('oops', file=sys.stderr)"],
                           log, {}, tmp_path, seen.append)
    assert code == 0 and "hello" in log.read_text() and "oops" in log.read_text()
    assert build.run_child([sys.executable, "-c", "raise SystemExit(7)"], log, {}, tmp_path, seen.append) == 7
    assert build.tail_line(log).startswith("$ ")


def test_batch_files_run_through_cmd_on_windows(monkeypatch):
    monkeypatch.setattr(build.os, "name", "nt")
    assert build.popen_argv(["C:/x/tool.cmd", "a"]) == ["cmd", "/c", "C:/x/tool.cmd", "a"]
    assert build.popen_argv(["C:/x/tool.exe", "a"]) == ["C:/x/tool.exe", "a"]


# ---------------------------------------------------------------- calibration text


def test_calibration_export_reproduces_the_authors_file_layout(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    pa = pytest.importorskip("pyarrow")
    source = tmp_path / "c.parquet"
    content = "Don\u2019t panic.\nSecond line \u00e9\n"
    pq.write_table(pa.table({"content": [content]}), source)
    out = tmp_path / "source.txt"
    digest = build.export_author_text(source, out, expected_sha256=None)
    data = out.read_bytes()
    assert data.startswith(b"\xef\xbb\xbf") and b"\r\n" in data and data.endswith(b"\r\n")
    assert data.decode("utf-8-sig").replace("\r\n", "\n") == content.encode("utf-8").decode("cp437") + "\n"
    assert digest == hashlib.sha256(data).hexdigest()
    with pytest.raises(RuntimeError, match="expected"):
        build.export_author_text(source, tmp_path / "other.txt", expected_sha256="0" * 64)
    assert not (tmp_path / "other.txt").exists()


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_download_is_hash_pinned_and_never_silent(tmp_path, monkeypatch):
    payload = b"parquet bytes"
    monkeypatch.setattr(build.urllib.request, "urlopen", lambda request, timeout: FakeResponse(payload))
    out = tmp_path / "f.parquet"
    messages = []
    build.download_pinned("https://example.invalid/f", out, hashlib.sha256(payload).hexdigest(), messages.append)
    assert out.read_bytes() == payload and messages
    with pytest.raises(RuntimeError, match="refusing the download"):
        build.download_pinned("https://example.invalid/f", tmp_path / "bad", "0" * 64, messages.append)
    assert not (tmp_path / "bad").exists() and not (tmp_path / "bad.part").exists()

    def broken(request, timeout):
        raise OSError("network down")

    monkeypatch.setattr(build.urllib.request, "urlopen", broken)
    with pytest.raises(RuntimeError, match="--calibration-text"):
        build.download_pinned("https://example.invalid/f", tmp_path / "x", "0" * 64, messages.append)


def test_the_pinned_corpus_constants_are_the_authors():
    assert build.CALIBRATION_REVISION == "e87ed55dcba9d9c3a3e41539f3e728e981b1daa4"
    assert build.CALIBRATION_PARQUET_SHA256 == "8a804421c074713aaa537a35da8bdb12e93b94b44344c09a9942d047c25d6e03"
    assert build.CALIBRATION_TEXT_SHA256 == "82b7e7804f2486a34c23feef43c5356e542724102321cd53c52e2c4d0c316c8a"
    assert build.CALIBRATION_FILE in build.CALIBRATION_URL


# ---------------------------------------------------------------- launcher


REM_HEAD = [
    "@echo off",
    "setlocal DisableDelayedExpansion",
]
REM_MTP = {
    True: "rem Generated by python -m tools.convert.gemma4_build; qualified serve settings of the Gemma 4 31B guide; "
    "MTP assistant enabled.",
    False: "rem Generated by python -m tools.convert.gemma4_build; qualified serve settings of the Gemma 4 31B guide; "
    "no MTP assistant in this artifact.",
}
REM_MATCH = {
    "any": "rem The server accepts any API key (none is configured). --model-match any: OpenAI requests may send "
    "any model name.",
    "exact": "rem The server accepts any API key (none is configured). OpenAI endpoints answer only to --model-id.",
}
HOST_LINE = {
    "any": "  --host 127.0.0.1 --port 8080 --model-id 131k --model-match any ^",
    "exact": "  --host 127.0.0.1 --port 8080 --model-id 131k ^",
}




def test_launcher_defaults_to_model_match_any_and_exact_emits_no_flag():
    default = launcher.launcher_lines("S", "A", mtp=True, vision=False)
    assert any("--model-id 131k --model-match any" in line for line in default)
    exact = "\n".join(launcher.launcher_lines("S", "A", mtp=True, vision=False, model_match="exact"))
    assert "--model-match" not in exact
    with pytest.raises(ValueError, match="--model-match"):
        launcher.launcher_lines("S", "A", mtp=True, vision=False, model_match="none")


def test_launcher_settings_flow_into_the_text():
    lines = launcher.launcher_lines("S", "A", mtp=True, vision=False, host="0.0.0.0", port=8001, alias="160k",
                                    context=65536, cuda_bin="C:\\cuda\\bin")
    assert launcher.default_alias(131072) == "131k" and launcher.default_alias(262144) == "262k"


def test_serve_argv_matches_the_launcher_flags():
    argv = launcher.serve_argv(Path("S"), Path("A"), host="127.0.0.1", port=8080, alias="131k", context=131072,
                               kv_dtype="fp8", mtp=True, vision=True)
    assert argv[argv.index("--draft-tokens") + 1] == "1"
    exact = launcher.serve_argv(Path("S"), Path("A"), host="h", port=1, alias="a", context=2, kv_dtype="fp8",
                                mtp=False, vision=False, model_match="exact")
    assert "--model-match" not in exact

@pytest.mark.parametrize("draft", ["1", "2", "7", "auto"])
def test_launcher_explicit_drafts_and_mtp_disable(draft):
    argv = launcher.serve_argv(Path("S"), Path("A"), host="h", port=1, alias="a", context=2,
                               kv_dtype="fp8", mtp=True, vision=False, draft_tokens=draft)
    assert argv[argv.index("--draft-tokens") + 1] == draft
    lines = launcher.launcher_lines("S", "A", mtp=False, vision=False, draft_tokens=draft)
    assert not any("--draft-tokens" in line for line in lines)


@pytest.mark.parametrize("draft", ["0", "8", "-1", "six", ""])
def test_launcher_rejects_invalid_drafts(draft):
    with pytest.raises(ValueError, match="--draft-tokens must be"):
        launcher.launcher_lines("S", "A", mtp=True, vision=False, draft_tokens=draft)


def test_gemma4_qk_flag_is_emitted_only_when_requested():
    default = "\n".join(launcher.launcher_lines("S", "A", mtp=False, vision=False))
    assert "--gemma4-qk" not in default
    explicit_default = launcher.launcher_lines("S", "A", mtp=True, vision=True, gemma4_qk="bf16")
    assert explicit_default == launcher.launcher_lines("S", "A", mtp=True, vision=True)
    lines = launcher.launcher_lines("S", "A", mtp=False, vision=False, gemma4_qk="fp8")
    assert lines[4].startswith("rem Gemma attention QK in FP8") and lines[5] == '"S" "A" ^'
    assert lines[7] == ("  --max-context 131072 --kv-capacity auto --kv-headroom-mib 1024 --kv-dtype fp8 "
                        "--gemma4-qk fp8 ^")
    argv = launcher.serve_argv(Path("S"), Path("A"), host="h", port=1, alias="a", context=2, kv_dtype="fp8",
                               mtp=False, vision=False, gemma4_qk="fp8")
    assert argv[argv.index("--kv-dtype") + 1:argv.index("--kv-dtype") + 4] == ["fp8", "--gemma4-qk", "fp8"]


def test_gemma4_qk_fp8_is_refused_without_fp8_kv_and_bad_values_are_named():
    with pytest.raises(ValueError, match=r"--gemma4-qk fp8 needs --kv-dtype fp8 .*bf16 KV"):
        launcher.launcher_lines("S", "A", mtp=False, vision=False, kv_dtype="bf16", gemma4_qk="fp8")
    assert "--gemma4-qk" not in "\n".join(launcher.launcher_lines("S", "A", mtp=False, vision=False, kv_dtype="bf16"))
    with pytest.raises(ValueError, match="--gemma4-qk must be bf16 or fp8"):
        launcher.launcher_lines("S", "A", mtp=False, vision=False, gemma4_qk="int8")


@pytest.mark.parametrize("chunk", ["128", "256", "2048", "16384", "auto"])
def test_prefill_chunk_accepts_multiples_of_128_and_auto(chunk):
    lines = launcher.launcher_lines("S", "A", mtp=False, vision=False, prefill_chunk=chunk)
    assert f"  --max-concurrency 1 --prefill-chunk {chunk} ^" in lines


@pytest.mark.parametrize("chunk", ["", "0", "64", "100", "129", "16512", "-128", "Auto", "2048k"])
def test_prefill_chunk_refuses_everything_else(chunk):
    with pytest.raises(ValueError, match="--prefill-chunk must be 'auto' or a multiple of 128 in 128..16384"):
        launcher.launcher_lines("S", "A", mtp=False, vision=False, prefill_chunk=chunk)


def test_launcher_refuses_quotes_and_neutralizes_percent():
    with pytest.raises(ValueError, match="quotes"):
        launcher.cmd_path(Path('C:/a"b'))
    with pytest.raises(ValueError, match="quotes"):
        launcher.cmd_path(Path("C:/a\nb"))
    assert "%%PATH%%" in launcher.cmd_path(Path("C:/a%PATH%b"))
    for bad in ("bad alias", 'a"b', "%x%", "", "-lead"):
        with pytest.raises(ValueError, match="--alias"):
            launcher.check_serve_settings(host="127.0.0.1", port=1, alias=bad, context=1, kv_dtype="fp8")
    with pytest.raises(ValueError, match="--host"):
        launcher.check_serve_settings(host="a b", port=1, alias="x", context=1, kv_dtype="fp8")


def test_write_launcher_detects_components_and_never_overwrites(tmp_path):
    server = tmp_path / "s.exe"
    artifact = tmp_path / "a.ninfer"
    server.write_bytes(b"x")
    artifact.write_bytes(b"x")
    out = tmp_path / "serve.cmd"
    info = launcher.ArtifactInfo(mtp=True, vision=True, max_context=262144)
    launcher.write_launcher(out, artifact, server, info=info)
    data = out.read_bytes()
    assert b"\r\n" in data and b"--spec mtp --draft-tokens 1" in data and data.rstrip().endswith(b"%errorlevel%")
    assert b"--vision" in data and b"--model-id 131k" in data
    with pytest.raises(FileExistsError):
        launcher.write_launcher(out, artifact, server, info=info)
    with pytest.raises(ValueError, match="position limit"):
        launcher.write_launcher(tmp_path / "x.cmd", artifact, server, info=info, context=300000)
    off = tmp_path / "nomtp.cmd"
    launcher.write_launcher(off, artifact, server, info=info, no_mtp=True)
    assert b"--spec" not in off.read_bytes()


def test_launcher_subcommand(tmp_path, monkeypatch, capsys):
    server, artifact = tmp_path / "s.exe", tmp_path / "a.ninfer"
    server.write_bytes(b"x")
    artifact.write_bytes(b"x")
    monkeypatch.setattr(launcher, "artifact_info", lambda path: launcher.ArtifactInfo(False, False, 262144))
    out = tmp_path / "serve.cmd"
    code = cli.main(["launcher", "--artifact", str(artifact), "--server-exe", str(server), "--out", str(out),
                     "--port", "8001", "--alias", "262k", "--context", "262144"])
    assert code == 0 and "MTP off" in capsys.readouterr().out
    text = out.read_text()
    assert "--port 8001 --model-id 262k" in text and "--spec" not in text and "--max-context 262144" in text
    assert cli.main(["launcher", "--artifact", str(artifact), "--server-exe", str(server), "--out", str(out)]) == 1


def test_launcher_subcommand_model_match_choices(tmp_path, monkeypatch):
    server, artifact = tmp_path / "s.exe", tmp_path / "a.ninfer"
    server.write_bytes(b"x")
    artifact.write_bytes(b"x")
    monkeypatch.setattr(launcher, "artifact_info", lambda path: launcher.ArtifactInfo(False, False, 262144))
    base = ["launcher", "--artifact", str(artifact), "--server-exe", str(server)]
    assert cli.main(base + ["--out", str(tmp_path / "any.cmd")]) == 0
    assert cli.main(base + ["--out", str(tmp_path / "exact.cmd"), "--model-match", "exact"]) == 0
    assert "--model-match any" in (tmp_path / "any.cmd").read_text()
    assert "--model-match" not in (tmp_path / "exact.cmd").read_text()
    with pytest.raises(SystemExit):
        cli.main(base + ["--out", str(tmp_path / "bad.cmd"), "--model-match", "some"])


def test_launcher_subcommand_qk_and_prefill_chunk(tmp_path, monkeypatch, capsys):
    server, artifact = tmp_path / "s.exe", tmp_path / "a.ninfer"
    server.write_bytes(b"x")
    artifact.write_bytes(b"x")
    monkeypatch.setattr(launcher, "artifact_info", lambda path: launcher.ArtifactInfo(False, False, 262144))
    base = ["launcher", "--artifact", str(artifact), "--server-exe", str(server)]
    default = tmp_path / "default.cmd"
    assert cli.main(base + ["--out", str(default)]) == 0
    assert "--gemma4-qk" not in default.read_text() and "--prefill-chunk 128 ^" in default.read_text()
    fast = tmp_path / "fast.cmd"
    assert cli.main(base + ["--out", str(fast), "--gemma4-qk", "fp8", "--prefill-chunk", "auto"]) == 0
    assert "--kv-dtype fp8 --gemma4-qk fp8 ^" in fast.read_text() and "--prefill-chunk auto ^" in fast.read_text()
    refused = tmp_path / "refused.cmd"
    capsys.readouterr()
    assert cli.main(base + ["--out", str(refused), "--gemma4-qk", "fp8", "--kv-dtype", "bf16"]) == 1
    assert "--gemma4-qk fp8 needs --kv-dtype fp8" in capsys.readouterr().err and not refused.exists()
    assert cli.main(base + ["--out", str(refused), "--prefill-chunk", "100"]) == 1
    assert "multiple of 128" in capsys.readouterr().err and not refused.exists()


def test_build_summary_serve_command_carries_the_model_match_setting(env):
    cfg = env.config("draft")
    builder, _, _ = make_builder(env, cfg)
    builder.run()
    assert "--model-match any" in (env.out / "BUILD-SUMMARY.md").read_text()
    assert "--model-match any" in (env.out / f"serve-{cfg.name}.cmd").read_text()
    exact_cfg = env.config("draft", out_dir=env.root / "o2", work_dir=env.root / "o2" / "w", name="ex")
    steps = build.build_plan(exact_cfg)
    other = build.Builder(
        exact_cfg, steps, build.build_identity(exact_cfg), serve_settings(env, model_match="exact"),
        server_exe=env.server, runner=FakeRunner(exact_cfg, steps), out=lambda line: None,
        artifact_info=launcher.ArtifactInfo(True, False, 262144),
    )
    other.run()
    assert "--model-match" not in (env.root / "o2" / "BUILD-SUMMARY.md").read_text()
    assert "--model-match" not in (env.root / "o2" / "serve-ex.cmd").read_text()


def test_one_command_build_passes_qk_and_prefill_chunk_to_launcher_and_summary(env, capsys):
    cfg = env.config("draft")
    builder, _, _ = make_builder(env, cfg)
    builder.run()
    default = (env.out / f"serve-{cfg.name}.cmd").read_bytes()
    assert b"--gemma4-qk" not in default and b"--prefill-chunk 128 ^" in default
    fast_cfg = env.config("draft", out_dir=env.root / "o3", work_dir=env.root / "o3" / "w", name="fast")
    steps = build.build_plan(fast_cfg)
    build.Builder(
        fast_cfg, steps, build.build_identity(fast_cfg),
        serve_settings(env, gemma4_qk="fp8", prefill_chunk="auto"),
        server_exe=env.server, runner=FakeRunner(fast_cfg, steps), out=lambda line: None,
        artifact_info=launcher.ArtifactInfo(True, False, 262144),
    ).run()
    for text in ((env.root / "o3" / "serve-fast.cmd").read_text(), (env.root / "o3" / "BUILD-SUMMARY.md").read_text()):
        assert "--kv-dtype fp8 --gemma4-qk fp8" in text and "--prefill-chunk auto" in text
    args = cli.build_parser().parse_args(env.argv("draft", "--gemma4-qk", "fp8", "--prefill-chunk", "2048"))
    assert cli.resolve_config(args)[1].gemma4_qk == "fp8"
    for extra, message in ((("--gemma4-qk", "fp8", "--kv-dtype", "bf16"), "--gemma4-qk fp8 needs --kv-dtype fp8"),
                           (("--prefill-chunk", "100"), "--prefill-chunk must be")):
        assert cli.main(env.argv("draft", *extra, "--dry-run", "--yes"), probes=FakeProbes()) == 1
        assert message in capsys.readouterr().err


def test_find_server_exe_prefers_a_build_that_knows_the_format(tmp_path):
    fork = tmp_path / "fork"
    fork.mkdir()
    old, new = tmp_path / "build-spo-old" / "apps", tmp_path / "build-spo-new" / "apps"
    for folder, payload in ((old, b"q4_g256_fp16"), (new, b"q4_g32_fp16 only")):
        folder.mkdir(parents=True)
        (folder / launcher.exe_name("ninfer-spo-serve")).write_bytes(payload)
    import os

    os.utime(old / launcher.exe_name("ninfer-spo-serve"), (1, 1))
    assert launcher.find_server_exe(fork) == new / launcher.exe_name("ninfer-spo-serve")
    assert launcher.find_server_exe(fork, ("q4_g256_fp16",)) == old / launcher.exe_name("ninfer-spo-serve")
    assert launcher.find_server_exe(tmp_path / "a" / "nowhere") is None


def test_missing_dlls_lists_patterns_not_found(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "")
    exe = tmp_path / "s.exe"
    exe.write_bytes(b"x")
    (tmp_path / "avcodec-63.dll").write_bytes(b"x")
    missing = launcher.missing_dlls(exe)
    assert "avcodec-*.dll" not in missing and "cudart64_*.dll" in missing


# ---------------------------------------------------------------- setup-gptq


def test_setup_gptq_dry_run_prints_the_documented_commands(tmp_path, capsys):
    code = cli.main(["setup-gptq", "--venv", str(tmp_path / "v"), "--dry-run"])
    out = capsys.readouterr().out
    assert code == 0 and not (tmp_path / "v").exists()
    assert "venv --python 3.13" in out and "https://download.pytorch.org/whl/cu130" in out
    assert "--extra-index-url https://pypi.org/simple" in out and "--index-strategy unsafe-best-match" in out
    assert "requirements.txt" in out and "-r" in out


def test_setup_gptq_needs_uv_and_verifies_versions(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    assert cli.main(["setup-gptq", "--venv", str(tmp_path / "v")]) == 1
    assert "uv is not installed" in capsys.readouterr().err
    monkeypatch.setattr(cli.shutil, "which", lambda name: "uv")
    calls = []
    monkeypatch.setattr(cli.subprocess, "run", lambda command, **kw: calls.append(command) or subprocess.CompletedProcess(command, 0))
    ok = FakeProbes()
    assert cli.main(["setup-gptq", "--venv", str(tmp_path / "v")], probes=ok) == 0
    assert calls[0][:2] == ["uv", "venv"] and calls[1][:3] == ["uv", "pip", "install"] and "--python" in calls[1]
    assert "Ready." in capsys.readouterr().out
    wrong = FakeProbes(gptq_versions={"transformers": "5.15.1"})
    assert cli.main(["setup-gptq", "--venv", str(tmp_path / "v")], probes=wrong) == 1
    assert "transformers 5.15.1 (pinned 5.14.1)" in capsys.readouterr().err
    assert cli.main(["setup-gptq", "--venv", str(tmp_path / "v")], probes=FakeProbes(cuda=False)) == 1
    assert "CUDA" in capsys.readouterr().err


def test_requirements_pins_are_parsed():
    pins = checks.parse_pins(build.FORK_ROOT / "tools" / "gptq" / "requirements.txt", checks.GPTQ_PINS)
    assert pins == {"torch": "2.13.0+cu130", "transformers": "5.14.1", "llmcompressor": "0.13.0",
                    "compressed-tensors": "0.18.0"}


# ---------------------------------------------------------------- serve-check


FAKE_SERVER_SCRIPT = '''
"""Fake ninfer-spo-serve for the serve-check tests: the OpenAI endpoints only, no GPU, no model."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

parser = argparse.ArgumentParser()
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--model-id", required=True)
parser.add_argument("--model-match", default="exact")
parser.add_argument("--max-context", type=int)
parser.add_argument("--kv-dtype")
parser.add_argument("--break-check", action="append", default=[])
args, _ = parser.parse_known_args()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/v1/models":
            data = [] if "models" in args.break_check else [
                {"id": args.model_id, "max_context": args.max_context, "kv_dtype": args.kv_dtype}]
            self._send(200, {"object": "list", "data": data})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except ValueError:
            body = {}
        if self.path != "/v1/chat/completions":
            self._send(404, {"error": "not found"})
        elif "chat" in args.break_check:
            self._send(500, {"error": "broken on purpose"})
        elif "bearer" in args.break_check and self.headers.get("Authorization"):
            self._send(401, {"error": "unauthorized"})
        elif "model" not in body:
            self._send(200 if "missing" in args.break_check else 400, {"error": "model is required"})
        elif args.model_match != "any" and body["model"] != args.model_id:
            self._send(404, {"error": {"message": "model '" + str(body["model"]) + "' not found",
                                       "code": "model_not_found"}})
        else:
            self._send(200, {"object": "chat.completion", "model": body["model"],
                             "choices": [{"index": 0, "message": {"role": "assistant", "content": "pong"},
                                          "finish_reason": "stop"}]})

    def log_message(self, *parts):
        pass


print("fake server on {}:{} model {} match {}".format(args.host, args.port, args.model_id, args.model_match),
      flush=True)
ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
'''


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def write_fake_launcher(tmp_path: Path, *, alias: str = "131k", match: str = "any", breaks: tuple = (),
                        host: str = "127.0.0.1") -> Path:
    """A .cmd shaped like a generated launcher whose server is the fake script under this Python."""
    script = tmp_path / "fake_server.py"
    script.write_text(FAKE_SERVER_SCRIPT, encoding="utf-8")
    match_flags = ["--model-match", "any"] if match == "any" else []
    break_flags = [word for name in breaks for word in ("--break-check", name)]
    lines = [
        "@echo off",
        "rem fake launcher for the serve-check tests",
        f'"{sys.executable}" "{script}" ^',
        "  --host {} --port {} --model-id {} {} ^".format(host, free_port(), alias, " ".join(match_flags)),
        "  --max-context 131072 --kv-dtype fp8 --spec mtp " + " ".join(break_flags),
        "exit /b %errorlevel%",
        "",
    ]
    cmd = tmp_path / "serve-m.cmd"
    cmd.write_text("\r\n".join(lines), encoding="utf-8", newline="")
    return cmd


def process_alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def test_parse_launcher_reads_a_generated_cmd(tmp_path):
    cmd = tmp_path / "serve.cmd"
    cmd.write_text("\r\n".join(launcher.launcher_lines(
        "C:/bin/ninfer-spo-serve.exe", "D:/ninfer/a.ninfer", alias="131k", mtp=True, vision=False)),
        encoding="utf-8", newline="")
    plan = build.parse_launcher(cmd)
    assert plan.server == Path("C:/bin/ninfer-spo-serve.exe")
    assert plan.artifact == Path("D:/ninfer/a.ninfer")
    assert (plan.host, plan.port, plan.alias, plan.model_match, plan.kv_dtype, plan.context) == \
        ("127.0.0.1", 8080, "131k", "any", "fp8", 131072)
    assert "--spec" in plan.flags and "mtp" in plan.flags


def test_serve_check_help_prints_usage(capsys):
    with pytest.raises(SystemExit) as stop:
        cli.main(["serve-check", "--help"])
    assert stop.value.code == 0
    text = capsys.readouterr().out
    assert text.startswith("usage:") and "--launcher" in text and "--out" in text


def test_serve_check_passes_every_check_and_writes_a_receipt(tmp_path, capsys):
    cmd = write_fake_launcher(tmp_path)
    out = tmp_path / "receipt.json"
    assert cli.main(["serve-check", "--launcher", str(cmd), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert text.count("PASS") == 6 and "FAIL" not in text and f"Receipt: {out}" in text
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema"] == build.SERVE_CHECK_SCHEMA
    assert [item["status"] for item in data["checks"]] == ["pass"] * 6
    assert data["checks"][0]["name"] == "ready" and data["checks"][0]["http_code"] == 200
    names = {item["name"]: item for item in data["checks"]}
    assert names["chat with another model name"]["http_code"] == 200
    assert names["chat without a model field"]["http_code"] == 400
    assert names["chat with an arbitrary bearer token"]["http_code"] == 200
    assert data["launcher"].endswith("serve-m.cmd") and data["url"].startswith("http://127.0.0.1:")
    assert data["artifact"]["size"] == (tmp_path / "fake_server.py").stat().st_size
    assert data["started"].endswith("Z") and data["finished"] >= data["started"]
    assert data["engine"] == {"model_id": "131k", "model_match": "any", "context": 131072, "kv_dtype": "fp8",
                              "kv_line": ""}


def test_serve_check_probes_loopback_when_the_launcher_binds_all_interfaces(tmp_path, capsys):
    cmd = write_fake_launcher(tmp_path, host="0.0.0.0")
    out = tmp_path / "receipt.json"
    assert cli.main(["serve-check", "--launcher", str(cmd), "--out", str(out)]) == 0
    assert "FAIL" not in capsys.readouterr().out
    assert json.loads(out.read_text(encoding="utf-8"))["url"].startswith("http://127.0.0.1:")


def test_serve_check_exact_match_wants_404_for_another_model_name(tmp_path):
    cmd = write_fake_launcher(tmp_path, match="exact")
    out = tmp_path / "receipt.json"
    assert cli.main(["serve-check", "--launcher", str(cmd), "--out", str(out)]) == 0
    checks = {item["name"]: item for item in json.loads(out.read_text(encoding="utf-8"))["checks"]}
    assert checks["chat with another model name"]["http_code"] == 404
    assert checks["chat with another model name"]["status"] == "pass"


def test_serve_check_failure_exits_nonzero_and_names_the_check(tmp_path, capsys):
    cmd = write_fake_launcher(tmp_path, breaks=("chat",))
    out = tmp_path / "receipt.json"
    assert cli.main(["serve-check", "--launcher", str(cmd), "--out", str(out)]) == 1
    captured = capsys.readouterr()
    assert "FAIL chat with the launcher model id" in captured.out
    assert "-> the server refuses its own --model-id" in captured.out
    assert "serve-check failed: chat with the launcher model id" in captured.err
    assert json.loads(out.read_text(encoding="utf-8"))["checks"][2]["status"] == "fail"


def test_serve_check_refuses_a_missing_launcher(tmp_path, capsys):
    assert cli.main(["serve-check", "--launcher", str(tmp_path / "nope.cmd")]) == 1
    assert "launcher not found" in capsys.readouterr().err


def test_serve_check_stops_the_child_process(tmp_path):
    cmd = write_fake_launcher(tmp_path)
    out = tmp_path / "receipt.json"
    assert cli.main(["serve-check", "--launcher", str(cmd), "--out", str(out)]) == 0
    pid = json.loads(out.read_text(encoding="utf-8"))["pid"]
    deadline = time.monotonic() + 10
    while process_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not process_alive(pid), f"child {pid} still runs after serve-check"


# ---------------------------------------------------------------- docs stay in sync


def test_docs_preset_table_matches_the_presets_and_list_presets():
    # The llama-local-tooling repository keeps its own rendered copy of this table in its docs; that copy is NOT
    # this fork's to guard. Only docs/gemma4-31b-conversion.md here is checked against the builder.
    doc = (build.FORK_ROOT / "docs" / "gemma4-31b-conversion.md").read_text(encoding="utf-8")
    rows = [line for line in doc.splitlines() if line.startswith("| `")]
    rows = [line for line in rows if line.strip().strip("|").split("|")[0].strip().strip("`") in build.PRESETS]
    assert len(rows) == len(build.PRESETS)
    table = cli.preset_table()  # the data `--list-presets` prints
    for line, preset in zip(rows, build.PRESETS.values()):
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        assert cells[0] == f"`{preset.id}`", line
        assert cells[2] == f"{preset.file_gb:g} GB", line
        assert cells[4] == f"{preset.kl:.4f}", line
        assert cells[5] == preset.wait, line
        assert f"{preset.id:<12}" in table and f"{preset.file_gb:g} GB" in table
        assert f"{preset.kl:<7.4f}" in table and preset.wait in table
