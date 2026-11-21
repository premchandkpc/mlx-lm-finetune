"""Real end-to-end run on a tiny random-weight model (CPU, offline, ~1 minute).

Exercises prepare_data -> mlx_lm.lora -> compare -> sweep with the actual mlx-lm.
Skipped automatically when mlx / mlx-lm are not importable.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
pytest.importorskip("tokenizers")

import cfgtools
import compare
import prepare_data
import sweep
from make_tiny_model import make_tiny_model

ROOT = Path(__file__).resolve().parent.parent


def train(cfgp):
    r = subprocess.run([sys.executable, "-m", "mlx_lm", "lora", "--config", str(cfgp)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("e2e")
    model = make_tiny_model(tmp / "tiny", ROOT / "data/sample")
    data = tmp / "data"
    assert prepare_data.main(["--input", str(ROOT / "data/sample"), "--output", str(data),
                              "--format", "chat", "--test-split", "0.05"]) == 0
    cfg = {
        "model": str(model), "train": True, "data": str(data), "optimizer": "adamw",
        "learning_rate": 1e-4, "iters": 8, "batch_size": 4, "val_batches": 1,
        "steps_per_report": 4, "steps_per_eval": 4, "save_every": 8, "num_layers": 2,
        "lora_parameters": {"rank": 4, "scale": 20.0, "dropout": 0.0},
        "max_seq_length": 256, "mask_prompt": True, "seed": 0,
        "lr_schedule": {"name": "cosine_decay", "warmup": 2, "warmup_init": 1e-7,
                        "arguments": [1e-4, 6, 1e-6]},
        "adapter_path": str(tmp / "adapters"),
    }
    cfgp = tmp / "cfg.yaml"
    cfgtools.dump_config(cfg, cfgp)
    train(cfgp)  # every test starts from a trained adapter, in any order
    return tmp, cfgp


def test_train_produces_the_adapter_files_the_tooling_expects(env):
    tmp, cfgp = env
    ad = tmp / "adapters"
    assert (ad / "adapter_config.json").exists()
    # the real mlx-lm filename is `adapters.safetensors` (plural), not `adapter.safetensors`
    assert (ad / "adapters.safetensors").exists()
    compare.check_adapter(str(ad))


def test_compare_runs_and_trained_beats_base_on_valid(env):
    tmp, cfgp = env
    out = tmp / "cmp"
    assert compare.main(["--config", str(cfgp), "--split", "valid", "--gen-limit", "2",
                         "--max-tokens", "5", "--out", str(out)]) == 0
    rep = json.loads((out / "report.json").read_text())
    assert rep["tuned"]["loss"] < rep["base"]["loss"]
    assert (out / "report.md").read_text().startswith("# Base vs tuned")
    assert len(rep["samples"]) == 2


def test_compare_loss_matches_mlx_lm_own_test(env):
    """compare.py must measure the same thing `mlx_lm.lora --test` does."""
    tmp, cfgp = env
    cfg = cfgtools.load_config(cfgp)
    r = subprocess.run(
        [sys.executable, "-m", "mlx_lm", "lora", "--model", cfg["model"], "--adapter-path",
         cfg["adapter_path"], "--data", cfg["data"], "--test", "--batch-size", "4",
         "--max-seq-length", "256", "--mask-prompt"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1500:]
    m = re.search(r"Test loss ([0-9.]+)", re.sub(r"\x1b\[[0-9;]*m", "", r.stdout + r.stderr))
    assert m, r.stdout[-800:]
    out = tmp / "cmp_test"
    compare.main(["--config", str(cfgp), "--split", "test", "--gen-limit", "0",
                  "--out", str(out)])
    ours = json.loads((out / "report.json").read_text())["tuned"]["loss"]
    assert ours == pytest.approx(float(m.group(1)), abs=1e-3)


def test_sweep_end_to_end(env, capsys):
    tmp, cfgp = env
    out_root = tmp / "sweeps"
    rc = sweep.main(["--config", str(cfgp), "--grid", "learning_rate=1e-5,3e-4",
                     "--name", "t", "--out-root", str(out_root), "--iters", "8"])
    assert rc == 0
    results = (out_root / "t" / "results.md").read_text()
    assert "valid loss" in results and results.count("| 1 |") == 1
    assert (out_root / "t" / "winner_test" / "report.json").exists()
    # LR 3e-4 should beat 1e-5 in 8 steps on this toy; ranking must reflect it
    first_row = [l for l in results.splitlines() if l.startswith("| 1 |")][0]
    assert "0.0003" in first_row
    # resume: nothing retrains
    sweep.main(["--config", str(cfgp), "--grid", "learning_rate=1e-5,3e-4",
                "--name", "t", "--out-root", str(out_root), "--iters", "8"])
    assert capsys.readouterr().out.count("already done") == 2
