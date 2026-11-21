import json
from pathlib import Path

import pytest
import yaml

import cfgtools
import resolve_config

ROOT = Path(__file__).resolve().parent.parent


def test_pyyaml_really_parses_1e4_as_string():
    """The trap cfgtools.coerce exists for. If PyYAML ever fixes this, drop coerce."""
    assert isinstance(yaml.safe_load("lr: 1e-4")["lr"], str)
    assert cfgtools.coerce(yaml.safe_load("lr: 1e-4"))["lr"] == 1e-4


def test_coerce_is_selective():
    out = cfgtools.coerce({"a": "1e-4", "b": "mlx-community/x", "c": ["2E3", "abc"], "d": 5})
    assert out == {"a": 1e-4, "b": "mlx-community/x", "c": [2000.0, "abc"], "d": 5}


def test_parse_value():
    assert cfgtools.parse_value("1e-4") == 1e-4
    assert cfgtools.parse_value("16") == 16 and isinstance(cfgtools.parse_value("16"), int)
    assert cfgtools.parse_value("-1") == -1
    assert cfgtools.parse_value("true") is True
    assert cfgtools.parse_value("dora") == "dora"


def test_deep_set_and_overrides_do_not_mutate():
    cfg = {"lora_parameters": {"rank": 8, "scale": 20.0}}
    out = cfgtools.apply_overrides(cfg, {"lora_parameters.rank": 32, "x.y.z": 1})
    assert out["lora_parameters"] == {"rank": 32, "scale": 20.0}
    assert out["x"]["y"]["z"] == 1
    assert cfg == {"lora_parameters": {"rank": 8, "scale": 20.0}}


def test_deep_merge_replaces_lists_merges_dicts():
    base = {"a": {"b": 1, "c": 2}, "l": [1, 2, 3]}
    out = cfgtools.deep_merge(base, {"a": {"c": 9}, "l": [7]})
    assert out == {"a": {"b": 1, "c": 9}, "l": [7]}


def test_extends_chain_and_relative_paths(tmp_path):
    (tmp_path / "base.yaml").write_text("iters: 100\nlora_parameters: {rank: 8, scale: 20.0}\n")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "mid.yaml").write_text("extends: ../base.yaml\nlearning_rate: 1e-4\n")
    (sub / "leaf.yaml").write_text("extends: mid.yaml\nlora_parameters: {rank: 32}\n")
    cfg = cfgtools.load_config(sub / "leaf.yaml")
    assert cfg == {"iters": 100, "learning_rate": 1e-4,
                   "lora_parameters": {"rank": 32, "scale": 20.0}}
    assert "extends" not in cfg


def test_extends_cycle_detected(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n")
    with pytest.raises(ValueError, match="circular"):
        cfgtools.load_config(tmp_path / "a.yaml")


def test_all_shipped_presets_load_and_are_sane():
    for p in sorted((ROOT / "configs").glob("*.yaml")):
        cfg = cfgtools.load_config(p)
        assert "extends" not in cfg
        assert cfg["model"] and cfg["data"] and cfg["adapter_path"]
        assert isinstance(cfg["learning_rate"], float), f"{p.name}: LR must be a float"
        assert cfg["mask_prompt"] is True
        if "lr_schedule" in cfg:  # the base recipe runs at a constant LR
            synced = cfgtools.sync_schedule(cfg)["lr_schedule"]
            assert synced["arguments"][0] == cfg["learning_rate"]
            assert synced["arguments"][1] == max(1, cfg["iters"] - cfg["lr_schedule"]["warmup"])
            assert synced["name"]


def test_presets_use_distinct_adapter_dirs():
    paths = {p.name: cfgtools.load_config(p)["adapter_path"]
             for p in (ROOT / "configs").glob("*.yaml")}
    assert len(set(paths.values())) == len(paths), paths  # no preset clobbers another


def test_sync_schedule_tracks_iters_and_lr():
    cfg = {"learning_rate": 2e-4, "iters": 200,
           "lr_schedule": {"name": "cosine_decay", "warmup": 20,
                           "arguments": [1e-4, 500, 1e-6]}}
    out = cfgtools.sync_schedule(cfg)
    assert out["lr_schedule"]["arguments"] == [2e-4, 180, 1e-6]
    assert cfg["lr_schedule"]["arguments"] == [1e-4, 500, 1e-6]  # input untouched


def test_sync_schedule_noop_without_schedule_and_never_zero_steps():
    assert cfgtools.sync_schedule({"iters": 5}) == {"iters": 5}
    out = cfgtools.sync_schedule({"learning_rate": 1e-4, "iters": 3,
                                  "lr_schedule": {"warmup": 10, "arguments": [1e-4, 9, 0]}})
    assert out["lr_schedule"]["arguments"][1] == 1


def test_parse_cli_overrides_both_spellings():
    ov = resolve_config.parse_cli_overrides(
        ["--iters", "50", "--learning-rate=2e-4", "--model", "x", "--batch-size", "2"])
    assert ov == {"iters": 50, "learning_rate": 2e-4, "batch_size": 2}


def test_resolve_applies_cli_to_schedule(tmp_path):
    cfgp = tmp_path / "c.yaml"
    cfgp.write_text(yaml.safe_dump({
        "learning_rate": 1e-4, "iters": 500,
        "lr_schedule": {"name": "cosine_decay", "warmup": 10, "arguments": [1e-4, 490, 1e-6]}}))
    out = resolve_config.resolve(str(cfgp), ["--iters", "100"])
    assert out["lr_schedule"]["arguments"] == [1e-4, 90, 1e-6]


def test_resolve_cli_main_prints_only_path(tmp_path, capsys):
    cfgp = tmp_path / "c.yaml"
    cfgp.write_text("iters: 10\nlearning_rate: 1e-4\ndata: nowhere\n")
    out = tmp_path / "r" / "c.yaml"
    assert resolve_config.main([str(cfgp), "--out", str(out), "--", "--iters", "20"]) == 0
    assert capsys.readouterr().out.strip() == str(out)  # shell captures stdout: keep it clean
    assert cfgtools.load_config(out)["iters"] == 20


# ---------- preflight ----------

def _data(tmp_path, n, row):
    d = tmp_path / "data"
    d.mkdir()
    (d / "train.jsonl").write_text("\n".join(json.dumps(row) for _ in range(n)) + "\n")
    return str(d)


def test_epoch_estimate():
    assert cfgtools.epoch_estimate({"iters": 500, "batch_size": 4,
                                    "grad_accumulation_steps": 2}, 10) == 400
    assert cfgtools.epoch_estimate({"iters": 100, "batch_size": 4}, 400) == 1


def test_preflight_flags_memorization(tmp_path):
    msgs = cfgtools.preflight({"data": _data(tmp_path, 10, {"messages": []}), "iters": 500,
                               "batch_size": 4, "mask_prompt": True})
    assert any("memorize" in m for m in msgs)


def test_preflight_quiet_on_healthy_config(tmp_path):
    msgs = cfgtools.preflight({"data": _data(tmp_path, 1000, {"messages": []}), "iters": 500,
                               "batch_size": 4, "mask_prompt": True})
    assert msgs == []


def test_preflight_mask_prompt_and_under_one_epoch(tmp_path):
    msgs = cfgtools.preflight({"data": _data(tmp_path, 1000, {"messages": []}), "iters": 10,
                               "batch_size": 4})
    assert any("mask_prompt" in m for m in msgs)
    assert any("not see every example" in m for m in msgs)


def test_preflight_full_ft_on_quantized():
    msgs = cfgtools.preflight({"fine_tune_type": "full", "model": "x/Qwen-0.5B-4bit"})
    assert any("non-quantized" in m for m in msgs)


def test_preflight_never_raises_without_data():
    assert cfgtools.preflight({}) == []
    assert cfgtools.preflight({"data": "mlx-community/wikisql"}) == []
