import json

import pytest

import cfgtools
import sweep


def test_parse_grid_infers_types():
    g = sweep.parse_grid(["learning_rate=1e-5,1e-4", "lora_parameters.rank=8,16",
                          "num_layers=-1,8", "fine_tune_type=lora,dora"])
    assert g["learning_rate"] == [1e-5, 1e-4]
    assert g["lora_parameters.rank"] == [8, 16]
    assert g["num_layers"] == [-1, 8]
    assert g["fine_tune_type"] == ["lora", "dora"]


@pytest.mark.parametrize("bad", ["learning_rate", "=1,2", "k="])
def test_parse_grid_rejects_garbage(bad):
    with pytest.raises(SystemExit):
        sweep.parse_grid([bad])


def test_expand_is_cartesian_and_ordered():
    combos = sweep.expand({"a": [1, 2], "b": ["x", "y", "z"]})
    assert len(combos) == 6
    assert combos[0] == {"a": 1, "b": "x"} and combos[-1] == {"a": 2, "b": "z"}
    assert sweep.expand({}) == [{}]


def test_run_ids_unique_and_filesystem_safe():
    ids = [sweep.run_id(i, ov) for i, ov in enumerate(
        sweep.expand({"lora_parameters.rank": [8, 16], "learning_rate": [1e-4]}))]
    assert len(set(ids)) == 2
    assert all("/" not in i and " " not in i for i in ids)
    assert sweep.run_id(0, {}) == "00_base"


def test_build_run_config_isolation_and_schedule(tmp_path):
    base = {"learning_rate": 1e-4, "iters": 100, "adapter_path": "shared", "test": True,
            "lora_parameters": {"rank": 8, "scale": 20.0},
            "lr_schedule": {"name": "cosine_decay", "warmup": 10, "arguments": [1e-4, 90, 1e-6]}}
    cfg = sweep.build_run_config(base, {"learning_rate": 3e-4, "lora_parameters.rank": 32},
                                 tmp_path / "r0", iters=50)
    assert cfg["adapter_path"] == str(tmp_path / "r0" / "adapters")  # own dir per run
    assert cfg["test"] is False                                      # test split reserved
    assert cfg["lora_parameters"] == {"rank": 32, "scale": 20.0}
    assert cfg["lr_schedule"]["arguments"] == [3e-4, 40, 1e-6]       # synced to lr + iters
    assert base["adapter_path"] == "shared" and base["iters"] == 100  # base untouched


def test_rank_orders_by_loss_and_sinks_failures():
    rs = [{"run": "a", "tuned_loss": 2.0}, {"run": "b", "tuned_loss": None},
          {"run": "c", "tuned_loss": 1.0}]
    assert [r["run"] for r in sweep.rank(rs)] == ["c", "a", "b"]


def test_table_and_csv(tmp_path):
    ranked = [{"run": "r1", "overrides": {"lr": 1}, "tuned_loss": 1.0, "base_loss": 2.0,
               "tuned_f1": 0.4},
              {"run": "r2", "overrides": {"lr": 2}, "tuned_loss": None}]
    md = sweep.render_table(ranked, "valid")
    assert "| 1 | r1 |" in md and "-1.0000" in md and "FAILED" in md
    sweep.write_results(tmp_path, ranked, "valid")
    assert (tmp_path / "results.csv").read_text().startswith("run,overrides,")
    assert "valid loss" in (tmp_path / "results.md").read_text()


def test_read_report(tmp_path):
    p = tmp_path / "report.json"
    p.write_text(json.dumps({"tuned": {"loss": 1.5, "gen": {"token_f1": 0.3}},
                             "base": {"loss": 2.5, "gen": {"n": 0}}}))
    assert sweep.read_report(p) == {"tuned_loss": 1.5, "base_loss": 2.5, "tuned_f1": 0.3}


def test_dry_run_prints_plan_and_touches_nothing(tmp_path, capsys):
    cfgp = tmp_path / "c.yaml"
    cfgp.write_text("data: nowhere\nlearning_rate: 1e-4\niters: 10\n")
    rc = sweep.main(["--config", str(cfgp), "--grid", "learning_rate=1e-5,1e-4",
                     "--grid", "lora_parameters.rank=8,16", "--dry-run",
                     "--out-root", str(tmp_path / "out")])
    out = capsys.readouterr().out
    assert rc == 0 and "4 run(s)" in out
    assert not (tmp_path / "out").exists()


def test_max_runs_guard(tmp_path):
    cfgp = tmp_path / "c.yaml"
    cfgp.write_text("data: nowhere\n")
    with pytest.raises(SystemExit, match="max-runs"):
        sweep.main(["--config", str(cfgp), "--grid", "a=1,2,3", "--grid", "b=1,2,3",
                    "--max-runs", "5", "--dry-run"])


def test_refuses_to_run_without_a_validation_split(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    (d / "train.jsonl").write_text("{}\n")
    cfgp = tmp_path / "c.yaml"
    cfgp.write_text(f"data: {d}\n")
    with pytest.raises(SystemExit, match="valid.jsonl"):
        sweep.main(["--config", str(cfgp), "--grid", "a=1", "--out-root", str(tmp_path / "o")])
