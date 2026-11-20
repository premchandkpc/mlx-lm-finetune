#!/usr/bin/env python3
"""Hyperparameter ablation harness: train every grid point, rank, confirm the winner.

Each run gets its own resolved config and adapter dir (the LR schedule is kept
in sync with `iters`/`learning_rate`). Runs are ranked by held-out loss on the
VALIDATION split; only the winner is then evaluated on the TEST split, so test
numbers are never used to pick hyperparameters.

Usage:
  python scripts/sweep.py --config configs/lora.yaml \\
      --grid learning_rate=2e-5,5e-5,1e-4 --grid lora_parameters.rank=8,16
  python scripts/sweep.py --config configs/lora.yaml --grid num_layers=8,16,-1 --dry-run

Keys are dotted config paths. Re-running the same --name resumes: finished runs
(those with a report.json) are skipped.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import cfgtools

SCRIPTS = Path(__file__).resolve().parent


def parse_grid(items: list[str]) -> dict[str, list]:
    grid: dict[str, list] = {}
    for item in items:
        key, sep, vals = item.partition("=")
        if not sep or not key.strip() or not vals.strip():
            raise SystemExit(f"error: bad --grid '{item}', expected key=v1,v2,...")
        grid[key.strip()] = [cfgtools.parse_value(v.strip()) for v in vals.split(",")]
    return grid


def expand(grid: dict[str, list]) -> list[dict]:
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


def run_id(idx: int, overrides: dict) -> str:
    bits = [f"{k.split('.')[-1]}-{v}" for k, v in overrides.items()]
    slug = re.sub(r"[^A-Za-z0-9._-]+", "", "_".join(bits))
    return f"{idx:02d}_{slug}" if slug else f"{idx:02d}_base"


def build_run_config(base_cfg: dict, overrides: dict, run_dir: Path,
                     iters: int | None) -> dict:
    cfg = cfgtools.apply_overrides(base_cfg, overrides)
    if iters:
        cfg["iters"] = iters
    cfg["adapter_path"] = str(run_dir / "adapters")
    cfg["test"] = False  # test split is reserved for the winner's confirmation
    return cfgtools.sync_schedule(cfg)


def train_cmd(cfg_path: Path) -> list[str]:
    return [sys.executable, "-m", "mlx_lm", "lora", "--train", "--config", str(cfg_path)]


def compare_cmd(cfg_path: Path, out: Path, split: str, gen_limit: int) -> list[str]:
    return [sys.executable, str(SCRIPTS / "compare.py"), "--config", str(cfg_path),
            "--out", str(out), "--split", split, "--gen-limit", str(gen_limit)]


def rank(results: list[dict]) -> list[dict]:
    ok = [r for r in results if r.get("tuned_loss") is not None]
    bad = [r for r in results if r.get("tuned_loss") is None]
    return sorted(ok, key=lambda r: r["tuned_loss"]) + bad


def render_table(ranked: list[dict], select_split: str) -> str:
    lines = [f"| rank | run | overrides | {select_split} loss | vs base | token F1 |",
             "|---|---|---|---|---|---|"]
    for i, r in enumerate(ranked, 1):
        if r.get("tuned_loss") is None:
            lines.append(f"| - | {r['run']} | `{r['overrides']}` | FAILED | | |")
            continue
        delta = r["tuned_loss"] - r["base_loss"]
        f1 = f"{r['tuned_f1']:.3f}" if r.get("tuned_f1") is not None else ""
        lines.append(f"| {i} | {r['run']} | `{r['overrides']}` | "
                     f"{r['tuned_loss']:.4f} | {delta:+.4f} | {f1} |")
    return "\n".join(lines) + "\n"


def write_results(out: Path, ranked: list[dict], select_split: str) -> None:
    fields = ["run", "overrides", "tuned_loss", "base_loss", "tuned_f1"]
    with (out / "results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in ranked:
            w.writerow({**r, "overrides": json.dumps(r["overrides"])})
    (out / "results.md").write_text(render_table(ranked, select_split), encoding="utf-8")


def read_report(path: Path) -> dict:
    rep = json.loads(path.read_text(encoding="utf-8"))
    return {
        "tuned_loss": rep["tuned"]["loss"],
        "base_loss": rep["base"]["loss"],
        "tuned_f1": (rep["tuned"].get("gen") or {}).get("token_f1"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/lora.yaml")
    ap.add_argument("--grid", action="append", default=[],
                    help="dotted.key=v1,v2,... (repeatable; cartesian product)")
    ap.add_argument("--iters", type=int, default=None,
                    help="override iters for every run (keeps runs comparable)")
    ap.add_argument("--name", default=None, help="sweep name (default: timestamp)")
    ap.add_argument("--out-root", default="outputs/sweeps")
    ap.add_argument("--gen-limit", type=int, default=0,
                    help="generations per run for ranking (default 0 = loss only, fast)")
    ap.add_argument("--max-runs", type=int, default=24,
                    help="refuse grids larger than this (default 24)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = ap.parse_args(argv)

    base_cfg = cfgtools.load_config(args.config)
    combos = expand(parse_grid(args.grid))
    if len(combos) > args.max_runs:
        raise SystemExit(f"error: grid has {len(combos)} runs (> --max-runs "
                         f"{args.max_runs}). Shrink it or raise --max-runs.")

    data_dir = Path(base_cfg["data"])
    has_valid = (data_dir / "valid.jsonl").exists()
    has_test = (data_dir / "test.jsonl").exists()
    select_split = "valid" if has_valid else ("test" if has_test else None)
    if not args.dry_run and select_split is None:
        raise SystemExit(f"error: need {data_dir}/valid.jsonl (or test.jsonl) to rank runs")
    if select_split == "test":
        print("warning: no valid.jsonl; ranking on test.jsonl (selection will leak "
              "into the test estimate)", file=sys.stderr)

    out = Path(args.out_root) / (args.name or time.strftime("%Y%m%d-%H%M%S"))
    plan = []
    for i, ov in enumerate(combos):
        rid = run_id(i, ov)
        plan.append((rid, ov, out / rid))

    print(f"sweep: {len(plan)} run(s) -> {out}")
    for rid, ov, _ in plan:
        print(f"  {rid}: {ov or '(base config)'}")
    if args.dry_run:
        return 0

    out.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for n, (rid, ov, run_dir) in enumerate(plan, 1):
        run_dir.mkdir(parents=True, exist_ok=True)
        report_dir = run_dir / "compare"
        rep_path = report_dir / "report.json"
        entry = {"run": rid, "overrides": ov}
        if rep_path.exists():
            print(f"[{n}/{len(plan)}] {rid}: already done, skipping")
            entry.update(read_report(rep_path))
            results.append(entry)
            continue

        cfg = build_run_config(base_cfg, ov, run_dir, args.iters)
        cfg_path = run_dir / "config.yaml"
        cfgtools.dump_config(cfg, cfg_path)

        print(f"[{n}/{len(plan)}] {rid}: training ...", flush=True)
        with (run_dir / "train.log").open("w", encoding="utf-8") as log:
            rc = subprocess.run(train_cmd(cfg_path), stdout=log, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            print(f"    FAILED (exit {rc}); see {run_dir / 'train.log'}", file=sys.stderr)
            results.append(entry)
            continue

        print(f"[{n}/{len(plan)}] {rid}: evaluating on {select_split} ...", flush=True)
        with (run_dir / "compare.log").open("w", encoding="utf-8") as log:
            rc = subprocess.run(
                compare_cmd(cfg_path, report_dir, select_split, args.gen_limit),
                stdout=log, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            print(f"    eval FAILED (exit {rc}); see {run_dir / 'compare.log'}", file=sys.stderr)
            results.append(entry)
            continue
        entry.update(read_report(rep_path))
        results.append(entry)

    ranked = rank(results)
    write_results(out, ranked, select_split)
    print("\n" + render_table(ranked, select_split))

    winner = next((r for r in ranked if r.get("tuned_loss") is not None), None)
    if winner is None:
        print("error: every run failed", file=sys.stderr)
        return 1
    print(f"winner: {winner['run']}  {winner['overrides']}")

    if has_test and select_split == "valid":
        wdir = out / winner["run"]
        confirm = out / "winner_test"
        print(f"confirming winner on the held-out test split -> {confirm}/report.md")
        confirm.mkdir(parents=True, exist_ok=True)
        with (confirm / "compare.log").open("w", encoding="utf-8") as log:
            rc = subprocess.run(compare_cmd(wdir / "config.yaml", confirm, "test", 20),
                                stdout=log, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            print(f"test confirmation failed; see {confirm / 'compare.log'}", file=sys.stderr)
            return rc
        for line in json.loads((confirm / "report.json").read_text())["verdict"]:
            print(f"  test: {line}")
    print("\nNext: copy the winning overrides into configs/lora.yaml and train for real.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
