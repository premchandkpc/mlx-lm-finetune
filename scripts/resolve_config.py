#!/usr/bin/env python3
"""Write a resolved copy of a training config and print its path.

Applies the CLI overrides that influence the LR schedule (--iters,
--learning-rate) *before* training, so `lr_schedule` stays in sync. Everything
else on the command line is still passed straight to mlx_lm.lora by train.sh.

Usage (from scripts/train.sh):
  CFG=$(python scripts/resolve_config.py configs/lora.yaml -- --iters 200)
  mlx_lm.lora --config "$CFG" --iters 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cfgtools

# CLI flag -> config key, for flags that feed the schedule or the epoch math.
FLAG_TO_KEY = {
    "--iters": "iters",
    "--learning-rate": "learning_rate",
    "--batch-size": "batch_size",
    "--grad-accumulation-steps": "grad_accumulation_steps",
}


def parse_cli_overrides(extra: list[str]) -> dict:
    out: dict = {}
    i = 0
    while i < len(extra):
        a = extra[i]
        if "=" in a and a.split("=", 1)[0] in FLAG_TO_KEY:
            flag, raw = a.split("=", 1)
            out[FLAG_TO_KEY[flag]] = cfgtools.parse_value(raw)
        elif a in FLAG_TO_KEY and i + 1 < len(extra):
            out[FLAG_TO_KEY[a]] = cfgtools.parse_value(extra[i + 1])
            i += 1
        i += 1
    return out


def resolve(config: str, extra: list[str]) -> dict:
    cfg = cfgtools.load_config(config)
    cfg = cfgtools.apply_overrides(cfg, parse_cli_overrides(extra))
    return cfgtools.sync_schedule(cfg)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra: list[str] = []
    if "--" in argv:
        k = argv.index("--")
        argv, extra = argv[:k], argv[k + 1:]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("--out", default="outputs/.resolved_config.yaml")
    ap.add_argument("--quiet", action="store_true", help="suppress preflight warnings")
    args = ap.parse_args(argv)

    cfg = resolve(args.config, extra)
    cfgtools.dump_config(cfg, args.out)
    if not args.quiet:
        for msg in cfgtools.preflight(cfg):
            print(f"WARNING: {msg}", file=sys.stderr)
    print(Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
