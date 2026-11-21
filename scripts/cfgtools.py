"""Small, dependency-light helpers for working with the training YAML.

Imported by resolve_config.py, compare.py and sweep.py (scripts/ is on
sys.path when those run). Pure Python + PyYAML so it is unit-testable anywhere.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import yaml

# PyYAML (YAML 1.1) parses `1e-4` as a *string*. mlx-lm patches its own loader to
# accept it; anything else that reads this config (this repo's tooling included)
# must do the same or it silently gets "1e-4" the string.
_SCI_FLOAT = re.compile(r"^[-+]?(?:\d+\.?\d*|\.\d+)[eE][-+]?\d+$")


def coerce(obj: Any) -> Any:
    """Recursively turn '1e-4'-style strings into floats."""
    if isinstance(obj, dict):
        return {k: coerce(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [coerce(v) for v in obj]
    if isinstance(obj, str) and _SCI_FLOAT.match(obj.strip()):
        return float(obj)
    return obj


def deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge; `override` wins. Lists are replaced, not concatenated."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path, _seen: tuple = ()) -> dict:
    """Load a YAML config, resolving an optional `extends: other.yaml` chain.

    `extends` is resolved relative to the file that names it. It is a feature of
    THIS repo's tooling (train.sh / compare.py / sweep.py resolve it); the merged
    result is what gets handed to mlx-lm, so never pass an `extends` preset to
    `mlx_lm.lora` directly.
    """
    path = Path(path)
    real = str(path.resolve())
    if real in _seen:
        raise ValueError(f"circular `extends` involving {path}")
    with open(path, encoding="utf-8") as f:
        cfg = coerce(yaml.safe_load(f) or {})
    parent = cfg.pop("extends", None)
    if parent:
        base = load_config(path.parent / parent, _seen + (real,))
        cfg = deep_merge(base, cfg)
    return cfg


def dump_config(cfg: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def parse_value(raw: str) -> Any:
    """Infer a Python value from a CLI string: 1e-4 -> float, 16 -> int, true -> bool."""
    val = yaml.safe_load(raw)
    return coerce(val)


def deep_set(cfg: dict, dotted_key: str, value: Any) -> None:
    """cfg['a']['b'] = value for 'a.b'; creates intermediate dicts."""
    parts = dotted_key.split(".")
    cur = cfg
    for p in parts[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[p] = nxt
        cur = nxt
    cur[parts[-1]] = value


def apply_overrides(cfg: dict, overrides: dict[str, Any]) -> dict:
    out = copy.deepcopy(cfg)
    for k, v in overrides.items():
        deep_set(out, k, v)
    return out


def sync_schedule(cfg: dict) -> dict:
    """Keep `lr_schedule` consistent with `learning_rate` and `iters`.

    mlx-lm uses lr_schedule *instead of* learning_rate when it is set, and its
    `arguments` are literal: [peak_lr, decay_steps, end_lr]. If you change
    `--iters` or `--learning-rate` and leave the schedule alone it silently keeps
    decaying over the old step count / from the old peak. Here we rebuild it:
    peak = learning_rate, decay_steps = iters - warmup (so the decay finishes
    exactly at the last step), end_lr preserved.
    """
    sched = cfg.get("lr_schedule")
    if not sched:
        return cfg
    out = copy.deepcopy(cfg)
    sched = out["lr_schedule"]
    name = sched.get("name", "cosine_decay")
    warmup = int(sched.get("warmup", 0) or 0)
    lr = float(out.get("learning_rate", 1e-5))
    iters = int(out.get("iters", 1000))
    old_args = list(sched.get("arguments") or [])
    if name in ("cosine_decay", "linear_schedule", "exponential_decay"):
        end = old_args[2] if len(old_args) > 2 else 0.0
        if name == "exponential_decay":
            # [init, decay_rate]: nothing step-count dependent to sync except peak
            sched["arguments"] = [lr] + old_args[1:2]
        else:
            sched["arguments"] = [lr, max(1, iters - warmup), end]
    sched["name"] = name
    return out


def count_lines(path: Path) -> int:
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def epoch_estimate(cfg: dict, n_train: int) -> float:
    """Epochs implied by iters x batch x grad-accumulation over n_train rows."""
    per_step = int(cfg.get("batch_size", 4)) * int(cfg.get("grad_accumulation_steps", 1) or 1)
    return int(cfg.get("iters", 1000)) * per_step / max(1, n_train)


def preflight(cfg: dict) -> list[str]:
    """Human-readable warnings about a config that will train badly (never raises)."""
    msgs: list[str] = []
    data = Path(str(cfg.get("data", "")))
    train = data / "train.jsonl"
    if data.is_dir() and train.exists():
        n = count_lines(train)
        ep = epoch_estimate(cfg, n)
        if ep > 10:
            msgs.append(
                f"~{ep:.0f} epochs over only {n} training examples: the model will "
                f"memorize them. Lower `iters` (≈{max(1, round(3 * n / (int(cfg.get('batch_size', 4)) * int(cfg.get('grad_accumulation_steps', 1) or 1))))} "
                "for 3 epochs) or add data.")
        elif ep < 1:
            msgs.append(f"~{ep:.2f} epochs: training will not see every example once.")
        if not cfg.get("mask_prompt"):
            try:
                with open(train, encoding="utf-8") as f:
                    first = f.readline()
                if '"messages"' in first or '"prompt"' in first:
                    msgs.append("`mask_prompt` is off: loss also covers system/user tokens, "
                                "not just the answers you want the model to learn.")
            except OSError:
                pass
    if cfg.get("fine_tune_type") == "full" and "bit" in str(cfg.get("model", "")).lower():
        msgs.append("fine_tune_type=full on a quantized model: full fine-tuning needs "
                    "non-quantized weights.")
    return msgs
