#!/usr/bin/env python3
"""Prepare JSONL datasets for mlx-lm fine-tuning.

Pipeline: load -> convert -> validate -> filter length -> dedup -> split -> write.

Input styles (auto-detected per file):
  1. .jsonl with {"messages": [...]}       -> chat format
  2. .jsonl with {"text": "..."}           -> pretraining/text format
  3. .jsonl with {"prompt","completion"}   -> completions format
  4. Plain .txt / .md files                -> one example per paragraph block

What this does beyond a naive splitter:
  * validates every example (roles, empty content, chat must end on assistant)
  * exact dedup, or near-duplicate dedup (MinHash-LSH + verified Jaccard)
  * leakage-proof splits: examples sharing a prompt (or an explicit --group-key,
    e.g. a document id) always land in the SAME split; a final audit re-checks it
  * optional length filters (chars, or real tokens via --tokenizer)
  * writes stats.json (counts, drops per reason, length percentiles) and prints
    a suggested `iters` so you don't train 200 epochs on 10 rows by accident

Examples:
  python scripts/prepare_data.py --input data/sample --output data/processed
  python scripts/prepare_data.py --input my.jsonl --output data/processed \\
      --format chat --dedup near --group-key doc_id --test-split 0.05 \\
      --tokenizer mlx-community/Qwen2.5-0.5B-Instruct-4bit --max-tokens 1024
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path

VALID_SUFFIXES = {".jsonl", ".txt", ".md"}
VALID_ROLES = {"system", "user", "assistant", "tool"}
GROUP_FIELD = "_group"  # internal key; stripped before writing


# --------------------------------------------------------------------------- #
# Loading / conversion
# --------------------------------------------------------------------------- #

def _warn(msg: str) -> None:
    print(msg, file=sys.stderr)


def load_raw(path: Path, group_key: str | None, drops: Counter) -> list[dict]:
    """Load one file into mlx-lm-shaped dicts (original keys only + _group)."""
    examples: list[dict] = []

    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    _warn(f"  ! {path}:{line_no} invalid JSON, skipped ({e})")
                    drops["invalid_json"] += 1
                    continue
                if not isinstance(obj, dict):
                    drops["not_an_object"] += 1
                    continue

                if "messages" in obj:
                    ex = {"messages": obj["messages"]}
                elif "text" in obj:
                    ex = {"text": obj["text"]}
                elif "prompt" in obj and "completion" in obj:
                    ex = {"prompt": obj["prompt"], "completion": obj["completion"]}
                else:
                    _warn(f"  ! {path}:{line_no} no 'text', 'messages' or "
                          "'prompt'+'completion' key, skipped")
                    drops["unknown_schema"] += 1
                    continue
                if group_key and obj.get(group_key) is not None:
                    ex[GROUP_FIELD] = str(obj[group_key])
                examples.append(ex)
    else:
        content = path.read_text(encoding="utf-8")
        for block in content.split("\n\n"):
            block = block.strip()
            if block:
                examples.append({"text": block})
    return examples


def convert(ex: dict, fmt: str) -> dict | None:
    """Convert one example to `fmt` (text|chat|completions|auto). None = drop."""
    group = {GROUP_FIELD: ex[GROUP_FIELD]} if GROUP_FIELD in ex else {}

    if fmt == "auto":
        return ex

    if fmt == "text":
        if "text" in ex:
            return ex
        if "prompt" in ex:
            return {"text": f"{ex['prompt']}{ex['completion']}", **group}
        msgs = ex.get("messages")
        if not isinstance(msgs, list):
            return None
        try:
            transcript = "\n".join(f"{m['role']}: {m['content']}" for m in msgs)
        except (KeyError, TypeError):
            return None
        return {"text": transcript, **group}

    if fmt == "chat":
        if "messages" in ex:
            return ex
        if "prompt" in ex:
            return {"messages": [
                {"role": "user", "content": ex["prompt"]},
                {"role": "assistant", "content": ex["completion"]},
            ], **group}
        return {"messages": [
            {"role": "user", "content": "Continue the text."},
            {"role": "assistant", "content": ex["text"]},
        ], **group}

    if fmt == "completions":
        if "prompt" in ex:
            return ex
        msgs = ex.get("messages")
        if isinstance(msgs, list) and msgs and isinstance(msgs[-1], dict) \
                and msgs[-1].get("role") == "assistant":
            try:
                prompt = "\n".join(f"{m['role']}: {m['content']}" for m in msgs[:-1])
            except (KeyError, TypeError):
                return None
            return {"prompt": prompt, "completion": msgs[-1]["content"], **group}
        return None

    raise ValueError(f"unknown format: {fmt}")


def shape_of(ex: dict) -> str:
    """mlx-lm dataset class an example maps to (same key precedence as load_raw)."""
    if "messages" in ex:
        return "chat"
    if "text" in ex:
        return "text"
    if "prompt" in ex and "completion" in ex:
        return "completions"
    return "unknown"


def mixed_format_error(examples: list[dict]) -> str | None:
    """Message if `examples` mix shapes, else None.

    mlx-lm picks the dataset class from the FIRST row of each split
    (mlx_lm/tuner/datasets.py -> create_dataset), so a mixed file would die
    mid-training with a bare KeyError on row 2. Only relevant with --format auto:
    a forced --format rewrites every row to one shape.
    """
    counts = Counter(shape_of(e) for e in examples)
    if len(counts) <= 1:
        return None
    detail = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    return (f"mixed example formats in input: {detail}\n"
            "  mlx-lm infers the format from the first line of each split, so the "
            "remaining lines would fail mid-training.\n"
            "  Fix one of:\n"
            "    - pass --format chat|text|completions to convert every example to "
            "a single shape\n"
            "    - run prepare_data.py once per source format into separate output "
            "directories")


def validate(ex: dict) -> str | None:
    """Return a drop-reason string, or None if the example is usable."""
    if "messages" in ex:
        msgs = ex["messages"]
        if not isinstance(msgs, list) or not msgs:
            return "bad_messages"
        for m in msgs:
            if not isinstance(m, dict) or m.get("role") not in VALID_ROLES:
                return "bad_role"
            if not isinstance(m.get("content"), str) or not m["content"].strip():
                return "empty_content"
        if msgs[-1]["role"] != "assistant":
            return "chat_not_ending_in_assistant"
        if not any(m["role"] == "user" for m in msgs):
            return "chat_without_user_turn"
        return None
    if "prompt" in ex:
        if not isinstance(ex["prompt"], str) or not ex["prompt"].strip():
            return "empty_prompt"
        if not isinstance(ex["completion"], str) or not ex["completion"].strip():
            return "empty_completion"
        return None
    if "text" in ex:
        if not isinstance(ex["text"], str) or not ex["text"].strip():
            return "empty_text"
        return None
    return "unknown_schema"


# --------------------------------------------------------------------------- #
# Keys, text views, lengths
# --------------------------------------------------------------------------- #

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def prompt_text(ex: dict) -> str:
    """Everything the model is conditioned on (excludes the target)."""
    if "messages" in ex:
        return "\n".join(m["content"] for m in ex["messages"][:-1])
    if "prompt" in ex:
        return ex["prompt"]
    return ex["text"]


def full_text(ex: dict) -> str:
    if "messages" in ex:
        return "\n".join(m["content"] for m in ex["messages"])
    if "prompt" in ex:
        return ex["prompt"] + "\n" + ex["completion"]
    return ex["text"]


def prompt_key(ex: dict) -> str:
    return _norm(prompt_text(ex))


def exact_key(ex: dict) -> str:
    def canonical(value):
        if isinstance(value, str):
            return _norm(value)
        if isinstance(value, list):
            return [canonical(item) for item in value]
        if isinstance(value, dict):
            return {key: canonical(item) for key, item in value.items()}
        return value

    return json.dumps(canonical({k: v for k, v in ex.items() if k != GROUP_FIELD}),
                      sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def group_of(ex: dict) -> str:
    """Split unit: explicit group if given, else the normalized prompt."""
    return ex.get(GROUP_FIELD) or ("p:" + prompt_key(ex))


def char_len(ex: dict) -> int:
    return len(full_text(ex))


def make_token_counter(tokenizer_name: str):
    """Return fn(ex)->token count using a real tokenizer (lazy import)."""
    from transformers import AutoTokenizer  # heavy import only when requested

    tok = AutoTokenizer.from_pretrained(tokenizer_name)

    def count(ex: dict) -> int:
        if "messages" in ex:
            ids = tok.apply_chat_template(ex["messages"], tokenize=True)
            if isinstance(ids, dict):  # newer transformers return a BatchEncoding
                ids = ids["input_ids"]
            return len(ids)
        return len(tok.encode(full_text(ex)))

    return count


def percentiles(values: list[int]) -> dict:
    if not values:
        return {}
    v = sorted(values)

    def pct(p: float) -> int:
        return v[min(len(v) - 1, int(math.ceil(p * len(v))) - 1)]

    return {"min": v[0], "p50": pct(0.50), "p90": pct(0.90),
            "p99": pct(0.99), "max": v[-1]}


# --------------------------------------------------------------------------- #
# Dedup
# --------------------------------------------------------------------------- #

def dedup_exact(examples: list[dict]) -> tuple[list[dict], int]:
    seen: set[str] = set()
    out = []
    for ex in examples:
        k = exact_key(ex)
        if k in seen:
            continue
        seen.add(k)
        out.append(ex)
    return out, len(examples) - len(out)


def _shingles(text: str, n: int = 3) -> set[str]:
    words = _norm(text).split()
    if len(words) <= n:
        return {" ".join(words)}
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def dedup_near(examples: list[dict], threshold: float = 0.85,
               num_perm: int = 64, bands: int = 16, seed: int = 1) -> tuple[list[dict], int]:
    """Near-duplicate removal: MinHash-LSH candidates, verified by true Jaccard.

    Keeps the first occurrence. Candidates come from LSH banding (cheap, ~O(n));
    every candidate pair is then verified with exact Jaccard on word 3-grams, so
    LSH only affects recall, never precision.
    """
    import numpy as np

    if num_perm % bands:
        raise ValueError("num_perm must be divisible by bands")
    rows = num_perm // bands
    rng = np.random.RandomState(seed)
    prime = np.uint64((1 << 31) - 1)
    a = rng.randint(1, (1 << 31) - 2, size=num_perm).astype(np.uint64)
    b = rng.randint(0, (1 << 31) - 2, size=num_perm).astype(np.uint64)

    sets = [_shingles(full_text(ex)) for ex in examples]
    buckets: dict[tuple, list[int]] = defaultdict(list)
    kept: list[int] = []

    for i, sh in enumerate(sets):
        base = np.fromiter((zlib.crc32(s.encode()) for s in sh), dtype=np.uint64)
        sig = ((a[:, None] * base[None, :] + b[:, None]) % prime).min(axis=1)
        band_keys = [(band, tuple(sig[band * rows:(band + 1) * rows].tolist()))
                     for band in range(bands)]

        is_dup = False
        seen_cands: set[int] = set()
        for bk in band_keys:
            for j in buckets.get(bk, ()):
                if j in seen_cands:
                    continue
                seen_cands.add(j)
                if jaccard(sh, sets[j]) >= threshold:
                    is_dup = True
                    break
            if is_dup:
                break
        if not is_dup:
            kept.append(i)
            for bk in band_keys:
                buckets[bk].append(i)

    out = [examples[i] for i in kept]
    return out, len(examples) - len(out)


# --------------------------------------------------------------------------- #
# Splitting + leakage audit
# --------------------------------------------------------------------------- #

def split_examples(examples: list[dict], valid_split: float, test_split: float,
                   min_split: int, shuffle: bool, seed: int):
    """Group-aware split. Returns (train, valid, test, notes)."""
    notes: list[str] = []
    n = len(examples)
    min_split = max(1, min_split)

    n_valid = int(n * valid_split)
    n_test = int(n * test_split)
    if valid_split > 0:
        n_valid = max(n_valid, min_split)
    if test_split > 0:
        n_test = max(n_test, min_split)

    groups: dict[str, list[dict]] = {}
    for ex in examples:
        groups.setdefault(group_of(ex), []).append(ex)
    order = list(groups)
    if shuffle:
        random.Random(seed).shuffle(order)

    # Place larger groups first to avoid letting a single large group
    # consume an entire held-out split. Preserve seeded tie-breaking.
    order.sort(key=lambda g: len(groups[g]), reverse=True)
    valid: list[dict] = []
    test: list[dict] = []
    train: list[dict] = []
    targets = {"valid": n_valid, "test": n_test}
    buckets = {"valid": valid, "test": test}
    for g in order:
        size = len(groups[g])
        choices = [
            name for name, target in targets.items()
            if target and len(buckets[name]) < target
            and len(buckets[name]) + size <= target
        ]
        if choices:
            chosen = max(choices, key=lambda name: targets[name] - len(buckets[name]))
            buckets[chosen].extend(groups[g])
        else:
            train.extend(groups[g])

    # Priority train > valid > test: never starve training.
    if len(train) < min_split and test:
        notes.append(f"train < min-split ({min_split}); merged test.jsonl back into train")
        train.extend(test)
        test = []
    if len(train) < min_split and valid:
        notes.append(f"train < min-split ({min_split}); merged valid.jsonl back into train")
        train.extend(valid)
        valid = []

    # mlx-lm needs >= batch_size examples in EVERY split it reads: drop tiny ones.
    for name, split in (("valid.jsonl", valid), ("test.jsonl", test)):
        if 0 < len(split) < min_split:
            notes.append(f"moved {len(split)} example(s) from undersized {name} back to train (< --min-split {min_split})")
            train.extend(split)
            split.clear()
    return train, valid, test, notes


def audit_leakage(train: list[dict], valid: list[dict], test: list[dict]) -> dict:
    """Count prompt/group overlaps across splits. All values must be 0."""
    def keys(split, fn):
        return {fn(e) for e in split}

    report = {}
    for fn_name, fn in (("prompt", prompt_key), ("group", group_of)):
        tr, va, te = (keys(s, fn) for s in (train, valid, test))
        report[f"{fn_name}_train_valid"] = len(tr & va)
        report[f"{fn_name}_train_test"] = len(tr & te)
        report[f"{fn_name}_valid_test"] = len(va & te)
    return report


def suggest_iters(n_train: int, batch_size: int, epochs: float,
                  grad_accum: int = 1) -> int:
    per_step = max(1, batch_size * max(1, grad_accum))
    return max(1, math.ceil(epochs * n_train / per_step))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def write_split(path: Path, split: list[dict]) -> None:
    if not split:
        path.unlink(missing_ok=True)  # never leave a stale split behind
        return
    with path.open("w", encoding="utf-8") as f:
        for ex in split:
            clean = {k: v for k, v in ex.items() if k != GROUP_FIELD}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True,
                    help="Input file (.jsonl/.txt/.md) or directory of those files")
    ap.add_argument("--output", default="data/processed",
                    help="Output directory (default: data/processed)")
    ap.add_argument("--format", choices=["auto", "text", "chat", "completions"],
                    default="auto", help="Force output format (default: auto)")
    ap.add_argument("--valid-split", type=float, default=0.05,
                    help="Fraction held out for validation (default: 0.05)")
    ap.add_argument("--test-split", type=float, default=0.0,
                    help="Fraction held out for test.jsonl (default: 0 = none)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-split", type=int, default=4,
                    help="Min examples per written split; mlx-lm needs >= batch_size "
                         "per split it reads, so smaller splits are dropped (default: 4)")
    ap.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True,
                    help="Shuffle groups before splitting (default: on; the old "
                         "behaviour of carving valid/test off the file head is "
                         "available via --no-shuffle)")
    ap.add_argument("--dedup", choices=["none", "exact", "near"], default="exact",
                    help="Duplicate removal before splitting (default: exact)")
    ap.add_argument("--near-threshold", type=float, default=0.85,
                    help="Jaccard threshold for --dedup near (default: 0.85)")
    ap.add_argument("--group-key", default=None,
                    help="JSONL field whose value keeps rows together in one split "
                         "(e.g. doc_id, source). Default: group by prompt.")
    ap.add_argument("--min-chars", type=int, default=1,
                    help="Drop examples shorter than this many characters (default: 1)")
    ap.add_argument("--max-chars", type=int, default=0,
                    help="Drop examples longer than this many characters (0 = off)")
    ap.add_argument("--tokenizer", default=None,
                    help="HF repo/path of a tokenizer for token-accurate stats/filters")
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="Drop examples longer than this many tokens "
                         "(needs --tokenizer; set to your max_seq_length)")
    ap.add_argument("--batch-size", type=int, default=4,
                    help="Used only for the suggested-iters hint (default: 4)")
    ap.add_argument("--epochs", type=float, default=3.0,
                    help="Target epochs for the suggested-iters hint (default: 3)")
    return ap


def run(args: argparse.Namespace) -> int:
    src = Path(args.input)
    if not src.exists():
        _warn(f"error: input not found: {src}")
        return 1
    if not (0 <= args.valid_split < 1 and 0 <= args.test_split < 1 and args.valid_split + args.test_split < 1):
        _warn("error: invalid split fractions")
        return 1
    if args.max_tokens and not args.tokenizer:
        _warn("error: --max-tokens requires --tokenizer")
        return 1

    files = [src] if src.is_file() else sorted(
        p for p in src.rglob("*")
        if p.suffix.lower() in VALID_SUFFIXES and p.is_file())
    if not files:
        _warn(f"error: no {sorted(VALID_SUFFIXES)} files under {src}")
        return 1

    drops: Counter = Counter()
    loaded: list[dict] = []
    for f in files:
        before = len(loaded)
        loaded.extend(load_raw(f, args.group_key, drops))
        print(f"  + {f}: {len(loaded) - before} examples")
    n_loaded = len(loaded)

    # convert + validate
    examples: list[dict] = []
    for ex in loaded:
        conv = convert(ex, args.format)
        if conv is None:
            drops["not_convertible_to_" + args.format] += 1
            continue
        reason = validate(conv)
        if reason:
            drops[reason] += 1
            continue
        examples.append(conv)

    if args.format == "auto":
        msg = mixed_format_error(examples)
        if msg:
            _warn(f"error: {msg}")
            return 1

    # length filters
    token_count = make_token_counter(args.tokenizer) if args.tokenizer else None
    tok_lens: list[int] = []
    kept: list[dict] = []
    for ex in examples:
        c = char_len(ex)
        if c < args.min_chars:
            drops["too_short_chars"] += 1
            continue
        if args.max_chars and c > args.max_chars:
            drops["too_long_chars"] += 1
            continue
        if token_count:
            t = token_count(ex)
            if args.max_tokens and t > args.max_tokens:
                drops["too_long_tokens"] += 1
                continue
            tok_lens.append(t)
        kept.append(ex)
    examples = kept

    # dedup
    if args.dedup == "exact":
        examples, n = dedup_exact(examples)
        drops["duplicate_exact"] += n
    elif args.dedup == "near":
        examples, n = dedup_near(examples, args.near_threshold)
        drops["duplicate_near"] += n

    if not examples:
        _warn("error: no usable examples left after filtering "
              f"(drops: {dict(drops)})")
        return 1

    train, valid, test, notes = split_examples(
        examples, args.valid_split, args.test_split, args.min_split,
        args.shuffle, args.seed)

    leak = audit_leakage(train, valid, test)
    if any(leak.values()):
        _warn(f"error: split leakage detected: { {k: v for k, v in leak.items() if v} }")
        return 2

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    write_split(out / "train.jsonl", train)
    write_split(out / "valid.jsonl", valid)
    write_split(out / "test.jsonl", test)

    stats = {
        "loaded": n_loaded,
        "usable": len(examples),
        "dropped": dict(drops),
        "splits": {"train": len(train), "valid": len(valid), "test": len(test)},
        "format": args.format,
        "dedup": args.dedup,
        "chars": percentiles([char_len(e) for e in examples]),
        "tokens": percentiles(tok_lens),
        "leakage_audit": leak,
        "suggested_iters": suggest_iters(len(train), args.batch_size, args.epochs),
        "suggestion_basis": {"epochs": args.epochs, "batch_size": args.batch_size},
        "notes": notes,
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"\nWrote {len(train)} train / {len(valid)} valid / {len(test)} test -> {out}/")
    if drops:
        print("Dropped: " + ", ".join(f"{k}={v}" for k, v in sorted(drops.items())))
    for note in notes:
        print(f"note: {note}")
    if len(train) < args.batch_size:
        print(f"note: train split has only {len(train)} examples "
              f"(< batch size {args.batch_size}); lower --batch-size when training.")
    if len(train) < 200:
        print(f"note: only {len(train)} training examples. For a real run use far "
              "more data; with this little, extra epochs just memorize it.")
    print(f"hint: {args.epochs:g} epochs at batch_size={args.batch_size} "
          f"≈ iters: {stats['suggested_iters']}  (stats: {out}/stats.json)")
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
