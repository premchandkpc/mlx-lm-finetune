#!/usr/bin/env python3
"""Prepare JSONL datasets for mlx-lm fine-tuning.

Supports four input styles (auto-detected per file):
  1. .jsonl with {"messages": [...]}       -> chat format
  2. .jsonl with {"text": "..."}           -> pretraining/text format
  3. .jsonl with {"prompt","completion"}   -> completions format
  4. Plain .txt / .md files                -> one example per paragraph block

Examples:
  python scripts/prepare_data.py --input data/sample --output data/processed
  python scripts/prepare_data.py --input my_data.jsonl --output data/processed \\
      --format chat --valid-split 0.1 --test-split 0.05
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

VALID_SUFFIXES = {".jsonl", ".txt", ".md"}


def shape_of(example: dict) -> str:
    """mlx-lm dataset format an example maps to.

    Uses the same key precedence as load_examples(), so it describes the shape
    of the dict that will actually be written out.
    """
    if "messages" in example:
        return "chat"
    if "text" in example:
        return "text"
    if "prompt" in example and "completion" in example:
        return "completions"
    return "unknown"


def load_examples(path: Path, fmt: str | None) -> list[dict]:
    """Load examples from a single file into mlx-lm format."""
    examples: list[dict] = []

    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"  ! {path}:{line_no} invalid JSON, skipped ({e})",
                          file=sys.stderr)
                    continue

                if "messages" in obj:
                    examples.append({"messages": obj["messages"]})
                elif "text" in obj:
                    examples.append({"text": obj["text"]})
                elif "prompt" in obj and "completion" in obj:
                    examples.append({"prompt": obj["prompt"],
                                     "completion": obj["completion"]})
                else:
                    print(f"  ! {path}:{line_no} no 'text', 'messages' or "
                          "'prompt'+'completion' key, skipped", file=sys.stderr)
    else:
        # Plain text/markdown: split on blank lines into chunks.
        content = path.read_text(encoding="utf-8")
        for block in content.split("\n\n"):
            block = block.strip()
            if block:
                examples.append({"text": block})

    # Enforce requested format if possible.
    if fmt == "text":
        out = []
        for ex in examples:
            if "text" in ex:
                out.append(ex)
            elif "prompt" in ex:
                out.append({"text": ex["prompt"] + ex["completion"]})
            else:
                # Flatten chat messages into a plain transcript.
                transcript = "\n".join(
                    f"{m['role']}: {m['content']}" for m in ex["messages"]
                )
                out.append({"text": transcript})
        return out

    if fmt == "chat":
        out = []
        for ex in examples:
            if "messages" in ex:
                out.append(ex)
            elif "prompt" in ex:
                out.append({"messages": [
                    {"role": "user", "content": ex["prompt"]},
                    {"role": "assistant", "content": ex["completion"]},
                ]})
            else:
                # Wrap plain text as a single-turn user/assistant pair is not
                # meaningful, so store it as a system-style chat example.
                out.append({
                    "messages": [
                        {"role": "user", "content": "Continue the text."},
                        {"role": "assistant", "content": ex["text"]},
                    ]
                })
        return out

    if fmt == "completions":
        out = []
        dropped = 0
        for ex in examples:
            if "prompt" in ex:
                out.append(ex)
            elif "messages" in ex and ex["messages"] and \
                    ex["messages"][-1].get("role") == "assistant":
                msgs = ex["messages"]
                prompt = "\n".join(
                    f"{m['role']}: {m['content']}" for m in msgs[:-1]
                )
                out.append({"prompt": prompt,
                            "completion": msgs[-1]["content"]})
            else:
                dropped += 1
        if dropped:
            print(f"  ! dropped {dropped} example(s) with no assistant "
                  "completion (completions format)", file=sys.stderr)
        return out

    return examples  # auto (keep original format)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True,
                    help="Input file (.jsonl/.txt/.md) or directory of those files")
    ap.add_argument("--output", default="data/processed",
                    help="Output directory (default: data/processed)")
    ap.add_argument("--format", choices=["auto", "text", "chat", "completions"],
                    default="auto",
                    help="Force output format (default: auto, keep as-is)")
    ap.add_argument("--valid-split", type=float, default=0.05,
                    help="Fraction held out for validation (default: 0.05)")
    ap.add_argument("--test-split", type=float, default=0.0,
                    help="Fraction held out for test.jsonl, used by eval.sh "
                         "(default: 0.0 = no test split)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-split", type=int, default=4,
                    help="Minimum examples per written split. mlx-lm needs at "
                         "least batch_size (default 4) examples in every split "
                         "it reads, so smaller splits are dropped rather than "
                         "written (default: 4)")
    ap.add_argument("--shuffle", action="store_true",
                    help="Shuffle before splitting (recommended for mixed sources)")
    args = ap.parse_args()

    src = Path(args.input)
    if not src.exists():
        print(f"error: input not found: {src}", file=sys.stderr)
        return 1

    files = [src] if src.is_file() else sorted(
        p for p in src.rglob("*")
        if p.suffix.lower() in VALID_SUFFIXES and p.is_file()
    )
    if not files:
        print(f"error: no {sorted(VALID_SUFFIXES)} files under {src}",
              file=sys.stderr)
        return 1

    examples: list[dict] = []
    for f in files:
        before = len(examples)
        examples.extend(load_examples(f, None if args.format == "auto" else args.format))
        print(f"  + {f}: {len(examples) - before} examples")

    if not examples:
        print("error: no usable examples found", file=sys.stderr)
        return 1

    # mlx-lm picks the dataset class from the FIRST row of each split
    # (mlx_lm/tuner/datasets.py -> create_dataset), so a mixed-format file
    # dies later with a bare KeyError on row 2. Catch it here, while the
    # source files are still named. Runs with a forced --format are already
    # homogeneous because load_examples() rewrites every example.
    if args.format == "auto":
        counts: dict[str, int] = {}
        for ex in examples:
            key = shape_of(ex)
            counts[key] = counts.get(key, 0) + 1
        if len(counts) > 1:
            detail = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            print(f"error: mixed example formats in input: {detail}",
                  file=sys.stderr)
            print("  mlx-lm infers the format from the first line of each "
                  "split, so the remaining lines would fail mid-training.",
                  file=sys.stderr)
            print("  Fix one of:", file=sys.stderr)
            print("    - pass --format chat|text|completions to convert every "
                  "example to a single shape", file=sys.stderr)
            print("    - run prepare_data.py once per source format into "
                  "separate output directories", file=sys.stderr)
            return 1

    if args.shuffle:
        random.Random(args.seed).shuffle(examples)

    n = len(examples)
    min_split = max(1, args.min_split)
    n_valid = int(n * args.valid_split)
    n_test = int(n * args.test_split)
    if args.valid_split > 0:
        n_valid = max(n_valid, min_split)
    if args.test_split > 0:
        n_test = max(n_test, min_split)

    # Priority: train > valid > test — always keep min_split for training.
    if n - n_valid - n_test < min_split:
        overflow = min_split - (n - n_valid - n_test)
        take = min(n_test, overflow)
        n_test -= take
        overflow -= take
        n_valid = max(0, n_valid - overflow)

    # A split smaller than min_split would break mlx-lm (it requires at least
    # batch_size examples per split it reads), so drop it instead of writing it.
    dropped = []
    if 0 < n_valid < min_split:
        n_valid = 0
        dropped.append("valid.jsonl")
    if 0 < n_test < min_split:
        n_test = 0
        dropped.append("test.jsonl")

    valid = examples[:n_valid]
    test = examples[n_valid:n_valid + n_test]
    train = examples[n_valid + n_test:]

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for name, split in (("train.jsonl", train), ("valid.jsonl", valid),
                        ("test.jsonl", test)):
        path = out / name
        if not split:
            # Remove a stale file from a previous run so eval.sh doesn't
            # silently evaluate against old data.
            path.unlink(missing_ok=True)
            continue
        with path.open("w", encoding="utf-8") as f:
            for ex in split:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(train)} train / {len(valid)} valid / "
          f"{len(test)} test -> {out}/")
    for name in dropped:
        print(f"note: dropped {name} — fewer than --min-split {min_split} "
              "examples available (mlx-lm needs batch_size examples per split).")
    if len(train) < min_split:
        print(f"note: train split has only {len(train)} examples (< "
              f"--min-split {min_split}); lower --batch-size when training.")
    if len(train) < 20:
        print("note: very small dataset — expect overfitting; use a larger "
              "corpus for real runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
