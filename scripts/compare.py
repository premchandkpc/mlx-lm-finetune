#!/usr/bin/env python3
"""Base-vs-tuned evaluation on held-out data.

Answers the only question that matters after a run: *did fine-tuning help?*

For the base model and the base+adapter model it reports, on the same held-out
split (test.jsonl if present, else valid.jsonl):
  * loss / perplexity, computed exactly like training (respects `mask_prompt`)
  * greedy-generation quality vs the reference answers: exact match, token F1,
    and length ratio (catches "got chattier/terser than the data")
  * a few side-by-side samples

Writes report.md + report.json (default: outputs/compare/) and prints the report.

Usage:
  python scripts/compare.py --config configs/lora.yaml
  python scripts/compare.py --config configs/lora.yaml --split valid --gen-limit 50
"""

from __future__ import annotations

import argparse
import json
import math
import re
import string
import sys
import types
from collections import Counter
from pathlib import Path

import cfgtools


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested; no MLX needed)
# --------------------------------------------------------------------------- #

def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return re.sub(r"\s+", " ", s).strip()


def exact_match(pred: str, ref: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(ref))


def token_f1(pred: str, ref: str) -> float:
    p, r = normalize_answer(pred).split(), normalize_answer(ref).split()
    if not p and not r:
        return 1.0
    if not p or not r:
        return 0.0
    common = Counter(p) & Counter(r)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(r)
    return 2 * precision * recall / (precision + recall)


def perplexity(loss: float) -> float:
    if not math.isfinite(loss):
        return float("inf")
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


def split_prompt_references(ex: dict) -> list[tuple[list[dict], str]]:
    """Evaluate each assistant turn against only the preceding conversation.

    A final-turn-only evaluation hides earlier mistakes in multi-turn data.
    Never include the target assistant response in its own prompt.
    """
    if "messages" in ex:
        messages = ex["messages"]
        return [
            (messages[:i], message["content"])
            for i, message in enumerate(messages)
            if message["role"] == "assistant"
            and any(m["role"] == "user" for m in messages[:i])
        ]
    if "prompt" in ex and "completion" in ex:
        return [([{"role": "user", "content": ex["prompt"]}], ex["completion"])]
    return []


def split_prompt_reference(ex: dict):
    """Backward-compatible final assistant turn helper."""
    pairs = split_prompt_references(ex)
    return pairs[-1] if pairs else None


def summarize_generations(rows: list[dict]) -> dict:
    """rows: [{'pred','ref'}...] -> aggregate metrics."""
    if not rows:
        return {"n": 0}
    em = sum(exact_match(r["pred"], r["ref"]) for r in rows) / len(rows)
    f1 = sum(token_f1(r["pred"], r["ref"]) for r in rows) / len(rows)
    pw = sum(len(r["pred"].split()) for r in rows) / len(rows)
    rw = sum(len(r["ref"].split()) for r in rows) / len(rows)
    return {"n": len(rows), "exact_match": em, "token_f1": f1,
            "avg_pred_words": pw, "avg_ref_words": rw,
            "length_ratio": (pw / rw) if rw else None}


def verdict(base: dict, tuned: dict) -> list[str]:
    """Plain-language conclusions. Deliberately skeptical."""
    out = []
    bl, tl = base["loss"], tuned["loss"]
    rel = (bl - tl) / bl * 100 if bl else 0.0
    if tl < bl:
        out.append(f"Held-out loss improved {bl:.3f} -> {tl:.3f} ({rel:.1f}% lower).")
    else:
        out.append(f"Held-out loss did NOT improve: {bl:.3f} -> {tl:.3f}. "
                   "Check learning rate, data quality, or overfitting (too many epochs).")
    bg, tg = base.get("gen", {}), tuned.get("gen", {})
    if bg.get("n") and tg.get("n"):
        df1 = tg["token_f1"] - bg["token_f1"]
        out.append(f"Generation token-F1 {bg['token_f1']:.3f} -> {tg['token_f1']:.3f} "
                   f"({df1:+.3f}).")
        if tl < bl and df1 <= 0:
            out.append("Loss improved but generations did not: read the samples; "
                       "the model may be fitting style/format, not content.")
        lr_ = tg.get("length_ratio")
        if lr_ is not None and (lr_ > 1.5 or lr_ < 0.5):
            out.append(f"Tuned outputs are {lr_:.1f}x the reference length: "
                       "check EOS/stop behaviour and the system prompt.")
    out.append("Small held-out sets are noisy: treat deltas under a few percent "
               "as inconclusive.")
    return out


def _clip(s: str, limit: int = 400) -> str:
    s = s.replace("\n", " ").strip()
    return s if len(s) <= limit else s[:limit] + " …"


def render_report(res: dict) -> str:
    b, t = res["base"], res["tuned"]
    lines = [
        "# Base vs tuned",
        "",
        f"- model: `{res['model']}`",
        f"- adapter: `{res['adapter']}`",
        f"- split: `{res['split']}` ({res['n_examples']} examples), "
        f"loss over {'assistant tokens only' if res['mask_prompt'] else 'all tokens'}",
        "",
        "| metric | base | tuned | delta |",
        "|---|---|---|---|",
        f"| loss | {b['loss']:.4f} | {t['loss']:.4f} | {t['loss'] - b['loss']:+.4f} |",
        f"| perplexity | {b['ppl']:.3f} | {t['ppl']:.3f} | {t['ppl'] - b['ppl']:+.3f} |",
    ]
    bg, tg = b.get("gen", {}), t.get("gen", {})
    if bg.get("n") and tg.get("n"):
        for key, label in (("exact_match", "exact match"), ("token_f1", "token F1"),
                           ("avg_pred_words", "avg words out")):
            lines.append(f"| {label} (n={tg['n']}) | {bg[key]:.3f} | {tg[key]:.3f} | "
                         f"{tg[key] - bg[key]:+.3f} |")
        lines.append(f"| reference words | {bg['avg_ref_words']:.1f} | "
                     f"{tg['avg_ref_words']:.1f} | |")
    lines += ["", "## Verdict", ""] + [f"- {v}" for v in res["verdict"]]
    if res.get("samples"):
        lines += ["", "## Samples", ""]
        for i, s in enumerate(res["samples"], 1):
            lines += [f"**{i}. prompt:** {_clip(s['prompt'])}", "",
                      f"- reference: {_clip(s['ref'])}", f"- base: {_clip(s['base'])}",
                      f"- tuned: {_clip(s['tuned'])}", ""]
    return "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------------------- #
# MLX-dependent part (imported lazily so the pure helpers stay testable)
# --------------------------------------------------------------------------- #

def _free(mx) -> None:
    for name in ("clear_cache",):
        fn = getattr(mx, name, None) or getattr(getattr(mx, "metal", None), name, None)
        if fn:
            fn()


def evaluate_variant(model_path: str, adapter_path: str | None, data_dir: str,
                     split: str, cfg: dict, prompts: list[list[dict]],
                     max_tokens: int, temp: float) -> dict:
    import mlx.core as mx
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.tuner.datasets import CacheDataset, load_dataset
    from mlx_lm.tuner.trainer import evaluate

    model, tokenizer = load(model_path, adapter_path=adapter_path)

    ns = types.SimpleNamespace(
        data=data_dir, hf_dataset=False, train=False, test=(split == "test"),
        mask_prompt=bool(cfg.get("mask_prompt", False)),
        max_seq_length=int(cfg.get("max_seq_length", 2048)),
    )
    train_set, valid_set, test_set = load_dataset(ns, tokenizer)
    ds = test_set if split == "test" else valid_set
    bs = max(1, min(int(cfg.get("batch_size", 4)), len(ds)))
    loss = evaluate(model=model, dataset=CacheDataset(ds), batch_size=bs,
                    num_batches=-1, max_seq_length=ns.max_seq_length)

    sampler = make_sampler(temp=temp)
    preds = []
    for messages in prompts:
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)
        preds.append(generate(model, tokenizer, prompt=text, max_tokens=max_tokens,
                              sampler=sampler, verbose=False).strip())

    del model
    _free(mx)
    return {"loss": float(loss), "ppl": perplexity(float(loss)), "preds": preds}


def pick_split(data_dir: Path, requested: str | None) -> str:
    if requested:
        if not (data_dir / f"{requested}.jsonl").exists():
            raise SystemExit(f"error: {data_dir}/{requested}.jsonl not found")
        return requested
    for name in ("test", "valid"):
        if (data_dir / f"{name}.jsonl").exists():
            return name
    raise SystemExit(f"error: no test.jsonl or valid.jsonl in {data_dir}. "
                     "Re-run prepare_data.py with --test-split (or --valid-split).")


def check_adapter(adapter: str) -> None:
    p = Path(adapter)
    if not (p / "adapter_config.json").exists() or not list(p.glob("*adapters.safetensors")):
        raise SystemExit(f"error: no trained adapter in {p}. Run `make train` first.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/lora.yaml")
    ap.add_argument("--model", default=None, help="override config model")
    ap.add_argument("--adapter-path", default=None, help="override config adapter_path")
    ap.add_argument("--data", default=None, help="override config data dir")
    ap.add_argument("--split", choices=["test", "valid"], default=None)
    ap.add_argument("--gen-limit", type=int, default=20,
                    help="held-out prompts to generate for (0 = loss only)")
    ap.add_argument("--show", type=int, default=3, help="samples shown in the report")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--temp", type=float, default=0.0, help="0 = greedy (default)")
    ap.add_argument("--out", default="outputs/compare")
    args = ap.parse_args(argv)

    if args.gen_limit < 0 or args.show < 0 or args.max_tokens < 1 or not math.isfinite(args.temp) or args.temp < 0:
        raise SystemExit("error: invalid generation arguments")
    cfg = cfgtools.load_config(args.config)
    model = args.model or cfg["model"]
    adapter = args.adapter_path or cfg.get("adapter_path", "adapters")
    data_dir = Path(args.data or cfg["data"])
    if not data_dir.is_dir():
        raise SystemExit(f"error: data dir not found: {data_dir} "
                         "(HF-hosted datasets are not supported by compare.py)")
    split = pick_split(data_dir, args.split)
    check_adapter(adapter)

    rows = read_jsonl(data_dir / f"{split}.jsonl")
    pairs = [pair for row in rows for pair in split_prompt_references(row)]
    pairs = pairs[: args.gen_limit] if args.gen_limit else []
    prompts = [p for p, _ in pairs]
    refs = [r for _, r in pairs]

    print(f"==> base model ({model})", file=sys.stderr)
    base = evaluate_variant(model, None, str(data_dir), split, cfg, prompts,
                            args.max_tokens, args.temp)
    print(f"==> tuned model (+ {adapter})", file=sys.stderr)
    tuned = evaluate_variant(model, adapter, str(data_dir), split, cfg, prompts,
                             args.max_tokens, args.temp)

    base_preds, tuned_preds = base.pop("preds"), tuned.pop("preds")
    for variant, preds in ((base, base_preds), (tuned, tuned_preds)):
        variant["gen"] = summarize_generations(
            [{"pred": p, "ref": r} for p, r in zip(preds, refs)]
        ) if refs else {"n": 0}

    res = {
        "model": model, "adapter": adapter, "split": split,
        "n_examples": len(rows), "mask_prompt": bool(cfg.get("mask_prompt", False)),
        "base": base, "tuned": tuned,
    }
    res["verdict"] = verdict(base, tuned)
    res["samples"] = [
        {"prompt": prompts[i][-1]["content"], "ref": refs[i],
         "base": base_preds[i], "tuned": tuned_preds[i]}
        for i in range(min(args.show, len(prompts)))
    ]
    res_path = Path(args.out)
    res_path.mkdir(parents=True, exist_ok=True)
    (res_path / "report.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    report = render_report(res)
    (res_path / "report.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
