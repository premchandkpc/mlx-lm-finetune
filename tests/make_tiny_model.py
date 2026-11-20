#!/usr/bin/env python3
"""Build a tiny random-weight Llama-style model + tokenizer, fully offline.

Purpose: end-to-end tests of the training/eval/sweep tooling without
downloading anything. The model is gibberish by design; it only needs to be
loadable by mlx-lm so that `mlx_lm.lora`, `scripts/compare.py` and
`scripts/sweep.py` can be exercised on any machine (including CI on Linux CPU).

Usage:
  python tests/make_tiny_model.py --out /tmp/tiny-model [--corpus data/sample]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SPECIAL_TOKENS = ["<pad>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]

CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}<|end|>"
    "{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}"
)


def _corpus_texts(corpus: Path | None) -> list[str]:
    texts = ["hello world this is a tiny tokenizer corpus for offline tests"]
    if corpus is None:
        return texts
    files = [corpus] if corpus.is_file() else sorted(corpus.rglob("*.jsonl"))
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            for m in obj.get("messages", []):
                texts.append(m["content"])
            for key in ("text", "prompt", "completion"):
                if key in obj:
                    texts.append(obj[key])
    return texts


def build_tokenizer(out: Path, corpus: Path | None, vocab_size: int = 512):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(_corpus_texts(corpus), trainer=trainer)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="<pad>",
        eos_token="<|end|>",
        chat_template=CHAT_TEMPLATE,
    )
    fast.save_pretrained(out)
    return len(fast)


def build_model(out: Path, vocab_size: int, seed: int = 0):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm.models.llama import Model, ModelArgs

    cfg = {
        "model_type": "llama",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-5,
        "vocab_size": vocab_size,
        "max_position_embeddings": 512,
        "tie_word_embeddings": True,
    }
    mx.random.seed(seed)
    model = Model(ModelArgs(**cfg))
    mx.eval(model.parameters())
    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(out / "model.safetensors"), weights)
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def make_tiny_model(out: Path, corpus: Path | None = None) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    vocab = build_tokenizer(out, corpus)
    build_model(out, vocab)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus", default=None,
                    help="optional jsonl file/dir to train the tokenizer on")
    args = ap.parse_args()
    path = make_tiny_model(Path(args.out), Path(args.corpus) if args.corpus else None)
    print(f"tiny model written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
