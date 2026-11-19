# mlx-lm-finetune

Fine-tune small language models (SLMs) on Apple silicon with
[MLX-LM](https://github.com/ml-explore/mlx-lm). Self-contained scaffold:
virtualenv setup, dataset preparation, LoRA training, evaluation, chat, and
adapter fusion — all driven by one YAML config.

Verified against `mlx==0.32.4` / `mlx-lm==0.32.2` on macOS (arm64).

## Layout

```
mlx-lm-finetune/
├── setup.sh                 # create .venv + install deps
├── Makefile                 # setup / data / train / eval / chat / fuse / clean
├── configs/
│   └── lora.yaml            # all training hyperparameters (the one file you edit)
├── data/
│   ├── sample/              # 60-example chat dataset (chat format)
│   └── processed/           # generated train/valid/test.jsonl (gitignored)
├── outputs/
│   └── adapters/            # trained adapter weights + config (gitignored)
└── scripts/
    ├── prepare_data.py      # convert raw files -> train/valid/test.jsonl
    ├── train.sh             # mlx_lm.lora  (LoRA / DoRA / full)
    ├── eval.sh              # test perplexity + sample generation
    ├── chat.sh              # interactive chat with the adapter
    └── fuse.sh              # merge adapter into a standalone model
```

## Quickstart

```bash
./setup.sh                                   # 1. venv + dependencies
make data                                    # 2. build dataset from data/sample
make train                                   # 3. LoRA fine-tune (edit configs/lora.yaml)
make eval                                    # 4. base vs tuned perplexity + sample output
make chat                                    # 5. talk to your model
```

`make train` runs a short demo (25 steps ≈ 2 epochs over the bundled 60-row
sample — stopped where validation loss bottoms out; the measurements are in
`configs/lora.yaml`). For a real run, point `data:` in
`configs/lora.yaml` at your own data and raise `iters`.

## Requirements

- Apple silicon Mac (MLX does not run on Intel/CUDA)
- Python 3.12 (other 3.10+ should work; `setup.sh` uses 3.12)
- Disk: ~0.5–2 GB for a 4-bit SLM download (cached by Hugging Face)

## Data preparation

mlx-lm reads `train.jsonl` (required), `valid.jsonl` (optional, enables
validation-loss reporting), and `test.jsonl` (for `eval.sh`) from one
directory. Each line is one example. Supported formats:

| Format       | Line shape |
|--------------|------------|
| `chat`       | `{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}` |
| `text`       | `{"text": "..."}` (pretraining / completion-style corpora) |
| `completions`| `{"prompt": "...", "completion": "..."}` |

Build the splits from raw files:

```bash
# From a directory of .jsonl / .txt / .md files (auto format detection)
.venv/bin/python scripts/prepare_data.py --input path/to/raw --output data/processed

# Force chat format, 10% validation, 5% test, shuffled
.venv/bin/python scripts/prepare_data.py --input data.jsonl --output data/processed \
    --format chat --valid-split 0.10 --test-split 0.05 --shuffle
```

**Important:** mlx-lm requires *at least `batch_size` examples* (default 4)
in every split it reads — smaller splits crash training with
`Dataset must have at least batch_size=4 examples`. `prepare_data.py` guards
against this with `--min-split` (default 4): each requested split gets at
least that many examples when data allows, and splits that would end up
smaller are dropped (with a note) instead of written. If you raise
`batch_size` in the config, raise `--min-split` to match. `scripts/train.sh`
also preflights the splits and fails with an actionable error before
training starts.

For prompt/completion tuning, `--format completions` flattens multi-turn
chats (all but the last message become the prompt). Tip: keep each example
short (`--max-seq-length` in the config caps token length anyway); long
examples cost memory quadratically.

Hugging Face datasets also work directly — either pass a pre-formatted repo
as `data:` (e.g. `mlx-community/wikisql`) or map features in the config:

```yaml
hf_dataset:
  path: "billsum"
  prompt_feature: "text"
  completion_feature: "summary"
  train_split: "train[:90%]"
  valid_split: "train[-90%:]"
```

## Training

```bash
./scripts/train.sh                                # configs/lora.yaml
./scripts/train.sh --iters 200 --batch-size 2     # CLI flags override the YAML
./scripts/train.sh --model mlx-community/SmolLM2-360M-Instruct-4bit
```

Key knobs in `configs/lora.yaml`:

| Key | Meaning | Notes |
|-----|---------|-------|
| `model` | HF repo or local path | 4-bit `mlx-community/*` models recommended |
| `iters` | optimizer steps | demo ships `25` (~2 epochs on the sample; validation minimum — see measured curve in the config); use 500–2000 for real datasets and watch `valid.jsonl` loss |
| `learning_rate` | base LR | default `1e-5` — mlx-lm applies `scale` as a raw multiplier, so effective step ≈ lr × scale; the official example pairs scale 20 with 1e-5. **1e-4 made tuned worse than base** (see `configs/lora.yaml` comment) |
| `mask_prompt` | loss on assistant reply only | default `true` for chat/completions data; **must be `false` for `text` data** (mlx-lm raises) |
| `num_layers` | last N layers get LoRA | `-1` = all layers |
| `lora_parameters` | `rank` / `scale` / `dropout` | **mlx-lm uses `scale`, not `alpha`** (≈ `alpha/rank` elsewhere) |
| `batch_size` | micro-batch | raise `grad_accumulation_steps` for a bigger effective batch without more memory |
| `max_seq_length` | token cap per example | lower it if you hit memory limits |
| `grad_checkpoint` | recompute activations | less memory, slower |
| `fine_tune_type` | `lora` \| `dora` \| `full` | `full` needs a non-quantized model and lots of memory — only for tiny models |

Other useful flags:

```bash
./scripts/train.sh --resume-adapter-file outputs/adapters/adapters.safetensors
./scripts/train.sh --report-to wandb --project-name my-slm   # needs `pip install wandb`
```

Checkpoints land in `outputs/adapters/` (`adapters.safetensors` +
`adapter_config.json`), re-saved every `save_every` steps.

## Evaluation

```bash
make eval                 # or: ./scripts/eval.sh
PROMPT="Your question here" ./scripts/eval.sh
```

Runs two things: test-set perplexity for **both the base model and the tuned
adapter** via `mlx_lm.lora --test` (needs `data/processed/test.jsonl`, created
with `--test-split`), printed side by side with the relative change, plus a
sample generation with the adapter loaded:

```text
==> Test perplexity (data/processed/test.jsonl)
  base  Test loss 4.674, Test ppl 107.097.
  tuned Test loss 3.101, Test ppl 22.223.
  delta  -79.2% perplexity (tuned vs base; lower is better)
```

`eval.sh` accepts only a config path — no training or generation flags. The
two mlx-lm subcommands take different options, so forwarding shared flags
would break one of them; set `PROMPT=...` to change the sample prompt.

## Chat and fusion

```bash
make chat                          # REPL with the adapter applied
SYSTEM="You are a terse assistant." ./scripts/chat.sh

make fuse                          # -> fused_model/, a standalone model
mlx_lm.generate --model fused_model --prompt "Hello"   # no adapter needed
```

Fusing bakes the low-rank weights into the base model for deployment (the
adapter file is no longer required at inference time).

## Recommended SLM bases (4-bit)

| Model | Size | Good for |
|-------|------|----------|
| `mlx-community/Qwen2.5-0.5B-Instruct-4bit` | ~0.5B | default; solid all-rounder |
| `mlx-community/Qwen3-0.6B-4bit` | ~0.6B | newer reasoning-style base |
| `mlx-community/SmolLM2-360M-Instruct-4bit` | 360M | very fast iteration |
| `mlx-community/Llama-3.2-1B-Instruct-4bit` | 1B | higher quality, more memory |
| `mlx-community/gemma-3-270m-it-4bit` | 270M | smallest fast on-device option |

## Memory tips (from MLX docs)

1. Use a 4-bit model (QLoRA) — this config already defaults to one.
2. Lower `batch_size` to 1–2, or raise `grad_accumulation_steps` to keep the
   effective batch size.
3. Lower `num_layers` (16 → 8 → 4).
4. Shorter examples (`max_seq_length`, and split long docs in
   `prepare_data.py`).
5. Turn on `grad_checkpoint: true`.

## Troubleshooting

- **`ValueError: Dataset must have at least batch_size=N examples`** — one of
  your splits (`train/valid/test.jsonl`) is smaller than `batch_size`. Re-run
  `scripts/prepare_data.py --min-split <batch_size>`, lower `batch_size`, or
  pass `--batch-size` ≤ the smallest split. `train.sh` catches this before
  training starts.
- **`KeyError: LoRA config is missing required key 'scale'`** — mlx-lm's
  `lora_parameters` takes `rank`/`scale`/`dropout`, not PEFT-style `alpha`.
  Set `scale: 20.0` (or `alpha / rank` from other frameworks).
- **`ModuleNotFoundError: lm_eval`** — `mlx_lm.evaluate` needs the optional
  `lm-eval` package; this scaffold evaluates via `mlx_lm.lora --test`
  instead (`make eval`).
- **Loss goes to NaN** — lower `learning_rate` (the default `1e-5` is
  already conservative) or lower `lora_parameters.scale`, reduce
  `max_seq_length`, or check for malformed JSONL lines.
- **Tuned model evaluates worse than the base model** — you overfit: train
  loss near zero while validation rises again. Lower `learning_rate` (or
  `scale`), reduce `iters`, or add data. `./scripts/eval.sh` prints
  base-vs-tuned perplexity precisely to catch this.
- **Out of memory** — see Memory tips above.
- **Training runs but generation looks unchanged** — increase `iters` and
  check the validation loss trend in the training log; also confirm the chat
  template matches (train and infer with the same model).
- **Upgrading mlx-lm** — config schemas change between releases (this
  scaffold matches 0.32.x). After upgrading, re-check `mlx_lm.lora --help`
  and the bundled `LORA.md`.
