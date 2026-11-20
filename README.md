# mlx-lm-finetune

Fine-tune small language models (SLMs) on Apple silicon with
[MLX-LM](https://github.com/ml-explore/mlx-lm): data preparation with leakage
protection, LoRA/DoRA training from tuned presets, a **base-vs-tuned evaluation**,
a hyperparameter **sweep**, chat, and adapter fusion.

Verified against `mlx==0.32.4` / `mlx-lm==0.32.2`. Training needs an Apple silicon
Mac; everything else (data prep, config tooling, the test suite, including an
end-to-end run on a tiny random model) also runs on Linux CPU.

## Quickstart

```bash
./setup.sh                 # venv + dependencies
make data                  # data/sample -> data/processed (+ stats.json)
make smoke                 # ~1 min plumbing check (NOT a real recipe)
make train                 # configs/lora.yaml
make eval                  # base vs tuned on held-out data -> outputs/compare/report.md
make chat
```

The bundled sample is 18 rows, enough to prove the pipeline works and nothing more.
`make train` on it prints a warning (~400 epochs over 10 rows = memorization).
For real work, point `data:` at your own corpus (see below).

## What changed vs. a plain `mlx_lm.lora` wrapper

| Problem | What this repo does |
|---|---|
| Demo looped ~200 epochs over 10 rows | `train.sh` preflight warns when epochs are absurd (>10) or <1; `prepare_data.py` prints a suggested `iters` |
| Loss included system/user tokens | `mask_prompt: true` in the base config (loss on answers only) |
| `lr_schedule` silently ignored `--iters` / `--learning-rate` | `resolve_config.py` re-syncs the schedule (peak LR, decay steps) before every run |
| `1e-4` read as a *string* by PyYAML | `cfgtools` coerces scientific-notation floats |
| No way to answer "did it help?" | `compare.py`: loss/perplexity + greedy-generation F1/EM, base vs tuned, same split |
| Hyperparameters by guesswork | `sweep.py`: grid, rank on **valid**, confirm only the winner on **test** |
| Splits could share prompts (leakage) | group-aware split + a final leakage audit that fails the run |
| `eval.sh` forwarded one flag set to two different CLIs | extra args go to `compare.py`; `GEN_ARGS` goes to `mlx_lm.generate` |

## Layout

```
configs/
  lora.yaml      recommended base recipe (QLoRA, all layers, rank 16, cosine LR)
  smoke.yaml     1-minute pipeline check
  quality.yaml   1.5B base, rank 32, 2k context, grad checkpointing
  dora.yaml      DoRA variant
  lowmem.yaml    8 GB-friendly (batch 1 + accumulation, 8 layers, 512 ctx)
data/sample/     tiny chat-format example
scripts/
  prepare_data.py   validate -> filter -> dedup -> group-aware split -> stats.json
  cfgtools.py       config loading (`extends:`, float coercion), schedule sync, preflight
  resolve_config.py merged config for a run (used by train/eval/fuse)
  train.sh eval.sh chat.sh fuse.sh
  compare.py        base vs tuned report
  sweep.py          hyperparameter grid
tests/              79 tests incl. a real end-to-end run on a tiny offline model
```

## Presets

Presets use `extends: lora.yaml` and override a few keys, so they stay in sync with
the base recipe. **The `extends:` key is understood by this repo's scripts only**:
use `./scripts/train.sh configs/quality.yaml`, never `mlx_lm.lora --config
configs/quality.yaml` directly.

| Preset | Use when |
|---|---|
| `lora.yaml` | default; start here |
| `quality.yaml` | you have ~16 GB+ and want the best result from a bigger base |
| `lowmem.yaml` | OOM, or an 8 GB machine |
| `dora.yaml` | you want to test DoRA against LoRA on your data (sweep it) |
| `smoke.yaml` | verifying the setup |

The values in `lora.yaml` are **reasoned starting points, not tuned results** (e.g.
`learning_rate: 5e-5` rather than `1e-4`, because `scale: 20` multiplies the
low-rank update strongly). Find yours with `make sweep`.

## Data

Local datasets are `train.jsonl` (required), `valid.jsonl` and `test.jsonl` in one
directory. Formats: `chat` (`{"messages": [...]}`), `text`, `completions`
(`{"prompt","completion"}`).

```bash
python scripts/prepare_data.py --input raw/ --output data/processed --format chat \
    --valid-split 0.05 --test-split 0.05 --dedup near \
    --group-key doc_id \
    --tokenizer mlx-community/Qwen2.5-0.5B-Instruct-4bit --max-tokens 1024
```

- **Validation**: bad JSON, unknown roles, empty content, and chats that do not end
  on an assistant turn are dropped and counted.
- **Dedup**: `--dedup exact` (default) or `near` (MinHash-LSH candidates verified by
  true Jaccard; `--near-threshold`, default 0.85).
- **No leakage**: rows are split by *group*: `--group-key` (e.g. a document id) or,
  by default, the normalized prompt, so the same question can never be in train and
  test. A final audit aborts the run if any overlap remains.
- **Shuffled by default** (`--no-shuffle` for the old head-of-file behaviour).
- `stats.json`: counts, drops per reason, length percentiles (tokens with
  `--tokenizer`), and a suggested `iters` for `--epochs` (default 3).
- mlx-lm needs ≥ `batch_size` rows in every split it reads; splits smaller than
  `--min-split` (default 4) are dropped rather than written.

Keep `--max-tokens` equal to `max_seq_length`: longer examples are silently
truncated by mlx-lm, which usually cuts off the answer.

## Train

```bash
./scripts/train.sh                                  # configs/lora.yaml
./scripts/train.sh configs/quality.yaml
./scripts/train.sh --iters 200 --learning-rate 1e-4 # schedule follows the overrides
./scripts/train.sh --resume-adapter-file outputs/adapters/adapters.safetensors
```

Rule of thumb: `iters ≈ epochs × n_train / (batch_size × grad_accumulation_steps)`,
with 2-5 epochs for small datasets.

## Evaluate: did it help?

```bash
make eval                                      # -> outputs/compare/report.md
./scripts/eval.sh configs/lora.yaml --split valid --gen-limit 50
```

Reports, for base and tuned on the **same** held-out split: loss/perplexity (same
masking as training; matches `mlx_lm.lora --test`), and greedy-generation exact
match, token F1 and length ratio vs the reference answers, plus side-by-side samples
and a plain-language verdict.

Read it skeptically. Loss can drop while answers get no better; a small held-out set
makes deltas under a few percent noise; token F1 is crude, so for a real task add a
task metric.

## Sweep

```bash
make sweep                                                  # 3 LRs x 2 ranks
python scripts/sweep.py --grid num_layers=8,16,-1 --grid fine_tune_type=lora,dora
python scripts/sweep.py --grid learning_rate=1e-5,1e-4 --dry-run
```

Each run trains in its own directory with its own resolved config. Runs are ranked
on **valid**; only the winner is then scored on **test**, so test data never picks
hyperparameters. Re-running the same `--name` resumes (finished runs are skipped).
Results land in `outputs/sweeps/<name>/results.{md,csv}`.

## Fuse and serve

```bash
make fuse                                   # -> fused_model/ (re-quantized like the base)
DEQUANTIZE=1 ./scripts/fuse.sh              # fp16 instead
mlx_lm.generate --model fused_model --prompt "Hello"
mlx_lm.server   --model fused_model         # OpenAI-compatible HTTP endpoint
```

## Tests

```bash
pip install -r requirements-dev.txt
make test
```

Unit tests cover the data pipeline, config tooling, report maths and sweep logic.
`tests/test_e2e_tiny.py` builds a tiny random-weight model offline and runs the real
`mlx_lm.lora`, `compare.py` and `sweep.py` on CPU; it skips itself when mlx is not
installed.

## Troubleshooting

- **`Dataset must have at least batch_size=N examples`**: a split is smaller than
  `batch_size`. Re-run `prepare_data.py --min-split N`, or lower `batch_size`.
- **`KeyError ... 'scale'`**: mlx-lm's `lora_parameters` takes `rank`/`scale`/
  `dropout`, not PEFT's `alpha`.
- **Loss goes to NaN**: lower `learning_rate`, reduce `max_seq_length`, check for
  malformed JSONL.
- **Out of memory**: use `configs/lowmem.yaml`, or lower `batch_size` / `num_layers` /
  `max_seq_length` and turn on `grad_checkpoint`.
- **Generation looks unchanged**: check `make eval`; raise `iters` only if you have
  the data to support it; make sure train and inference use the same chat template.
- **Adapter file name**: mlx-lm writes `adapters.safetensors` (plural) plus numbered
  checkpoints `0000100_adapters.safetensors`.
- **Upgrading mlx-lm**: config schemas change between releases; re-check
  `mlx_lm.lora --help` and run `make test`.

## Requirements

Apple silicon Mac for training (MLX does not run on Intel/CUDA); Python 3.12
recommended; ~0.5-2 GB disk per 4-bit SLM download (cached by Hugging Face).
