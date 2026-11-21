# SLM fine-tuning with MLX-LM — Makefile convenience targets.
# All targets assume ./setup.sh has been run first.

PY      := .venv/bin/python
CONFIG  := configs/lora.yaml
SAMPLE  := data/sample

# Default sweep: 3 learning rates x 2 ranks = 6 runs. Override: make sweep GRID="--grid num_layers=8,-1"
GRID    := --grid learning_rate=2e-5,5e-5,1e-4 --grid lora_parameters.rank=8,16

.PHONY: setup data smoke train eval chat fuse sweep test clean help

help:  ## Show this help
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-8s %s\n", $$1, $$2}'

setup:  ## Create venv and install dependencies
	./setup.sh

data:  ## Build train/valid/test.jsonl (+ stats.json) from data/sample
	$(PY) scripts/prepare_data.py --input $(SAMPLE) --output data/processed \
		--format chat --test-split 0.05

smoke:  ## ~1 min pipeline check on the sample data (not a real recipe)
	./scripts/train.sh configs/smoke.yaml

train:  ## LoRA fine-tune using $(CONFIG)
	./scripts/train.sh $(CONFIG)

eval:  ## Base vs tuned on held-out data + a sample generation
	./scripts/eval.sh $(CONFIG)

chat:  ## Interactive chat with the fine-tuned model
	./scripts/chat.sh $(CONFIG)

fuse:  ## Merge adapter into a standalone model (fused_model/)
	./scripts/fuse.sh $(CONFIG)

sweep:  ## Ablate hyperparameters; rank on valid, confirm winner on test
	$(PY) scripts/sweep.py --config $(CONFIG) $(GRID)

test:  ## Run the unit tests (the e2e test needs mlx; it skips itself if absent)
	$(PY) -m pytest -q tests

clean:  ## Remove generated data and training outputs
	rm -rf data/processed/*.jsonl data/processed/stats.json outputs/adapters* \
		outputs/smoke-adapters outputs/compare outputs/sweeps outputs/.resolved fused_model
