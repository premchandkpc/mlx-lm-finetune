# SLM fine-tuning with MLX-LM — Makefile convenience targets.
# All targets assume ./setup.sh has been run first.

PY      := .venv/bin/python
CONFIG  := configs/lora.yaml
SAMPLE  := data/sample

.PHONY: setup data train eval chat fuse clean help

help:  ## Show this help
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-8s %s\n", $$1, $$2}'

setup:  ## Create venv and install dependencies
	./setup.sh

data:  ## Build train/valid/test.jsonl from data/sample
	$(PY) scripts/prepare_data.py --input $(SAMPLE) --output data/processed \
		--format chat --test-split 0.05 --shuffle

train:  ## LoRA fine-tune using $(CONFIG)
	./scripts/train.sh $(CONFIG)

eval:  ## Test-set perplexity + sample generation
	./scripts/eval.sh $(CONFIG)

chat:  ## Interactive chat with the fine-tuned model
	./scripts/chat.sh $(CONFIG)

fuse:  ## Merge adapter into a standalone model (fused_model/)
	./scripts/fuse.sh $(CONFIG)

clean:  ## Remove generated data and training outputs
	rm -rf data/processed/*.jsonl outputs/adapters fused_model
