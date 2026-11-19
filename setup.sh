#!/usr/bin/env bash
# Create a virtualenv and install MLX-LM fine-tuning dependencies.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ ! -d .venv ]]; then
  if command -v uv >/dev/null 2>&1; then
    echo "==> Creating venv with uv"
    uv venv .venv --python 3.12
    uv pip install --python .venv/bin/python -r requirements.txt
  else
    echo "==> Creating venv with python3"
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -r requirements.txt
  fi
else
  echo "==> .venv already exists, skipping creation"
fi

echo "==> Installed versions:"
.venv/bin/python -c "import mlx.core as mx; print('mlx:', mx.__version__)"
.venv/bin/python -c "import mlx_lm; print('mlx-lm:', getattr(mlx_lm, '__version__', 'unknown'))"
echo "Done. Activate with: source .venv/bin/activate"
