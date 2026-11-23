#!/usr/bin/env bash
# Install the official Mamba-3 kernels into .venv (pinned commit, CUDA only, no nvcc needed for SISO).
# --no-deps keeps the torch/triton versions pinned by uv.lock; tilelang (MIMO only) breaks `import mamba_ssm`.
# Re-run this after every `uv sync` / environment rebuild: uv removes packages that are not in uv.lock.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
uv pip install --python "$PY" einops packaging ninja wheel "setuptools>=68" huggingface_hub transformers
uv pip install --python "$PY" --no-build-isolation --no-deps \
  git+https://github.com/state-spaces/mamba.git@e9594ce1c732d97440f0332fdc43170a2294dbfa
uv pip uninstall --python "$PY" tilelang 2>/dev/null || true
"$PY" -c "from mamba_ssm import Mamba3; print('mamba_ssm OK')"
