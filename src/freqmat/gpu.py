"""Pick the GPU to train on: physical GPU 1 when the machine has one, otherwise GPU 0.

RecBole restricts the process to ``gpu_id`` through ``CUDA_VISIBLE_DEVICES``, so choosing is just a matter of giving it
the right id before CUDA starts. The GPUs are listed with ``nvidia-smi`` (never through torch) so that no CUDA state is
created before RecBole sets the variable. Override with ``FREQMAT_GPU=<id>`` or ``-s gpu_id=<id>``.
"""

import os
import subprocess

PREFERRED, FALLBACK = "1", "0"
ENV_OVERRIDE = "FREQMAT_GPU"


def physical_gpus() -> list[str]:
    """Ids of the GPUs this process may use (``nvidia-smi`` order, narrowed by an existing CUDA_VISIBLE_DEVICES)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    ids = out.split()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        ids = [i for i in (v.strip() for v in visible.split(",")) if i in ids]
    return ids


def choose_gpu(available: list[str] | None = None) -> str | None:
    """``FREQMAT_GPU`` if set, else GPU 1 if present, else GPU 0, else the first GPU; ``None`` when there is no GPU."""
    if os.environ.get(ENV_OVERRIDE):
        return os.environ[ENV_OVERRIDE]
    ids = physical_gpus() if available is None else available
    if not ids:
        return None
    for wanted in (PREFERRED, FALLBACK):
        if wanted in ids:
            return wanted
    return ids[0]
