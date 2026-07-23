#!/usr/bin/env bash
set -euo pipefail

# CUDA 12.8 base images include a forward-compatibility libcuda. RunPod's
# RTX 4090 hosts currently use a CUDA 12.4-capable 550 driver, so the host
# driver must take precedence over the image compatibility library.
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
if [[ -e /usr/local/nvidia/lib64/libcuda.so.1 ]]; then
  export LD_PRELOAD="/usr/local/nvidia/lib64/libcuda.so.1${LD_PRELOAD:+:${LD_PRELOAD}}"
fi

/app/.venv/bin/python - <<'PY'
import torch

print(
    "torch",
    torch.__version__,
    "cuda_build",
    torch.version.cuda,
    "available",
    torch.cuda.is_available(),
    flush=True,
)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available; refusing to start the TTS API")
PY

exec /app/scripts/runpod_start_lb.sh
