#!/usr/bin/env bash
set -euo pipefail

# CUDA 12.8 base images include a forward-compatibility libcuda and cuDNN.
# RunPod's RTX 4090 hosts currently use a CUDA 12.4-capable 550 driver, so use
# the host driver while allowing PyTorch to load the cuDNN bundled in its wheel.
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib"
if [[ -e /usr/local/nvidia/lib64/libcuda.so.1 ]]; then
  export LD_PRELOAD="/usr/local/nvidia/lib64/libcuda.so.1${LD_PRELOAD:+:${LD_PRELOAD}}"
fi

/app/.venv/bin/python - <<'PY'
import torch

cudnn_version = torch.backends.cudnn.version()
print(
    "torch",
    torch.__version__,
    "cuda_build",
    torch.version.cuda,
    "cudnn",
    cudnn_version,
    "available",
    torch.cuda.is_available(),
    flush=True,
)
if not str(torch.__version__).startswith("2.6.0+cu124"):
    raise SystemExit(f"Unexpected PyTorch build: {torch.__version__}")
if torch.version.cuda != "12.4":
    raise SystemExit(f"Unexpected CUDA build: {torch.version.cuda}")
if cudnn_version is None:
    raise SystemExit("cuDNN is not available")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available; refusing to start the TTS API")
PY

: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"

if [[ -z "${IRODORI_MODEL_REGISTRY:-}" && -f "/workspace/irodori_artifacts/configs/model_registry.runpod.json" ]]; then
  export IRODORI_MODEL_REGISTRY="/workspace/irodori_artifacts/configs/model_registry.runpod.json"
fi

if [[ -n "${IRODORI_ARTIFACT_SYNC_CMD:-}" ]]; then
  bash -lc "${IRODORI_ARTIFACT_SYNC_CMD}"
fi

if [[ -z "${IRODORI_MODEL_REGISTRY:-}" ]]; then
  echo "IRODORI_MODEL_REGISTRY is not set and no workspace registry was found." >&2
  exit 1
fi

# Do not use `uv run` here. It would resync pyproject.toml and replace the
# CUDA 12.4-compatible PyTorch installed in this deployment image.
exec /app/.venv/bin/python -m uvicorn api_server:app \
  --host "${HOST}" \
  --port "${PORT}" \
  --workers 1
