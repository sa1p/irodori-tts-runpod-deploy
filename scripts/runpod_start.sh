#!/usr/bin/env bash
set -euo pipefail

: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"
: "${IRODORI_MODEL_DEVICE:=cuda}"
: "${IRODORI_MODEL_PRECISION:=bf16}"
: "${IRODORI_CODEC_DEVICE:=cuda}"
: "${IRODORI_CODEC_PRECISION:=fp32}"
: "${IRODORI_LORA_LOAD_MODE:=merged}"
: "${IRODORI_MAX_CACHED_RUNTIMES:=1}"
: "${IRODORI_PRELOAD_MODELS:=false}"

if [[ -z "${IRODORI_MODEL_REGISTRY:-}" && -f "/workspace/irodori_artifacts/configs/model_registry.runpod.json" ]]; then
  export IRODORI_MODEL_REGISTRY="/workspace/irodori_artifacts/configs/model_registry.runpod.json"
fi

if [[ -n "${IRODORI_ARTIFACT_SYNC_CMD:-}" ]]; then
  bash -lc "${IRODORI_ARTIFACT_SYNC_CMD}"
fi

if [[ -z "${IRODORI_MODEL_REGISTRY:-}" ]]; then
  echo "IRODORI_MODEL_REGISTRY is not set and /workspace/irodori_artifacts/configs/model_registry.runpod.json was not found." >&2
  exit 1
fi

exec uv run uvicorn api_server:app --host "${HOST}" --port "${PORT}" --workers 1
