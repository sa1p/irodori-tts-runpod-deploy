# RunPod Production Deployment

This deployment keeps the Docker image small and stores model artifacts on a
RunPod Network Volume.

## Artifact Layout

The production layout uses one base Irodori checkpoint plus per-voice LoRA
adapters:

```text
/workspace/irodori_artifacts/
  configs/model_registry.runpod.json
  models/Irodori-TTS-500M-v2/model.safetensors
  outputs/irodori_loras/<model_id>/checkpoint/checkpoint_final/
    adapter_config.json
    adapter_model.safetensors
    config.json
    irodori_lora_metadata.json
  refs/<model_id>/reference.wav
```

This keeps the artifact volume small. In production the API defaults to
`IRODORI_LORA_LOAD_MODE=delta`: it keeps the base checkpoint resident and applies
the selected LoRA delta in-place while still storing only lightweight adapters on
the Network Volume.

## Prepare Artifacts

Validate the local registry and output paths without copying files:

```powershell
uv run python scripts\prepare_runpod_artifacts.py --no-copy
```

Create the upload directory:

```powershell
uv run python scripts\prepare_runpod_artifacts.py
```

Upload the contents of `runpod_artifacts` to the Network Volume so the root
inside the Pod is `/workspace/irodori_artifacts`.

## Build And Push Image

Build from this repository and push to a registry RunPod can pull:

```powershell
docker build -f Dockerfile.runpod -t <registry>/irodori-tts-kohaku-api:<tag> .
docker push <registry>/irodori-tts-kohaku-api:<tag>
```

Do not bake `models`, `outputs`, `data`, or `runpod_artifacts` into the image.
Those paths are intentionally excluded by `.dockerignore`.

## Create Pod

Set the API key only in the current shell:

```powershell
$env:RUNPOD_API_KEY = "<your key>"
```

Create a Pod after the image exists and the Network Volume has the artifacts:

```powershell
.\scripts\runpod_deploy_pod.ps1 `
  -Image "<registry>/irodori-tts-kohaku-api:<tag>" `
  -NetworkVolumeId "<runpod-network-volume-id>" `
  -GpuId "NVIDIA GeForce RTX 4090"
```

The container starts `scripts/runpod_start.sh`, reads
`/workspace/irodori_artifacts/configs/model_registry.runpod.json`, and serves:

```text
GET  /health
GET  /readyz
GET  /v1/models
POST /v1/tts
```

## Runtime Defaults

The RunPod entrypoint defaults to:

```text
IRODORI_MODEL_DEVICE=cuda
IRODORI_MODEL_PRECISION=bf16
IRODORI_CODEC_DEVICE=cuda
IRODORI_CODEC_PRECISION=fp32
IRODORI_LORA_LOAD_MODE=delta
API num_steps=6
API t_schedule_mode=sway
API sway_coeff=-1.0
API trim_tail=true
IRODORI_CHUNK_BATCH_SIZE=2
IRODORI_MAX_CACHED_RUNTIMES=1
IRODORI_PRELOAD_MODELS=false
```

`IRODORI_LORA_LOAD_MODE=delta` keeps a single base model resident and applies the
selected LoRA delta in-place, which is the production switching path for this API.
The request body can still override `num_steps`, `t_schedule_mode`, `sway_coeff`,
and `trim_tail` for targeted quality checks.

## Production Notes

- Use a 24GB GPU class for the first production Pod. 16GB can work, but leaves
  less headroom for codec, longer chunks, and concurrent overhead.
- Keep the Network Volume as the source of truth for model artifacts and mirror
  `runpod_artifacts/artifact_manifest.json` to external storage after each
  release.
- Keep the image tag immutable per release.
- Expose the Pod through a private proxy or gateway if this becomes public API;
  this repository currently assumes local or trusted-network access.
