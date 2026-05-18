# Kohaku Irodori-TTS LoRA API

This repository is a local fine-tuning/API wrapper around Irodori-TTS v3 for the
Kohaku dataset at `D:\sbv2\Style-Bert-VITS2\Data\kohaku-haishin_v2`.

## Multi-Model Production Shape

The local API now supports multiple Irodori checkpoints through a registry file.
Each model entry points to a merged Irodori `.safetensors` checkpoint and a fixed
reference WAV. The runtime uses an LRU cache, so `IRODORI_MAX_CACHED_RUNTIMES=1`
keeps VRAM low and reloads on model switches, while larger values trade VRAM for
lower switch latency.

Within each loaded runtime, reference audio is also encoded once and reused from
an in-process LRU cache. This avoids re-encoding the same reference WAV on every
chunk/request. Tune it with `IRODORI_REF_LATENT_CACHE_SIZE` (default: `64`, set
`0` to disable).

```powershell
$env:IRODORI_MODEL_REGISTRY = "configs/model_registry.example.json"
$env:IRODORI_MAX_CACHED_RUNTIMES = "1"
$env:IRODORI_PRELOAD_MODELS = "true"
uv run uvicorn api_server:app --host 127.0.0.1 --port 8000
```

List models and generate audio:

```powershell
curl.exe http://127.0.0.1:8000/v1/models

curl.exe -X POST http://127.0.0.1:8000/v1/tts `
  -H "Content-Type: application/json" `
  -d "{\"model_id\":\"kohaku\",\"text\":\"今日も来てくれてありがとう。\",\"format\":\"mp3\"}" `
  --output api_smoke.mp3
```

For lowest latency in production, prefer one hot model per Pod when traffic is
predictable. Use the multi-model cache for admin tools, low-QPS models, or a
small set of frequently used voices that can fit in VRAM together.

## Local Setup

```powershell
cd D:\github.com\Trippy-inc\irodori-tts-kohaku-api
uv sync
```

## Prepare Data

Convert the Style-Bert-VITS2 `esd.list` into JSONL for `datasets`.
The initial run excludes `kohaku-haishin-60.wav` because it is 36.75 seconds.

```powershell
uv run python scripts\convert_esd_to_jsonl.py
```

Precompute DACVAE latents. This repository uses a local helper instead of
upstream `prepare_manifest.py` because `datasets.Audio` can depend on
`torchcodec` FFmpeg DLL loading on Windows.

```powershell
uv run python scripts\prepare_local_manifest.py `
  --input-jsonl data/kohaku/train.jsonl `
  --output-manifest data/kohaku/train_manifest.jsonl `
  --latent-dir data/kohaku/latents `
  --device cuda `
  --target-sample-rate 48000
```

## Fine-Tune

Download the v3 base checkpoint:

```powershell
uv run python scripts\download_base_model.py
```

Run LoRA fine-tuning:

```powershell
uv run python train.py `
  --config configs/train_500m_v3_lora.yaml `
  --manifest data/kohaku/train_manifest.jsonl `
  --output-dir outputs/kohaku_lora `
  --init-checkpoint models/Irodori-TTS-500M-v3/model.safetensors `
  --device cuda
```

Convert the final LoRA adapter to an inference checkpoint:

```powershell
uv run python convert_checkpoint_to_safetensors.py `
  outputs/kohaku_lora/checkpoint_final `
  --base-checkpoint models/Irodori-TTS-500M-v3/model.safetensors `
  --output outputs/kohaku_lora/kohaku_lora_merged.safetensors
```

## Local API

The API uses `kohaku-haishin-2.wav` as the default fixed reference voice.
By default, API synthesis uses the fast production sampler:
`num_steps=6`, `t_schedule_mode=sway`, `sway_coeff=-1.0`, and
`trim_tail=true`. When `seconds` and `chunk_seconds` are omitted, v3
checkpoints use automatic duration prediction; use `duration_scale` to make the
predicted length longer or shorter.

For v3 reference-voice tuning, `/v1/tts` and `/v1/tts/stream` accept optional
`reference_wav`, `reference_latent`, `cfg_scale_speaker`, `speaker_kv_scale`,
`speaker_kv_min_t`, and `speaker_kv_max_layers`. If `reference_wav` and
`reference_latent` are omitted, the registry/default reference WAV is used, so
existing clients remain compatible.

```powershell
$env:IRODORI_CHECKPOINT = "outputs/kohaku_lora/kohaku_lora_merged.safetensors"
$env:IRODORI_REF_WAV = "D:\sbv2\Style-Bert-VITS2\Data\kohaku-haishin_v2\raw\haishin\kohaku-haishin-2.wav"
$env:IRODORI_MODEL_DEVICE = "cuda"
$env:IRODORI_MODEL_PRECISION = "bf16"
$env:IRODORI_CODEC_DEVICE = "cuda"
$env:IRODORI_CODEC_PRECISION = "fp32"
uv run uvicorn api_server:app --host 127.0.0.1 --port 8000
```

Generate WAV:

```powershell
curl.exe -X POST http://127.0.0.1:8000/v1/tts `
  -H "Content-Type: application/json" `
  -d "{\"text\":\"今日も来てくれてありがとう。\"}" `
  --output api_smoke.wav
```

Recommended MP3 streaming:

```powershell
curl.exe -N -X POST http://127.0.0.1:8000/v1/tts/stream `
  -H "Content-Type: application/json" `
  -d "{\"model_id\":\"kohaku004\",\"text\":\"今日も来てくれてありがとう。\",\"auto_split\":true,\"duration_scale\":1.0,\"max_chunk_chars\":100}" `
  --output stream_smoke.mp3
```

`/v1/tts/stream` always returns `audio/mpeg`. The server keeps one ffmpeg
encoder open per request, writes each generated chunk as raw PCM, and streams
MP3 bytes as soon as the encoder emits them. This avoids concatenating separate
MP3 files and keeps the output compatible with normal HTTP audio clients.

## SBV2 Inventory And Distillation

The target SBV2 model folder list is defined in `scripts/irodori_targets.py`.
First inventory the 38 folders and detect which models already have recorded
`esd.list` datasets:

```powershell
uv run python scripts\inventory_irodori_targets.py --pretty
```

Models without recorded datasets are treated as merged SBV2 teachers. Generate a
distilled dataset by asking SBV2 to synthesize a neutral prompt set. The default
layout expects the SBV2 source checkout at `C:\sbv2\Style-Bert-VITS2` and the
model/data assets at `D:\sbv2\Style-Bert-VITS2`.

```powershell
uv run python scripts\distill_sbv2_dataset.py `
  --models hiyoko_tentyuou,nana02 `
  --mode subprocess `
  --sbv2-root C:\sbv2\Style-Bert-VITS2 `
  --sbv2-python C:\sbv2\Style-Bert-VITS2\venv\Scripts\python.exe `
  --limit 160 `
  --use-gpu
```

`subprocess` mode keeps SBV2 dependencies in the SBV2 venv instead of installing
them into the Irodori uv environment. If an existing SBV2 API is already running,
HTTP mode is also available:

```powershell
uv run python scripts\distill_sbv2_dataset.py `
  --models hiyoko_tentyuou `
  --mode http `
  --http-url http://127.0.0.1:5000 `
  --limit 160
```

Outputs are written under `data/distilled/<model_id>/` as `train.jsonl`,
`esd.list`, and `raw/*.wav`.

## Batch Irodori LoRA Training

After inventory and any needed distillation, run the Irodori preparation,
training, conversion, and registry generation pipeline. Use `--max-models` or
`--models` for pilot runs before launching all 38.

```powershell
uv run python scripts\batch_train_irodori_loras.py `
  --models kohaku004,hachiroku_20251208_ml `
  --skip-existing `
  --device cuda
```

For a short smoke run:

```powershell
uv run python scripts\batch_train_irodori_loras.py `
  --models hiyoko_tentyuou `
  --stages jsonl,manifest,train,convert,registry `
  --max-steps 100 `
  --skip-existing `
  --device cuda
```

The generated API registry is `configs/model_registry.local.json`. Start the API
with it:

```powershell
$env:IRODORI_MODEL_REGISTRY = "configs/model_registry.local.json"
uv run uvicorn api_server:app --host 127.0.0.1 --port 8000
```

## Full Production Pipeline

To run the whole local production pipeline for all configured models, use the
PowerShell orchestrator. It inventories assets, distills SBV2-only voices through
the SBV2 venv, prepares DACVAE latents, trains LoRA adapters, converts them to
merged safetensors, writes `configs/model_registry.local.json`, and restarts the
local API when complete.

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass `
  -File scripts\run_full_production_pipeline.ps1 `
  -DistillLimit 160 `
  -MaxSteps 1000 `
  -StartApiWhenDone
```

Check progress while it runs:

```powershell
uv run python scripts\pipeline_status.py `
  --inventory logs\<run>\inventory.json `
  --distill-target-rows 160
```

## RunPod Pod Deployment

Build the image:

```powershell
docker build -f Dockerfile.runpod -t irodori-tts-api:runpod .
```

Recommended runtime layout:

- Put merged checkpoints, registry JSON, and reference WAV files on a persistent
  RunPod volume or sync them from object storage at boot.
- Set `IRODORI_MODEL_REGISTRY` to the mounted registry path.
- Keep `--workers 1`; the API already serializes GPU synthesis with a lock.
- For failover, keep the same artifact bundle in object storage and run a second
  Pod from the same image. A load balancer or client retry can switch to the
  warm standby.
- For high-QPS voices, deploy one Pod per voice and preload that single model.
  For long-tail voices, a shared multi-model Pod with `IRODORI_MAX_CACHED_RUNTIMES=1`
  is cheaper but has model switch latency.

---

# Upstream Irodori-TTS

[![Model](https://img.shields.io/badge/Model-HuggingFace-yellow)](https://huggingface.co/Aratako/Irodori-TTS-500M-v3)
[![VoiceDesign](https://img.shields.io/badge/VoiceDesign-HuggingFace-orange)](https://huggingface.co/Aratako/Irodori-TTS-500M-v2-VoiceDesign)
[![Demo](https://img.shields.io/badge/Demo-HuggingFace%20Space-blue)](https://huggingface.co/spaces/Aratako/Irodori-TTS-500M-v3-Demo)
[![VoiceDesign Demo](https://img.shields.io/badge/VoiceDesign%20Demo-HuggingFace%20Space-red)](https://huggingface.co/spaces/Aratako/Irodori-TTS-500M-v2-VoiceDesign-Demo)
[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-green.svg)](LICENSE)

Training and inference code for **Irodori-TTS**, a Flow Matching-based Text-to-Speech model. The architecture and training design largely follow [Echo-TTS](https://jordandarefsky.com/blog/2025/echo/), using [DACVAE](https://github.com/facebookresearch/dacvae) continuous latents as the generation target.

For an OpenAI-compatible inference API server, see [Irodori-TTS-Server](https://github.com/Aratako/Irodori-TTS-Server).

> [!IMPORTANT]
> `main` tracks the **v3** codebase and is intended for use with the **Irodori-TTS-500M-v3** base model release.
> The current code remains backward-compatible with **Irodori-TTS-500M-v2** checkpoints, including **Irodori-TTS-500M-v2-VoiceDesign**.
> If you need the previous v2 codebase state, use the `v2` tag. If you need the previous v1 code, use the `v1` tag.
> v1 checkpoints / preprocessing are not compatible with v2/v3.
> The previous public v1 model is available at [Aratako/Irodori-TTS-500M](https://huggingface.co/Aratako/Irodori-TTS-500M).

For model weights and audio samples, please refer to the [base model card](https://huggingface.co/Aratako/Irodori-TTS-500M-v3) and the [VoiceDesign model card](https://huggingface.co/Aratako/Irodori-TTS-500M-v2-VoiceDesign).

## Features

- **Flow Matching TTS**: Rectified Flow Diffusion Transformer (RF-DiT) over continuous DACVAE latents
- **Voice Cloning**: Zero-shot voice cloning from reference audio
- **Voice Design**: Caption-conditioned style control
- **Automatic Duration Prediction**: v3 base checkpoints estimate output length without manual `--seconds`
- **Automatic Watermarking**: Generated audio is watermarked with [SilentCipher](https://github.com/sony/silentcipher) when available
- **Multi-GPU Training**: Distributed training via `uv run torchrun` with gradient accumulation, mixed precision (bf16), and W&B logging
- **PEFT LoRA Fine-Tuning**: Parameter-efficient adaptation with PEFT/LoRA for released checkpoints
- **Flexible Inference**: CLI, Gradio Web UI, and HuggingFace Hub checkpoint support

## Architecture

The current codebase supports two closely related checkpoint families:

1. **Base model (`Aratako/Irodori-TTS-500M-v3`)**:
   Text encoder + reference latent encoder + diffusion transformer + duration predictor. The reference latent encoder consumes patched DACVAE latents from reference audio for speaker/style conditioning. v2 base checkpoints remain supported for inference.
2. **VoiceDesign model (`Aratako/Irodori-TTS-500M-v2-VoiceDesign`)**:
   Text encoder + caption encoder + diffusion transformer. The caption encoder consumes style-control text and the speaker/reference branch is disabled. A v3 VoiceDesign release is not available yet, so this path still uses the v2 checkpoint.

Shared building blocks:

1. **Text Encoder**: Token embeddings initialized from a pretrained LLM, followed by self-attention + SwiGLU transformer layers with RoPE
2. **Condition Encoder**: Either a reference latent encoder for the base model or a caption encoder for the VoiceDesign model
3. **Diffusion Transformer**: Joint-attention DiT blocks with Low-Rank AdaLN (timestep-conditioned adaptive layer normalization), half-RoPE, and SwiGLU MLPs
4. **Duration Predictor**: v3 base checkpoints include an integrated predictor for automatic output length estimation

Audio is represented as continuous latent sequences via the codec configured by the checkpoint. The released v2/v3 checkpoints use the 32-dim [Semantic-DACVAE-Japanese-32dim](https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim) codec for 48kHz waveform reconstruction.

## Installation

```bash
git clone https://github.com/Aratako/Irodori-TTS.git
cd Irodori-TTS
uv sync
```

**Note**: For Linux/Windows with CUDA, PyTorch is automatically installed from the cu128 index. For macOS (MPS) or CPU-only usage, `uv sync` will install the default PyTorch build.

## Quick Start

### Simple Inference

```bash
uv run python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v3 \
  --text "こんにちは、私はAIです。これは音声合成のテストです。" \
  --ref-wav path/to/reference.wav \
  --output-wav outputs/sample.wav
```

### Inference without Reference Audio

```bash
uv run python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v3 \
  --text "こんにちは、私はAIです。これは音声合成のテストです。" \
  --no-ref \
  --output-wav outputs/sample.wav
```

### VoiceDesign Inference

```bash
uv run python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v2-VoiceDesign \
  --text "こんにちは、私はAIです。これは音声合成のテストです。" \
  --caption "落ち着いた女性の声で、近い距離感でやわらかく自然に読み上げてください。" \
  --no-ref \
  --output-wav outputs/sample_voice_design.wav
```

### Gradio Web UI

```bash
uv run python gradio_app.py --server-name 0.0.0.0 --server-port 7860
```

Then access the UI at `http://localhost:7860`.
The hosted v3 demo is available at [Aratako/Irodori-TTS-500M-v3-Demo](https://huggingface.co/spaces/Aratako/Irodori-TTS-500M-v3-Demo).

For the VoiceDesign checkpoint, use the dedicated UI:

```bash
uv run python gradio_app_voicedesign.py --server-name 0.0.0.0 --server-port 7861
```

The hosted VoiceDesign demo is available at [Aratako/Irodori-TTS-500M-v2-VoiceDesign-Demo](https://huggingface.co/spaces/Aratako/Irodori-TTS-500M-v2-VoiceDesign-Demo).

`gradio_app.py` is for `Aratako/Irodori-TTS-500M-v3`. `gradio_app_voicedesign.py` is for `Aratako/Irodori-TTS-500M-v2-VoiceDesign`.

## Inference

### CLI

```bash
uv run python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v3 \
  --text "こんにちは、私はAIです。これは音声合成のテストです。" \
  --ref-wav path/to/reference.wav \
  --output-wav outputs/sample.wav
```

Local checkpoints (`.pt` or `.safetensors`) are also supported:

```bash
uv run python infer.py \
  --checkpoint outputs/checkpoint_final.safetensors \
  --text "こんにちは、私はAIです。これは音声合成のテストです。" \
  --ref-wav path/to/reference.wav \
  --output-wav outputs/sample.wav
```

VoiceDesign checkpoints also support caption conditioning:

```bash
uv run python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v2-VoiceDesign \
  --text "こんにちは、私はAIです。これは音声合成のテストです。" \
  --caption "落ち着いた、近い距離感の女性話者" \
  --no-ref \
  --output-wav outputs/sample_voice_design.wav
```

LoRA adapter directories can be loaded dynamically at inference time without
exporting a merged checkpoint:

```bash
uv run python infer.py \
  --checkpoint path/to/base_model.safetensors \
  --lora-adapter outputs/irodori_tts_lora/checkpoint_final \
  --text "こんにちは、私はAIです。これはLoRA推論のテストです。" \
  --ref-wav path/to/reference.wav \
  --output-wav outputs/sample_lora.wav
```

### Output Duration

The v3 base model integrates duration prediction into inference.
When `--seconds` is omitted, the runtime estimates the output length from the input
text and, for speaker-conditioned checkpoints, the reference audio, then generates
audio for that estimated duration. Use `--duration-scale` to multiply the predicted
length (`>1` longer, `<1` shorter). For exact control, pass `--seconds` manually.

Older v2 checkpoints were trained with fixed-length 30-second targets. They remain
supported by the v3 codebase and still accept manual `--seconds`, but forcing a
non-default duration can reduce audio quality; prefer the v3 base model for automatic
or scaled duration control.

### Sway Sampling

For faster experimental inference, Sway Sampling can be combined with fewer Euler
steps:

```bash
uv run python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v3 \
  --text "こんにちは、私はAIです。これは音声合成のテストです。" \
  --ref-wav path/to/reference.wav \
  --num-steps 6 \
  --t-schedule-mode sway \
  --sway-coeff -1.0 \
  --output-wav outputs/sample_sway.wav
```

### Additional Inference Notes

For tuning guidance and detailed explanations of inference options, see the
[Parameter Guide](docs/parameters.md).

Generated audio is passed through [SilentCipher](https://github.com/sony/silentcipher) watermarking automatically when the dependency and model files are available.

## Training

### 1. Prepare Manifest (Precompute DACVAE Latents)

Encodes audio from a Hugging Face dataset into DACVAE latents and produces a JSONL manifest for training.

```bash
uv run python prepare_manifest.py \
  --dataset myorg/my_dataset \
  --split train \
  --audio-column audio \
  --text-column text \
  --output-manifest data/train_manifest.jsonl \
  --latent-dir data/latents \
  --device cuda
```

To include `speaker_id` in the manifest (for speaker-conditioned training):

```bash
uv run python prepare_manifest.py \
  --dataset myorg/my_dataset \
  --split train \
  --audio-column audio \
  --text-column text \
  --speaker-column speaker \
  --output-manifest data/train_manifest.jsonl \
  --latent-dir data/latents \
  --device cuda
```

To include `caption` in the manifest (for caption-conditioned voice design training):

```bash
uv run python prepare_manifest.py \
  --dataset myorg/my_dataset \
  --split train \
  --audio-column audio \
  --text-column text \
  --caption-column caption \
  --speaker-column speaker \
  --output-manifest data/train_manifest.jsonl \
  --latent-dir data/latents \
  --device cuda
```

When training the caption-conditioned voice-design model, `speaker_id` is optional. The
voice-design path disables speaker/reference conditioning and learns from `text + caption`.

This produces a JSONL manifest with entries like:

```json
{"text": "こんにちは", "caption": "落ち着いた、近い距離感の女性話者", "latent_path": "data/latents/00001.pt", "speaker_id": "myorg/my_dataset:speaker_001", "num_frames": 750}
```

### 2. Training

Single-GPU training:

```bash
uv run python train.py \
  --config configs/train_500m_v3_phase1_body.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts
```

v3 release training uses two phases. After training the body, initialize the integrated
duration predictor from the phase-1 checkpoint:

```bash
uv run python train.py \
  --config configs/train_500m_v3_phase2_duration.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts_duration \
  --init-checkpoint outputs/irodori_tts/checkpoint_final.pt
```

VoiceDesign training uses a dedicated config:

```bash
uv run python train.py \
  --config configs/train_500m_v2_voice_design.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts_voice_design
```

`configs/train_500m_v2_voice_design.yaml` sets `use_caption_condition: true` and disables the
speaker/reference branch. Caption-free configs continue to use speaker conditioning when
`speaker_id` / reference inputs are available.

The VoiceDesign config also enables `caption_warmup: true` for optional caption-branch warmup.
`warmup_steps` controls the LR scheduler, while `caption_warmup_steps` controls how long
non-caption gradients are discarded before normal joint training resumes.

### v3 Duration Predictor Training

v3 training uses two phases: `configs/train_500m_v3_phase1_body.yaml` trains the
variable-length DiT body, then `configs/train_500m_v3_phase2_duration.yaml` freezes the
body and trains the duration predictor.

The duration predictor regresses `log1p(num_frames)` with Huber loss. The current v3 phase2
config uses the token-sum duration predictor selected from ablations; see the parameter
guide for the architecture details.

Multi-GPU DDP training:

```bash
uv run torchrun --nproc_per_node 4 train.py \
  --config configs/train_500m_v3_phase1_body.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts \
  --device cuda
```

Training supports YAML config files with `model` and `train` sections. CLI arguments take precedence over YAML values. See `uv run python train.py --help` for all available options.
For a more detailed explanation of model and training config fields, see [Parameter Guide](docs/parameters.md).

#### Fine-Tuning from Released Weights

Start a new training run from released inference weights (`.safetensors`). This initializes only the model weights; optimizer / scheduler state starts fresh. For the v3 base release, the LoRA config keeps the duration predictor as part of the saved adapter by default.

```bash
uv run python train.py \
  --config configs/train_500m_v3_lora.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts_lora \
  --init-checkpoint path/to/Irodori-TTS-500M-v3.safetensors
```

Caption-conditioned voice-design LoRA fine-tuning:

```bash
uv run python train.py \
  --config configs/train_500m_v2_voice_design_lora.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts_voice_design_lora \
  --init-checkpoint path/to/Irodori-TTS-500M-v2-VoiceDesign.safetensors
```

LoRA target presets, adapter saving behavior, and resume details are covered in the
[Parameter Guide](docs/parameters.md).

#### Resuming Interrupted Training

Resume an existing training run from a training checkpoint. Full-model runs use `.pt`; LoRA runs use checkpoint directories. Both restore optimizer, scheduler, and step state.

```bash
uv run python train.py \
  --config configs/train_500m_v3_phase1_body.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts \
  --resume outputs/irodori_tts/checkpoint_0010000.pt
```

LoRA resume example:

```bash
uv run python train.py \
  --config configs/train_500m_v3_lora.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts_lora \
  --resume outputs/irodori_tts_lora/checkpoint_0010000
```

If you move a LoRA checkpoint to another environment and the original base-checkpoint path is no longer valid, pass `--init-checkpoint path/to/base_model.safetensors` together with `--resume` to override the saved base-model path.

### 3. Checkpoint Conversion

Convert a training checkpoint to inference-only safetensors format:

```bash
uv run python convert_checkpoint_to_safetensors.py outputs/checkpoint_final.pt
```

LoRA adapter checkpoints can also be converted directly:

```bash
uv run python convert_checkpoint_to_safetensors.py outputs/irodori_tts_lora/checkpoint_final
```

LoRA adapter checkpoints are merged into the base model automatically during conversion, so the exported `.safetensors` file is directly usable for inference. If you do not want to merge the adapter, pass the adapter directory directly to `infer.py --lora-adapter` or the matching Gradio field.

## Project Structure

```text
Irodori-TTS/
├── train.py                    # Training entry point (DDP support)
├── infer.py                    # CLI inference
├── gradio_app.py               # Gradio web UI
├── gradio_app_voicedesign.py   # Gradio web UI for VoiceDesign checkpoints
├── prepare_manifest.py         # Dataset -> DACVAE latent preprocessing
├── convert_checkpoint_to_safetensors.py  # Checkpoint converter
│
├── docs/
│   └── parameters.md         # Detailed parameter guide
│
├── irodori_tts/                # Core library
│   ├── model.py                # TextToLatentRFDiT architecture
│   ├── rf.py                   # Rectified Flow utilities & Euler CFG sampling
│   ├── codec.py                # DACVAE codec wrapper
│   ├── dataset.py              # Dataset and collator
│   ├── tokenizer.py            # Pretrained LLM tokenizer wrapper
│   ├── config.py               # Model / Train / Sampling config dataclasses
│   ├── inference_runtime.py    # Cached, thread-safe inference runtime
│   ├── lora.py                 # PEFT LoRA integration helpers
│   ├── text_normalization.py   # Japanese text normalization
│   ├── optim.py                # Muon + AdamW optimizer
│   └── progress.py             # Training progress tracker
│
└── configs/
    ├── train_500m_v3_phase1_body.yaml        # 500M v3 body training config
    ├── train_500m_v3_phase2_duration.yaml    # 500M v3 duration-predictor training config
    ├── train_500m_v3_lora.yaml               # 500M v3 LoRA fine-tuning config
    ├── train_500m_v2.yaml                    # 500M v2 backward-compatible model config
    ├── train_500m_v2_lora.yaml               # 500M v2 LoRA fine-tuning config
    ├── train_500m_v2_voice_design.yaml       # 500M v2 VoiceDesign full fine-tuning config
    ├── train_500m_v2_voice_design_lora.yaml  # 500M v2 VoiceDesign LoRA fine-tuning config
    ├── train_500m.yaml                       # 500M v1 model config
    └── train_2.5b.yaml                       # 2.5B parameter model config
```

## License

- **Code**: [MIT License](LICENSE)
- **Model Weights**: Please refer to the [base model card](https://huggingface.co/Aratako/Irodori-TTS-500M-v3) and the [VoiceDesign model card](https://huggingface.co/Aratako/Irodori-TTS-500M-v2-VoiceDesign) for licensing details

## Acknowledgments

This project builds upon the following works:

- [Echo-TTS](https://jordandarefsky.com/blog/2025/echo/) — Architecture and training design reference
- [DACVAE](https://github.com/facebookresearch/dacvae) — Audio VAE
- [SilentCipher](https://github.com/sony/silentcipher) — Audio watermarking

## Citation

```bibtex
@misc{irodori-tts,
  author = {Chihiro Arata},
  title = {Irodori-TTS: A Flow Matching-based Text-to-Speech Model with Emoji-driven Style Control},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Aratako/Irodori-TTS}}
}
```
