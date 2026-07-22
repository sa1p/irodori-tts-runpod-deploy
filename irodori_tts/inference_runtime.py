from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import secrets
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torchaudio
from safetensors import safe_open
from safetensors.torch import load_file as load_safetensors_file

from .codec import DACVAECodec, patchify_latent, unpatchify_latent
from .config import ModelConfig
from .duration import build_duration_features
from .lora import checkpoint_state_uses_lora, is_lora_adapter_dir, load_lora_adapter
from .model import TextToLatentRFDiT
from .rf import sample_euler_rf_cfg
from .text_normalization import normalize_text
from .tokenizer import PretrainedTextTokenizer
from .watermark import SilentCipherWatermarker


def _is_mps_available() -> bool:
    backends = getattr(torch, "backends", None)
    if backends is None or not hasattr(backends, "mps"):
        return False
    return bool(torch.backends.mps.is_available())


def resolve_runtime_device(device: str | torch.device) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cpu":
        return resolved
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA device requested but torch.cuda.is_available() is False.")
        return resolved
    if resolved.type == "mps":
        if resolved.index is not None:
            raise ValueError("MPS device index is not supported. Use 'mps'.")
        if not _is_mps_available():
            raise ValueError("MPS device requested but torch.backends.mps.is_available() is False.")
        return torch.device("mps")
    raise ValueError(f"Unsupported inference device={resolved!s}. Expected one of: cpu, cuda, mps.")


def list_available_runtime_devices() -> list[str]:
    devices: list[str] = []
    if torch.cuda.is_available():
        devices.append("cuda")
    if _is_mps_available():
        devices.append("mps")
    devices.append("cpu")
    return devices


def default_runtime_device() -> str:
    return list_available_runtime_devices()[0]


def list_available_runtime_precisions(device: str | torch.device) -> list[str]:
    resolved = resolve_runtime_device(device)
    if resolved.type == "cuda":
        return ["fp32", "bf16"]
    return ["fp32"]


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        mps = getattr(torch, "mps", None)
        if mps is not None and hasattr(mps, "synchronize"):
            mps.synchronize()


def _sync_devices(*devices: torch.device) -> None:
    seen: set[tuple[str, int | None]] = set()
    for device in devices:
        key = (device.type, device.index)
        if key in seen:
            continue
        _sync_device(device)
        seen.add(key)


def _measure_start(device: torch.device, *extra_devices: torch.device) -> float:
    _sync_devices(device, *extra_devices)
    return time.perf_counter()


def _measure_end(device: torch.device, t0: float, *extra_devices: torch.device) -> float:
    _sync_devices(device, *extra_devices)
    return time.perf_counter() - t0


def _coerce_latent_shape(latent: torch.Tensor, latent_dim: int) -> torch.Tensor:
    if latent.ndim == 3 and latent.shape[0] == 1:
        latent = latent[0]
    if latent.ndim != 2:
        raise ValueError(f"Unsupported latent shape: {tuple(latent.shape)}")
    if latent.shape[1] == latent_dim:
        return latent
    if latent.shape[0] == latent_dim:
        return latent.transpose(0, 1).contiguous()
    raise ValueError(
        f"Could not infer latent layout for shape={tuple(latent.shape)} and latent_dim={latent_dim}"
    )


def find_flattening_point(
    latent: torch.Tensor,
    target_value: float = 0.0,
    window_size: int = 20,
    std_threshold: float = 0.05,
    mean_threshold: float = 0.1,
) -> int:
    """
    Echo-style heuristic: find first index where a trailing window becomes near-flat and near-zero.

    Args:
      latent: (T, D) latent sequence.
    Returns:
      Flattening index in [0, T].
    """
    if latent.ndim != 2:
        raise ValueError(f"Expected latent shape (T, D), got {tuple(latent.shape)}")
    total_steps = int(latent.shape[0])
    if total_steps <= 0 or window_size <= 0:
        return total_steps

    pad = torch.zeros(
        (window_size, latent.shape[1]),
        device=latent.device,
        dtype=latent.dtype,
    )
    padded = torch.cat([latent, pad], dim=0)
    for i in range(padded.shape[0] - window_size):
        window = padded[i : i + window_size]
        window_std = window.std(unbiased=False)
        window_mean = window.mean()
        if window_std < std_threshold and torch.abs(window_mean - target_value) < mean_threshold:
            return int(i)
    return total_steps


@dataclass(frozen=True)
class RuntimeKey:
    checkpoint: str
    model_device: str
    lora_adapter: str | None = None
    lora_load_mode: str = "none"
    codec_repo: str = "Aratako/Semantic-DACVAE-Japanese-32dim"
    model_precision: str = "fp32"
    codec_device: str = "cpu"
    codec_precision: str = "fp32"
    codec_deterministic_encode: bool = True
    codec_deterministic_decode: bool = True
    compile_model: bool = False
    compile_dynamic: bool = False


def _normalize_lora_load_mode(mode: str | None) -> str:
    value = (mode or "none").strip().lower()
    if value in {"", "none", "off", "disabled"}:
        return "none"
    if value in {"dynamic", "adapter"}:
        return "dynamic"
    if value in {"reload", "replace", "single", "single_adapter", "dynamic_reload"}:
        return "reload"
    if value in {"delta", "hot_swap", "hotswap", "manual", "manual_merge"}:
        return "delta"
    if value in {"merged", "merge", "merge_on_load"}:
        return "merged"
    raise ValueError(
        f"Unsupported lora_load_mode={mode!r}. Expected one of: none, dynamic, reload, delta, merged."
    )


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


@dataclass
class SamplingRequest:
    text: str
    caption: str | None = None
    ref_wav: str | None = None
    ref_latent: str | None = None
    no_ref: bool = False
    ref_normalize_db: float | None = -16.0
    ref_ensure_max: bool = True
    num_candidates: int = 1
    decode_mode: str = "sequential"
    seconds: float | None = None
    duration_scale: float = 1.0
    min_seconds: float = 0.5
    max_seconds: float = 30.0
    max_ref_seconds: float | None = 30.0
    max_text_len: int | None = None
    max_caption_len: int | None = None
    num_steps: int = 40
    cfg_scale_text: float = 3.0
    cfg_scale_caption: float = 3.0
    cfg_scale_speaker: float = 5.0
    cfg_guidance_mode: str = "independent"
    cfg_scale: float | None = None
    cfg_min_t: float = 0.5
    cfg_max_t: float = 1.0
    truncation_factor: float | None = None
    rescale_k: float | None = None
    rescale_sigma: float | None = None
    context_kv_cache: bool = True
    speaker_kv_scale: float | None = None
    speaker_kv_min_t: float | None = None
    speaker_kv_max_layers: int | None = None
    seed: int | None = None
    t_schedule_mode: str = "linear"
    sway_coeff: float = -1.0
    trim_tail: bool = True
    tail_window_size: int = 20
    tail_std_threshold: float = 0.05
    tail_mean_threshold: float = 0.1
    lora_adapter: str | None = None


@dataclass(frozen=True)
class _ReferenceLatentCacheKey:
    source_kind: str
    source_path: str
    source_size: int | None
    source_mtime_ns: int | None
    max_ref_seconds: float | None
    ref_normalize_db: float | None
    ref_ensure_max: bool


@dataclass
class SamplingResult:
    audio: torch.Tensor
    audios: list[torch.Tensor]
    sample_rate: int
    stage_timings: list[tuple[str, float]]
    total_to_decode: float
    used_seed: int
    messages: list[str]


@dataclass(frozen=True)
class _LoraDeltaAdapter:
    path: str
    scaling: float
    weights: dict[str, tuple[torch.Tensor, torch.Tensor]]


def _load_lora_delta_adapter(adapter_dir: Path) -> _LoraDeltaAdapter:
    config_path = adapter_dir / "adapter_config.json"
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    if bool(raw_config.get("fan_in_fan_out", False)):
        raise ValueError(f"LoRA delta switching does not support fan_in_fan_out=True: {adapter_dir}")
    r = int(raw_config["r"])
    if r <= 0:
        raise ValueError(f"LoRA rank must be positive: {adapter_dir}")
    scaling = float(raw_config["lora_alpha"]) / float(r)

    state_path = adapter_dir / "adapter_model.safetensors"
    if state_path.is_file():
        state = load_safetensors_file(str(state_path), device="cpu")
    else:
        state_path = adapter_dir / "adapter_model.bin"
        state = torch.load(state_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"LoRA adapter state is empty: {adapter_dir}")

    grouped: dict[str, dict[str, torch.Tensor]] = {}
    prefix = "base_model.model."
    for raw_key, tensor in state.items():
        key = str(raw_key)
        if key.startswith(prefix):
            key = key[len(prefix) :]
        if key.endswith(".lora_A.weight"):
            target = key[: -len(".lora_A.weight")]
            grouped.setdefault(target, {})["A"] = tensor.detach().cpu()
        elif key.endswith(".lora_B.weight"):
            target = key[: -len(".lora_B.weight")]
            grouped.setdefault(target, {})["B"] = tensor.detach().cpu()

    weights: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for target, pair in grouped.items():
        if "A" not in pair or "B" not in pair:
            raise ValueError(f"Incomplete LoRA A/B pair for {target!r}: {adapter_dir}")
        weights[target] = (pair["A"], pair["B"])
    if not weights:
        raise ValueError(f"No LoRA delta weights found: {adapter_dir}")
    return _LoraDeltaAdapter(path=str(adapter_dir.resolve()), scaling=scaling, weights=weights)


def _maybe_compile_inference_model(
    model: TextToLatentRFDiT,
    *,
    enabled: bool,
    dynamic: bool,
) -> TextToLatentRFDiT:
    if not enabled:
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("compile_model=True requires torch.compile (PyTorch 2+).")
    compile_kwargs = {"dynamic": bool(dynamic)}
    model.encode_conditions = torch.compile(model.encode_conditions, **compile_kwargs)
    model.build_context_kv_cache = torch.compile(model.build_context_kv_cache, **compile_kwargs)
    model.forward_with_encoded_conditions = torch.compile(
        model.forward_with_encoded_conditions,
        **compile_kwargs,
    )
    return model


def _move_inference_module(
    module: torch.nn.Module,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    module.to(device=device)
    with torch.no_grad():
        for param in module.parameters():
            if param.is_floating_point() and param.dtype != dtype:
                param.data = param.data.to(device=device, dtype=dtype)
                if param.grad is not None:
                    param.grad.data = param.grad.data.to(device=device, dtype=dtype)
        for child in module.modules():
            for name, buffer in child._buffers.items():
                if buffer is None:
                    continue
                if buffer.is_floating_point() and buffer.dtype != dtype:
                    child._buffers[name] = buffer.to(device=device, dtype=dtype)
                elif buffer.device != device:
                    child._buffers[name] = buffer.to(device=device)
    return module


def resolve_runtime_dtype(*, precision: str, device: torch.device) -> torch.dtype:
    mode = str(precision).strip().lower()
    if mode == "fp32":
        return torch.float32
    if mode == "bf16":
        if device.type != "cuda":
            raise ValueError("precision='bf16' currently requires CUDA device.")
        return torch.bfloat16
    raise ValueError(f"Unsupported precision={precision!r}. Expected one of: fp32, bf16.")


def resolve_cfg_scales(
    *,
    cfg_guidance_mode: str,
    cfg_scale_text: float,
    cfg_scale_caption: float,
    cfg_scale_speaker: float,
    cfg_scale: float | None,
    use_caption_condition: bool = True,
    use_speaker_condition: bool = True,
) -> tuple[float, float, float, list[str]]:
    """Normalize/validate CFG scales for guidance mode."""
    messages: list[str] = []
    text_val = float(cfg_scale_text)
    caption_val = float(cfg_scale_caption)
    speaker_val = float(cfg_scale_speaker)

    if cfg_scale is not None:
        text_val = float(cfg_scale)
        caption_val = float(cfg_scale)
        speaker_val = float(cfg_scale)
    if not use_speaker_condition:
        if speaker_val > 0.0:
            messages.append(
                "info: speaker conditioning is disabled for this checkpoint; ignoring cfg_scale_speaker."
            )
        speaker_val = 0.0

    mode = str(cfg_guidance_mode).strip().lower()
    enabled_vals = [value for value in (text_val, speaker_val) if value > 0.0]
    if use_caption_condition and caption_val > 0.0:
        enabled_vals.append(caption_val)
    if mode == "joint" and enabled_vals and (max(enabled_vals) - min(enabled_vals) > 1e-6):
        raise ValueError(
            "cfg_guidance_mode='joint' requires equal enabled cfg_scale_text/cfg_scale_caption/cfg_scale_speaker, "
            "or set cfg_scale."
        )

    return text_val, caption_val, speaker_val, messages


def _load_torch_checkpoint_payload(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported checkpoint payload type: {type(payload)!r}")
    return payload


_CONFIG_META_KEY = "config_json"
_INFERENCE_CONFIG_KEYS = {"max_text_len", "max_caption_len", "fixed_target_latent_steps"}


def _load_checkpoint_from_pt(path: Path) -> tuple[dict[str, torch.Tensor], dict, dict | None]:
    ckpt = _load_torch_checkpoint_payload(path)
    model_state = ckpt.get("model")
    model_cfg = ckpt.get("model_config")
    train_cfg = ckpt.get("train_config")

    if not isinstance(model_state, dict):
        raise ValueError(f"Checkpoint missing model weights dictionary: {path}")
    if not isinstance(model_cfg, dict):
        raise ValueError(f"Checkpoint missing model_config dictionary: {path}")
    if train_cfg is not None and not isinstance(train_cfg, dict):
        raise ValueError(f"Checkpoint train_config must be a dictionary when present: {path}")

    if checkpoint_state_uses_lora(model_state):
        raise ValueError(
            f"LoRA checkpoints must be loaded from adapter directories or merged safetensors: {path}"
        )
    return model_state, model_cfg, _extract_inference_train_config(train_cfg)


def _parse_json_mapping(
    raw: str | None,
    *,
    field: str,
    path: Path,
    required: bool = False,
) -> dict | None:
    if raw is None:
        if required:
            raise ValueError(f"Missing required metadata field '{field}' in checkpoint: {path}")
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in '{field}' metadata for checkpoint: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Metadata field '{field}' must decode to an object: {path}")
    return payload


def _extract_inference_train_config(raw: dict | None) -> dict | None:
    if raw is None:
        return None

    inference_cfg: dict[str, int] = {}
    for key in _INFERENCE_CONFIG_KEYS:
        value = raw.get(key)
        if value is None:
            continue
        if not isinstance(value, int):
            raise ValueError(f"Inference config key '{key}' must be int, got {type(value)!r}.")
        inference_cfg[key] = int(value)

    return inference_cfg or None


def _split_flat_checkpoint_config(path: Path, flat_config: dict) -> tuple[dict, dict | None]:
    model_cfg: dict[str, object] = {}
    inference_cfg: dict[str, int] = {}
    for key, value in flat_config.items():
        if key in _INFERENCE_CONFIG_KEYS:
            if not isinstance(value, int):
                raise ValueError(
                    f"Inference config key '{key}' must be int in checkpoint metadata: {path}"
                )
            inference_cfg[key] = int(value)
            continue
        model_cfg[key] = value
    return model_cfg, (inference_cfg or None)


def _load_checkpoint_from_safetensors(
    path: Path,
) -> tuple[dict[str, torch.Tensor], dict, dict | None]:
    model_state = load_safetensors_file(str(path), device="cpu")
    if not isinstance(model_state, dict) or not model_state:
        raise ValueError(f"Safetensors checkpoint has no model weights: {path}")

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}

    flat_config = _parse_json_mapping(
        metadata.get(_CONFIG_META_KEY),
        field=_CONFIG_META_KEY,
        path=path,
        required=True,
    )
    model_cfg, inference_cfg = _split_flat_checkpoint_config(path=path, flat_config=flat_config)
    return model_state, model_cfg, inference_cfg


def _load_checkpoint_for_inference(path: Path) -> tuple[dict[str, torch.Tensor], dict, dict | None]:
    if path.suffix.lower() == ".safetensors":
        return _load_checkpoint_from_safetensors(path)
    return _load_checkpoint_from_pt(path)


class InferenceRuntime:
    def __init__(
        self,
        *,
        key: RuntimeKey,
        model_cfg: ModelConfig,
        train_cfg: dict | None,
        model: torch.nn.Module,
        tokenizer: PretrainedTextTokenizer,
        caption_tokenizer: PretrainedTextTokenizer | None,
        codec: DACVAECodec,
        default_text_max_len: int,
        default_caption_max_len: int,
        model_dtype: torch.dtype,
    ) -> None:
        self.key = key
        self.model_device = resolve_runtime_device(key.model_device)
        self.codec_device = resolve_runtime_device(key.codec_device)
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.model = model
        self.tokenizer = tokenizer
        self.caption_tokenizer = caption_tokenizer
        self.codec = codec
        self.default_text_max_len = default_text_max_len
        self.default_caption_max_len = default_caption_max_len
        self.model_dtype = model_dtype
        self._loaded_lora_adapters: set[str] = set()
        self._active_lora_adapter: str | None = None
        self._lora_delta_cache: dict[str, _LoraDeltaAdapter] = {}
        self._active_lora_delta: _LoraDeltaAdapter | None = None
        self.watermarker = SilentCipherWatermarker(device=str(self.codec_device))
        self._infer_lock = threading.Lock()
        self._model_dtype = model_dtype
        self._lora_adapter_names: dict[str, str] = {}
        self._reference_latent_cache: OrderedDict[
            _ReferenceLatentCacheKey, torch.Tensor
        ] = OrderedDict()
        self._reference_latent_cache_size = _env_int("IRODORI_REF_LATENT_CACHE_SIZE", 64)

    def _clear_model_memory(self) -> None:
        gc.collect()
        if self.model_device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.model_device.type == "mps":
            mps = getattr(torch, "mps", None)
            if mps is not None and hasattr(mps, "empty_cache"):
                mps.empty_cache()

    def _reload_base_model(self) -> None:
        del self.model
        self._clear_model_memory()

        model_state, model_cfg_dict, train_cfg = _load_checkpoint_for_inference(
            Path(self.key.checkpoint)
        )
        model_cfg = ModelConfig(**model_cfg_dict)
        model = TextToLatentRFDiT(model_cfg).to(self.model_device)
        model.load_state_dict(model_state)
        del model_state
        gc.collect()
        model = _move_inference_module(model, device=self.model_device, dtype=self.model_dtype)
        model.eval()

        self.model_cfg = model_cfg
        self.train_cfg = train_cfg if isinstance(train_cfg, dict) else None
        self.model = _maybe_compile_inference_model(
            model,
            enabled=bool(self.key.compile_model),
            dynamic=bool(self.key.compile_dynamic),
        )
        self._loaded_lora_adapters.clear()
        self._active_lora_adapter = None
        self._active_lora_delta = None
        self._lora_adapter_names.clear()
        self._clear_model_memory()

    def _get_lora_delta_adapter(self, adapter_dir: Path) -> _LoraDeltaAdapter:
        key = str(adapter_dir.resolve())
        adapter = self._lora_delta_cache.get(key)
        if adapter is None:
            adapter = _load_lora_delta_adapter(adapter_dir)
            self._lora_delta_cache[key] = adapter
        return adapter

    def _apply_lora_delta_adapter(self, adapter: _LoraDeltaAdapter, *, sign: float) -> None:
        with torch.no_grad():
            for target, (weight_a, weight_b) in adapter.weights.items():
                module = self.model.get_submodule(target)
                if not isinstance(module, torch.nn.Linear):
                    raise TypeError(
                        f"LoRA delta target is not torch.nn.Linear: {target} ({type(module)!r})"
                    )
                if module.weight.ndim != 2:
                    raise ValueError(f"LoRA delta target weight must be 2D: {target}")

                weight_a_device = weight_a.to(
                    device=module.weight.device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                weight_b_device = weight_b.to(
                    device=module.weight.device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                delta = weight_b_device @ weight_a_device
                if tuple(delta.shape) != tuple(module.weight.shape):
                    raise ValueError(
                        f"LoRA delta shape mismatch for {target}: "
                        f"delta={tuple(delta.shape)} weight={tuple(module.weight.shape)}"
                    )
                delta.mul_(float(adapter.scaling) * float(sign))

                updated = module.weight.data.float()
                updated.add_(delta)
                module.weight.data.copy_(updated.to(dtype=module.weight.dtype))

    def ensure_lora_delta_adapter(self, *, adapter_name: str, adapter_path: str | Path) -> None:
        adapter_dir = Path(adapter_path)
        if not is_lora_adapter_dir(adapter_dir):
            raise ValueError(f"LoRA adapter directory is invalid: {adapter_dir}")
        if not adapter_name:
            raise ValueError("adapter_name must not be empty.")

        next_adapter = self._get_lora_delta_adapter(adapter_dir)
        if self._active_lora_adapter == adapter_name and self._active_lora_delta == next_adapter:
            return

        if self._loaded_lora_adapters:
            self._reload_base_model()

        if self._active_lora_delta is not None:
            self._apply_lora_delta_adapter(self._active_lora_delta, sign=-1.0)
            self._active_lora_delta = None

        self._apply_lora_delta_adapter(next_adapter, sign=1.0)
        self._active_lora_delta = next_adapter
        self._active_lora_adapter = adapter_name
        self._clear_model_memory()

    @classmethod
    def from_key(cls, key: RuntimeKey) -> InferenceRuntime:
        model_device = resolve_runtime_device(key.model_device)
        codec_device = resolve_runtime_device(key.codec_device)
        lora_load_mode = _normalize_lora_load_mode(key.lora_load_mode)
        if lora_load_mode == "none" and key.lora_adapter is not None:
            raise ValueError("RuntimeKey.lora_adapter requires lora_load_mode='merged' or 'dynamic'.")
        if lora_load_mode in {"dynamic", "reload", "delta"} and key.lora_adapter is not None:
            raise ValueError("Runtime-switched LoRA adapters must be loaded with ensure_lora_adapter().")
        model_dtype = resolve_runtime_dtype(
            precision=key.model_precision,
            device=model_device,
        )
        codec_dtype = resolve_runtime_dtype(
            precision=key.codec_precision,
            device=codec_device,
        )

        model_state, model_cfg_dict, train_cfg = _load_checkpoint_for_inference(
            Path(key.checkpoint)
        )
        model_cfg = ModelConfig(**model_cfg_dict)

        model = TextToLatentRFDiT(model_cfg).to(model_device)
        model.load_state_dict(model_state)
        del model_state
        gc.collect()
        if lora_load_mode == "merged":
            if key.lora_adapter is None:
                raise ValueError("lora_load_mode='merged' requires RuntimeKey.lora_adapter.")
            adapter_dir = Path(key.lora_adapter)
            if not is_lora_adapter_dir(adapter_dir):
                raise ValueError(f"LoRA adapter directory is invalid: {adapter_dir}")
            model = load_lora_adapter(
                model,
                adapter_dir,
                is_trainable=False,
                adapter_name="merged",
                torch_device=str(model_device),
            )
            if not hasattr(model, "merge_and_unload"):
                raise RuntimeError("Loaded LoRA model does not support merge_and_unload().")
            model = model.merge_and_unload()
        model = _move_inference_module(model, device=model_device, dtype=model_dtype)
        model.eval()
        model = _maybe_compile_inference_model(
            model,
            enabled=bool(key.compile_model),
            dynamic=bool(key.compile_dynamic),
        )

        tokenizer = PretrainedTextTokenizer.from_pretrained(
            repo_id=model_cfg.text_tokenizer_repo,
            add_bos=bool(model_cfg.text_add_bos),
            local_files_only=False,
        )
        if tokenizer.vocab_size != model_cfg.text_vocab_size:
            raise ValueError(
                f"text_vocab_size mismatch: checkpoint text_vocab_size={model_cfg.text_vocab_size} but tokenizer "
                f"({model_cfg.text_tokenizer_repo}) vocab_size={tokenizer.vocab_size}."
            )
        caption_tokenizer = None
        if model_cfg.use_caption_condition:
            caption_tokenizer = PretrainedTextTokenizer.from_pretrained(
                repo_id=model_cfg.caption_tokenizer_repo_resolved,
                add_bos=model_cfg.caption_add_bos_resolved,
                local_files_only=False,
            )
            if caption_tokenizer.vocab_size != model_cfg.caption_vocab_size_resolved:
                raise ValueError(
                    f"caption_vocab_size mismatch: checkpoint caption_vocab_size={model_cfg.caption_vocab_size_resolved} but tokenizer ({model_cfg.caption_tokenizer_repo_resolved}) "
                    f"vocab_size={caption_tokenizer.vocab_size}."
                )

        default_text_max_len = 256
        default_caption_max_len = default_text_max_len
        if isinstance(train_cfg, dict):
            ckpt_text_max_len = train_cfg.get("max_text_len")
            if isinstance(ckpt_text_max_len, int) and ckpt_text_max_len > 0:
                default_text_max_len = int(ckpt_text_max_len)
            ckpt_caption_max_len = train_cfg.get("max_caption_len")
            if isinstance(ckpt_caption_max_len, int) and ckpt_caption_max_len > 0:
                default_caption_max_len = int(ckpt_caption_max_len)
            else:
                default_caption_max_len = default_text_max_len

        codec = DACVAECodec.load(
            repo_id=key.codec_repo,
            device=str(codec_device),
            dtype=codec_dtype,
            deterministic_encode=bool(key.codec_deterministic_encode),
            deterministic_decode=bool(key.codec_deterministic_decode),
        )
        if model_cfg.latent_dim != codec.latent_dim:
            raise ValueError(
                f"Latent dimension mismatch: checkpoint latent_dim={model_cfg.latent_dim} but codec latent_dim={codec.latent_dim}. "
                "Use a compatible codec/checkpoint pair."
            )

        return cls(
            key=key,
            model_cfg=model_cfg,
            train_cfg=train_cfg if isinstance(train_cfg, dict) else None,
            model=model,
            tokenizer=tokenizer,
            caption_tokenizer=caption_tokenizer,
            codec=codec,
            default_text_max_len=default_text_max_len,
            default_caption_max_len=default_caption_max_len,
            model_dtype=model_dtype,
        )

    def ensure_lora_adapter(
        self,
        *,
        adapter_name: str,
        adapter_path: str | Path,
        replace_existing: bool = False,
    ) -> None:
        adapter_dir = Path(adapter_path)
        if not is_lora_adapter_dir(adapter_dir):
            raise ValueError(f"LoRA adapter directory is invalid: {adapter_dir}")
        if not adapter_name:
            raise ValueError("adapter_name must not be empty.")

        if (
            replace_existing
            and self._loaded_lora_adapters
            and self._active_lora_adapter != adapter_name
        ):
            self._reload_base_model()

        if adapter_name not in self._loaded_lora_adapters:
            if not self._loaded_lora_adapters:
                self.model = load_lora_adapter(
                    self.model,
                    adapter_dir,
                    is_trainable=False,
                    adapter_name=adapter_name,
                    torch_device=str(self.model_device),
                )
            else:
                if not hasattr(self.model, "load_adapter"):
                    raise RuntimeError("Current model does not support multiple LoRA adapters.")
                self.model.load_adapter(
                    str(adapter_dir),
                    adapter_name=adapter_name,
                    is_trainable=False,
                    torch_device=str(self.model_device),
                )
            self.model = _move_inference_module(
                self.model,
                device=self.model_device,
                dtype=self.model_dtype,
            )
            self.model.eval()
            self._loaded_lora_adapters.add(adapter_name)

        if self._active_lora_adapter != adapter_name:
            if not hasattr(self.model, "set_adapter"):
                raise RuntimeError("Current model does not support LoRA adapter switching.")
            self.model.set_adapter(adapter_name)
            self._active_lora_adapter = adapter_name

    def _resolve_lora_adapter_path(self, adapter_path: str | None) -> str | None:
        if adapter_path is None:
            return None
        raw = str(adapter_path).strip()
        if raw.lower() in {"", "none", "null", "off", "disable", "disabled", "base"}:
            return None

        path = Path(raw).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(f"LoRA adapter directory not found: {path}")
        if not is_lora_adapter_dir(path):
            raise ValueError(
                f"LoRA adapter directory must contain adapter_config.json and adapter weights: {path}"
            )
        return str(path.resolve())

    @staticmethod
    def _adapter_name_for_path(path: str) -> str:
        digest = hashlib.sha1(path.encode("utf-8")).hexdigest()[:16]
        return f"runtime_{digest}"

    def _prepare_lora_for_request(
        self,
        adapter_path: str | None,
        *,
        messages: list[str],
        stage_timings: list[tuple[str, float]],
        log_fn: Callable[[str], None],
    ) -> Any:
        should_time = adapter_path is not None and str(adapter_path).strip() != ""
        t0 = _measure_start(self.model_device) if should_time else None
        try:
            return self._prepare_lora_for_request_inner(
                adapter_path,
                messages=messages,
                log_fn=log_fn,
            )
        finally:
            if t0 is not None:
                stage_sec = _measure_end(self.model_device, t0)
                stage_timings.append(("prepare_lora", stage_sec))
                log_fn(f"[runtime] prepare_lora: {stage_sec * 1000.0:.1f} ms")

    def _prepare_lora_for_request_inner(
        self,
        adapter_path: str | None,
        *,
        messages: list[str],
        log_fn: Callable[[str], None],
    ) -> Any:
        resolved_path = self._resolve_lora_adapter_path(adapter_path)
        if resolved_path is None:
            disable_adapter = getattr(self.model, "disable_adapter", None)
            if callable(disable_adapter):
                msg = "info: dynamic LoRA disabled for this request; using base model."
                messages.append(msg)
                log_fn(msg)
                return disable_adapter()
            return nullcontext()

        if self.key.compile_model:
            raise RuntimeError("Dynamic LoRA loading is not compatible with compile_model=True.")

        adapter_name = self._lora_adapter_names.get(resolved_path)
        if adapter_name is None:
            adapter_name = self._adapter_name_for_path(resolved_path)
            msg = f"info: loading LoRA adapter: {resolved_path}"
            messages.append(msg)
            log_fn(msg)
        else:
            msg = f"info: using cached LoRA adapter: {resolved_path}"
            messages.append(msg)
            log_fn(msg)

        self.model = load_lora_adapter(
            self.model,
            resolved_path,
            is_trainable=False,
            adapter_name=adapter_name,
            torch_device=str(self.model_device),
        )
        self._lora_adapter_names[resolved_path] = adapter_name
        self.model = _move_inference_module(
            self.model,
            device=self.model_device,
            dtype=self._model_dtype,
        )
        self.model.eval()
        return nullcontext()

    def _reference_latent_cache_key(
        self, req: SamplingRequest
    ) -> _ReferenceLatentCacheKey | None:
        if self._reference_latent_cache_size <= 0:
            return None
        source_kind: str
        source_path_raw: str | None
        if req.ref_latent is not None:
            source_kind = "latent"
            source_path_raw = req.ref_latent
        elif req.ref_wav is not None:
            source_kind = "wav"
            source_path_raw = req.ref_wav
        else:
            return None

        path = Path(source_path_raw).expanduser()
        try:
            resolved = path.resolve(strict=False)
        except OSError:
            resolved = path.absolute()
        try:
            stat = resolved.stat()
            source_size: int | None = int(stat.st_size)
            source_mtime_ns: int | None = int(stat.st_mtime_ns)
        except OSError:
            source_size = None
            source_mtime_ns = None

        return _ReferenceLatentCacheKey(
            source_kind=source_kind,
            source_path=str(resolved),
            source_size=source_size,
            source_mtime_ns=source_mtime_ns,
            max_ref_seconds=None
            if req.max_ref_seconds is None
            else float(req.max_ref_seconds),
            ref_normalize_db=None
            if req.ref_normalize_db is None
            else float(req.ref_normalize_db),
            ref_ensure_max=bool(req.ref_ensure_max),
        )

    def _get_cached_reference_latent(
        self, cache_key: _ReferenceLatentCacheKey | None
    ) -> torch.Tensor | None:
        if cache_key is None:
            return None
        ref_latent = self._reference_latent_cache.get(cache_key)
        if ref_latent is None:
            return None
        self._reference_latent_cache.move_to_end(cache_key)
        return ref_latent

    def _remember_reference_latent(
        self,
        cache_key: _ReferenceLatentCacheKey | None,
        ref_latent: torch.Tensor,
    ) -> None:
        if cache_key is None or self._reference_latent_cache_size <= 0:
            return
        self._reference_latent_cache[cache_key] = ref_latent.detach().cpu()
        self._reference_latent_cache.move_to_end(cache_key)
        while len(self._reference_latent_cache) > self._reference_latent_cache_size:
            self._reference_latent_cache.popitem(last=False)

    def _load_reference_latent(
        self,
        *,
        req: SamplingRequest,
        batch_size: int,
        messages: list[str],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        runtime_dtype = next(self.model.parameters()).dtype
        if not self.model_cfg.use_speaker_condition:
            if req.ref_wav is not None or req.ref_latent is not None:
                messages.append(
                    "info: speaker conditioning is disabled for this checkpoint; ignoring reference input."
                )
            return None, None
        if req.no_ref:
            ref_len = max(1, int(self.model_cfg.speaker_patch_size))
            ref_latent_patched = torch.zeros(
                (
                    batch_size,
                    ref_len,
                    self.model_cfg.latent_dim * self.model_cfg.latent_patch_size,
                ),
                device=self.model_device,
                dtype=runtime_dtype,
            )
            ref_mask = torch.zeros(
                (batch_size, ref_len), dtype=torch.bool, device=self.model_device
            )
            return ref_latent_patched, ref_mask

        if req.ref_wav is None and req.ref_latent is None:
            raise ValueError("Specify either ref_wav/ref_latent, or set no_ref=True.")

        max_ref_latent_steps = None
        if req.max_ref_seconds is not None and req.max_ref_seconds > 0:
            max_ref_latent_steps = max(
                1,
                math.ceil(
                    float(req.max_ref_seconds)
                    * float(self.codec.sample_rate)
                    / float(int(self.codec.model.hop_length))
                ),
            )

        cache_key = self._reference_latent_cache_key(req)
        cached_ref_latent = self._get_cached_reference_latent(cache_key)
        if cached_ref_latent is not None:
            messages.append("info: using cached reference latent.")
            ref_latent = cached_ref_latent.to(dtype=runtime_dtype)
        elif req.ref_latent is not None:
            latent_raw = torch.load(req.ref_latent, map_location="cpu", weights_only=True)
            ref_latent = _coerce_latent_shape(
                latent_raw, latent_dim=self.model_cfg.latent_dim
            ).unsqueeze(0)
            ref_latent = ref_latent.to(dtype=runtime_dtype)
        else:
            wav, sr = _load_audio(req.ref_wav)
            if req.max_ref_seconds is not None and req.max_ref_seconds > 0:
                max_ref_samples = max(1, int(float(req.max_ref_seconds) * float(sr)))
                if wav.shape[1] > max_ref_samples:
                    messages.append(
                        f"warning: reference audio exceeds max_ref_seconds ({req.max_ref_seconds}s). "
                        f"Trimming from {float(wav.shape[1]) / float(sr):.2f}s to {float(max_ref_samples) / float(sr):.2f}s."
                    )
                    wav = wav[:, :max_ref_samples]
            if req.ref_normalize_db is not None:
                messages.append(
                    f"info: reference loudness normalize enabled (target_db={float(req.ref_normalize_db):.2f}, includes peak safety scaling)."
                )
            elif req.ref_ensure_max:
                messages.append("info: reference peak safety scaling enabled (ensure_max=True).")
            ref_latent = self.codec.encode_waveform(
                wav.unsqueeze(0),
                sample_rate=int(sr),
                normalize_db=req.ref_normalize_db,
                ensure_max=bool(req.ref_ensure_max),
            ).cpu()

        if max_ref_latent_steps is not None and ref_latent.shape[1] > max_ref_latent_steps:
            messages.append(
                f"warning: reference latent steps ({ref_latent.shape[1]}) exceed max_ref_seconds bound ({max_ref_latent_steps} steps). "
                "Trimming reference latent."
            )
            ref_latent = ref_latent[:, :max_ref_latent_steps]

        if cached_ref_latent is None:
            self._remember_reference_latent(cache_key, ref_latent)

        ref_latent_patched = patchify_latent(ref_latent, self.model_cfg.latent_patch_size).to(
            device=self.model_device,
            dtype=runtime_dtype,
        )
        if ref_latent_patched.shape[1] == 0:
            raise ValueError(
                "Reference latent length became zero after patchify. Use longer reference audio."
            )
        if batch_size > 1:
            ref_latent_patched = ref_latent_patched.repeat(batch_size, 1, 1)
        ref_mask = torch.ones(
            (batch_size, ref_latent_patched.shape[1]), dtype=torch.bool, device=self.model_device
        )
        return ref_latent_patched, ref_mask

    def synthesize(
        self,
        req: SamplingRequest,
        *,
        log_fn: Callable[[str], None] | None = None,
    ) -> SamplingResult:
        def _log(msg: str) -> None:
            if log_fn is not None:
                log_fn(msg)

        messages: list[str] = []
        _log(
            (
                "[runtime] start synthesize "
                "model_device={} model_precision={} codec_device={} codec_precision={} "
                "silentcipher_watermark={} mode={} seconds={} steps={} schedule={} sway_coeff={} "
                "seed={} candidates={} decode_mode={}"
            ).format(
                self.key.model_device,
                self.key.model_precision,
                self.key.codec_device,
                self.key.codec_precision,
                self.watermarker.ready,
                req.cfg_guidance_mode,
                req.seconds,
                req.num_steps,
                req.t_schedule_mode,
                req.sway_coeff,
                "random" if req.seed is None else int(req.seed),
                req.num_candidates,
                req.decode_mode,
            )
        )

        manual_seconds = None if req.seconds is None else float(req.seconds)
        if manual_seconds is not None and manual_seconds <= 0:
            raise ValueError(f"seconds must be > 0 when provided, got {req.seconds}")
        duration_scale = float(req.duration_scale)
        if duration_scale <= 0:
            raise ValueError(f"duration_scale must be > 0, got {duration_scale}")
        min_seconds = float(req.min_seconds)
        max_seconds = float(req.max_seconds)
        if min_seconds <= 0:
            raise ValueError(f"min_seconds must be > 0, got {min_seconds}")
        if max_seconds < min_seconds:
            raise ValueError(
                f"max_seconds must be >= min_seconds, got min={min_seconds} max={max_seconds}"
            )
        num_candidates = int(req.num_candidates)
        if num_candidates <= 0:
            raise ValueError(f"num_candidates must be > 0, got {num_candidates}")
        decode_mode = str(req.decode_mode).strip().lower()
        if decode_mode not in {"sequential", "batch"}:
            raise ValueError(
                f"Unsupported decode_mode={req.decode_mode!r}. Expected one of: sequential, batch."
            )

        raw_text = str(req.text)
        normalized_text = normalize_text(raw_text).strip()
        if normalized_text == "":
            raise ValueError("text became empty after normalization.")

        text_max_len = (
            self.default_text_max_len if req.max_text_len is None else int(req.max_text_len)
        )
        if text_max_len <= 0:
            raise ValueError(f"max_text_len must be > 0, got {text_max_len}")
        caption_max_len = (
            self.default_caption_max_len
            if req.max_caption_len is None
            else int(req.max_caption_len)
        )
        if self.model_cfg.use_caption_condition and caption_max_len <= 0:
            raise ValueError(f"max_caption_len must be > 0, got {caption_max_len}")
        has_caption_text = bool(
            self.model_cfg.use_caption_condition
            and req.caption is not None
            and str(req.caption).strip() != ""
        )

        truncation_factor = None if req.truncation_factor is None else float(req.truncation_factor)
        rescale_k = None if req.rescale_k is None else float(req.rescale_k)
        rescale_sigma = None if req.rescale_sigma is None else float(req.rescale_sigma)
        if truncation_factor is not None and truncation_factor <= 0:
            raise ValueError(f"truncation_factor must be > 0, got {truncation_factor}")
        if (rescale_k is None) != (rescale_sigma is None):
            raise ValueError("rescale_k and rescale_sigma must be set together.")
        if rescale_k is not None and rescale_k <= 0:
            raise ValueError(f"rescale_k must be > 0, got {rescale_k}")
        if rescale_sigma is not None and rescale_sigma <= 0:
            raise ValueError(f"rescale_sigma must be > 0, got {rescale_sigma}")

        speaker_kv_scale = None if req.speaker_kv_scale is None else float(req.speaker_kv_scale)
        speaker_kv_min_t = None
        speaker_kv_max_layers = (
            None if req.speaker_kv_max_layers is None else int(req.speaker_kv_max_layers)
        )
        if speaker_kv_scale is not None:
            if not self.model_cfg.use_speaker_condition:
                messages.append(
                    "info: speaker conditioning is disabled for this checkpoint; ignoring speaker_kv_scale."
                )
                speaker_kv_scale = None
            else:
                if speaker_kv_scale <= 0:
                    raise ValueError(f"speaker_kv_scale must be > 0, got {speaker_kv_scale}")
                speaker_kv_min_t = (
                    0.9 if req.speaker_kv_min_t is None else float(req.speaker_kv_min_t)
                )
                if not (0.0 <= speaker_kv_min_t <= 1.0):
                    raise ValueError(f"speaker_kv_min_t must be in [0, 1], got {speaker_kv_min_t}")
                if speaker_kv_max_layers is not None and speaker_kv_max_layers < 0:
                    raise ValueError(
                        f"speaker_kv_max_layers must be >= 0 when specified, got {speaker_kv_max_layers}"
                    )

        cfg_mode = str(req.cfg_guidance_mode).strip().lower()
        if cfg_mode not in {"independent", "joint", "alternating"}:
            raise ValueError(
                f"Unsupported cfg_guidance_mode={req.cfg_guidance_mode!r}. "
                "Expected one of: independent, joint, alternating."
            )

        cfg_scale_text, cfg_scale_caption, cfg_scale_speaker, scale_messages = resolve_cfg_scales(
            cfg_guidance_mode=cfg_mode,
            cfg_scale_text=req.cfg_scale_text,
            cfg_scale_caption=req.cfg_scale_caption,
            cfg_scale_speaker=req.cfg_scale_speaker,
            cfg_scale=req.cfg_scale,
            use_caption_condition=has_caption_text,
            use_speaker_condition=self.model_cfg.use_speaker_condition,
        )
        messages.extend(scale_messages)
        for msg in scale_messages:
            _log(msg)

        stage_timings: list[tuple[str, float]] = []
        if req.seed is None:
            used_seed = int(secrets.randbits(63))
            msg = f"info: seed not specified; using random seed {used_seed}."
            messages.append(msg)
            _log(msg)
        else:
            used_seed = int(req.seed)
            _log(f"[runtime] using seed: {used_seed}")
        post_load_t0 = _measure_start(self.model_device, self.codec_device)

        with (
            self._infer_lock,
            self._prepare_lora_for_request(
                req.lora_adapter,
                messages=messages,
                stage_timings=stage_timings,
                log_fn=_log,
            ),
            torch.inference_mode(),
        ):
            t0 = _measure_start(self.model_device)
            text_ids, text_mask = self.tokenizer.batch_encode(
                [normalized_text] * num_candidates,
                max_length=text_max_len,
            )
            stage_sec = _measure_end(self.model_device, t0)
            stage_timings.append(("tokenize_text", stage_sec))
            _log(f"[runtime] tokenize_text: {stage_sec * 1000.0:.1f} ms")
            text_ids = text_ids.to(self.model_device)
            text_mask = text_mask.to(self.model_device)
            caption_ids = None
            caption_mask = None
            if self.model_cfg.use_caption_condition:
                if self.caption_tokenizer is None:
                    raise RuntimeError(
                        "Caption conditioning is enabled but caption tokenizer is not loaded."
                    )
                caption_text = "" if req.caption is None else str(req.caption).strip()
                caption_ids, caption_mask = self.caption_tokenizer.batch_encode(
                    [caption_text] * num_candidates,
                    max_length=caption_max_len,
                )
                if caption_text == "":
                    caption_mask.zero_()
                caption_ids = caption_ids.to(self.model_device)
                caption_mask = caption_mask.to(self.model_device)

            t0 = _measure_start(self.model_device, self.codec_device)
            msg_count_before_ref = len(messages)
            ref_latent, ref_mask = self._load_reference_latent(
                req=req,
                batch_size=num_candidates,
                messages=messages,
            )
            stage_sec = _measure_end(self.model_device, t0, self.codec_device)
            stage_timings.append(("prepare_reference", stage_sec))
            for msg in messages[msg_count_before_ref:]:
                _log(msg)
            _log(f"[runtime] prepare_reference: {stage_sec * 1000.0:.1f} ms")

            hop_length = int(self.codec.model.hop_length)
            if manual_seconds is not None:
                clamped_seconds = min(max_seconds, max(min_seconds, manual_seconds))
                if clamped_seconds != manual_seconds:
                    duration_msg = (
                        f"warning: manual duration {manual_seconds:.3f}s was clamped to "
                        f"{clamped_seconds:.3f}s."
                    )
                    messages.append(duration_msg)
                    _log(duration_msg)
                target_samples = max(1, int(clamped_seconds * self.codec.sample_rate))
                latent_steps = math.ceil(target_samples / hop_length)
                duration_msg = f"info: using manual duration {clamped_seconds:.3f}s."
                messages.append(duration_msg)
                _log(duration_msg)
            elif self.model_cfg.use_duration_predictor:
                t0 = _measure_start(self.model_device)
                has_speaker_duration = torch.zeros(
                    (num_candidates,), dtype=torch.bool, device=self.model_device
                )
                if self.model_cfg.use_speaker_condition and ref_mask is not None:
                    has_speaker_duration = ref_mask.any(dim=1)
                duration_features = build_duration_features(
                    [normalized_text] * num_candidates,
                    token_counts=text_mask.sum(dim=1),
                    max_text_len=text_max_len,
                    has_speaker=has_speaker_duration,
                ).to(self.model_device)
                (
                    duration_text_state,
                    duration_text_mask,
                    duration_speaker_state,
                    _duration_speaker_mask,
                    _duration_caption_state,
                    _duration_caption_mask,
                ) = self.model.encode_conditions(
                    text_input_ids=text_ids,
                    text_mask=text_mask,
                    ref_latent=ref_latent,
                    ref_mask=ref_mask,
                    caption_input_ids=caption_ids,
                    caption_mask=caption_mask,
                )
                pred_log_frames = self.model.predict_duration_log_frames(
                    text_state=duration_text_state,
                    text_mask=duration_text_mask,
                    speaker_state=duration_speaker_state,
                    speaker_mask=_duration_speaker_mask,
                    duration_features=duration_features,
                    has_speaker=has_speaker_duration,
                )
                pred_frames = torch.expm1(pred_log_frames).float().mean().item()
                scaled_frames = pred_frames * duration_scale
                min_frames = max(1, math.ceil(min_seconds * self.codec.sample_rate / hop_length))
                max_frames = max(1, math.floor(max_seconds * self.codec.sample_rate / hop_length))
                latent_steps = int(round(scaled_frames))
                latent_steps = max(min_frames, min(max_frames, latent_steps))
                target_samples = int(latent_steps * hop_length)
                stage_sec = _measure_end(self.model_device, t0)
                stage_timings.append(("predict_duration", stage_sec))
                msg = (
                    f"info: predicted duration frames={pred_frames:.1f}, "
                    f"scale={duration_scale:.3f}, using_frames={latent_steps} "
                    f"({target_samples / float(self.codec.sample_rate):.3f}s)."
                )
                messages.append(msg)
                _log(msg)
                _log(f"[runtime] predict_duration: {stage_sec * 1000.0:.1f} ms")
            else:
                fallback_seconds = 30.0
                target_samples = int(fallback_seconds * self.codec.sample_rate)
                latent_steps = math.ceil(target_samples / hop_length)
                msg = "info: checkpoint has no duration predictor; falling back to 30.000s."
                messages.append(msg)
                _log(msg)
            patched_steps = math.ceil(latent_steps / self.model_cfg.latent_patch_size)

            if isinstance(self.train_cfg, dict):
                fixed_steps = self.train_cfg.get("fixed_target_latent_steps")
                if isinstance(fixed_steps, int) and fixed_steps > 0 and latent_steps > fixed_steps:
                    msg = (
                        f"warning: requested latent length ({latent_steps}) exceeds fixed_target_latent_steps ({fixed_steps}) "
                        "used in training. Long-tail stability may degrade."
                    )
                    messages.append(msg)
                    _log(msg)

            t0 = _measure_start(self.model_device)
            z_patched = sample_euler_rf_cfg(
                model=self.model,
                text_input_ids=text_ids,
                text_mask=text_mask,
                ref_latent=ref_latent,
                ref_mask=ref_mask,
                sequence_length=patched_steps,
                caption_input_ids=caption_ids,
                caption_mask=caption_mask,
                num_steps=int(req.num_steps),
                cfg_scale_text=cfg_scale_text,
                cfg_scale_caption=cfg_scale_caption,
                cfg_scale_speaker=cfg_scale_speaker,
                cfg_guidance_mode=cfg_mode,
                cfg_min_t=float(req.cfg_min_t),
                cfg_max_t=float(req.cfg_max_t),
                seed=used_seed,
                truncation_factor=truncation_factor,
                rescale_k=rescale_k,
                rescale_sigma=rescale_sigma,
                use_context_kv_cache=bool(req.context_kv_cache),
                speaker_kv_scale=speaker_kv_scale,
                speaker_kv_max_layers=speaker_kv_max_layers,
                speaker_kv_min_t=speaker_kv_min_t,
                t_schedule_mode=str(req.t_schedule_mode),
                sway_coeff=float(req.sway_coeff),
            )
            stage_sec = _measure_end(self.model_device, t0)
            stage_timings.append(("sample_rf", stage_sec))
            _log(f"[runtime] sample_rf: {stage_sec * 1000.0:.1f} ms")

            t0 = _measure_start(self.model_device)
            z = unpatchify_latent(
                z_patched,
                patch_size=self.model_cfg.latent_patch_size,
                latent_dim=self.model_cfg.latent_dim,
            )
            stage_sec = _measure_end(self.model_device, t0)
            stage_timings.append(("unpatchify_latent", stage_sec))
            _log(f"[runtime] unpatchify_latent: {stage_sec * 1000.0:.1f} ms")
            z = z[:, :latent_steps]

            t0 = _measure_start(self.model_device, self.codec_device)
            trimmed_audios: list[torch.Tensor] = []
            if decode_mode == "batch":
                audio_batch = self.codec.decode_latent(z).cpu()
                for i in range(num_candidates):
                    audio_i = audio_batch[i]
                    max_samples = target_samples
                    if bool(req.trim_tail):
                        flattening_point = find_flattening_point(
                            z[i],
                            window_size=max(1, int(req.tail_window_size)),
                            std_threshold=float(req.tail_std_threshold),
                            mean_threshold=float(req.tail_mean_threshold),
                        )
                        flattening_samples = int(
                            flattening_point * int(self.codec.model.hop_length)
                        )
                        if flattening_samples > 0:
                            max_samples = min(max_samples, flattening_samples)
                    trimmed_audios.append(audio_i[:, :max_samples])
            else:
                for i in range(num_candidates):
                    audio_i = self.codec.decode_latent(z[i : i + 1]).cpu()[0]
                    max_samples = target_samples
                    if bool(req.trim_tail):
                        flattening_point = find_flattening_point(
                            z[i],
                            window_size=max(1, int(req.tail_window_size)),
                            std_threshold=float(req.tail_std_threshold),
                            mean_threshold=float(req.tail_mean_threshold),
                        )
                        flattening_samples = int(
                            flattening_point * int(self.codec.model.hop_length)
                        )
                        if flattening_samples > 0:
                            max_samples = min(max_samples, flattening_samples)
                    trimmed_audios.append(audio_i[:, :max_samples])
            stage_sec = _measure_end(self.model_device, t0, self.codec_device)
            stage_timings.append(("decode_latent", stage_sec))
            _log(f"[runtime] decode_latent ({decode_mode}): {stage_sec * 1000.0:.1f} ms")

            if self.watermarker.ready:
                t0 = _measure_start(self.codec_device)
                trimmed_audios = self.watermarker.encode_batch(
                    trimmed_audios,
                    sample_rate=int(self.codec.sample_rate),
                )
                stage_sec = _measure_end(self.codec_device, t0)
                stage_timings.append(("silentcipher_watermark", stage_sec))
                _log(f"[runtime] silentcipher_watermark: {stage_sec * 1000.0:.1f} ms")
            else:
                msg = (
                    "warning: SilentCipher watermark is unavailable; generated audio was not "
                    "watermarked."
                )
                messages.append(msg)
                _log(msg)

            total_to_decode = _measure_end(self.model_device, post_load_t0, self.codec_device)
            _log(f"[runtime] total_to_decode: {total_to_decode:.3f} s")

        _log("[runtime] done synthesize")
        return SamplingResult(
            audio=trimmed_audios[0],
            audios=trimmed_audios,
            sample_rate=int(self.codec.sample_rate),
            stage_timings=stage_timings,
            total_to_decode=total_to_decode,
            used_seed=used_seed,
            messages=messages,
        )

    def synthesize_batch(
        self,
        reqs: list[SamplingRequest],
        *,
        log_fn: Callable[[str], None] | None = None,
    ) -> list[SamplingResult]:
        if not reqs:
            return []
        if len(reqs) == 1:
            return [self.synthesize(reqs[0], log_fn=log_fn)]

        def _log(msg: str) -> None:
            if log_fn is not None:
                log_fn(msg)

        req = reqs[0]
        batch_size = len(reqs)
        common_fields = (
            "no_ref",
            "ref_normalize_db",
            "ref_ensure_max",
            "decode_mode",
            "seconds",
            "duration_scale",
            "min_seconds",
            "max_seconds",
            "max_ref_seconds",
            "max_text_len",
            "max_caption_len",
            "num_steps",
            "cfg_scale_text",
            "cfg_scale_caption",
            "cfg_scale_speaker",
            "cfg_guidance_mode",
            "cfg_scale",
            "cfg_min_t",
            "cfg_max_t",
            "truncation_factor",
            "rescale_k",
            "rescale_sigma",
            "context_kv_cache",
            "speaker_kv_scale",
            "speaker_kv_min_t",
            "speaker_kv_max_layers",
            "t_schedule_mode",
            "sway_coeff",
            "trim_tail",
            "tail_window_size",
            "tail_std_threshold",
            "tail_mean_threshold",
            "lora_adapter",
        )
        for other in reqs[1:]:
            for field in common_fields:
                if getattr(other, field) != getattr(req, field):
                    raise ValueError(f"Batch chunk requests must share {field}.")

        messages: list[str] = []
        if req.seconds is not None and req.seconds <= 0:
            raise ValueError(f"seconds must be > 0, got {req.seconds}")
        duration_scale = float(req.duration_scale)
        if duration_scale <= 0:
            raise ValueError(f"duration_scale must be > 0, got {duration_scale}")
        min_seconds = float(req.min_seconds)
        max_seconds = float(req.max_seconds)
        if min_seconds <= 0:
            raise ValueError(f"min_seconds must be > 0, got {min_seconds}")
        if max_seconds < min_seconds:
            raise ValueError(
                f"max_seconds must be >= min_seconds, got min={min_seconds} max={max_seconds}"
            )
        decode_mode = str(req.decode_mode).strip().lower()
        if decode_mode not in {"sequential", "batch"}:
            raise ValueError(
                f"Unsupported decode_mode={req.decode_mode!r}. Expected one of: sequential, batch."
            )

        normalized_texts = [normalize_text(str(item.text)).strip() for item in reqs]
        if any(text == "" for text in normalized_texts):
            raise ValueError("one or more batch texts became empty after normalization.")

        text_max_len = (
            self.default_text_max_len if req.max_text_len is None else int(req.max_text_len)
        )
        if text_max_len <= 0:
            raise ValueError(f"max_text_len must be > 0, got {text_max_len}")
        caption_max_len = (
            self.default_caption_max_len
            if req.max_caption_len is None
            else int(req.max_caption_len)
        )
        if self.model_cfg.use_caption_condition and caption_max_len <= 0:
            raise ValueError(f"max_caption_len must be > 0, got {caption_max_len}")

        captions = ["" if item.caption is None else str(item.caption).strip() for item in reqs]
        has_caption_text = bool(self.model_cfg.use_caption_condition and any(captions))

        truncation_factor = None if req.truncation_factor is None else float(req.truncation_factor)
        rescale_k = None if req.rescale_k is None else float(req.rescale_k)
        rescale_sigma = None if req.rescale_sigma is None else float(req.rescale_sigma)
        if truncation_factor is not None and truncation_factor <= 0:
            raise ValueError(f"truncation_factor must be > 0, got {truncation_factor}")
        if (rescale_k is None) != (rescale_sigma is None):
            raise ValueError("rescale_k and rescale_sigma must be set together.")
        if rescale_k is not None and rescale_k <= 0:
            raise ValueError(f"rescale_k must be > 0, got {rescale_k}")
        if rescale_sigma is not None and rescale_sigma <= 0:
            raise ValueError(f"rescale_sigma must be > 0, got {rescale_sigma}")

        speaker_kv_scale = None if req.speaker_kv_scale is None else float(req.speaker_kv_scale)
        speaker_kv_min_t = None
        speaker_kv_max_layers = (
            None if req.speaker_kv_max_layers is None else int(req.speaker_kv_max_layers)
        )
        if speaker_kv_scale is not None:
            if not self.model_cfg.use_speaker_condition:
                messages.append(
                    "info: speaker conditioning is disabled for this checkpoint; ignoring speaker_kv_scale."
                )
                speaker_kv_scale = None
            else:
                if speaker_kv_scale <= 0:
                    raise ValueError(f"speaker_kv_scale must be > 0, got {speaker_kv_scale}")
                speaker_kv_min_t = (
                    0.9 if req.speaker_kv_min_t is None else float(req.speaker_kv_min_t)
                )
                if not (0.0 <= speaker_kv_min_t <= 1.0):
                    raise ValueError(f"speaker_kv_min_t must be in [0, 1], got {speaker_kv_min_t}")
                if speaker_kv_max_layers is not None and speaker_kv_max_layers < 0:
                    raise ValueError(
                        f"speaker_kv_max_layers must be >= 0 when specified, got {speaker_kv_max_layers}"
                    )

        cfg_mode = str(req.cfg_guidance_mode).strip().lower()
        if cfg_mode not in {"independent", "joint", "alternating"}:
            raise ValueError(
                f"Unsupported cfg_guidance_mode={req.cfg_guidance_mode!r}. "
                "Expected one of: independent, joint, alternating."
            )

        cfg_scale_text, cfg_scale_caption, cfg_scale_speaker, scale_messages = resolve_cfg_scales(
            cfg_guidance_mode=cfg_mode,
            cfg_scale_text=req.cfg_scale_text,
            cfg_scale_caption=req.cfg_scale_caption,
            cfg_scale_speaker=req.cfg_scale_speaker,
            cfg_scale=req.cfg_scale,
            use_caption_condition=has_caption_text,
            use_speaker_condition=self.model_cfg.use_speaker_condition,
        )
        messages.extend(scale_messages)
        for msg in scale_messages:
            _log(msg)

        used_seeds: list[int] = []
        for item in reqs:
            if item.seed is None:
                used_seeds.append(int(secrets.randbits(63)))
            else:
                used_seeds.append(int(item.seed))

        _log(
            f"[runtime] start synthesize_batch "
            f"model_device={self.key.model_device} model_precision={self.key.model_precision} "
            f"codec_device={self.key.codec_device} codec_precision={self.key.codec_precision} "
            f"silentcipher_watermark={self.watermarker.ready} mode={req.cfg_guidance_mode} "
            f"seconds={req.seconds} steps={req.num_steps} schedule={req.t_schedule_mode} "
            f"sway_coeff={req.sway_coeff} batch={batch_size} "
            f"decode_mode={req.decode_mode}"
        )

        stage_timings: list[tuple[str, float]] = []
        post_load_t0 = _measure_start(self.model_device, self.codec_device)

        with self._infer_lock, torch.inference_mode():
            t0 = _measure_start(self.model_device)
            text_ids, text_mask = self.tokenizer.batch_encode(
                normalized_texts,
                max_length=text_max_len,
            )
            stage_sec = _measure_end(self.model_device, t0)
            stage_timings.append(("tokenize_text", stage_sec))
            _log(f"[runtime] batch tokenize_text: {stage_sec * 1000.0:.1f} ms")
            text_ids = text_ids.to(self.model_device)
            text_mask = text_mask.to(self.model_device)

            caption_ids = None
            caption_mask = None
            if self.model_cfg.use_caption_condition:
                if self.caption_tokenizer is None:
                    raise RuntimeError(
                        "Caption conditioning is enabled but caption tokenizer is not loaded."
                    )
                caption_ids, caption_mask = self.caption_tokenizer.batch_encode(
                    captions,
                    max_length=caption_max_len,
                )
                for index, caption_text in enumerate(captions):
                    if caption_text == "":
                        caption_mask[index].zero_()
                caption_ids = caption_ids.to(self.model_device)
                caption_mask = caption_mask.to(self.model_device)

            t0 = _measure_start(self.model_device, self.codec_device)
            msg_count_before_ref = len(messages)
            loaded_refs = [
                self._load_reference_latent(req=item, batch_size=1, messages=messages)
                for item in reqs
            ]
            if loaded_refs[0][0] is None:
                if any(latent is not None or mask is not None for latent, mask in loaded_refs):
                    raise ValueError("Batch reference inputs produced incompatible speaker states.")
                ref_latent = None
                ref_mask = None
            else:
                if any(latent is None or mask is None for latent, mask in loaded_refs):
                    raise ValueError("Batch reference inputs produced incompatible speaker states.")
                ref_latent_items = [latent for latent, _mask in loaded_refs if latent is not None]
                ref_mask_items = [mask for _latent, mask in loaded_refs if mask is not None]
                max_ref_steps = max(int(latent.shape[1]) for latent in ref_latent_items)
                ref_dim = int(ref_latent_items[0].shape[2])
                ref_latent = torch.zeros(
                    (batch_size, max_ref_steps, ref_dim),
                    device=self.model_device,
                    dtype=ref_latent_items[0].dtype,
                )
                ref_mask = torch.zeros(
                    (batch_size, max_ref_steps),
                    device=self.model_device,
                    dtype=torch.bool,
                )
                for index, (latent, mask) in enumerate(
                    zip(ref_latent_items, ref_mask_items, strict=True)
                ):
                    ref_steps = int(latent.shape[1])
                    ref_latent[index, :ref_steps] = latent[0]
                    ref_mask[index, :ref_steps] = mask[0]
            stage_sec = _measure_end(self.model_device, t0, self.codec_device)
            stage_timings.append(("prepare_reference", stage_sec))
            for msg in messages[msg_count_before_ref:]:
                _log(msg)
            _log(f"[runtime] batch prepare_reference: {stage_sec * 1000.0:.1f} ms")

            hop_length = int(self.codec.model.hop_length)
            if req.seconds is not None:
                manual_seconds = float(req.seconds)
                clamped_seconds = min(max_seconds, max(min_seconds, manual_seconds))
                if clamped_seconds != manual_seconds:
                    msg = (
                        f"warning: manual duration {manual_seconds:.3f}s was clamped to "
                        f"{clamped_seconds:.3f}s."
                    )
                    messages.append(msg)
                    _log(msg)
                target_samples_per_item = [
                    max(1, int(clamped_seconds * self.codec.sample_rate)) for _item in reqs
                ]
                latent_steps_per_item = [
                    math.ceil(target_samples / hop_length)
                    for target_samples in target_samples_per_item
                ]
            elif self.model_cfg.use_duration_predictor:
                t0 = _measure_start(self.model_device)
                has_speaker_duration = torch.zeros(
                    (batch_size,), dtype=torch.bool, device=self.model_device
                )
                if self.model_cfg.use_speaker_condition and ref_mask is not None:
                    has_speaker_duration = ref_mask.any(dim=1)
                duration_features = build_duration_features(
                    normalized_texts,
                    token_counts=text_mask.sum(dim=1),
                    max_text_len=text_max_len,
                    has_speaker=has_speaker_duration,
                ).to(self.model_device)
                (
                    duration_text_state,
                    duration_text_mask,
                    duration_speaker_state,
                    duration_speaker_mask,
                    _duration_caption_state,
                    _duration_caption_mask,
                ) = self.model.encode_conditions(
                    text_input_ids=text_ids,
                    text_mask=text_mask,
                    ref_latent=ref_latent,
                    ref_mask=ref_mask,
                    caption_input_ids=caption_ids,
                    caption_mask=caption_mask,
                )
                pred_log_frames = self.model.predict_duration_log_frames(
                    text_state=duration_text_state,
                    text_mask=duration_text_mask,
                    speaker_state=duration_speaker_state,
                    speaker_mask=duration_speaker_mask,
                    duration_features=duration_features,
                    has_speaker=has_speaker_duration,
                )
                predicted_frames = torch.expm1(pred_log_frames).float().reshape(-1)
                min_frames = max(
                    1, math.ceil(min_seconds * self.codec.sample_rate / hop_length)
                )
                max_frames = max(
                    1, math.floor(max_seconds * self.codec.sample_rate / hop_length)
                )
                latent_steps_per_item = [
                    max(
                        min_frames,
                        min(max_frames, int(round(float(frames.item()) * duration_scale))),
                    )
                    for frames in predicted_frames
                ]
                target_samples_per_item = [
                    latent_steps * hop_length for latent_steps in latent_steps_per_item
                ]
                stage_sec = _measure_end(self.model_device, t0)
                stage_timings.append(("predict_duration", stage_sec))
                _log(
                    "[runtime] batch predict_duration: "
                    f"{stage_sec * 1000.0:.1f} ms frames={latent_steps_per_item}"
                )
            else:
                fallback_seconds = 30.0
                target_samples_per_item = [
                    int(fallback_seconds * self.codec.sample_rate) for _item in reqs
                ]
                latent_steps_per_item = [
                    math.ceil(target_samples / hop_length)
                    for target_samples in target_samples_per_item
                ]
                msg = "info: checkpoint has no duration predictor; falling back to 30.000s."
                messages.append(msg)
                _log(msg)

            max_latent_steps = max(latent_steps_per_item)
            patched_steps = math.ceil(max_latent_steps / self.model_cfg.latent_patch_size)

            if isinstance(self.train_cfg, dict):
                fixed_steps = self.train_cfg.get("fixed_target_latent_steps")
                if (
                    isinstance(fixed_steps, int)
                    and fixed_steps > 0
                    and max_latent_steps > fixed_steps
                ):
                    msg = (
                        f"warning: requested latent length ({max_latent_steps}) exceeds "
                        f"fixed_target_latent_steps ({fixed_steps}) used in training. "
                        "Long-tail stability may degrade."
                    )
                    messages.append(msg)
                    _log(msg)

            t0 = _measure_start(self.model_device)
            z_patched = sample_euler_rf_cfg(
                model=self.model,
                text_input_ids=text_ids,
                text_mask=text_mask,
                ref_latent=ref_latent,
                ref_mask=ref_mask,
                sequence_length=patched_steps,
                caption_input_ids=caption_ids,
                caption_mask=caption_mask,
                num_steps=int(req.num_steps),
                cfg_scale_text=cfg_scale_text,
                cfg_scale_caption=cfg_scale_caption,
                cfg_scale_speaker=cfg_scale_speaker,
                cfg_guidance_mode=cfg_mode,
                cfg_min_t=float(req.cfg_min_t),
                cfg_max_t=float(req.cfg_max_t),
                seed=used_seeds[0],
                seeds=used_seeds,
                truncation_factor=truncation_factor,
                rescale_k=rescale_k,
                rescale_sigma=rescale_sigma,
                use_context_kv_cache=bool(req.context_kv_cache),
                speaker_kv_scale=speaker_kv_scale,
                speaker_kv_max_layers=speaker_kv_max_layers,
                speaker_kv_min_t=speaker_kv_min_t,
                t_schedule_mode=str(req.t_schedule_mode),
                sway_coeff=float(req.sway_coeff),
            )
            stage_sec = _measure_end(self.model_device, t0)
            stage_timings.append(("sample_rf", stage_sec))
            _log(f"[runtime] batch sample_rf: {stage_sec * 1000.0:.1f} ms")

            t0 = _measure_start(self.model_device)
            z = unpatchify_latent(
                z_patched,
                patch_size=self.model_cfg.latent_patch_size,
                latent_dim=self.model_cfg.latent_dim,
            )
            stage_sec = _measure_end(self.model_device, t0)
            stage_timings.append(("unpatchify_latent", stage_sec))
            _log(f"[runtime] batch unpatchify_latent: {stage_sec * 1000.0:.1f} ms")
            z = z[:, :max_latent_steps]

            t0 = _measure_start(self.model_device, self.codec_device)
            trimmed_audios: list[torch.Tensor] = []
            if decode_mode == "batch":
                audio_batch = self.codec.decode_latent(z).cpu()
                for i in range(batch_size):
                    audio_i = audio_batch[i]
                    latent_steps = latent_steps_per_item[i]
                    max_samples = target_samples_per_item[i]
                    if bool(req.trim_tail):
                        flattening_point = find_flattening_point(
                            z[i, :latent_steps],
                            window_size=max(1, int(req.tail_window_size)),
                            std_threshold=float(req.tail_std_threshold),
                            mean_threshold=float(req.tail_mean_threshold),
                        )
                        flattening_samples = int(
                            flattening_point * int(self.codec.model.hop_length)
                        )
                        if flattening_samples > 0:
                            max_samples = min(max_samples, flattening_samples)
                    trimmed_audios.append(audio_i[:, :max_samples])
            else:
                for i in range(batch_size):
                    latent_steps = latent_steps_per_item[i]
                    audio_i = self.codec.decode_latent(
                        z[i : i + 1, :latent_steps]
                    ).cpu()[0]
                    max_samples = target_samples_per_item[i]
                    if bool(req.trim_tail):
                        flattening_point = find_flattening_point(
                            z[i, :latent_steps],
                            window_size=max(1, int(req.tail_window_size)),
                            std_threshold=float(req.tail_std_threshold),
                            mean_threshold=float(req.tail_mean_threshold),
                        )
                        flattening_samples = int(
                            flattening_point * int(self.codec.model.hop_length)
                        )
                        if flattening_samples > 0:
                            max_samples = min(max_samples, flattening_samples)
                    trimmed_audios.append(audio_i[:, :max_samples])
            stage_sec = _measure_end(self.model_device, t0, self.codec_device)
            stage_timings.append(("decode_latent", stage_sec))
            _log(f"[runtime] batch decode_latent ({decode_mode}): {stage_sec * 1000.0:.1f} ms")

            if self.watermarker.ready:
                t0 = _measure_start(self.codec_device)
                trimmed_audios = self.watermarker.encode_batch(
                    trimmed_audios,
                    sample_rate=int(self.codec.sample_rate),
                )
                stage_sec = _measure_end(self.codec_device, t0)
                stage_timings.append(("silentcipher_watermark", stage_sec))
                _log(f"[runtime] batch silentcipher_watermark: {stage_sec * 1000.0:.1f} ms")

            total_to_decode = _measure_end(self.model_device, post_load_t0, self.codec_device)
            _log(f"[runtime] batch total_to_decode: {total_to_decode:.3f} s")

        _log("[runtime] done synthesize_batch")
        return [
            SamplingResult(
                audio=trimmed_audios[index],
                audios=[trimmed_audios[index]],
                sample_rate=int(self.codec.sample_rate),
                stage_timings=stage_timings,
                total_to_decode=total_to_decode,
                used_seed=used_seeds[index],
                messages=messages,
            )
            for index in range(batch_size)
        ]

    def unload(self) -> None:
        self._reference_latent_cache.clear()
        del self.model
        del self.tokenizer
        del self.codec
        gc.collect()
        for device in (self.model_device, self.codec_device):
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                mps = getattr(torch, "mps", None)
                if mps is not None and hasattr(mps, "empty_cache"):
                    mps.empty_cache()


_RUNTIME_CACHE_LOCK = threading.Lock()
_RUNTIME_CACHE_KEY: RuntimeKey | None = None
_RUNTIME_CACHE_VALUE: InferenceRuntime | None = None


def get_cached_runtime(key: RuntimeKey) -> tuple[InferenceRuntime, bool]:
    global _RUNTIME_CACHE_KEY, _RUNTIME_CACHE_VALUE
    with _RUNTIME_CACHE_LOCK:
        if _RUNTIME_CACHE_VALUE is not None and _RUNTIME_CACHE_KEY == key:
            return _RUNTIME_CACHE_VALUE, False

        old_runtime = _RUNTIME_CACHE_VALUE
        runtime = InferenceRuntime.from_key(key)
        _RUNTIME_CACHE_KEY = key
        _RUNTIME_CACHE_VALUE = runtime

    if old_runtime is not None:
        old_runtime.unload()

    return runtime, True


def clear_cached_runtime() -> None:
    global _RUNTIME_CACHE_KEY, _RUNTIME_CACHE_VALUE
    with _RUNTIME_CACHE_LOCK:
        runtime = _RUNTIME_CACHE_VALUE
        _RUNTIME_CACHE_KEY = None
        _RUNTIME_CACHE_VALUE = None

    if runtime is not None:
        runtime.unload()


def _load_audio(path: str | Path) -> tuple[torch.Tensor, int]:
    try:
        return torchaudio.load(str(path))
    except RuntimeError:
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="float32")
        wav = torch.from_numpy(data)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        else:
            wav = wav.T
        return wav, sr


def save_wav(path: str | Path, audio: torch.Tensor, sample_rate: int) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torchaudio.save(str(out_path), audio, sample_rate)
    except RuntimeError:
        import soundfile as sf

        sf.write(str(out_path), audio.squeeze(0).numpy(), sample_rate)
    return out_path
