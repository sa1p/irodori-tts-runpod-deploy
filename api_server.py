from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import queue
import subprocess
import tempfile
import threading
import time
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
)
from irodori_tts.lora import is_lora_adapter_dir

DEFAULT_CHECKPOINT = "outputs/kohaku_lora/kohaku_lora_merged.safetensors"
DEFAULT_REF_WAV = r"D:\sbv2\Style-Bert-VITS2\Data\kohaku-haishin_v2\raw\haishin\kohaku-haishin-2.wav"
LOGGER = logging.getLogger("uvicorn.error")


@dataclass(frozen=True)
class ModelSpec:
    id: str
    checkpoint: str
    ref_wav: str | None = None
    ref_latent: str | None = None
    lora_adapter: str | None = None
    enabled: bool = True
    preload: bool = False
    description: str | None = None

    @property
    def mode(self) -> str:
        if self.ref_latent:
            return "reference_latent"
        return "reference_wav"

    @property
    def available_modes(self) -> list[str]:
        modes = [self.mode]
        if self.lora_adapter:
            modes.append("lora")
        return modes


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    caption: str | None = Field(default=None, max_length=1000)
    model_id: str | None = None
    format: Literal["wav", "mp3"] = "wav"
    auto_split: bool = True
    max_chunk_chars: int = Field(default=100, ge=20, le=100)
    chunk_seconds: float | None = Field(default=None, gt=0.0, le=60.0)
    duration_scale: float = Field(default=1.0, gt=0.0, le=4.0)
    min_seconds: float = Field(default=0.5, gt=0.0, le=60.0)
    max_seconds: float = Field(default=30.0, gt=0.0, le=60.0)
    chunk_silence_ms: int = Field(default=350, ge=0, le=2000)
    chunk_tail_padding_ms: int = Field(default=250, ge=0, le=2000)
    trim_tail: bool = True
    seed: int | None = None
    num_steps: int = Field(default=40, ge=4, le=80)
    t_schedule_mode: Literal["linear", "sway"] = "sway"
    sway_coeff: float = Field(default=-1.0, ge=-2.0, le=2.0)
    cfg_scale_text: float = Field(default=3.0, ge=0.0, le=8.0)
    cfg_scale_caption: float = Field(default=3.0, ge=0.0, le=8.0)
    cfg_scale_speaker: float = Field(default=5.0, ge=0.0, le=10.0)
    speaker_kv_scale: float | None = Field(default=None, gt=0.0)
    speaker_kv_min_t: float | None = Field(default=None, ge=0.0, le=1.0)
    speaker_kv_max_layers: int | None = Field(default=None, ge=0)
    use_lora: bool = False
    reference_wav: str | None = None
    reference_wav_base64: str | None = None
    reference_wav_mime: str | None = None
    reference_latent: str | None = None
    seconds: float | None = Field(default=None, gt=0.0, le=60.0)


class ModelRegistry:
    def __init__(self, models: dict[str, ModelSpec], default_model_id: str):
        self.models = models
        self.default_model_id = default_model_id

    @classmethod
    def from_env(cls) -> ModelRegistry:
        registry_path_raw = os.getenv("IRODORI_MODEL_REGISTRY")
        if registry_path_raw:
            return cls.from_file(Path(registry_path_raw))
        return cls.single_model_from_env()

    @classmethod
    def from_file(cls, path: Path) -> ModelRegistry:
        if not path.is_file():
            raise FileNotFoundError(f"IRODORI_MODEL_REGISTRY not found: {path}")
        root = json.loads(path.read_text(encoding="utf-8"))
        base_dir = path.parent
        default_model_id = str(root.get("default_model_id") or "").strip()
        models: dict[str, ModelSpec] = {}
        for item in root.get("models", []):
            model_id = str(item["id"]).strip()
            if not model_id:
                raise ValueError(f"empty model id in registry: {path}")
            raw_adapter = item.get("lora_adapter") or item.get("adapter")
            if raw_adapter:
                raw_checkpoint = item.get("base_checkpoint") or item.get("checkpoint")
                if raw_checkpoint is None:
                    raise ValueError(f"LoRA model is missing base_checkpoint: {model_id}")
                checkpoint = _resolve_maybe_relative(str(raw_checkpoint), base_dir)
                lora_adapter = _resolve_maybe_relative(str(raw_adapter), base_dir)
            else:
                checkpoint = _resolve_maybe_relative(str(item["checkpoint"]), base_dir)
                lora_adapter = None
            raw_ref_wav = item.get("ref_wav") or item.get("reference_wav")
            raw_ref_latent = item.get("ref_latent") or item.get("reference_latent")
            if raw_ref_wav is None and raw_ref_latent is None:
                raise ValueError(
                    f"model is missing ref_wav/reference_wav or ref_latent/reference_latent: {model_id}"
                )
            ref_wav = (
                _resolve_maybe_relative(str(raw_ref_wav), base_dir)
                if raw_ref_wav is not None
                else None
            )
            ref_latent = (
                _resolve_maybe_relative(str(raw_ref_latent), base_dir)
                if raw_ref_latent is not None
                else None
            )
            spec = ModelSpec(
                id=model_id,
                checkpoint=checkpoint,
                ref_wav=ref_wav,
                ref_latent=ref_latent,
                lora_adapter=lora_adapter,
                enabled=bool(item.get("enabled", True)),
                preload=bool(item.get("preload", False)),
                description=item.get("description"),
            )
            models[model_id] = spec

        enabled_ids = [model_id for model_id, spec in models.items() if spec.enabled]
        if not enabled_ids:
            raise ValueError(f"registry has no enabled models: {path}")
        if not default_model_id:
            default_model_id = enabled_ids[0]
        if default_model_id not in models or not models[default_model_id].enabled:
            raise ValueError(f"default_model_id is not enabled in registry: {default_model_id}")
        return cls(models=models, default_model_id=default_model_id)

    @classmethod
    def single_model_from_env(cls) -> ModelRegistry:
        checkpoint = _resolve_path(os.getenv("IRODORI_CHECKPOINT", DEFAULT_CHECKPOINT))
        ref_wav = _resolve_path(os.getenv("IRODORI_REF_WAV", DEFAULT_REF_WAV))
        default_id = os.getenv("IRODORI_DEFAULT_MODEL_ID", "kohaku")
        return cls(
            models={
                default_id: ModelSpec(
                    id=default_id,
                    checkpoint=checkpoint,
                    ref_wav=ref_wav,
                    enabled=True,
                    preload=True,
                    description="single-model environment fallback",
                )
            },
            default_model_id=default_id,
        )

    def get(self, model_id: str | None) -> ModelSpec:
        resolved_id = model_id or self.default_model_id
        spec = self.models.get(resolved_id)
        if spec is None or not spec.enabled:
            raise KeyError(resolved_id)
        return spec

    def list_models(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for spec in self.models.values():
            rows.append(
                {
                    **asdict(spec),
                    "is_default": spec.id == self.default_model_id,
                    "mode": spec.mode,
                    "available_modes": spec.available_modes,
                    "checkpoint_exists": Path(spec.checkpoint).is_file(),
                    "lora_adapter_exists": (
                        is_lora_adapter_dir(spec.lora_adapter)
                        if spec.lora_adapter is not None
                        else None
                    ),
                    "ref_wav_exists": (
                        Path(spec.ref_wav).is_file() if spec.ref_wav is not None else None
                    ),
                    "ref_latent_exists": (
                        Path(spec.ref_latent).is_file() if spec.ref_latent is not None else None
                    ),
                }
            )
        return rows


class RuntimeCache:
    def __init__(self, max_items: int):
        self.max_items = max(1, int(max_items))
        self._lock = threading.RLock()
        self._cache: OrderedDict[RuntimeKey, InferenceRuntime] = OrderedDict()

    def get(self, key: RuntimeKey) -> tuple[InferenceRuntime, bool]:
        evicted_before_load: list[InferenceRuntime] = []
        with self._lock:
            runtime = self._cache.get(key)
            if runtime is not None:
                self._cache.move_to_end(key)
                return runtime, False

            while len(self._cache) >= self.max_items:
                _old_key, evicted = self._cache.popitem(last=False)
                if evicted is not None:
                    evicted_before_load.append(evicted)

        for evicted in evicted_before_load:
            evicted.unload()

        runtime = InferenceRuntime.from_key(key)
        evicted_after_load: list[InferenceRuntime] = []
        with self._lock:
            existing = self._cache.get(key)
            if existing is not None:
                self._cache.move_to_end(key)
                runtime.unload()
                return existing, False
            while len(self._cache) >= self.max_items:
                _old_key, evicted = self._cache.popitem(last=False)
                if evicted is not None:
                    evicted_after_load.append(evicted)
            self._cache[key] = runtime
            self._cache.move_to_end(key)
        for evicted in evicted_after_load:
            evicted.unload()
        return runtime, True

    def keys(self) -> list[dict[str, Any]]:
        with self._lock:
            return [asdict(key) for key in self._cache.keys()]

    def clear(self) -> None:
        with self._lock:
            runtimes = list(self._cache.values())
            self._cache.clear()
        for runtime in runtimes:
            runtime.unload()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    raw = os.getenv(name)
    if raw is None:
        value = default
    else:
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if minimum is not None:
        value = max(minimum, value)
    return value


def _env_float(name: str, default: float, *, minimum: float | None = None) -> float:
    raw = os.getenv(name)
    if raw is None:
        value = default
    else:
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if minimum is not None:
        value = max(minimum, value)
    return value


def _lora_load_mode() -> str:
    raw = os.getenv("IRODORI_LORA_LOAD_MODE", "merged")
    value = raw.strip().lower()
    if value in {"dynamic", "adapter"}:
        return "dynamic"
    if value in {"reload", "replace", "single", "single_adapter", "dynamic_reload"}:
        return "reload"
    if value in {"delta", "hot_swap", "hotswap", "manual", "manual_merge"}:
        return "delta"
    if value in {"merged", "merge", "merge_on_load"}:
        return "merged"
    raise ValueError(
        f"Unsupported IRODORI_LORA_LOAD_MODE={raw!r}. Expected 'merged', 'dynamic', 'reload', or 'delta'."
    )


def _resolve_path(raw: str) -> str:
    return str(Path(raw).expanduser().resolve())


def _resolve_maybe_relative(raw: str, base_dir: Path) -> str:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def build_runtime_key(spec: ModelSpec, *, use_lora: bool = False) -> RuntimeKey:
    lora_mode = _lora_load_mode() if use_lora and spec.lora_adapter is not None else "none"
    runtime_kwargs: dict[str, Any] = dict(
        checkpoint=spec.checkpoint,
        model_device=os.getenv("IRODORI_MODEL_DEVICE", "cuda"),
        codec_repo=os.getenv("IRODORI_CODEC_REPO", "Aratako/Semantic-DACVAE-Japanese-32dim"),
        model_precision=os.getenv("IRODORI_MODEL_PRECISION", "bf16"),
        codec_device=os.getenv("IRODORI_CODEC_DEVICE", "cuda"),
        codec_precision=os.getenv("IRODORI_CODEC_PRECISION", "fp32"),
        codec_deterministic_encode=_env_bool("IRODORI_CODEC_DETERMINISTIC_ENCODE", True),
        codec_deterministic_decode=_env_bool("IRODORI_CODEC_DETERMINISTIC_DECODE", True),
        compile_model=_env_bool("IRODORI_COMPILE_MODEL", False),
        compile_dynamic=_env_bool("IRODORI_COMPILE_DYNAMIC", False),
    )
    runtime_fields = RuntimeKey.__dataclass_fields__
    if "lora_adapter" in runtime_fields:
        runtime_kwargs["lora_adapter"] = spec.lora_adapter if lora_mode == "merged" else None
    if "lora_load_mode" in runtime_fields:
        runtime_kwargs["lora_load_mode"] = lora_mode
    return RuntimeKey(**runtime_kwargs)


SENTENCE_END_CHARS = {".", "!", "?", "\u3002", "\uff01", "\uff1f"}
QUOTED_NON_BREAK_SENTENCE_END_CHARS = {"!", "?", "\uff01", "\uff1f"}
QUOTE_OPENERS = {"\u300c": "\u300d", "\u300e": "\u300f"}
QUOTE_CLOSERS = set(QUOTE_OPENERS.values())
SOFT_BREAK_CHARS = {
    ",",
    ";",
    ":",
    "/",
    "\\",
    "\uff1b",
    "\uff1a",
    "\u30fb",
    "\u2026",
    "\u2025",
    "\u2014",
    "\u2661",
    "\u2665",
    "\u266a",
    "\u266b",
    "\u2764",
}
HARD_BREAK_CHARS = {" ", "\u3000"}
SAFE_BACKTRACK_PUNCTUATION = {"\u3001", "\uff0c"}
PRIMARY_BACKTRACK_CHARS = SOFT_BREAK_CHARS | SAFE_BACKTRACK_PUNCTUATION
EMOJI_JOINERS = {"\ufe0e", "\ufe0f", "\u200d"}
EXPRESSIVE_BREAK_CHARS = {
    "/",
    "\\",
    "\u2661",
    "\u2665",
    "\u266a",
    "\u266b",
    "\u2764",
}
TTS_SYMBOL_REPLACEMENTS = (
    ("\u2665\ufe0f", "\U0001f60f"),
    ("\u2764\ufe0f", "\U0001f60f"),
    ("\u2661", "\U0001f60f"),
    ("\u2764", "\U0001f60f"),
    ("\u2665", "\U0001f60f"),
    ("\U0001f495", "\U0001f60f"),
)
NON_SPEECH_CHARS = SENTENCE_END_CHARS | SOFT_BREAK_CHARS | EMOJI_JOINERS | {
    "\u3000",
    " ",
    "\t",
    "\n",
}
SAFE_BACKTRACK_CHARS = SOFT_BREAK_CHARS | SAFE_BACKTRACK_PUNCTUATION | {
    "\u3000",
    " ",
    "\u306f",
    "\u304c",
    "\u3092",
    "\u306b",
    "\u3067",
    "\u3068",
    "\u3082",
    "\u3084",
    "\u306e",
    "\u3078",
}


def _is_emoji_codepoint(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x1F000 <= cp <= 0x1FAFF
        or 0x2600 <= cp <= 0x27BF
        or 0x2300 <= cp <= 0x23FF
    )


def _has_speakable_text(chunk: str) -> bool:
    return any(
        ch not in NON_SPEECH_CHARS and not _is_emoji_codepoint(ch)
        for ch in str(chunk)
    )


def _chunk_len(chunk: str) -> int:
    return len(str(chunk).strip())


def _join_chunks(left: str, right: str) -> str:
    return f"{left}{right}".strip()


def _replace_tts_expression_symbols(text: str) -> str:
    normalized = str(text)
    for source, target in TTS_SYMBOL_REPLACEMENTS:
        normalized = normalized.replace(source, target)
    return normalized


def _merge_non_speech_chunks(chunks: list[str], max_chunk_chars: int) -> list[str]:
    pending = [chunk for chunk in chunks if chunk.strip()]
    merged: list[str] = []
    index = 0
    while index < len(pending):
        chunk = pending[index]
        if _has_speakable_text(chunk):
            merged.append(chunk)
            index += 1
            continue

        if merged:
            merged[-1] = _join_chunks(merged[-1], chunk)
        elif index + 1 < len(pending):
            pending[index + 1] = _join_chunks(chunk, pending[index + 1])
        index += 1

    return [chunk for chunk in merged if _has_speakable_text(chunk)]


def _merge_short_chunks(chunks: list[str], max_chunk_chars: int) -> list[str]:
    if len(chunks) <= 1:
        return chunks

    min_chunk_chars = min(12, max(6, int(max_chunk_chars * 0.12)))
    pending = list(chunks)
    merged: list[str] = []
    index = 0
    while index < len(pending):
        chunk = pending[index]
        if _chunk_len(chunk) >= min_chunk_chars or len(pending) == 1:
            merged.append(chunk)
            index += 1
            continue

        should_prepend_next = chunk[-1:] in SOFT_BREAK_CHARS
        if should_prepend_next and index + 1 < len(pending):
            candidate = _join_chunks(chunk, pending[index + 1])
            if _chunk_len(candidate) <= max_chunk_chars:
                pending[index + 1] = candidate
                index += 1
                continue

        if merged:
            candidate = _join_chunks(merged[-1], chunk)
            if _chunk_len(candidate) <= max_chunk_chars:
                merged[-1] = candidate
                index += 1
                continue

        if index + 1 < len(pending):
            candidate = _join_chunks(chunk, pending[index + 1])
            if _chunk_len(candidate) <= max_chunk_chars:
                pending[index + 1] = candidate
                index += 1
                continue

        if merged:
            candidate = _join_chunks(merged[-1], chunk)
            if _chunk_len(candidate) <= max_chunk_chars:
                merged[-1] = candidate
            else:
                merged.append(chunk)
        else:
            merged.append(chunk)
        index += 1

    return merged


def _postprocess_tts_chunks(chunks: list[str], max_chunk_chars: int) -> list[str]:
    chunks = _merge_non_speech_chunks(chunks, max_chunk_chars)
    chunks = _merge_short_chunks(chunks, max_chunk_chars)
    chunks = _merge_non_speech_chunks(chunks, max_chunk_chars)
    return chunks


def _prepare_tts_chunks(text: str, max_chunk_chars: int, auto_split: bool) -> list[str]:
    raw_chunks = split_tts_text(text, max_chunk_chars) if auto_split else [str(text).strip()]
    return [_replace_tts_expression_symbols(chunk) for chunk in raw_chunks if chunk.strip()]


def _natural_break_min_chars(max_chunk_chars: int) -> int:
    return max(14, min(32, int(max_chunk_chars * 0.18)))


def _is_expressive_break_end(prev: str, ch: str, next_ch: str) -> bool:
    if ch in EXPRESSIVE_BREAK_CHARS:
        return next_ch not in EXPRESSIVE_BREAK_CHARS and next_ch not in EMOJI_JOINERS
    if ch in EMOJI_JOINERS and prev in EXPRESSIVE_BREAK_CHARS:
        return next_ch not in EMOJI_JOINERS
    return False


def _is_non_break_sentence_end(ch: str, quote_stack: list[str]) -> bool:
    return bool(quote_stack) and ch in QUOTED_NON_BREAK_SENTENCE_END_CHARS


def _split_tts_text_line(text: str, max_chunk_chars: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    quote_stack: list[str] = []
    max_chunk_chars = max(20, int(max_chunk_chars))
    natural_break_min_chars = _natural_break_min_chars(max_chunk_chars)

    def flush() -> None:
        chunk = "".join(current).strip()
        current.clear()
        if chunk:
            chunks.append(chunk)

    def flush_at_safe_boundary() -> None:
        if not current:
            return
        candidate = "".join(current)
        min_prefix = max(12, int(max_chunk_chars * 0.45))
        split_at = -1
        for index in range(len(candidate) - 1, min_prefix - 1, -1):
            if candidate[index] in PRIMARY_BACKTRACK_CHARS:
                candidate_split_at = index + 1
                if 0 < candidate_split_at < len(candidate):
                    split_at = candidate_split_at
                    break
        for index in range(len(candidate) - 1, min_prefix - 1, -1):
            if split_at > 0:
                break
            if candidate[index] in SAFE_BACKTRACK_CHARS:
                candidate_split_at = index + 1
                if 0 < candidate_split_at < len(candidate):
                    split_at = candidate_split_at
                    break
        if split_at <= 0 or split_at >= len(candidate):
            flush()
            return

        prefix = candidate[:split_at].strip()
        suffix = candidate[split_at:].strip()
        current.clear()
        if prefix:
            chunks.append(prefix)
        if suffix:
            current.extend(suffix)

    for position, ch in enumerate(text):
        next_ch = text[position + 1] if position + 1 < len(text) else ""
        prev = current[-1] if current else ""
        quote_stack_before = list(quote_stack)
        if (
            current
            and _is_emoji_codepoint(ch)
            and prev not in EMOJI_JOINERS
            and not _is_emoji_codepoint(prev)
        ):
            flush()

        current.append(ch)

        if ch in QUOTE_OPENERS:
            quote_stack.append(QUOTE_OPENERS[ch])
        elif quote_stack and ch == quote_stack[-1]:
            quote_stack.pop()
        elif ch in QUOTE_CLOSERS:
            quote_stack.clear()

        if ch in SENTENCE_END_CHARS and not _is_non_break_sentence_end(ch, quote_stack_before):
            flush()
            continue

        current_len = len("".join(current).strip())
        if _is_expressive_break_end(prev, ch, next_ch):
            flush()
        elif (
            current_len >= natural_break_min_chars
            and ch in SOFT_BREAK_CHARS
            and ch not in EXPRESSIVE_BREAK_CHARS
        ):
            flush()
        elif current_len >= max_chunk_chars and ch not in EMOJI_JOINERS and not _is_emoji_codepoint(ch):
            flush_at_safe_boundary()

    flush()
    return _postprocess_tts_chunks(chunks, max_chunk_chars)


def split_tts_text(text: str, max_chunk_chars: int) -> list[str]:
    max_chunk_chars = max(20, int(max_chunk_chars))
    normalized = str(text).replace("\r\n", "\n").replace("\r", "\n")
    chunks: list[str] = []
    for line in normalized.split("\n"):
        segments = []
        current: list[str] = []
        for ch in line:
            if ch in HARD_BREAK_CHARS:
                segment = "".join(current).strip()
                current.clear()
                if segment:
                    segments.append(segment)
                continue
            current.append(ch)
        segment = "".join(current).strip()
        if segment:
            segments.append(segment)
        if not segments:
            continue
        for segment in segments:
            chunks.extend(_split_tts_text_line(segment, max_chunk_chars))
    return chunks


def _audio_to_channels_first(audio: torch.Tensor) -> torch.Tensor:
    audio = audio.detach().float().cpu()
    if audio.ndim == 1:
        return audio.unsqueeze(0).contiguous()
    if audio.ndim == 2:
        if audio.shape[0] <= 8:
            return audio.contiguous()
        if audio.shape[1] <= 8:
            return audio.transpose(0, 1).contiguous()
    raise ValueError(f"Unsupported audio shape: {tuple(audio.shape)}")


def _concat_audio_segments(
    segments: list[torch.Tensor],
    sample_rate: int,
    silence_ms: int,
    tail_padding_ms: int,
) -> torch.Tensor:
    if not segments:
        raise ValueError("No audio segments were generated")
    normalized = [_audio_to_channels_first(segment) for segment in segments]
    channels = normalized[0].shape[0]
    for segment in normalized:
        if segment.shape[0] != channels:
            raise ValueError("Generated chunks have different channel counts")

    silence_samples = int(sample_rate * max(0, silence_ms) / 1000)
    tail_padding_samples = int(sample_rate * max(0, tail_padding_ms) / 1000)
    if tail_padding_samples > 0:
        normalized = [
            torch.cat(
                [
                    segment,
                    torch.zeros((segment.shape[0], tail_padding_samples), dtype=segment.dtype),
                ],
                dim=1,
            )
            for segment in normalized
        ]
    if silence_samples <= 0 or len(normalized) == 1:
        return torch.cat(normalized, dim=1)

    silence = torch.zeros((channels, silence_samples), dtype=normalized[0].dtype)
    parts: list[torch.Tensor] = []
    for index, segment in enumerate(normalized):
        if index:
            parts.append(silence)
        parts.append(segment)
    return torch.cat(parts, dim=1)


def _resolve_chunk_seconds(req: TTSRequest) -> float | None:
    if req.chunk_seconds is not None:
        return float(req.chunk_seconds)
    if req.seconds is not None:
        return float(req.seconds)
    return None


def _validate_request_limits(
    req: TTSRequest,
    *,
    chunks: list[str],
    resolved_chunk_seconds: float | None,
) -> None:
    max_text_chars = _env_int("IRODORI_MAX_TEXT_CHARS", 0, minimum=0)
    if max_text_chars > 0 and len(req.text) > max_text_chars:
        raise HTTPException(
            status_code=413,
            detail=f"text is too long; max chars is {max_text_chars}",
        )

    max_chunks = _env_int("IRODORI_MAX_CHUNKS", 0, minimum=0)
    if max_chunks > 0 and len(chunks) > max_chunks:
        raise HTTPException(
            status_code=413,
            detail=f"too many chunks after splitting; max chunks is {max_chunks}",
        )

    max_total_seconds = _env_float("IRODORI_MAX_TOTAL_SECONDS", 0.0, minimum=0.0)
    if max_total_seconds > 0 and resolved_chunk_seconds is not None:
        requested_seconds = resolved_chunk_seconds * len(chunks)
        if requested_seconds > max_total_seconds:
            raise HTTPException(
                status_code=413,
                detail=(
                    "requested audio duration is too long; "
                    f"max total seconds is {max_total_seconds:g}"
                ),
            )


def _max_reference_upload_bytes() -> int:
    raw = os.getenv("IRODORI_MAX_REFERENCE_UPLOAD_BYTES", str(32 * 1024 * 1024))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"IRODORI_MAX_REFERENCE_UPLOAD_BYTES must be an integer, got {raw!r}"
        ) from exc
    return max(1, value)


def _reference_upload_dir() -> Path:
    raw = os.getenv("IRODORI_REFERENCE_UPLOAD_DIR")
    root = Path(raw) if raw else Path(tempfile.gettempdir()) / "irodori_reference_uploads"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _decode_reference_wav_base64(raw: str, mime: str | None) -> str:
    payload = str(raw).strip()
    if payload.startswith("data:"):
        header, separator, data = payload.partition(",")
        if not separator:
            raise ValueError("reference_wav_base64 data URL is missing comma separator")
        if ";base64" not in header:
            raise ValueError("reference_wav_base64 data URL must be base64 encoded")
        if mime is None:
            mime = header[5:].split(";", 1)[0] or None
        payload = data

    normalized_mime = (mime or "").strip().lower()
    if normalized_mime and normalized_mime not in {
        "audio/wav",
        "audio/wave",
        "audio/x-wav",
        "audio/vnd.wave",
        "application/octet-stream",
    }:
        raise ValueError(f"reference_wav_base64 must be WAV audio, got {mime!r}")

    compact = "".join(payload.split())
    max_bytes = _max_reference_upload_bytes()
    if len(compact) > max_bytes * 2:
        raise ValueError(
            f"reference_wav_base64 is too large; max decoded bytes is {max_bytes}"
        )

    try:
        data = base64.b64decode(compact, validate=True)
    except Exception as exc:
        raise ValueError("reference_wav_base64 is not valid base64") from exc

    if not data:
        raise ValueError("reference_wav_base64 decoded to an empty file")
    if len(data) > max_bytes:
        raise ValueError(f"reference_wav_base64 is too large; max decoded bytes is {max_bytes}")

    digest = hashlib.sha256(data).hexdigest()
    upload_dir = _reference_upload_dir()
    path = upload_dir / f"{digest}.wav"
    if not path.exists():
        tmp_file = tempfile.NamedTemporaryFile(
            delete=False,
            dir=upload_dir,
            prefix=f"{digest}.",
            suffix=".tmp",
        )
        tmp_path = Path(tmp_file.name)
        try:
            with tmp_file:
                tmp_file.write(data)
            tmp_path.replace(path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
    return str(path)


def _resolve_reference_inputs(spec: ModelSpec, req: TTSRequest) -> tuple[str | None, str | None]:
    if req.reference_latent:
        return None, req.reference_latent
    if req.reference_wav_base64:
        return _decode_reference_wav_base64(req.reference_wav_base64, req.reference_wav_mime), None
    if req.reference_wav:
        return req.reference_wav, None
    if spec.ref_latent:
        return None, spec.ref_latent
    return spec.ref_wav, None


def _format_seconds_header(seconds: float | None) -> str:
    return "auto" if seconds is None else f"{seconds:.2f}"


def _chunk_batch_size() -> int:
    raw = os.getenv("IRODORI_CHUNK_BATCH_SIZE", "1")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"IRODORI_CHUNK_BATCH_SIZE must be an integer, got {raw!r}") from exc
    return max(1, value)


def _num_steps_cap() -> int | None:
    """Optional server-side ceiling on sampling steps.

    Clients hardcode num_steps=40, which is well past the point of diminishing
    returns for this model. Capping here applies the change to every caller
    without a client release, and can be lifted by clearing the variable.
    """
    raw = os.getenv("IRODORI_NUM_STEPS_CAP", "").strip()
    if raw == "" or raw.lower() in {"off", "none", "0"}:
        return None
    value = int(raw)
    return value if value > 0 else None


def _resolve_num_steps(req: TTSRequest) -> int:
    cap = _num_steps_cap()
    requested = int(req.num_steps)
    if cap is None or requested <= cap:
        return requested
    return cap


def _preview_chunk(chunk: str, max_chars: int = 80) -> str:
    normalized = " ".join(str(chunk).split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 1] + "..."


def _audio_to_wav_bytes(audio: torch.Tensor, sample_rate: int) -> bytes:
    audio_cpu = audio.detach().float().cpu()
    if audio_cpu.ndim == 2 and audio_cpu.shape[0] <= 8:
        audio_cpu = audio_cpu.transpose(0, 1).contiguous()
    elif audio_cpu.ndim > 2:
        raise ValueError(f"Unsupported audio shape: {tuple(audio_cpu.shape)}")

    buffer = BytesIO()
    sf.write(buffer, audio_cpu.numpy(), sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _wav_to_mp3_bytes(wav_bytes: bytes) -> bytes:
    ffmpeg_path = os.getenv("IRODORI_FFMPEG_PATH", "ffmpeg")
    bitrate = os.getenv("IRODORI_MP3_BITRATE", "192k")
    proc = subprocess.run(
        [
            ffmpeg_path,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-f",
            "mp3",
            "-b:a",
            bitrate,
            "pipe:1",
        ],
        input=wav_bytes,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="replace"))
    return proc.stdout


def _channels_first_to_pcm_s16le_bytes(audio: torch.Tensor) -> bytes:
    audio_cpu = audio.detach().float().cpu()
    if audio_cpu.ndim != 2:
        raise ValueError(f"Unsupported audio shape: {tuple(audio_cpu.shape)}")
    interleaved = audio_cpu.clamp(-1.0, 1.0).transpose(0, 1).contiguous()
    return (interleaved * 32767.0).round().to(torch.int16).numpy().tobytes()


def _silence_pcm_s16le_bytes(channels: int, sample_rate: int, silence_ms: int) -> bytes:
    silence_samples = int(sample_rate * max(0, silence_ms) / 1000)
    if silence_samples <= 0:
        return b""
    return bytes(silence_samples * channels * 2)


def _start_mp3_stream_process(sample_rate: int, channels: int) -> subprocess.Popen[bytes]:
    ffmpeg_path = os.getenv("IRODORI_FFMPEG_PATH", "ffmpeg")
    bitrate = os.getenv("IRODORI_MP3_BITRATE", "192k")
    return subprocess.Popen(
        [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "s16le",
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            "-i",
            "pipe:0",
            "-f",
            "mp3",
            "-b:a",
            bitrate,
            "pipe:1",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )


_STREAM_END = object()


def _iter_mp3_stream(
    spec: ModelSpec,
    key: RuntimeKey,
    req: TTSRequest,
    chunks: list[str],
    chunk_seconds: list[float | None],
    *,
    release_synthesis_lock: bool = False,
) -> Iterator[bytes]:
    request_started_at = time.perf_counter()
    stream_max_seconds = _max_request_seconds()
    stream_timer: threading.Timer | None = None
    if stream_max_seconds > 0:

        def kill_overdue_stream() -> None:
            LOGGER.critical(
                "[tts:stream] response_watchdog_exit max_seconds=%.1f model_id=%s",
                stream_max_seconds,
                spec.id,
            )
            os._exit(_env_int("IRODORI_MAX_REQUEST_EXIT_CODE", 124, minimum=1))

        stream_timer = threading.Timer(stream_max_seconds, kill_overdue_stream)
        stream_timer.daemon = True
        stream_timer.start()

    output_queue: queue.Queue[bytes | object] = queue.Queue(
        maxsize=_env_int("IRODORI_STREAM_QUEUE_MAX_ITEMS", 8, minimum=1)
    )
    stop_event = threading.Event()
    read_size = _env_int("IRODORI_STREAM_READ_SIZE", 65536, minimum=1)
    proc_lock = threading.Lock()
    active_proc: subprocess.Popen[bytes] | None = None

    def queue_put(item: bytes | object) -> bool:
        while not stop_event.is_set():
            try:
                output_queue.put(item, timeout=0.5)
                return True
            except queue.Full:
                continue
        return False

    def put_end() -> None:
        queue_put(_STREAM_END)

    def set_active_proc(proc: subprocess.Popen[bytes] | None) -> None:
        nonlocal active_proc
        with proc_lock:
            active_proc = proc

    def kill_active_proc() -> None:
        with proc_lock:
            proc = active_proc
        if proc is not None and proc.poll() is None:
            proc.kill()

    def read_stdout(proc: subprocess.Popen[bytes]) -> None:
        try:
            if proc.stdout is None:
                return
            while True:
                data = proc.stdout.read(read_size)
                if not data:
                    break
                if not queue_put(data):
                    break
        finally:
            put_end()

    def generate_and_encode() -> None:
        proc: subprocess.Popen[bytes] | None = None
        reader: threading.Thread | None = None
        try:
            with _release_synthesis_lock_on_exit(release_synthesis_lock):
                with _track_synthesis_request(
                    route="tts:stream",
                    model_id=spec.id,
                    text_chars=len(req.text),
                    chunk_count=len(chunks),
                ):
                    _update_active_request(phase="loading_runtime")
                    runtime, reloaded = RUNTIME_CACHE.get(key)
                    _ensure_lora_adapter_for_request(runtime, spec, key)

                    LOGGER.info(
                        "[tts:stream] split model_id=%s mode=%s auto_split=%s chunks=%d max_chunk_chars=%d",
                        spec.id,
                        _format_effective_mode(spec, key),
                        req.auto_split,
                        len(chunks),
                        req.max_chunk_chars,
                    )

                    sample_rate: int | None = None
                    channels: int | None = None
                    used_seeds: list[int | None] = []
                    ref_wav, ref_latent = _resolve_reference_inputs(spec, req)
                    for index, (chunk, seconds) in enumerate(
                        zip(chunks, chunk_seconds, strict=True),
                        start=1,
                    ):
                        if stop_event.is_set():
                            LOGGER.warning(
                                "[tts:stream] cancelled_before_chunk model_id=%s chunk=%d/%d",
                                spec.id,
                                index,
                                len(chunks),
                            )
                            return
                        _update_active_request(
                            phase="synthesizing",
                            chunk_index=index,
                            chunk_batch_end=index,
                        )
                        chunk_seed = None if req.seed is None else req.seed + index - 1
                        LOGGER.info(
                            "[tts:stream] chunk %d/%d chars=%d seconds=%s seed=%s text=%r",
                            index,
                            len(chunks),
                            len(chunk),
                            _format_seconds_header(seconds),
                            "random" if chunk_seed is None else chunk_seed,
                            _preview_chunk(chunk),
                        )
                        result = runtime.synthesize(
                            SamplingRequest(
                                text=chunk,
                                caption=req.caption,
                                ref_wav=ref_wav,
                                ref_latent=ref_latent,
                                no_ref=False,
                                ref_normalize_db=-16.0,
                                ref_ensure_max=True,
                                num_candidates=1,
                                decode_mode="sequential",
                                seconds=seconds,
                                duration_scale=req.duration_scale,
                                min_seconds=req.min_seconds,
                                max_seconds=req.max_seconds,
                                max_ref_seconds=None,
                                max_text_len=None,
                                num_steps=_resolve_num_steps(req),
                                t_schedule_mode=req.t_schedule_mode,
                                sway_coeff=req.sway_coeff,
                                seed=chunk_seed,
                                cfg_guidance_mode="independent",
                                cfg_scale_text=req.cfg_scale_text,
                                cfg_scale_caption=req.cfg_scale_caption,
                                cfg_scale_speaker=req.cfg_scale_speaker,
                                speaker_kv_scale=req.speaker_kv_scale,
                                speaker_kv_min_t=req.speaker_kv_min_t,
                                speaker_kv_max_layers=req.speaker_kv_max_layers,
                                trim_tail=req.trim_tail,
                                lora_adapter=_sampling_request_lora_adapter(spec, key),
                            )
                        )
                        if stop_event.is_set():
                            LOGGER.warning(
                                "[tts:stream] cancelled_after_synthesize model_id=%s chunk=%d/%d",
                                spec.id,
                                index,
                                len(chunks),
                            )
                            return
                        used_seeds.append(result.used_seed)
                        audio = _audio_to_channels_first(result.audio)
                        if sample_rate is None:
                            sample_rate = result.sample_rate
                            channels = int(audio.shape[0])
                            proc = _start_mp3_stream_process(sample_rate, channels)
                            set_active_proc(proc)
                            reader = threading.Thread(target=read_stdout, args=(proc,), daemon=True)
                            reader.start()
                        elif sample_rate != result.sample_rate:
                            raise ValueError("Generated chunks have different sample rates")

                        if channels is None or audio.shape[0] != channels:
                            raise ValueError("Generated chunks have different channel counts")
                        if proc is None or proc.stdin is None:
                            raise RuntimeError("MP3 stream encoder is not available")

                        proc.stdin.write(_channels_first_to_pcm_s16le_bytes(audio))
                        proc.stdin.write(
                            _silence_pcm_s16le_bytes(
                                channels,
                                sample_rate,
                                req.chunk_tail_padding_ms,
                            )
                        )
                        if index < len(chunks):
                            proc.stdin.write(
                                _silence_pcm_s16le_bytes(
                                    channels,
                                    sample_rate,
                                    req.chunk_silence_ms,
                                )
                            )
                        proc.stdin.flush()

                    if proc is None:
                        raise ValueError("No audio segments were generated")
                    _update_active_request(phase="encoding")
                    if proc.stdin is not None:
                        proc.stdin.close()
                    stderr = b""
                    if proc.stderr is not None:
                        stderr = proc.stderr.read()
                    return_code = proc.wait()
                    if return_code != 0:
                        raise RuntimeError(stderr.decode("utf-8", errors="replace"))
                    LOGGER.info(
                        "[tts:stream] complete model_id=%s chunks=%d seeds=%s reloaded=%s generation_sec=%.1f",
                        spec.id,
                        len(chunks),
                        ",".join(str(seed) for seed in used_seeds),
                        reloaded,
                        time.perf_counter() - request_started_at,
                    )
        except Exception:
            LOGGER.exception(
                "[tts:stream] failed model_id=%s generation_sec=%.1f",
                spec.id,
                time.perf_counter() - request_started_at,
            )
            if proc is not None and proc.poll() is None:
                proc.kill()
            if reader is None:
                put_end()
        finally:
            set_active_proc(None)

    worker = threading.Thread(target=generate_and_encode, daemon=True)
    worker.start()
    try:
        while True:
            try:
                item = output_queue.get(timeout=0.5)
            except queue.Empty:
                if not worker.is_alive():
                    break
                continue
            if item is _STREAM_END:
                break
            yield item
    finally:
        if stream_timer is not None:
            stream_timer.cancel()
        stop_event.set()
        kill_active_proc()
        worker.join(timeout=1.0)
        if worker.is_alive():
            LOGGER.warning(
                "[tts:stream] generator_closed_worker_still_running model_id=%s age_sec=%.1f",
                spec.id,
                time.perf_counter() - request_started_at,
            )


def _request_uses_lora(spec: ModelSpec, req: TTSRequest) -> bool:
    if not req.use_lora:
        return False
    if spec.lora_adapter is None:
        raise HTTPException(status_code=400, detail=f"LoRA adapter is not configured: {spec.id}")
    return True


def _format_effective_mode(spec: ModelSpec, key: RuntimeKey) -> str:
    lora_load_mode = getattr(key, "lora_load_mode", "none")
    if lora_load_mode != "none":
        return f"lora:{lora_load_mode}"
    return spec.mode


def _sampling_request_lora_adapter(spec: ModelSpec, key: RuntimeKey) -> str | None:
    if getattr(key, "lora_load_mode", "none") in {"dynamic", "reload"}:
        return spec.lora_adapter
    return None


def _ensure_lora_adapter_for_request(
    runtime: InferenceRuntime,
    spec: ModelSpec,
    key: RuntimeKey,
) -> None:
    lora_load_mode = getattr(key, "lora_load_mode", "none")
    if spec.lora_adapter is None or lora_load_mode == "none":
        return
    if lora_load_mode == "delta":
        runtime.ensure_lora_delta_adapter(
            adapter_name=spec.id,
            adapter_path=spec.lora_adapter,
        )
    elif lora_load_mode in {"dynamic", "reload"}:
        runtime.ensure_lora_adapter(
            adapter_name=spec.id,
            adapter_path=spec.lora_adapter,
            replace_existing=True,
        )


def _validate_model_assets(spec: ModelSpec, *, use_lora: bool = False) -> None:
    if not Path(spec.checkpoint).is_file():
        raise HTTPException(status_code=503, detail=f"Checkpoint not found: {spec.checkpoint}")
    if use_lora and spec.lora_adapter is not None and not is_lora_adapter_dir(spec.lora_adapter):
        raise HTTPException(
            status_code=503,
            detail=f"LoRA adapter not found or invalid: {spec.lora_adapter}",
        )
    has_ref_wav = spec.ref_wav is not None and Path(spec.ref_wav).is_file()
    has_ref_latent = spec.ref_latent is not None and Path(spec.ref_latent).is_file()
    if not has_ref_wav and not has_ref_latent:
        raise HTTPException(
            status_code=503,
            detail=(
                "Reference asset not found: "
                f"ref_wav={spec.ref_wav!r}, ref_latent={spec.ref_latent!r}"
            ),
        )


REGISTRY = ModelRegistry.from_env()
RUNTIME_CACHE = RuntimeCache(max_items=int(os.getenv("IRODORI_MAX_CACHED_RUNTIMES", "1")))

app = FastAPI(title="Irodori-TTS Multi-Model API", version="0.2.0")
_synthesis_lock = threading.Lock()
_active_request_lock = threading.Lock()
_active_request: dict[str, Any] | None = None
_active_request_seq = 0


def _synthesis_busy() -> bool:
    return _synthesis_lock.locked()


def _worker_id() -> str | None:
    for key in ("RUNPOD_WORKER_ID", "RUNPOD_POD_ID", "HOSTNAME"):
        value = os.getenv(key)
        if value:
            return value
    return None


def _busy_retry_after_seconds() -> str:
    raw = os.getenv("IRODORI_BUSY_RETRY_AFTER_SECONDS", "1")
    try:
        parsed = int(raw)
    except ValueError:
        parsed = 1
    return str(max(parsed, 1))


def _busy_wait_timeout_seconds() -> float:
    return _env_float("IRODORI_BUSY_WAIT_TIMEOUT_SECONDS", 30.0, minimum=0.0)


def _max_request_seconds() -> float:
    return _env_float("IRODORI_MAX_REQUEST_SECONDS", 0.0, minimum=0.0)


def _ping_fails_when_busy() -> bool:
    return _env_bool("IRODORI_PING_FAILS_WHEN_BUSY", False)


def _new_request_id(route: str) -> str:
    global _active_request_seq
    with _active_request_lock:
        _active_request_seq += 1
        sequence = _active_request_seq
    worker_id = _worker_id() or "local"
    normalized_route = route.replace(":", "-").replace("/", "-").strip("-") or "request"
    return f"{worker_id}-{normalized_route}-{sequence}"


def _active_request_payload() -> dict[str, Any] | None:
    with _active_request_lock:
        if _active_request is None:
            return None
        payload = dict(_active_request)
    payload["age_seconds"] = round(time.perf_counter() - float(payload["started_at_perf"]), 3)
    return payload


def _set_active_request(
    *,
    request_id: str,
    route: str,
    model_id: str | None,
    text_chars: int,
    chunk_count: int | None,
) -> None:
    global _active_request
    with _active_request_lock:
        _active_request = {
            "request_id": request_id,
            "route": route,
            "model_id": model_id,
            "worker_id": _worker_id(),
            "text_chars": text_chars,
            "chunk_count": chunk_count,
            "chunk_index": None,
            "phase": "started",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "started_at_perf": time.perf_counter(),
        }


def _update_active_request(**updates: Any) -> None:
    with _active_request_lock:
        if _active_request is not None:
            _active_request.update(updates)


def _clear_active_request(request_id: str) -> None:
    global _active_request
    with _active_request_lock:
        if _active_request is not None and _active_request.get("request_id") == request_id:
            _active_request = None


@contextmanager
def _track_synthesis_request(
    *,
    route: str,
    model_id: str | None,
    text_chars: int,
    chunk_count: int | None,
) -> Iterator[str]:
    request_id = _new_request_id(route)
    _set_active_request(
        request_id=request_id,
        route=route,
        model_id=model_id,
        text_chars=text_chars,
        chunk_count=chunk_count,
    )
    max_seconds = _max_request_seconds()
    timer: threading.Timer | None = None
    if max_seconds > 0:

        def kill_overdue_request() -> None:
            payload = _active_request_payload()
            LOGGER.critical(
                "[tts] request_watchdog_exit max_seconds=%.1f active_request=%s",
                max_seconds,
                payload,
            )
            os._exit(_env_int("IRODORI_MAX_REQUEST_EXIT_CODE", 124, minimum=1))

        timer = threading.Timer(max_seconds, kill_overdue_request)
        timer.daemon = True
        timer.start()
    try:
        yield request_id
    finally:
        if timer is not None:
            timer.cancel()
        _clear_active_request(request_id)


def _worker_headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    headers = dict(extra or {})
    worker_id = _worker_id()
    if worker_id:
        headers["X-Irodori-Worker-Id"] = worker_id
    return headers


def _busy_headers() -> dict[str, str]:
    return _worker_headers(
        {
            "Retry-After": _busy_retry_after_seconds(),
            "X-Irodori-Busy": "1",
            "X-Irodori-Busy-Wait-Timeout": f"{_busy_wait_timeout_seconds():g}",
        }
    )


def _raise_worker_busy(route: str, model_id: str | None) -> None:
    LOGGER.warning("[tts] busy route=%s model_id=%s worker_id=%s", route, model_id, _worker_id())
    raise HTTPException(
        status_code=503,
        detail="TTS worker is busy. Retry this request on another worker.",
        headers=_busy_headers(),
    )


def _model_assets_ready_rows() -> tuple[bool, list[dict[str, Any]]]:
    rows = REGISTRY.list_models()
    enabled = [row for row in rows if row["enabled"]]
    ready = all(
        row["checkpoint_exists"]
        and (row["ref_wav_exists"] or row["ref_latent_exists"])
        and (row["lora_adapter_exists"] is not False)
        for row in enabled
    )
    return ready, rows


def _readiness_payload() -> dict[str, Any]:
    assets_ready, rows = _model_assets_ready_rows()
    busy = _synthesis_busy()
    return {
        "ready": assets_ready and not busy,
        "assets_ready": assets_ready,
        "busy": busy,
        "active_request": _active_request_payload(),
        "worker_id": _worker_id(),
        "models": rows,
        "cache": RUNTIME_CACHE.keys(),
    }


def _acquire_synthesis_lock(route: str, model_id: str | None) -> None:
    if _synthesis_lock.acquire(blocking=False):
        return

    wait_timeout = _busy_wait_timeout_seconds()
    if wait_timeout <= 0:
        _raise_worker_busy(route, model_id)

    started_at = time.perf_counter()
    LOGGER.warning(
        "[tts] busy_wait route=%s model_id=%s worker_id=%s timeout_sec=%.1f",
        route,
        model_id,
        _worker_id(),
        wait_timeout,
    )
    if not _synthesis_lock.acquire(timeout=wait_timeout):
        _raise_worker_busy(route, model_id)
    LOGGER.info(
        "[tts] busy_wait_acquired route=%s model_id=%s worker_id=%s waited_sec=%.3f",
        route,
        model_id,
        _worker_id(),
        time.perf_counter() - started_at,
    )


@contextmanager
def _release_synthesis_lock_on_exit(enabled: bool) -> Iterator[None]:
    try:
        yield
    finally:
        if enabled:
            _synthesis_lock.release()


@contextmanager
def _try_synthesis_lock(route: str, model_id: str | None) -> Iterator[None]:
    _acquire_synthesis_lock(route, model_id)
    try:
        yield
    finally:
        _synthesis_lock.release()


def _warmup_seconds() -> list[float]:
    """Utterance lengths (seconds) to pre-capture CUDA graphs for."""
    raw = os.getenv("IRODORI_WARMUP_SECONDS", "1,2,3,4,5,6")
    if raw.strip().lower() in {"", "off", "none", "0"}:
        return []
    values: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        seconds = float(part)
        if seconds > 0:
            values.append(seconds)
    return values


def _warmup_runtime(spec: ModelSpec, runtime: InferenceRuntime) -> None:
    """Run throwaway synthesis so the first real request hits warm graphs.

    Sampling is CUDA-graph accelerated; each new latent-length bucket costs a
    one-off capture. Doing that here keeps it out of user-visible latency.
    """
    seconds_list = _warmup_seconds()
    if not seconds_list:
        return
    ref_wav, ref_latent = _resolve_reference_inputs(spec, TTSRequest(text="warmup"))
    for seconds in seconds_list:
        try:
            runtime.synthesize(
                SamplingRequest(
                    text="ウォームアップ",
                    ref_wav=ref_wav,
                    ref_latent=ref_latent,
                    seconds=seconds,
                    seed=0,
                    trim_tail=False,
                )
            )
        except Exception as exc:  # pragma: no cover - warmup must never block start
            LOGGER.warning("warmup failed for model=%s seconds=%s: %s", spec.id, seconds, exc)
            return
    LOGGER.info(
        "warmup complete model=%s seconds=%s",
        spec.id,
        ",".join(f"{value:g}" for value in seconds_list),
    )


@app.on_event("startup")
def preload_models() -> None:
    if not _env_bool("IRODORI_PRELOAD_MODELS", True):
        return
    for spec in REGISTRY.models.values():
        if not spec.enabled or not spec.preload:
            continue
        _validate_model_assets(spec, use_lora=False)
        runtime, _loaded = RUNTIME_CACHE.get(build_runtime_key(spec, use_lora=False))
        _warmup_runtime(spec, runtime)


@app.get("/ping")
async def ping() -> Response:
    if _ping_fails_when_busy() and _synthesis_busy():
        return Response(status_code=503, headers=_busy_headers())
    return Response(
        status_code=200,
        headers=_worker_headers({"X-Irodori-Busy": "1" if _synthesis_busy() else "0"}),
    )


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "default_model_id": REGISTRY.default_model_id,
        "enabled_models": len([spec for spec in REGISTRY.models.values() if spec.enabled]),
        "worker_id": _worker_id(),
        "busy": _synthesis_busy(),
        "active_request": _active_request_payload(),
        "busy_wait_timeout_seconds": _busy_wait_timeout_seconds(),
        "max_request_seconds": _max_request_seconds(),
    }


@app.get("/readyz")
async def readyz() -> dict[str, Any]:
    payload = _readiness_payload()
    if not payload["ready"]:
        headers = _busy_headers() if payload["busy"] else _worker_headers()
        raise HTTPException(status_code=503, detail=payload, headers=headers)
    return payload


@app.get("/ready")
async def ready() -> dict[str, Any]:
    return await readyz()


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    return {
        "default_model_id": REGISTRY.default_model_id,
        "models": REGISTRY.list_models(),
        "cache": RUNTIME_CACHE.keys(),
    }


@app.post("/v1/cache/clear")
def clear_cache() -> dict[str, Any]:
    RUNTIME_CACHE.clear()
    return {"ok": True}


@app.post("/v1/tts")
def tts(req: TTSRequest) -> Response:
    request_started_at = time.perf_counter()
    try:
        spec = REGISTRY.get(req.model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown model_id: {req.model_id}") from exc

    use_lora = _request_uses_lora(spec, req)
    _validate_model_assets(spec, use_lora=use_lora)
    key = build_runtime_key(spec, use_lora=use_lora)

    try:
        with _try_synthesis_lock("tts", spec.id):
            with _track_synthesis_request(
                route="tts",
                model_id=spec.id,
                text_chars=len(req.text),
                chunk_count=None,
            ):
                chunks = _prepare_tts_chunks(req.text, req.max_chunk_chars, req.auto_split)
                if not chunks:
                    raise ValueError("text is empty after splitting")

                resolved_chunk_seconds = _resolve_chunk_seconds(req)
                _validate_request_limits(
                    req,
                    chunks=chunks,
                    resolved_chunk_seconds=resolved_chunk_seconds,
                )
                chunk_seconds: list[float | None] = [resolved_chunk_seconds for _chunk in chunks]
                _update_active_request(chunk_count=len(chunks), phase="loading_runtime")

                runtime, reloaded = RUNTIME_CACHE.get(key)
                _ensure_lora_adapter_for_request(runtime, spec, key)

                audios: list[torch.Tensor] = []
                used_seeds: list[int | None] = []
                sample_rate: int | None = None
                ref_wav, ref_latent = _resolve_reference_inputs(spec, req)
                LOGGER.info(
                    "[tts] split model_id=%s mode=%s auto_split=%s chunks=%d max_chunk_chars=%d",
                    spec.id,
                    _format_effective_mode(spec, key),
                    req.auto_split,
                    len(chunks),
                    req.max_chunk_chars,
                )
                for index, (chunk, seconds) in enumerate(
                    zip(chunks, chunk_seconds, strict=True),
                    start=1,
                ):
                    LOGGER.info(
                        "[tts] chunk %d/%d chars=%d seconds=%s seed=%s text=%r",
                        index,
                        len(chunks),
                        len(chunk),
                        _format_seconds_header(seconds),
                        "random" if req.seed is None else req.seed + index - 1,
                        _preview_chunk(chunk),
                    )

                auto_duration = any(seconds is None for seconds in chunk_seconds)
                chunk_batch_size = 1 if auto_duration else min(_chunk_batch_size(), len(chunks))
                LOGGER.info("[tts] chunk_batch_size=%d", chunk_batch_size)
                for batch_start in range(0, len(chunks), chunk_batch_size):
                    batch_end = min(len(chunks), batch_start + chunk_batch_size)
                    _update_active_request(
                        phase="synthesizing",
                        chunk_index=batch_start + 1,
                        chunk_batch_end=batch_end,
                    )
                    batch_reqs: list[SamplingRequest] = []
                    for index in range(batch_start, batch_end):
                        chunk_seed = None if req.seed is None else req.seed + index
                        batch_reqs.append(
                            SamplingRequest(
                                text=chunks[index],
                                caption=req.caption,
                                ref_wav=ref_wav,
                                ref_latent=ref_latent,
                                no_ref=False,
                                ref_normalize_db=-16.0,
                                ref_ensure_max=True,
                                num_candidates=1,
                                decode_mode="batch" if batch_end - batch_start > 1 else "sequential",
                                seconds=chunk_seconds[index],
                                duration_scale=req.duration_scale,
                                min_seconds=req.min_seconds,
                                max_seconds=req.max_seconds,
                                max_ref_seconds=None,
                                max_text_len=None,
                                num_steps=_resolve_num_steps(req),
                                t_schedule_mode=req.t_schedule_mode,
                                sway_coeff=req.sway_coeff,
                                seed=chunk_seed,
                                cfg_guidance_mode="independent",
                                cfg_scale_text=req.cfg_scale_text,
                                cfg_scale_caption=req.cfg_scale_caption,
                                cfg_scale_speaker=req.cfg_scale_speaker,
                                speaker_kv_scale=req.speaker_kv_scale,
                                speaker_kv_min_t=req.speaker_kv_min_t,
                                speaker_kv_max_layers=req.speaker_kv_max_layers,
                                trim_tail=req.trim_tail,
                                lora_adapter=_sampling_request_lora_adapter(spec, key),
                            )
                        )
                    if auto_duration or not hasattr(runtime, "synthesize_batch"):
                        results = [runtime.synthesize(batch_req) for batch_req in batch_reqs]
                    else:
                        results = runtime.synthesize_batch(batch_reqs)
                    for result in results:
                        if sample_rate is None:
                            sample_rate = result.sample_rate
                        elif sample_rate != result.sample_rate:
                            raise ValueError("Generated chunks have different sample rates")
                        audios.append(result.audio)
                        used_seeds.append(result.used_seed)

                _update_active_request(phase="encoding")
                assert sample_rate is not None
                audio = _concat_audio_segments(
                    audios,
                    sample_rate,
                    req.chunk_silence_ms,
                    req.chunk_tail_padding_ms,
                )
                wav_bytes = _audio_to_wav_bytes(audio, sample_rate)
                if req.format == "mp3":
                    body = _wav_to_mp3_bytes(wav_bytes)
                    media_type = "audio/mpeg"
                else:
                    body = wav_bytes
                    media_type = "audio/wav"
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    generation_sec = time.perf_counter() - request_started_at
    LOGGER.info(
        "[tts] complete model_id=%s chunks=%d seeds=%s reloaded=%s generation_sec=%.1f",
        spec.id,
        len(chunks),
        ",".join(str(seed) for seed in used_seeds),
        reloaded,
        generation_sec,
    )
    headers = {
        "X-Irodori-Model-Id": spec.id,
        "X-Irodori-Model-Mode": _format_effective_mode(spec, key),
        "X-Irodori-Lora-Load-Mode": getattr(key, "lora_load_mode", "none"),
        "X-Irodori-Use-Lora": "1" if use_lora else "0",
        "X-Irodori-Seed": str(used_seeds[0]),
        "X-Irodori-Seeds": ",".join(str(seed) for seed in used_seeds),
        "X-Irodori-Num-Steps": str(_resolve_num_steps(req)),
        "X-Irodori-T-Schedule-Mode": req.t_schedule_mode,
        "X-Irodori-Sway-Coeff": str(req.sway_coeff),
        "X-Irodori-Cfg-Scale-Speaker": str(req.cfg_scale_speaker),
        "X-Irodori-Speaker-Kv-Scale": "" if req.speaker_kv_scale is None else str(req.speaker_kv_scale),
        "X-Irodori-Speaker-Kv-Min-T": "" if req.speaker_kv_min_t is None else str(req.speaker_kv_min_t),
        "X-Irodori-Speaker-Kv-Max-Layers": ""
        if req.speaker_kv_max_layers is None
        else str(req.speaker_kv_max_layers),
        "X-Irodori-Chunk-Count": str(len(chunks)),
        "X-Irodori-Chunk-Batch-Size": str(chunk_batch_size),
        "X-Irodori-Chunk-Seconds": ",".join(
            _format_seconds_header(seconds) for seconds in chunk_seconds
        ),
        "X-Irodori-Duration-Scale": str(req.duration_scale),
        "X-Irodori-Min-Seconds": str(req.min_seconds),
        "X-Irodori-Max-Seconds": str(req.max_seconds),
        "X-Irodori-Trim-Tail": "1" if req.trim_tail else "0",
        "X-Irodori-Runtime-Reloaded": "1" if reloaded else "0",
    }
    headers.update(_worker_headers())
    return Response(content=body, media_type=media_type, headers=headers)


@app.post("/v1/tts/stream")
def tts_stream(req: TTSRequest) -> StreamingResponse:
    try:
        spec = REGISTRY.get(req.model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown model_id: {req.model_id}") from exc

    use_lora = _request_uses_lora(spec, req)
    _validate_model_assets(spec, use_lora=use_lora)
    key = build_runtime_key(spec, use_lora=use_lora)
    try:
        chunks = _prepare_tts_chunks(req.text, req.max_chunk_chars, req.auto_split)
        if not chunks:
            raise ValueError("text is empty after splitting")
        resolved_chunk_seconds = _resolve_chunk_seconds(req)
        _validate_request_limits(
            req,
            chunks=chunks,
            resolved_chunk_seconds=resolved_chunk_seconds,
        )
        chunk_seconds = [resolved_chunk_seconds for _chunk in chunks]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    seed_header = "random" if req.seed is None else str(req.seed)
    seeds_header = (
        "random"
        if req.seed is None
        else ",".join(str(req.seed + index) for index in range(len(chunks)))
    )
    headers = {
        "Cache-Control": "no-store",
        "X-Irodori-Stream-Mode": "ffmpeg-pcm-mp3",
        "X-Irodori-Model-Id": spec.id,
        "X-Irodori-Model-Mode": _format_effective_mode(spec, key),
        "X-Irodori-Lora-Load-Mode": getattr(key, "lora_load_mode", "none"),
        "X-Irodori-Use-Lora": "1" if use_lora else "0",
        "X-Irodori-Seed": seed_header,
        "X-Irodori-Seeds": seeds_header,
        "X-Irodori-Num-Steps": str(_resolve_num_steps(req)),
        "X-Irodori-T-Schedule-Mode": req.t_schedule_mode,
        "X-Irodori-Sway-Coeff": str(req.sway_coeff),
        "X-Irodori-Cfg-Scale-Speaker": str(req.cfg_scale_speaker),
        "X-Irodori-Speaker-Kv-Scale": "" if req.speaker_kv_scale is None else str(req.speaker_kv_scale),
        "X-Irodori-Speaker-Kv-Min-T": "" if req.speaker_kv_min_t is None else str(req.speaker_kv_min_t),
        "X-Irodori-Speaker-Kv-Max-Layers": ""
        if req.speaker_kv_max_layers is None
        else str(req.speaker_kv_max_layers),
        "X-Irodori-Chunk-Count": str(len(chunks)),
        "X-Irodori-Chunk-Seconds": ",".join(
            _format_seconds_header(seconds) for seconds in chunk_seconds
        ),
        "X-Irodori-Duration-Scale": str(req.duration_scale),
        "X-Irodori-Min-Seconds": str(req.min_seconds),
        "X-Irodori-Max-Seconds": str(req.max_seconds),
        "X-Irodori-Trim-Tail": "1" if req.trim_tail else "0",
    }
    headers.update(_worker_headers())
    _acquire_synthesis_lock("tts:stream", spec.id)
    try:
        return StreamingResponse(
            _iter_mp3_stream(
                spec,
                key,
                req,
                chunks,
                chunk_seconds,
                release_synthesis_lock=True,
            ),
            media_type="audio/mpeg",
            headers=headers,
        )
    except Exception:
        _synthesis_lock.release()
        raise
