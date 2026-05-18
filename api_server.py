from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import threading
import time
from collections import OrderedDict
from collections.abc import Iterator
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
    ref_wav: str
    lora_adapter: str | None = None
    enabled: bool = True
    preload: bool = False
    description: str | None = None

    @property
    def mode(self) -> str:
        return "lora" if self.lora_adapter else "checkpoint"


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
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
    num_steps: int = Field(default=6, ge=4, le=80)
    t_schedule_mode: Literal["linear", "sway"] = "sway"
    sway_coeff: float = Field(default=-1.0, ge=-2.0, le=2.0)
    cfg_scale_text: float = Field(default=3.0, ge=0.0, le=8.0)
    cfg_scale_speaker: float = Field(default=5.0, ge=0.0, le=10.0)
    speaker_kv_scale: float | None = Field(default=None, gt=0.0)
    speaker_kv_min_t: float | None = Field(default=None, ge=0.0, le=1.0)
    speaker_kv_max_layers: int | None = Field(default=None, ge=0)
    reference_wav: str | None = None
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
            ref_wav = _resolve_maybe_relative(str(item.get("ref_wav") or item["reference_wav"]), base_dir)
            spec = ModelSpec(
                id=model_id,
                checkpoint=checkpoint,
                ref_wav=ref_wav,
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
                    "checkpoint_exists": Path(spec.checkpoint).is_file(),
                    "lora_adapter_exists": (
                        is_lora_adapter_dir(spec.lora_adapter)
                        if spec.lora_adapter is not None
                        else None
                    ),
                    "ref_wav_exists": Path(spec.ref_wav).is_file(),
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


def build_runtime_key(spec: ModelSpec) -> RuntimeKey:
    lora_mode = _lora_load_mode() if spec.lora_adapter is not None else "none"
    return RuntimeKey(
        checkpoint=spec.checkpoint,
        model_device=os.getenv("IRODORI_MODEL_DEVICE", "cuda"),
        lora_adapter=spec.lora_adapter if lora_mode == "merged" else None,
        lora_load_mode=lora_mode,
        codec_repo=os.getenv("IRODORI_CODEC_REPO", "Aratako/Semantic-DACVAE-Japanese-32dim"),
        model_precision=os.getenv("IRODORI_MODEL_PRECISION", "bf16"),
        codec_device=os.getenv("IRODORI_CODEC_DEVICE", "cuda"),
        codec_precision=os.getenv("IRODORI_CODEC_PRECISION", "fp32"),
        codec_deterministic_encode=_env_bool("IRODORI_CODEC_DETERMINISTIC_ENCODE", True),
        codec_deterministic_decode=_env_bool("IRODORI_CODEC_DETERMINISTIC_DECODE", True),
        compile_model=_env_bool("IRODORI_COMPILE_MODEL", False),
        compile_dynamic=_env_bool("IRODORI_COMPILE_DYNAMIC", False),
    )


SENTENCE_END_CHARS = {".", "!", "?", "\u3002", "\uff01", "\uff1f"}
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


def _split_tts_text_line(text: str, max_chunk_chars: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
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
        if (
            current
            and _is_emoji_codepoint(ch)
            and prev not in EMOJI_JOINERS
            and not _is_emoji_codepoint(prev)
        ):
            flush()

        current.append(ch)

        if ch in SENTENCE_END_CHARS:
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


def _resolve_reference_inputs(spec: ModelSpec, req: TTSRequest) -> tuple[str | None, str | None]:
    if req.reference_latent:
        return None, req.reference_latent
    return req.reference_wav or spec.ref_wav, None


def _format_seconds_header(seconds: float | None) -> str:
    return "auto" if seconds is None else f"{seconds:.2f}"


def _chunk_batch_size() -> int:
    raw = os.getenv("IRODORI_CHUNK_BATCH_SIZE", "1")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"IRODORI_CHUNK_BATCH_SIZE must be an integer, got {raw!r}") from exc
    return max(1, value)


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
) -> Iterator[bytes]:
    request_started_at = time.perf_counter()
    output_queue: queue.Queue[bytes | object] = queue.Queue()
    read_size = int(os.getenv("IRODORI_STREAM_READ_SIZE", "65536"))

    def put_end() -> None:
        output_queue.put(_STREAM_END)

    def read_stdout(proc: subprocess.Popen[bytes]) -> None:
        try:
            if proc.stdout is None:
                return
            while True:
                data = proc.stdout.read(read_size)
                if not data:
                    break
                output_queue.put(data)
        finally:
            put_end()

    def generate_and_encode() -> None:
        proc: subprocess.Popen[bytes] | None = None
        reader: threading.Thread | None = None
        try:
            with _synthesis_lock:
                runtime, reloaded = RUNTIME_CACHE.get(key)
                if spec.lora_adapter is not None and key.lora_load_mode == "delta":
                    runtime.ensure_lora_delta_adapter(
                        adapter_name=spec.id,
                        adapter_path=spec.lora_adapter,
                    )
                elif spec.lora_adapter is not None and key.lora_load_mode in {"dynamic", "reload"}:
                    runtime.ensure_lora_adapter(
                        adapter_name=spec.id,
                        adapter_path=spec.lora_adapter,
                        replace_existing=True,
                    )

                LOGGER.info(
                    "[tts:stream] split model_id=%s mode=%s auto_split=%s chunks=%d max_chunk_chars=%d",
                    spec.id,
                    f"{spec.mode}:{key.lora_load_mode}" if spec.lora_adapter else spec.mode,
                    req.auto_split,
                    len(chunks),
                    req.max_chunk_chars,
                )

                sample_rate: int | None = None
                channels: int | None = None
                used_seeds: list[int | None] = []
                ref_wav, ref_latent = _resolve_reference_inputs(spec, req)
                for index, (chunk, seconds) in enumerate(zip(chunks, chunk_seconds, strict=True), start=1):
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
                            max_ref_seconds=30.0,
                            max_text_len=None,
                            num_steps=req.num_steps,
                            t_schedule_mode=req.t_schedule_mode,
                            sway_coeff=req.sway_coeff,
                            seed=chunk_seed,
                            cfg_guidance_mode="independent",
                            cfg_scale_text=req.cfg_scale_text,
                            cfg_scale_speaker=req.cfg_scale_speaker,
                            speaker_kv_scale=req.speaker_kv_scale,
                            speaker_kv_min_t=req.speaker_kv_min_t,
                            speaker_kv_max_layers=req.speaker_kv_max_layers,
                            trim_tail=req.trim_tail,
                        )
                    )
                    used_seeds.append(result.used_seed)
                    audio = _audio_to_channels_first(result.audio)
                    if sample_rate is None:
                        sample_rate = result.sample_rate
                        channels = int(audio.shape[0])
                        proc = _start_mp3_stream_process(sample_rate, channels)
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
                        _silence_pcm_s16le_bytes(channels, sample_rate, req.chunk_tail_padding_ms)
                    )
                    if index < len(chunks):
                        proc.stdin.write(
                            _silence_pcm_s16le_bytes(channels, sample_rate, req.chunk_silence_ms)
                        )
                    proc.stdin.flush()

                if proc is None:
                    raise ValueError("No audio segments were generated")
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

    worker = threading.Thread(target=generate_and_encode, daemon=True)
    worker.start()
    while True:
        item = output_queue.get()
        if item is _STREAM_END:
            break
        yield item
    worker.join(timeout=1.0)


def _validate_model_assets(spec: ModelSpec) -> None:
    if not Path(spec.checkpoint).is_file():
        raise HTTPException(status_code=503, detail=f"Checkpoint not found: {spec.checkpoint}")
    if spec.lora_adapter is not None and not is_lora_adapter_dir(spec.lora_adapter):
        raise HTTPException(
            status_code=503,
            detail=f"LoRA adapter not found or invalid: {spec.lora_adapter}",
        )
    if not Path(spec.ref_wav).is_file():
        raise HTTPException(status_code=503, detail=f"Reference wav not found: {spec.ref_wav}")


REGISTRY = ModelRegistry.from_env()
RUNTIME_CACHE = RuntimeCache(max_items=int(os.getenv("IRODORI_MAX_CACHED_RUNTIMES", "1")))

app = FastAPI(title="Irodori-TTS Multi-Model API", version="0.2.0")
_synthesis_lock = threading.Lock()


@app.on_event("startup")
def preload_models() -> None:
    if not _env_bool("IRODORI_PRELOAD_MODELS", True):
        return
    for spec in REGISTRY.models.values():
        if not spec.enabled or not spec.preload:
            continue
        _validate_model_assets(spec)
        runtime, _ = RUNTIME_CACHE.get(build_runtime_key(spec))
        lora_mode = _lora_load_mode()
        if spec.lora_adapter is not None and lora_mode == "delta":
            runtime.ensure_lora_delta_adapter(adapter_name=spec.id, adapter_path=spec.lora_adapter)
        elif spec.lora_adapter is not None and lora_mode in {"dynamic", "reload"}:
            runtime.ensure_lora_adapter(
                adapter_name=spec.id,
                adapter_path=spec.lora_adapter,
                replace_existing=True,
            )


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "default_model_id": REGISTRY.default_model_id,
        "enabled_models": len([spec for spec in REGISTRY.models.values() if spec.enabled]),
    }


@app.get("/readyz")
def readyz() -> dict[str, Any]:
    rows = REGISTRY.list_models()
    enabled = [row for row in rows if row["enabled"]]
    ready = all(
        row["checkpoint_exists"]
        and row["ref_wav_exists"]
        and (row["lora_adapter_exists"] is not False)
        for row in enabled
    )
    if not ready:
        raise HTTPException(status_code=503, detail={"ready": False, "models": rows})
    return {"ready": True, "models": rows, "cache": RUNTIME_CACHE.keys()}


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

    _validate_model_assets(spec)
    key = build_runtime_key(spec)

    try:
        with _synthesis_lock:
            runtime, reloaded = RUNTIME_CACHE.get(key)
            if spec.lora_adapter is not None and key.lora_load_mode == "delta":
                runtime.ensure_lora_delta_adapter(
                    adapter_name=spec.id,
                    adapter_path=spec.lora_adapter,
                )
            elif spec.lora_adapter is not None and key.lora_load_mode in {"dynamic", "reload"}:
                runtime.ensure_lora_adapter(
                    adapter_name=spec.id,
                    adapter_path=spec.lora_adapter,
                    replace_existing=True,
                )
            chunks = _prepare_tts_chunks(req.text, req.max_chunk_chars, req.auto_split)
            if not chunks:
                raise ValueError("text is empty after splitting")

            audios: list[torch.Tensor] = []
            used_seeds: list[int | None] = []
            sample_rate: int | None = None
            resolved_chunk_seconds = _resolve_chunk_seconds(req)
            chunk_seconds: list[float | None] = [resolved_chunk_seconds for _chunk in chunks]
            ref_wav, ref_latent = _resolve_reference_inputs(spec, req)
            LOGGER.info(
                "[tts] split model_id=%s mode=%s auto_split=%s chunks=%d max_chunk_chars=%d",
                spec.id,
                f"{spec.mode}:{key.lora_load_mode}" if spec.lora_adapter else spec.mode,
                req.auto_split,
                len(chunks),
                req.max_chunk_chars,
            )
            for index, (chunk, seconds) in enumerate(zip(chunks, chunk_seconds, strict=True), start=1):
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
                batch_reqs: list[SamplingRequest] = []
                for index in range(batch_start, batch_end):
                    chunk_seed = None if req.seed is None else req.seed + index
                    batch_reqs.append(
                        SamplingRequest(
                            text=chunks[index],
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
                            max_ref_seconds=30.0,
                            max_text_len=None,
                            num_steps=req.num_steps,
                            t_schedule_mode=req.t_schedule_mode,
                            sway_coeff=req.sway_coeff,
                            seed=chunk_seed,
                            cfg_guidance_mode="independent",
                            cfg_scale_text=req.cfg_scale_text,
                            cfg_scale_speaker=req.cfg_scale_speaker,
                            speaker_kv_scale=req.speaker_kv_scale,
                            speaker_kv_min_t=req.speaker_kv_min_t,
                            speaker_kv_max_layers=req.speaker_kv_max_layers,
                            trim_tail=req.trim_tail,
                        )
                    )
                if auto_duration:
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
        "X-Irodori-Model-Mode": spec.mode,
        "X-Irodori-Lora-Load-Mode": key.lora_load_mode,
        "X-Irodori-Seed": str(used_seeds[0]),
        "X-Irodori-Seeds": ",".join(str(seed) for seed in used_seeds),
        "X-Irodori-Num-Steps": str(req.num_steps),
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
    return Response(content=body, media_type=media_type, headers=headers)


@app.post("/v1/tts/stream")
def tts_stream(req: TTSRequest) -> StreamingResponse:
    try:
        spec = REGISTRY.get(req.model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown model_id: {req.model_id}") from exc

    _validate_model_assets(spec)
    key = build_runtime_key(spec)
    try:
        chunks = _prepare_tts_chunks(req.text, req.max_chunk_chars, req.auto_split)
        if not chunks:
            raise ValueError("text is empty after splitting")
        resolved_chunk_seconds = _resolve_chunk_seconds(req)
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
        "X-Irodori-Model-Mode": spec.mode,
        "X-Irodori-Lora-Load-Mode": key.lora_load_mode,
        "X-Irodori-Seed": seed_header,
        "X-Irodori-Seeds": seeds_header,
        "X-Irodori-Num-Steps": str(req.num_steps),
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
    return StreamingResponse(
        _iter_mp3_stream(spec, key, req, chunks, chunk_seconds),
        media_type="audio/mpeg",
        headers=headers,
    )
