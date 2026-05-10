from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ADAPTER_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "adapter_model.bin",
    "config.json",
    "irodori_lora_metadata.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Irodori model artifacts into a RunPod volume-ready directory."
    )
    parser.add_argument(
        "--registry",
        default="configs/model_registry.local.json",
        help="Local registry containing the completed models.",
    )
    parser.add_argument(
        "--output-dir",
        default="runpod_artifacts",
        help="Output artifact root to upload to a RunPod network volume.",
    )
    parser.add_argument(
        "--runpod-root",
        default="/workspace/irodori_artifacts",
        help="Absolute artifact root path inside the RunPod container.",
    )
    parser.add_argument(
        "--base-checkpoint",
        default="models/Irodori-TTS-500M-v2/model.safetensors",
        help="Base Irodori checkpoint used by all LoRA adapters.",
    )
    parser.add_argument(
        "--artifact-mode",
        choices=("lora", "merged"),
        default="lora",
        help="Upload compact LoRA adapters or full merged checkpoints.",
    )
    parser.add_argument("--default-model-id", default=None)
    parser.add_argument("--copy", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def resolve_registry_path(raw: str, registry_dir: Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = registry_dir / path
    return path.resolve()


def copy_file(src: Path, dst: Path, *, copy_enabled: bool) -> None:
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_enabled:
        if not dst.is_file() or src.stat().st_size != dst.stat().st_size:
            shutil.copy2(src, dst)


def copy_adapter_dir(src: Path, dst: Path, *, copy_enabled: bool) -> None:
    if not src.is_dir():
        raise FileNotFoundError(src)
    for name in ADAPTER_FILES:
        candidate = src / name
        if candidate.is_file():
            copy_file(candidate, dst / name, copy_enabled=copy_enabled)
    if not (src / "adapter_config.json").is_file():
        raise FileNotFoundError(src / "adapter_config.json")
    if not (src / "adapter_model.safetensors").is_file() and not (src / "adapter_model.bin").is_file():
        raise FileNotFoundError(src / "adapter_model.safetensors")


def main() -> None:
    args = parse_args()
    registry_path = Path(args.registry).resolve()
    registry_dir = registry_path.parent
    output_root = Path(args.output_dir).resolve()
    runpod_root = args.runpod_root.rstrip("/")
    payload = json.loads(registry_path.read_text(encoding="utf-8"))

    output_models = output_root / "outputs" / "irodori_loras"
    output_base = output_root / "models" / "Irodori-TTS-500M-v2" / "model.safetensors"
    output_refs = output_root / "refs"
    output_configs = output_root / "configs"
    output_configs.mkdir(parents=True, exist_ok=True)

    base_checkpoint_src = resolve_registry_path(args.base_checkpoint, Path.cwd())
    if args.artifact_mode == "lora":
        copy_file(base_checkpoint_src, output_base, copy_enabled=args.copy)

    runpod_models: list[dict] = []
    for item in payload.get("models", []):
        if not item.get("enabled", True):
            continue
        model_id = str(item["id"])
        ref_src = resolve_registry_path(str(item["ref_wav"]), registry_dir)

        ref_suffix = ref_src.suffix or ".wav"
        ref_dst = output_refs / model_id / f"reference{ref_suffix}"
        copy_file(ref_src, ref_dst, copy_enabled=args.copy)

        if args.artifact_mode == "lora":
            if item.get("lora_adapter"):
                adapter_src = resolve_registry_path(str(item["lora_adapter"]), registry_dir)
            elif item.get("checkpoint"):
                checkpoint_src = resolve_registry_path(str(item["checkpoint"]), registry_dir)
                adapter_src = checkpoint_src.parent / "checkpoint" / "checkpoint_final"
            else:
                adapter_src = Path("outputs") / "irodori_loras" / model_id / "checkpoint" / "checkpoint_final"
                adapter_src = adapter_src.resolve()
            adapter_dst = output_models / model_id / "checkpoint" / "checkpoint_final"
            copy_adapter_dir(adapter_src, adapter_dst, copy_enabled=args.copy)
            model_entry = {
                "id": model_id,
                "base_checkpoint": f"{runpod_root}/models/Irodori-TTS-500M-v2/model.safetensors",
                "lora_adapter": f"{runpod_root}/outputs/irodori_loras/{model_id}/checkpoint/checkpoint_final",
                "ref_wav": f"{runpod_root}/refs/{model_id}/reference{ref_suffix}",
                "enabled": True,
                "preload": False,
                "description": item.get("description") or "Irodori LoRA adapter",
            }
        else:
            checkpoint_src = resolve_registry_path(str(item["checkpoint"]), registry_dir)
            checkpoint_dst = output_models / model_id / f"{model_id}_merged.safetensors"
            copy_file(checkpoint_src, checkpoint_dst, copy_enabled=args.copy)
            model_entry = {
                "id": model_id,
                "checkpoint": f"{runpod_root}/outputs/irodori_loras/{model_id}/{model_id}_merged.safetensors",
                "ref_wav": f"{runpod_root}/refs/{model_id}/reference{ref_suffix}",
                "enabled": True,
                "preload": False,
                "description": item.get("description") or "Irodori merged checkpoint",
            }

        runpod_models.append(
            model_entry
        )

    default_model_id = args.default_model_id or payload.get("default_model_id")
    if not default_model_id and runpod_models:
        default_model_id = runpod_models[0]["id"]
    for row in runpod_models:
        row["preload"] = row["id"] == default_model_id

    runpod_registry = {
        "default_model_id": default_model_id,
        "models": runpod_models,
    }
    runpod_registry_path = output_configs / "model_registry.runpod.json"
    runpod_registry_path.write_text(
        json.dumps(runpod_registry, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "source_registry": str(registry_path),
        "output_root": str(output_root),
        "runpod_root": runpod_root,
        "artifact_mode": args.artifact_mode,
        "registry": str(runpod_registry_path),
        "models": len(runpod_models),
        "total_bytes": sum(p.stat().st_size for p in output_root.rglob("*") if p.is_file()),
    }
    (output_root / "artifact_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
