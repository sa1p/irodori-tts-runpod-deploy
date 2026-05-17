from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from irodori_targets import DEFAULT_MODEL_FOLDERS, parse_model_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an Irodori API model registry JSON.")
    parser.add_argument("--inventory", default=None)
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model ids, or @file. Defaults to inventory items or configured targets.",
    )
    parser.add_argument("--checkpoint-root", default="outputs/irodori_loras")
    parser.add_argument("--base-checkpoint", default="models/Irodori-TTS-500M-v3/model.safetensors")
    parser.add_argument("--training-data-root", default="data/irodori_training")
    parser.add_argument("--distilled-root", default="data/distilled")
    parser.add_argument("--output", default="configs/model_registry.local.json")
    parser.add_argument("--registry-mode", choices=["lora", "merged"], default="merged")
    parser.add_argument("--default-model-id", default=None)
    parser.add_argument("--include-missing", action="store_true")
    parser.add_argument("--path-mode", choices=["absolute", "relative"], default="relative")
    parser.add_argument("--preload-default", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_model_ids(args: argparse.Namespace) -> list[str]:
    if args.models:
        return parse_model_ids(args.models)
    if args.inventory:
        payload = json.loads(Path(args.inventory).read_text(encoding="utf-8"))
        return [str(item["model_id"]) for item in payload.get("items", [])]
    return list(DEFAULT_MODEL_FOLDERS)


def first_audio_from_jsonl(path: Path) -> str | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            audio = row.get("audio")
            if audio:
                return str(audio)
    return None


def display_path(path: Path | str, output_path: Path, mode: str) -> str:
    path_obj = Path(path).expanduser()
    if mode == "absolute":
        return str(path_obj.resolve())
    if not path_obj.is_absolute():
        path_obj = path_obj.resolve()
    return os.path.relpath(path_obj, start=output_path.parent.resolve()).replace("\\", "/")


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model_ids = load_model_ids(args)
    inventory_by_model: dict[str, dict[str, Any]] = {}
    if args.inventory:
        inventory_payload = json.loads(Path(args.inventory).read_text(encoding="utf-8"))
        inventory_by_model = {
            str(item["model_id"]): item for item in inventory_payload.get("items", [])
        }

    models: list[dict[str, Any]] = []
    for model_id in model_ids:
        checkpoint = Path(args.checkpoint_root) / model_id / f"{model_id}_merged.safetensors"
        adapter = Path(args.checkpoint_root) / model_id / "checkpoint" / "checkpoint_final"
        ref_wav = first_audio_from_jsonl(Path(args.training_data_root) / model_id / "train.jsonl")
        if ref_wav is None:
            ref_wav = first_audio_from_jsonl(Path(args.distilled_root) / model_id / "train.jsonl")

        if args.registry_mode == "lora":
            checkpoint_exists = Path(args.base_checkpoint).is_file() and (adapter / "adapter_config.json").is_file() and (
                (adapter / "adapter_model.safetensors").is_file()
                or (adapter / "adapter_model.bin").is_file()
            )
        else:
            checkpoint_exists = checkpoint.is_file()
        ref_exists = bool(ref_wav and Path(ref_wav).is_file())
        enabled = bool(checkpoint_exists and ref_exists)
        if not args.include_missing and not enabled:
            continue
        source_type = (inventory_by_model.get(model_id) or {}).get("source_type")
        row = {
            "id": model_id,
            "ref_wav": display_path(ref_wav, output, args.path_mode) if ref_wav else "",
            "enabled": enabled,
            "preload": False,
            "description": "Irodori LoRA distilled from SBV2"
            if source_type == "distill"
            else "Irodori LoRA trained from recorded dataset",
        }
        if args.registry_mode == "lora":
            row["base_checkpoint"] = display_path(args.base_checkpoint, output, args.path_mode)
            row["lora_adapter"] = display_path(adapter, output, args.path_mode)
        else:
            row["checkpoint"] = display_path(checkpoint, output, args.path_mode)
        models.append(row)

    default_model_id = args.default_model_id
    if default_model_id is None:
        first_enabled = next((row for row in models if row["enabled"]), None)
        default_model_id = first_enabled["id"] if first_enabled else (models[0]["id"] if models else "")
    for row in models:
        if row["id"] == default_model_id:
            row["preload"] = bool(args.preload_default and row["enabled"])

    payload = {"default_model_id": default_model_id, "models": models}
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"models={len(models)} default_model_id={default_model_id} output={output}")


if __name__ == "__main__":
    main()
