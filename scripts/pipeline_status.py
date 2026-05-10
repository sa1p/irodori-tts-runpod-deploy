from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize full production pipeline progress.")
    parser.add_argument("--inventory", default="outputs/irodori_training_inventory.json")
    parser.add_argument("--distilled-root", default="data/distilled")
    parser.add_argument("--training-data-root", default="data/irodori_training")
    parser.add_argument("--output-root", default="outputs/irodori_loras")
    parser.add_argument("--distill-target-rows", type=int, default=160)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def count_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def count_files(path: Path, pattern: str) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for _ in path.glob(pattern))


def summarize_model(item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    model_id = str(item["model_id"])
    source_type = str(item.get("source_type") or "unknown")
    distilled_jsonl = Path(args.distilled_root) / model_id / "train.jsonl"
    training_jsonl = Path(args.training_data_root) / model_id / "train.jsonl"
    manifest = Path(args.training_data_root) / model_id / "train_manifest.jsonl"
    latent_dir = Path(args.training_data_root) / model_id / "latents"
    output_dir = Path(args.output_root) / model_id
    checkpoint_dir = output_dir / "checkpoint"
    merged = output_dir / f"{model_id}_merged.safetensors"

    distill_rows = count_jsonl(distilled_jsonl)
    train_rows = count_jsonl(training_jsonl)
    manifest_rows = count_jsonl(manifest)
    latent_files = count_files(latent_dir, "*.pt")
    final_checkpoint = checkpoint_dir / "checkpoint_final"
    latest_checkpoint = None
    latest_step = 0
    for path in checkpoint_dir.glob("checkpoint_*") if checkpoint_dir.is_dir() else []:
        if path.is_dir() and path.name.removeprefix("checkpoint_").isdigit():
            step = int(path.name.removeprefix("checkpoint_"))
            if step > latest_step:
                latest_step = step
                latest_checkpoint = str(path)

    return {
        "model_id": model_id,
        "source_type": source_type,
        "distill_rows": distill_rows,
        "distill_ready": source_type != "distill" or distill_rows >= args.distill_target_rows,
        "train_jsonl_rows": train_rows,
        "manifest_rows": manifest_rows,
        "latent_files": latent_files,
        "data_ready": train_rows > 0 and manifest_rows > 0 and latent_files >= manifest_rows,
        "latest_step": latest_step,
        "latest_checkpoint": latest_checkpoint,
        "checkpoint_final": final_checkpoint.is_dir(),
        "merged": merged.is_file(),
    }


def main() -> None:
    args = parse_args()
    inventory_path = Path(args.inventory)
    if not inventory_path.is_file():
        raise FileNotFoundError(f"inventory not found: {inventory_path}")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    rows = [summarize_model(item, args) for item in inventory.get("items", [])]

    summary = {
        "total": len(rows),
        "distill_models": sum(1 for row in rows if row["source_type"] == "distill"),
        "recorded_models": sum(1 for row in rows if row["source_type"] == "recorded"),
        "distill_ready": sum(1 for row in rows if row["distill_ready"]),
        "data_ready": sum(1 for row in rows if row["data_ready"]),
        "checkpoint_final": sum(1 for row in rows if row["checkpoint_final"]),
        "merged": sum(1 for row in rows if row["merged"]),
    }
    payload = {"summary": summary, "models": rows}
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print(json.dumps(summary, ensure_ascii=False))
    for row in rows:
        print(
            f"{row['model_id']}: source={row['source_type']} "
            f"distill={row['distill_rows']} train={row['train_jsonl_rows']} "
            f"manifest={row['manifest_rows']} latents={row['latent_files']} "
            f"step={row['latest_step']} final={row['checkpoint_final']} merged={row['merged']}"
        )


if __name__ == "__main__":
    main()
