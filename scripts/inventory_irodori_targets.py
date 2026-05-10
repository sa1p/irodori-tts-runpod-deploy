from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from irodori_targets import (
    candidate_dataset_names,
    parse_model_ids,
    resolve_sbv2_model_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventory SBV2 source assets and available recorded datasets for Irodori FT."
    )
    parser.add_argument(
        "--model-assets-root",
        default=os.getenv("SBV2_MODEL_ASSETS_ROOT", r"D:\sbv2\Style-Bert-VITS2\model_assets"),
        help="Style-Bert-VITS2 model_assets root.",
    )
    parser.add_argument(
        "--data-root",
        default=os.getenv("SBV2_DATA_ROOT", r"D:\sbv2\Style-Bert-VITS2\Data"),
        help="Style-Bert-VITS2 Data root.",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model ids, or @file. Defaults to the configured 38 target models.",
    )
    parser.add_argument("--output", default="outputs/irodori_training_inventory.json")
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def _count_esd_with_root(esd_list: Path, audio_root: Path) -> dict[str, int]:
    total = 0
    missing = 0
    empty_text = 0
    malformed = 0
    with esd_list.open("r", encoding="utf-8-sig") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            total += 1
            parts = line.split("|", 3)
            if len(parts) != 4:
                malformed += 1
                continue
            rel_audio, _style, _language, text = parts
            if not text.strip():
                empty_text += 1
            if not (audio_root / rel_audio).is_file():
                missing += 1
    return {
        "total": total,
        "missing": missing,
        "empty_text": empty_text,
        "malformed": malformed,
    }


def summarize_esd(esd_list: Path) -> dict[str, Any]:
    parent = esd_list.parent
    candidates = [parent / "raw", parent]
    summaries: list[dict[str, Any]] = []
    for audio_root in candidates:
        counts = _count_esd_with_root(esd_list, audio_root)
        summaries.append(
            {
                "esd_list": str(esd_list.resolve()),
                "audio_root": str(audio_root.resolve()),
                **counts,
            }
        )
    return min(summaries, key=lambda row: (row["missing"], row["malformed"], -row["total"]))


def find_recorded_dataset(data_root: Path, model_id: str) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for dataset_name in candidate_dataset_names(model_id):
        dataset_dir = data_root / dataset_name
        if not dataset_dir.is_dir():
            continue
        for esd_list in sorted(dataset_dir.rglob("esd.list")):
            summary = summarize_esd(esd_list)
            summary["dataset_dir"] = str(dataset_dir.resolve())
            summary["dataset_name"] = dataset_name
            candidates.append(summary)
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (row["total"] - row["missing"] - row["malformed"], -row["empty_text"]),
    )


def main() -> None:
    args = parse_args()
    model_assets_root = Path(args.model_assets_root)
    data_root = Path(args.data_root)
    model_ids = parse_model_ids(args.models)

    items: list[dict[str, Any]] = []
    for model_id in model_ids:
        assets = resolve_sbv2_model_files(model_assets_root, model_id)
        dataset = find_recorded_dataset(data_root, model_id)
        ready_for_teacher = all(
            [
                assets["model_dir_exists"],
                assets["config"],
                assets["style_vectors"],
                assets["teacher_checkpoint"],
            ]
        )
        items.append(
            {
                "model_id": model_id,
                "source_type": "recorded" if dataset else "distill",
                "assets": assets,
                "dataset": dataset,
                "ready_for_distillation": bool(ready_for_teacher),
            }
        )

    summary = {
        "total": len(items),
        "recorded": sum(1 for item in items if item["source_type"] == "recorded"),
        "distill": sum(1 for item in items if item["source_type"] == "distill"),
        "missing_sbv2_assets": sum(1 for item in items if not item["ready_for_distillation"]),
    }
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_assets_root": str(model_assets_root.resolve()),
        "data_root": str(data_root.resolve()),
        "summary": summary,
        "items": items,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    print(f"wrote={output}")


if __name__ == "__main__":
    main()
