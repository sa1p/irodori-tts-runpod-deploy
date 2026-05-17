from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import re
from pathlib import Path
from typing import Any

from irodori_targets import parse_model_ids


ALL_STAGES = ("jsonl", "manifest", "train", "convert", "registry")
CHECKPOINT_RE = re.compile(r"^checkpoint_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare data and train multiple Irodori LoRA adapters from inventory."
    )
    parser.add_argument("--inventory", default="outputs/irodori_training_inventory.json")
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model ids, or @file. Defaults to all models in the inventory.",
    )
    parser.add_argument("--training-data-root", default="data/irodori_training")
    parser.add_argument("--distilled-root", default="data/distilled")
    parser.add_argument("--output-root", default="outputs/irodori_loras")
    parser.add_argument("--config", default="configs/train_500m_v3_lora.yaml")
    parser.add_argument(
        "--base-checkpoint",
        default="models/Irodori-TTS-500M-v3/model.safetensors",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-sample-rate", type=int, default=48000)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--stages", default=",".join(ALL_STAGES))
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--force-convert", action="store_true")
    parser.add_argument("--resume-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run(cmd: list[str], *, dry_run: bool) -> None:
    print(" ".join(str(part) for part in cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def load_inventory(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"inventory not found: {path}. Run scripts/inventory_irodori_targets.py first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def latest_training_checkpoint(output_dir: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint_*"):
        if not path.is_dir():
            continue
        match = CHECKPOINT_RE.match(path.name)
        if match is None:
            continue
        candidates.append((int(match.group(1)), path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def select_items(inventory: dict[str, Any], models_arg: str | None, max_models: int | None) -> list[dict[str, Any]]:
    items = list(inventory.get("items", []))
    if models_arg:
        wanted = set(parse_model_ids(models_arg))
        items = [item for item in items if item.get("model_id") in wanted]
    if max_models is not None:
        items = items[:max_models]
    return items


def copy_or_prepare_jsonl(
    item: dict[str, Any],
    train_jsonl: Path,
    distilled_root: Path,
    dry_run: bool,
) -> None:
    model_id = str(item["model_id"])
    train_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if item.get("source_type") == "recorded":
        dataset = item.get("dataset") or {}
        esd_list = dataset.get("esd_list")
        audio_root = dataset.get("audio_root")
        if not esd_list or not audio_root:
            raise ValueError(f"{model_id}: recorded source is missing esd_list/audio_root")
        run(
            [
                sys.executable,
                "scripts/convert_esd_to_jsonl.py",
                "--esd-list",
                str(esd_list),
                "--audio-root",
                str(audio_root),
                "--output",
                str(train_jsonl),
                "--speaker",
                model_id,
            ],
            dry_run=dry_run,
        )
        return

    distilled_jsonl = distilled_root / model_id / "train.jsonl"
    if not distilled_jsonl.is_file():
        if dry_run:
            print(f"would require distilled dataset: {distilled_jsonl}")
            return
        raise FileNotFoundError(
            f"{model_id}: distilled dataset not found: {distilled_jsonl}. "
            "Run scripts/distill_sbv2_dataset.py for this model first."
        )
    print(f"copy {distilled_jsonl} -> {train_jsonl}")
    if not dry_run:
        shutil.copyfile(distilled_jsonl, train_jsonl)


def main() -> None:
    args = parse_args()
    stages = tuple(stage.strip() for stage in args.stages.split(",") if stage.strip())
    invalid = [stage for stage in stages if stage not in ALL_STAGES]
    if invalid:
        raise ValueError(f"invalid stages: {', '.join(invalid)}")

    inventory = load_inventory(Path(args.inventory))
    items = select_items(inventory, args.models, args.max_models)
    training_data_root = Path(args.training_data_root)
    distilled_root = Path(args.distilled_root)
    output_root = Path(args.output_root)

    for item in items:
        model_id = str(item["model_id"])
        model_data_dir = training_data_root / model_id
        train_jsonl = model_data_dir / "train.jsonl"
        manifest = model_data_dir / "train_manifest.jsonl"
        latent_dir = model_data_dir / "latents"
        train_output_dir = output_root / model_id / "checkpoint"
        merged = output_root / model_id / f"{model_id}_merged.safetensors"

        if "jsonl" in stages and not (args.skip_existing and train_jsonl.is_file()):
            copy_or_prepare_jsonl(item, train_jsonl, distilled_root, args.dry_run)

        if "manifest" in stages and not (args.skip_existing and manifest.is_file()):
            run(
                [
                    sys.executable,
                    "scripts/prepare_local_manifest.py",
                    "--input-jsonl",
                    str(train_jsonl),
                    "--output-manifest",
                    str(manifest),
                    "--latent-dir",
                    str(latent_dir),
                    "--speaker-id-prefix",
                    model_id,
                    "--device",
                    args.device,
                    "--target-sample-rate",
                    str(args.target_sample_rate),
                ],
                dry_run=args.dry_run,
            )

        if "train" in stages and not (args.skip_existing and (train_output_dir / "checkpoint_final").exists()):
            resume_checkpoint = (
                latest_training_checkpoint(train_output_dir) if args.resume_existing else None
            )
            cmd = [
                sys.executable,
                "train.py",
                "--config",
                args.config,
                "--manifest",
                str(manifest),
                "--output-dir",
                str(train_output_dir),
                "--device",
                args.device,
            ]
            if resume_checkpoint is not None:
                cmd.extend(["--resume", str(resume_checkpoint)])
            else:
                cmd.extend(["--init-checkpoint", args.base_checkpoint])
            if args.max_steps is not None:
                cmd.extend(["--max-steps", str(args.max_steps)])
            run(cmd, dry_run=args.dry_run)

        if "convert" in stages and not (args.skip_existing and merged.is_file()):
            cmd = [
                sys.executable,
                "convert_checkpoint_to_safetensors.py",
                str(train_output_dir / "checkpoint_final"),
                "--base-checkpoint",
                args.base_checkpoint,
                "--output",
                str(merged),
            ]
            if args.force_convert:
                cmd.append("--force")
            run(cmd, dry_run=args.dry_run)

    if "registry" in stages:
        cmd = [
            sys.executable,
            "scripts/build_model_registry.py",
            "--inventory",
            args.inventory,
            "--models",
            ",".join(str(item["model_id"]) for item in items),
            "--training-data-root",
            args.training_data_root,
            "--checkpoint-root",
            args.output_root,
            "--output",
            "configs/model_registry.local.json",
        ]
        run(cmd, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
