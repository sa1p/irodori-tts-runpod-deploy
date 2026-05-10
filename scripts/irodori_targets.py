from __future__ import annotations

import re
from pathlib import Path


DEFAULT_MODEL_FOLDERS: list[str] = [
    "hiyoko_tentyuou",
    "nana02",
    "sakura04",
    "shiori02",
    "noa01",
    "amane04",
    "momo02",
    "fururu02",
    "chacha02",
    "suzuha001",
    "shinon002",
    "saki003",
    "ruri01",
    "wanko002",
    "nephil003",
    "mao005",
    "kohaku004",
    "hachiroku_20251208_marged",
    "hachiroku_20251208_ml",
    "sora003",
    "alisa002",
    "setsuho008",
    "toa002",
    "yui003",
    "fuu001",
    "neno003",
    "yuna002",
    "haruka003",
    "misato003",
    "reika004",
    "yuu002",
    "shizune003",
    "setsuna003",
    "M001_2f2ae696",
    "M003_2cf01874",
    "M006_b8b5fe66",
    "M004_4d8b14ad",
    "M005_5c25991f",
]


DATASET_ALIASES: dict[str, list[str]] = {
    "suzuha001": ["suzuha001", "suzuha001_ml"],
    "mao005": ["mao005_ml"],
    "kohaku004": ["kohaku004_ml"],
    "hachiroku_20251208_marged": ["hachiroku_20251208"],
    "hachiroku_20251208_ml": ["hachiroku_20251208_ml"],
}


STEP_RE = re.compile(r"(?:^|[_-])e(?P<epoch>\d+)[_-]s(?P<step>\d+)", re.IGNORECASE)


def parse_model_ids(raw: str | None) -> list[str]:
    if raw is None or not raw.strip():
        return list(DEFAULT_MODEL_FOLDERS)
    if raw.startswith("@"):
        path = Path(raw[1:]).expanduser()
        return [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    return [item.strip() for item in raw.split(",") if item.strip()]


def candidate_dataset_names(model_id: str) -> list[str]:
    names: list[str] = []
    for name in DATASET_ALIASES.get(model_id, []):
        names.append(name)
    names.append(model_id)
    if not model_id.endswith("_ml"):
        names.append(f"{model_id}_ml")
    else:
        names.append(model_id.removesuffix("_ml"))
    if model_id.endswith("_marged"):
        names.append(model_id.replace("_marged", ""))
    if model_id.endswith("_merged"):
        names.append(model_id.replace("_merged", ""))

    deduped: list[str] = []
    seen: set[str] = set()
    for name in names:
        key = name.casefold()
        if key not in seen:
            deduped.append(name)
            seen.add(key)
    return deduped


def find_safetensors(model_dir: Path) -> list[Path]:
    return sorted(path for path in model_dir.glob("*.safetensors") if path.is_file())


def pick_teacher_checkpoint(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    if len(paths) == 1:
        return paths[0]

    def score(path: Path) -> tuple[int, int, float, int, str]:
        match = STEP_RE.search(path.stem)
        epoch = int(match.group("epoch")) if match else -1
        step = int(match.group("step")) if match else -1
        stat = path.stat()
        return (step, epoch, stat.st_mtime, stat.st_size, path.name)

    return max(paths, key=score)


def resolve_sbv2_model_files(model_assets_root: Path, model_id: str) -> dict[str, object]:
    model_dir = model_assets_root / model_id
    safetensors = find_safetensors(model_dir) if model_dir.is_dir() else []
    teacher_checkpoint = pick_teacher_checkpoint(safetensors)
    return {
        "model_id": model_id,
        "model_dir": str(model_dir.resolve()) if model_dir.exists() else str(model_dir),
        "model_dir_exists": model_dir.is_dir(),
        "config": str((model_dir / "config.json").resolve())
        if (model_dir / "config.json").is_file()
        else None,
        "style_vectors": str((model_dir / "style_vectors.npy").resolve())
        if (model_dir / "style_vectors.npy").is_file()
        else None,
        "safetensors": [str(path.resolve()) for path in safetensors],
        "teacher_checkpoint": str(teacher_checkpoint.resolve()) if teacher_checkpoint else None,
    }
