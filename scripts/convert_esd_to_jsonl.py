from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a Style-Bert-VITS2 esd.list file to JSONL for HF datasets."
    )
    parser.add_argument(
        "--esd-list",
        default=r"D:\sbv2\Style-Bert-VITS2\Data\kohaku-haishin_v2\esd.list",
        help="Path to esd.list.",
    )
    parser.add_argument(
        "--audio-root",
        default=r"D:\sbv2\Style-Bert-VITS2\Data\kohaku-haishin_v2\raw",
        help="Root directory used to resolve relative audio paths from esd.list.",
    )
    parser.add_argument("--output", default="data/kohaku/train.jsonl")
    parser.add_argument("--speaker", default="kohaku")
    parser.add_argument(
        "--exclude-name",
        action="append",
        default=["kohaku-haishin-60.wav"],
        help="Audio basename to exclude. May be repeated.",
    )
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    esd_list = Path(args.esd_list)
    audio_root = Path(args.audio_root)
    output = Path(args.output)
    excluded = {str(name).casefold() for name in args.exclude_name}

    if not esd_list.is_file():
        raise FileNotFoundError(f"esd.list not found: {esd_list}")
    if not audio_root.is_dir():
        raise FileNotFoundError(f"audio root not found: {audio_root}")

    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    missing: list[str] = []

    with esd_list.open("r", encoding="utf-8-sig") as in_f, output.open(
        "w", encoding="utf-8", newline="\n"
    ) as out_f:
        for line_no, raw_line in enumerate(in_f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split("|", 3)
            if len(parts) != 4:
                raise ValueError(f"Invalid esd.list line {line_no}: {line}")

            rel_audio, _style, _language, text = parts
            audio_path = (audio_root / rel_audio).resolve()
            if audio_path.name.casefold() in excluded:
                skipped += 1
                continue
            if not text.strip():
                raise ValueError(f"Empty text at line {line_no}: {line}")
            if not audio_path.is_file():
                missing.append(str(audio_path))
                if not args.allow_missing:
                    continue

            payload = {
                "audio": audio_path.as_posix(),
                "text": text.strip(),
                "speaker": args.speaker,
            }
            out_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written += 1

    if missing and not args.allow_missing:
        output.unlink(missing_ok=True)
        sample = "\n".join(missing[:10])
        raise FileNotFoundError(f"Missing {len(missing)} audio files:\n{sample}")

    print(f"wrote={written} skipped={skipped} missing={len(missing)} output={output}")


if __name__ == "__main__":
    main()
