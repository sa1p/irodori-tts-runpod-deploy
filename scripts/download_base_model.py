from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the Irodori-TTS v3 base checkpoint.")
    parser.add_argument("--repo-id", default="Aratako/Irodori-TTS-500M-v3")
    parser.add_argument("--filename", default="model.safetensors")
    parser.add_argument("--output-dir", default="models/Irodori-TTS-500M-v3")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source = Path(
        hf_hub_download(
            repo_id=args.repo_id,
            filename=args.filename,
            local_dir=output_dir,
        )
    )
    print(source)


if __name__ == "__main__":
    main()
