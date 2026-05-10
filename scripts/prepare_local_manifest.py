from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from irodori_tts.codec import DACVAECodec
from irodori_tts.text_normalization import normalize_text


def parse_optional_float(value: str) -> float | None:
    raw = str(value).strip().lower()
    if raw in {"none", "null", "off", "disable", "disabled"}:
        return None
    out = float(raw)
    if not math.isfinite(out):
        raise argparse.ArgumentTypeError(f"normalize-db must be finite, got: {value}")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute DACVAE latents from local JSONL without datasets.Audio/torchcodec. "
            "Input rows need audio, text, and optionally speaker."
        )
    )
    parser.add_argument("--input-jsonl", default="data/kohaku/train.jsonl")
    parser.add_argument("--output-manifest", default="data/kohaku/train_manifest.jsonl")
    parser.add_argument("--latent-dir", default="data/kohaku/latents")
    parser.add_argument("--speaker-id-prefix", default="kohaku_haishin")
    parser.add_argument("--codec-repo", default="Aratako/Semantic-DACVAE-Japanese-32dim")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--target-sample-rate", type=int, default=48000)
    parser.add_argument("--normalize-db", type=parse_optional_float, default=-16.0)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--no-text-normalize", action="store_true")
    return parser.parse_args()


def load_audio(path: Path, target_sample_rate: int) -> tuple[torch.Tensor, int]:
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(samples).transpose(0, 1).contiguous()

    if sample_rate != target_sample_rate:
        wav = torchaudio.functional.resample(wav, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate
    return wav, sample_rate


def main() -> None:
    args = parse_args()
    input_jsonl = Path(args.input_jsonl)
    output_manifest = Path(args.output_manifest)
    latent_dir = Path(args.latent_dir)

    if not input_jsonl.is_file():
        raise FileNotFoundError(f"input JSONL not found: {input_jsonl}")

    rows: list[dict] = []
    with input_jsonl.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not item.get("audio") or not item.get("text"):
                raise ValueError(f"line {line_no} needs audio and text: {line.rstrip()}")
            rows.append(item)

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    latent_dir.mkdir(parents=True, exist_ok=True)
    manifest_base = output_manifest.parent.resolve()
    codec = DACVAECodec.load(
        repo_id=args.codec_repo,
        device=args.device,
        deterministic_encode=True,
        deterministic_decode=True,
        normalize_db=args.normalize_db,
    )

    written = 0
    with output_manifest.open("w", encoding="utf-8", newline="\n") as out_f:
        for idx, item in enumerate(tqdm(rows, desc="Precompute latents", unit="utt")):
            audio_path = Path(str(item["audio"])).expanduser()
            if not audio_path.is_file():
                raise FileNotFoundError(f"audio not found: {audio_path}")

            text = str(item["text"]).strip()
            if not args.no_text_normalize:
                text = normalize_text(text).strip()
            if not text:
                continue

            wav, sample_rate = load_audio(audio_path, args.target_sample_rate)
            if args.max_seconds is not None:
                wav = wav[:, : int(args.max_seconds * sample_rate)]
            if wav.numel() == 0:
                continue

            with torch.inference_mode():
                latent = codec.encode_waveform(wav, sample_rate=sample_rate)[0].cpu()

            latent_path = (latent_dir / f"{written:08d}_{idx:08d}.pt").resolve()
            torch.save(latent, latent_path)
            payload = {
                "text": text,
                "latent_path": os.path.relpath(latent_path, start=manifest_base),
                "num_frames": int(latent.shape[0]),
            }
            speaker = str(item.get("speaker") or "").strip()
            if speaker:
                payload["speaker_id"] = f"{args.speaker_id_prefix}:{speaker}"
            out_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written += 1

    print(f"written={written} manifest={output_manifest} latent_dir={latent_dir}")


if __name__ == "__main__":
    main()
