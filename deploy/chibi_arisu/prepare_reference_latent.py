from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from irodori_tts.codec import DACVAECodec


def optional_float(value: str) -> float | None:
    raw = value.strip().lower()
    if raw in {"none", "null", "off", "disabled"}:
        return None
    parsed = float(raw)
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("normalize-db must be finite")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode the chibi Arisu reference WAV as a DACVAE latent."
    )
    parser.add_argument("input_wav", type=Path)
    parser.add_argument("output_latent", type=Path)
    parser.add_argument(
        "--codec-repo",
        default="Aratako/Semantic-DACVAE-Japanese-32dim",
    )
    parser.add_argument("--target-sample-rate", type=int, default=48_000)
    parser.add_argument("--normalize-db", type=optional_float, default=-16.0)
    args = parser.parse_args()

    samples, sample_rate = sf.read(
        args.input_wav,
        dtype="float32",
        always_2d=True,
    )
    waveform = torch.from_numpy(samples).transpose(0, 1).contiguous()
    if sample_rate != args.target_sample_rate:
        waveform = torchaudio.functional.resample(
            waveform,
            sample_rate,
            args.target_sample_rate,
        )
        sample_rate = args.target_sample_rate
    if waveform.numel() == 0:
        raise ValueError(f"reference WAV is empty: {args.input_wav}")

    codec = DACVAECodec.load(
        repo_id=args.codec_repo,
        device="cpu",
        deterministic_encode=True,
        deterministic_decode=True,
        normalize_db=args.normalize_db,
    )
    with torch.inference_mode():
        latent = codec.encode_waveform(waveform, sample_rate=sample_rate)[0].cpu()

    if latent.ndim != 2 or latent.shape[-1] != 32:
        raise ValueError(f"unexpected latent shape: {tuple(latent.shape)}")
    if not torch.isfinite(latent).all():
        raise ValueError("reference latent contains non-finite values")

    args.output_latent.parent.mkdir(parents=True, exist_ok=True)
    torch.save(latent, args.output_latent)
    print(
        f"saved={args.output_latent} shape={tuple(latent.shape)} "
        f"dtype={latent.dtype}",
        flush=True,
    )


if __name__ == "__main__":
    main()
