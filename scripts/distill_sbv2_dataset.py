from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
import wave
from pathlib import Path
from typing import Any

from irodori_targets import parse_model_ids, resolve_sbv2_model_files


JP_BERT_MODEL_ID = "ku-nlp/deberta-v2-large-japanese-char-wwm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate distilled training audio from SBV2 models for Irodori LoRA FT."
    )
    parser.add_argument(
        "--model-assets-root",
        default=os.getenv("SBV2_MODEL_ASSETS_ROOT", r"D:\sbv2\Style-Bert-VITS2\model_assets"),
        help="Style-Bert-VITS2 model_assets root used by direct mode.",
    )
    parser.add_argument(
        "--sbv2-root",
        default=os.getenv("SBV2_ROOT", r"C:\sbv2\Style-Bert-VITS2"),
        help="Style-Bert-VITS2 source checkout used for direct/subprocess mode.",
    )
    parser.add_argument(
        "--sbv2-python",
        default=os.getenv(
            "SBV2_PYTHON",
            r"C:\sbv2\Style-Bert-VITS2\venv\Scripts\python.exe",
        ),
        help="Python executable with SBV2 dependencies for subprocess mode.",
    )
    parser.add_argument("--output-root", default="data/distilled")
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model ids, or @file. Defaults to the configured 38 target models.",
    )
    parser.add_argument("--prompts", default="assets/distill_prompts_ja.txt")
    parser.add_argument("--limit", type=int, default=160)
    parser.add_argument("--speaker-id", type=int, default=0)
    parser.add_argument(
        "--style",
        default="auto",
        help="SBV2 style name. Use 'auto' to select the first style exposed by the model.",
    )
    parser.add_argument("--style-weight", type=float, default=1.0)
    parser.add_argument("--sdp-ratio", type=float, default=0.2)
    parser.add_argument("--noise", type=float, default=0.6)
    parser.add_argument("--noise-w", type=float, default=0.8)
    parser.add_argument("--length", type=float, default=1.0)
    parser.add_argument("--line-split", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--split-interval", type=float, default=0.5)
    parser.add_argument("--use-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--mode",
        choices=["subprocess", "direct", "http"],
        default="subprocess",
        help=(
            "subprocess runs the SBV2 checkout's Python; direct imports style_bert_vits2 "
            "in this environment; http calls an already-running SBV2 API."
        ),
    )
    parser.add_argument("--http-url", default="http://127.0.0.1:5000")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_prompts(path: Path, limit: int) -> list[str]:
    prompts: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "|" in line:
            _label, line = line.split("|", 1)
            line = line.strip()
        if line:
            prompts.append(line)
        if len(prompts) >= limit:
            break
    if not prompts:
        raise ValueError(f"No prompts found: {path}")
    return prompts


def write_wav(path: Path, sample_rate: int, audio: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(audio, bytes):
        pcm_bytes = audio
    else:
        import numpy as np

        array = np.asarray(audio)
        if array.dtype.kind == "f":
            array = np.clip(array, -1.0, 1.0)
            array = (array * 32767.0).astype("<i2")
        else:
            array = array.astype("<i2", copy=False)
        pcm_bytes = array.tobytes()

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


def load_direct_dependencies(sbv2_root: str | None = None):
    if sbv2_root:
        root = Path(sbv2_root).expanduser()
        if root.is_dir() and str(root) not in sys.path:
            sys.path.insert(0, str(root))
    try:
        from style_bert_vits2.constants import Languages
        from style_bert_vits2.nlp import bert_models
        from style_bert_vits2.tts_model import TTSModel
    except ImportError as exc:
        raise RuntimeError(
            "direct mode requires style-bert-vits2. "
            "Install it in this environment or run with --mode http against sbv2-api-lite."
        ) from exc

    bert_models.load_model(Languages.JP, JP_BERT_MODEL_ID)
    bert_models.load_tokenizer(Languages.JP, JP_BERT_MODEL_ID)
    return Languages, TTSModel


def choose_style(tts_model: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    style2id = getattr(tts_model, "style2id", None)
    if isinstance(style2id, dict) and style2id:
        return next(iter(style2id))
    return "Neutral"


async def synthesize_direct(
    tts_model: Any,
    languages: Any,
    text: str,
    style: str,
    args: argparse.Namespace,
) -> tuple[int, Any]:
    return await asyncio.to_thread(
        tts_model.infer,
        text=text,
        speaker_id=args.speaker_id,
        style=style,
        language=languages.JP,
        sdp_ratio=args.sdp_ratio,
        noise=args.noise,
        noise_w=args.noise_w,
        length=args.length,
        line_split=args.line_split,
        split_interval=args.split_interval,
        style_weight=args.style_weight,
    )


def synthesize_http(model_id: str, text: str, style: str, args: argparse.Namespace) -> bytes:
    params: dict[str, Any] = {
        "text": text,
        "speaker_id": args.speaker_id,
        "model_name": model_id,
        "x_audio_format": "wave",
        "sdp_ratio": args.sdp_ratio,
        "noise": args.noise,
        "noisew": args.noise_w,
        "length": args.length,
        "auto_split": str(args.line_split).lower(),
        "split_interval": args.split_interval,
        "style_weight": args.style_weight,
    }
    if style != "auto":
        params["style"] = style
    url = f"{args.http_url.rstrip('/')}/voice?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=args.timeout) as response:
        return response.read()


def count_jsonl_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def distill_model_subprocess(
    model_id: str,
    prompts: list[str],
    assets: dict[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    sbv2_python = Path(args.sbv2_python).expanduser()
    sbv2_root = Path(args.sbv2_root).expanduser()
    if not sbv2_python.is_file():
        raise FileNotFoundError(f"SBV2 python not found: {sbv2_python}")
    if not sbv2_root.is_dir():
        raise FileNotFoundError(f"SBV2 root not found: {sbv2_root}")

    worker = Path(__file__).resolve().parent / "sbv2_synthesize_worker.py"
    job = {
        "model_id": model_id,
        "prompts": prompts,
        "output_dir": str(output_dir.resolve()),
        "assets": assets,
        "style": args.style,
        "speaker_id": args.speaker_id,
        "use_gpu": args.use_gpu,
        "sdp_ratio": args.sdp_ratio,
        "noise": args.noise,
        "noise_w": args.noise_w,
        "length": args.length,
        "line_split": args.line_split,
        "split_interval": args.split_interval,
        "style_weight": args.style_weight,
        "continue_on_error": args.continue_on_error,
    }

    with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8", delete=False) as f:
        json.dump(job, f, ensure_ascii=False)
        job_path = Path(f.name)

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(sbv2_root) if not existing_pythonpath else f"{sbv2_root}{os.pathsep}{existing_pythonpath}"
    )
    try:
        proc = subprocess.run(
            [str(sbv2_python), str(worker), "--job", str(job_path)],
            cwd=str(sbv2_root),
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
    finally:
        job_path.unlink(missing_ok=True)

    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.returncode != 0:
        if proc.stderr:
            print(proc.stderr.rstrip(), file=sys.stderr)
        raise RuntimeError(f"SBV2 subprocess failed for {model_id} with code {proc.returncode}")

    train_jsonl = output_dir / "train.jsonl"
    rows = count_jsonl_rows(train_jsonl)
    return {
        "model_id": model_id,
        "status": "ok",
        "written": rows,
        "train_jsonl": str(train_jsonl),
        "esd_list": str(output_dir / "esd.list"),
        "mode": "subprocess",
    }


async def distill_model(
    model_id: str,
    prompts: list[str],
    args: argparse.Namespace,
    direct_deps: tuple[Any, Any] | None,
) -> dict[str, Any]:
    output_dir = Path(args.output_root) / model_id
    raw_dir = output_dir / "raw"
    train_jsonl = output_dir / "train.jsonl"
    esd_list = output_dir / "esd.list"
    if args.skip_existing and count_jsonl_rows(train_jsonl) >= len(prompts):
        return {"model_id": model_id, "status": "skipped", "train_jsonl": str(train_jsonl)}

    assets = resolve_sbv2_model_files(Path(args.model_assets_root), model_id)
    if args.mode in {"direct", "subprocess"}:
        missing = [
            name
            for name in ("config", "style_vectors", "teacher_checkpoint")
            if not assets.get(name)
        ]
        if missing:
            raise FileNotFoundError(f"{model_id}: missing SBV2 assets: {', '.join(missing)}")

    if args.dry_run:
        return {
            "model_id": model_id,
            "status": "dry-run",
            "output_dir": str(output_dir),
            "prompts": len(prompts),
            "assets": assets,
        }

    if args.mode == "subprocess":
        return distill_model_subprocess(model_id, prompts, assets, output_dir, args)

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    tts_model = None
    style = args.style
    languages = None
    if args.mode == "direct":
        assert direct_deps is not None
        languages, TTSModel = direct_deps
        tts_model = TTSModel(
            model_path=Path(str(assets["teacher_checkpoint"])),
            config_path=Path(str(assets["config"])),
            style_vec_path=Path(str(assets["style_vectors"])),
            device="cuda" if args.use_gpu else "cpu",
        )
        style = choose_style(tts_model, args.style)

    written = 0
    failures: list[dict[str, str]] = []
    with train_jsonl.open("w", encoding="utf-8", newline="\n") as jsonl_f, esd_list.open(
        "w", encoding="utf-8", newline="\n"
    ) as esd_f:
        for idx, text in enumerate(prompts, start=1):
            wav_path = raw_dir / f"{model_id}_{idx:06d}.wav"
            try:
                if args.mode == "direct":
                    sample_rate, audio = await synthesize_direct(
                        tts_model, languages, text, style, args
                    )
                    write_wav(wav_path, sample_rate, audio)
                else:
                    wav_path.write_bytes(synthesize_http(model_id, text, style, args))

                payload = {
                    "audio": wav_path.resolve().as_posix(),
                    "text": text,
                    "speaker": model_id,
                }
                jsonl_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                esd_f.write(f"{wav_path.name}|{style}|JP|{text}\n")
                written += 1
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                failures.append({"text": text, "error": str(exc)})

    if failures:
        (output_dir / "errors.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in failures) + "\n",
            encoding="utf-8",
        )
    return {
        "model_id": model_id,
        "status": "ok",
        "written": written,
        "failures": len(failures),
        "train_jsonl": str(train_jsonl),
        "esd_list": str(esd_list),
        "style": style,
    }


async def amain() -> None:
    args = parse_args()
    prompts = read_prompts(Path(args.prompts), args.limit)
    model_ids = parse_model_ids(args.models)
    direct_deps = (
        load_direct_dependencies(args.sbv2_root)
        if args.mode == "direct" and not args.dry_run
        else None
    )

    results: list[dict[str, Any]] = []
    for model_id in model_ids:
        result = await distill_model(model_id, prompts, args, direct_deps)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))

    non_error_statuses = {"ok", "skipped", "dry-run"}
    failures = [row for row in results if row.get("status") not in non_error_statuses]
    if failures:
        sys.exit(1)


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
