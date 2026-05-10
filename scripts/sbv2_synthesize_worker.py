from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path
from typing import Any

import numpy as np

JP_BERT_MODEL_ID = "ku-nlp/deberta-v2-large-japanese-char-wwm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SBV2 synthesis worker for Irodori distillation.")
    parser.add_argument("--job", required=True)
    return parser.parse_args()


def write_wav(path: Path, sample_rate: int, audio: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(audio, bytes):
        pcm_bytes = audio
    else:
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


def choose_style(tts_model: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    style2id = getattr(tts_model, "style2id", None)
    if isinstance(style2id, dict) and style2id:
        return next(iter(style2id))
    return "Neutral"


def main() -> None:
    args = parse_args()
    job = json.loads(Path(args.job).read_text(encoding="utf-8"))

    from style_bert_vits2.constants import Languages
    from style_bert_vits2.nlp import bert_models
    from style_bert_vits2.tts_model import TTSModel

    bert_models.load_model(Languages.JP, JP_BERT_MODEL_ID)
    bert_models.load_tokenizer(Languages.JP, JP_BERT_MODEL_ID)

    model_id = str(job["model_id"])
    output_dir = Path(job["output_dir"])
    raw_dir = output_dir / "raw"
    train_jsonl = output_dir / "train.jsonl"
    esd_list = output_dir / "esd.list"
    assets = job["assets"]

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    tts_model = TTSModel(
        model_path=Path(str(assets["teacher_checkpoint"])),
        config_path=Path(str(assets["config"])),
        style_vec_path=Path(str(assets["style_vectors"])),
        device="cuda" if bool(job["use_gpu"]) else "cpu",
    )
    style = choose_style(tts_model, str(job["style"]))

    written = 0
    failures: list[dict[str, str]] = []
    with train_jsonl.open("w", encoding="utf-8", newline="\n") as jsonl_f, esd_list.open(
        "w", encoding="utf-8", newline="\n"
    ) as esd_f:
        for idx, text in enumerate(job["prompts"], start=1):
            wav_path = raw_dir / f"{model_id}_{idx:06d}.wav"
            try:
                sample_rate, audio = tts_model.infer(
                    text=text,
                    speaker_id=int(job["speaker_id"]),
                    style=style,
                    language=Languages.JP,
                    sdp_ratio=float(job["sdp_ratio"]),
                    noise=float(job["noise"]),
                    noise_w=float(job["noise_w"]),
                    length=float(job["length"]),
                    line_split=bool(job["line_split"]),
                    split_interval=float(job["split_interval"]),
                    style_weight=float(job["style_weight"]),
                )
                write_wav(wav_path, sample_rate, audio)
                jsonl_f.write(
                    json.dumps(
                        {
                            "audio": wav_path.resolve().as_posix(),
                            "text": text,
                            "speaker": model_id,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                esd_f.write(f"{wav_path.name}|{style}|JP|{text}\n")
                written += 1
            except Exception as exc:
                if not bool(job["continue_on_error"]):
                    raise
                failures.append({"text": text, "error": str(exc)})

    if failures:
        (output_dir / "errors.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in failures) + "\n",
            encoding="utf-8",
        )

    print(
        json.dumps(
            {
                "model_id": model_id,
                "written": written,
                "failures": len(failures),
                "style": style,
                "train_jsonl": str(train_jsonl),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
