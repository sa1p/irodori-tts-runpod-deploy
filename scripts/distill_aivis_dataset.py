from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.parse
import urllib.request
import wave
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate distilled Irodori training data from a local AivisSpeech/VOICEVOX-compatible API."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:10101")
    parser.add_argument("--model-id", required=True, help="Output dataset/model id.")
    parser.add_argument("--speaker-name", default=None, help="AivisSpeech speaker name to select.")
    parser.add_argument("--speaker-uuid", default=None, help="AivisSpeech speaker UUID to select.")
    parser.add_argument("--speaker-style-id", type=int, default=None, help="AivisSpeech/VOICEVOX style id.")
    parser.add_argument("--style-name", default="ノーマル")
    parser.add_argument("--prompts", default="assets/distill_prompts_ja.txt")
    parser.add_argument("--limit", type=int, default=160)
    parser.add_argument("--output-root", default="data/distilled")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--overwrite", action="store_true")
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


def request_json(url: str, *, method: str = "GET", body: Any | None = None, timeout: float) -> Any:
    data = None
    headers: dict[str, str] = {}
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def request_bytes(url: str, *, method: str = "POST", body: Any | None = None, timeout: float) -> bytes:
    data = None
    headers: dict[str, str] = {}
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


def select_style(
    speakers: list[dict[str, Any]],
    *,
    speaker_name: str | None,
    speaker_uuid: str | None,
    speaker_style_id: int | None,
    style_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if speaker_style_id is not None:
        for speaker in speakers:
            for style in speaker.get("styles", []):
                if int(style.get("id")) == speaker_style_id:
                    return speaker, style
        raise ValueError(f"style id not found: {speaker_style_id}")

    matches = []
    for speaker in speakers:
        if speaker_name is not None and speaker.get("name") != speaker_name:
            continue
        if speaker_uuid is not None and speaker.get("speaker_uuid") != speaker_uuid:
            continue
        matches.append(speaker)
    if not matches:
        raise ValueError(f"speaker not found: name={speaker_name!r} uuid={speaker_uuid!r}")
    if len(matches) > 1:
        names = ", ".join(str(item.get("name")) for item in matches)
        raise ValueError(f"speaker selection is ambiguous: {names}")

    speaker = matches[0]
    styles = list(speaker.get("styles", []))
    if not styles:
        raise ValueError(f"speaker has no styles: {speaker.get('name')}")
    for style in styles:
        if style.get("name") == style_name:
            return speaker, style
    return speaker, styles[0]


def validate_wav(path: Path) -> tuple[int, float]:
    with wave.open(str(path), "rb") as wf:
        sample_rate = wf.getframerate()
        duration = wf.getnframes() / float(sample_rate)
    return sample_rate, duration


def main() -> None:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    output_dir = Path(args.output_root) / args.model_id
    raw_dir = output_dir / "raw"
    train_jsonl = output_dir / "train.jsonl"
    esd_list = output_dir / "esd.list"

    speakers = request_json(f"{base_url}/speakers", timeout=args.timeout)
    speaker, style = select_style(
        speakers,
        speaker_name=args.speaker_name,
        speaker_uuid=args.speaker_uuid,
        speaker_style_id=args.speaker_style_id,
        style_name=args.style_name,
    )
    style_id = int(style["id"])
    style_label = str(style.get("name") or style_id)
    prompts = read_prompts(Path(args.prompts), args.limit)

    if output_dir.exists() and args.overwrite:
        backup_dir = output_dir.with_name(f"{output_dir.name}.backup_aivis")
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(output_dir, backup_dir)
        for path in output_dir.glob("*"):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

    raw_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    failures: list[dict[str, str]] = []

    with train_jsonl.open("w", encoding="utf-8", newline="\n") as jsonl_f, esd_list.open(
        "w", encoding="utf-8", newline="\n"
    ) as esd_f:
        for idx, text in enumerate(prompts, start=1):
            wav_path = raw_dir / f"{args.model_id}_{idx:06d}.wav"
            try:
                query_url = (
                    f"{base_url}/audio_query?"
                    + urllib.parse.urlencode({"speaker": style_id, "text": text})
                )
                query = request_json(query_url, method="POST", timeout=args.timeout)
                synth_url = f"{base_url}/synthesis?" + urllib.parse.urlencode({"speaker": style_id})
                wav_bytes = request_bytes(synth_url, method="POST", body=query, timeout=args.timeout)
                wav_path.write_bytes(wav_bytes)
                validate_wav(wav_path)

                jsonl_f.write(
                    json.dumps(
                        {
                            "audio": wav_path.resolve().as_posix(),
                            "text": text,
                            "speaker": args.model_id,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                esd_f.write(f"{wav_path.name}|{style_label}|JP|{text}\n")
                written += 1
                if written % 10 == 0 or written == len(prompts):
                    print(f"written={written}/{len(prompts)}", flush=True)
            except Exception as exc:
                failures.append({"text": text, "error": str(exc)})

    if failures:
        (output_dir / "errors.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in failures) + "\n",
            encoding="utf-8",
        )
    else:
        (output_dir / "errors.jsonl").unlink(missing_ok=True)

    metadata = {
        "model_id": args.model_id,
        "source": "aivis_speech",
        "base_url": base_url,
        "speaker_name": speaker.get("name"),
        "speaker_uuid": speaker.get("speaker_uuid"),
        "style_name": style.get("name"),
        "style_id": style_id,
        "written": written,
        "failures": len(failures),
    }
    (output_dir / "aivis_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False))
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
