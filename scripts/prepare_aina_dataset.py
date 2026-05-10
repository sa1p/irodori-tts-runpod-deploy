from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from prepare_longform_recorded_dataset import (
    build_speech_ranges,
    detect_silences,
    export_segment,
    ffprobe_duration,
    write_lines,
)


NUMBERED_LINE_RE = re.compile(r"^\s*\d+[.．]\s*")


@dataclass(frozen=True)
class SourceSpec:
    wav_name: str
    section: str
    style: str
    start_index: int
    end_index: int
    noise_db: float
    min_silence_sec: float


@dataclass
class AinaSegment:
    index: int
    source_wav: str
    section: str
    style: str
    source_line: int
    start: float
    end: float
    duration: float
    output: str
    text: str


DEFAULT_SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        wav_name="haisinn1.wav",
        section="default",
        style="haishin",
        start_index=0,
        end_index=43,
        noise_db=-55.0,
        min_silence_sec=0.74,
    ),
    SourceSpec(
        wav_name="haisinn2.wav",
        section="default",
        style="haishin",
        start_index=43,
        end_index=63,
        noise_db=-50.0,
        min_silence_sec=0.90,
    ),
    SourceSpec(
        wav_name="コハクささやきsweet.wav",
        section="sweet",
        style="sweet",
        start_index=0,
        end_index=50,
        noise_db=-48.0,
        min_silence_sec=2.00,
    ),
    SourceSpec(
        wav_name="コハクささやきdry.wav",
        section="dry",
        style="dry",
        start_index=0,
        end_index=50,
        noise_db=-45.0,
        min_silence_sec=2.00,
    ),
    SourceSpec(
        wav_name="コハクささやきsleepy.wav",
        section="sleepy",
        style="sleepy",
        start_index=0,
        end_index=50,
        noise_db=-50.0,
        min_silence_sec=1.60,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the Aina/Kohaku 2026-03-10 long-form recordings as aligned Irodori data."
    )
    parser.add_argument(
        "--audio-dir",
        default=r"D:\Downloads\9追加分-20260310T014049Z-1-001\3-9追加分",
    )
    parser.add_argument(
        "--script",
        default=r"D:\Downloads\9追加分-20260310T014049Z-1-001\3-9追加分\script.txt",
    )
    parser.add_argument("--output-dir", default=r"data\irodori_training\aina001")
    parser.add_argument("--speaker", default="aina001")
    parser.add_argument("--language", default="JP")
    parser.add_argument("--target-sample-rate", type=int, default=48000)
    parser.add_argument("--pad-sec", type=float, default=0.08)
    parser.add_argument("--min-segment-sec", type=float, default=0.70)
    parser.add_argument("--max-segment-sec", type=float, default=30.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def clean_script_line(line: str) -> str:
    return NUMBERED_LINE_RE.sub("", line).strip()


def parse_script_sections(path: Path) -> dict[str, list[tuple[int, str]]]:
    sections: dict[str, list[tuple[int, str]]] = {}
    current = "default"
    sections[current] = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            current = line.lstrip("#").strip().casefold()
            sections.setdefault(current, [])
            continue
        if not NUMBERED_LINE_RE.match(line):
            continue
        text = clean_script_line(line)
        if text:
            sections.setdefault(current, []).append((line_no, text))
    return sections


def validate_sources(audio_dir: Path, sections: dict[str, list[tuple[int, str]]]) -> None:
    missing_wavs = [spec.wav_name for spec in DEFAULT_SOURCES if not (audio_dir / spec.wav_name).is_file()]
    if missing_wavs:
        raise FileNotFoundError(f"missing source wavs: {', '.join(missing_wavs)}")
    missing_sections = [spec.section for spec in DEFAULT_SOURCES if spec.section not in sections]
    if missing_sections:
        raise ValueError(f"missing script sections: {', '.join(missing_sections)}")
    for spec in DEFAULT_SOURCES:
        rows = sections[spec.section]
        if len(rows) < spec.end_index:
            raise ValueError(
                f"{spec.section}: expected at least {spec.end_index} lines, found {len(rows)}"
            )


def main() -> None:
    args = parse_args()
    audio_dir = Path(args.audio_dir)
    script_path = Path(args.script)
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"audio dir not found: {audio_dir}")
    if not script_path.is_file():
        raise FileNotFoundError(f"script not found: {script_path}")

    sections = parse_script_sections(script_path)
    validate_sources(audio_dir, sections)

    segments: list[AinaSegment] = []
    next_index = 1
    for spec in DEFAULT_SOURCES:
        wav_path = audio_dir / spec.wav_name
        script_rows = sections[spec.section][spec.start_index : spec.end_index]
        duration = ffprobe_duration(wav_path)
        silences = detect_silences(wav_path, spec.noise_db, spec.min_silence_sec)
        ranges = build_speech_ranges(
            duration,
            silences,
            pad_sec=args.pad_sec,
            min_segment_sec=args.min_segment_sec,
            max_segment_sec=args.max_segment_sec,
            fallback_segment_sec=8.0,
        )
        if len(ranges) != len(script_rows):
            raise ValueError(
                f"{spec.wav_name}: segment/script count mismatch: "
                f"segments={len(ranges)} script_lines={len(script_rows)} "
                f"(noise={spec.noise_db}, min_silence={spec.min_silence_sec})"
            )

        for (start, end), (source_line, text) in zip(ranges, script_rows, strict=True):
            rel_name = f"{args.speaker}_{next_index:05d}.wav"
            output_path = raw_dir / rel_name
            export_segment(
                wav_path,
                output_path,
                start=start,
                end=end,
                sample_rate=args.target_sample_rate,
                force=args.force,
            )
            segments.append(
                AinaSegment(
                    index=next_index,
                    source_wav=str(wav_path.resolve()),
                    section=spec.section,
                    style=spec.style,
                    source_line=source_line,
                    start=round(start, 3),
                    end=round(end, 3),
                    duration=round(end - start, 3),
                    output=str(output_path.resolve()),
                    text=text,
                )
            )
            next_index += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "segments.jsonl").write_text(
        "\n".join(json.dumps(asdict(row), ensure_ascii=False) for row in segments) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_lines(
        output_dir / "segments.tsv",
        [
            "index\toutput\tsource_wav\tsection\tstyle\tsource_line\tstart\tend\tduration\ttext",
            *[
                "\t".join(
                    [
                        str(row.index),
                        row.output,
                        row.source_wav,
                        row.section,
                        row.style,
                        str(row.source_line),
                        f"{row.start:.3f}",
                        f"{row.end:.3f}",
                        f"{row.duration:.3f}",
                        row.text,
                    ]
                )
                for row in segments
            ],
        ],
    )
    write_lines(
        output_dir / "esd.list",
        [
            f"{Path(row.output).name}|{row.style}|{args.language}|{row.text}"
            for row in segments
        ],
    )
    with (output_dir / "train.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for row in segments:
            payload = {
                "audio": Path(row.output).resolve().as_posix(),
                "text": row.text,
                "speaker": args.speaker,
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    summary = {
        "audio_dir": str(audio_dir.resolve()),
        "script": str(script_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "speaker": args.speaker,
        "segments": len(segments),
        "sections": {
            spec.section
            if spec.section != "default"
            else f"{spec.section}:{spec.wav_name}": spec.end_index - spec.start_index
            for spec in DEFAULT_SOURCES
        },
        "total_duration_sec": round(sum(row.duration for row in segments), 3),
        "min_duration_sec": min(row.duration for row in segments),
        "max_duration_sec": max(row.duration for row in segments),
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
