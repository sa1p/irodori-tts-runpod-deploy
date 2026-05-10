from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


SILENCE_START_RE = re.compile(r"silence_start:\s*(?P<value>[0-9.]+)")
SILENCE_END_RE = re.compile(r"silence_end:\s*(?P<value>[0-9.]+)")
TIMESTAMP_RE = re.compile(
    r"(?P<h>\d{1,2}):(?P<m>\d{2}):(?P<s>\d{2})(?P<ms>[,.]\d{1,3})?"
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?])[\s　]+|[\r\n]+")


@dataclass
class Segment:
    index: int
    source: str
    start: float
    end: float
    duration: float
    output: str
    text: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split long recorded wav files by silence and optionally align transcript units "
            "to build SBV2-style esd.list plus Irodori train.jsonl."
        )
    )
    parser.add_argument("--audio-dir", required=True, help="Directory containing long wav files.")
    parser.add_argument(
        "--transcript",
        default=None,
        help=(
            "Transcript file. txt/md/list/csv/tsv/srt/vtt are supported. "
            "If omitted, only audio segments and segment metadata are written."
        ),
    )
    parser.add_argument("--output-dir", required=True, help="Output dataset directory.")
    parser.add_argument("--speaker", default="aina001")
    parser.add_argument("--language", default="JP")
    parser.add_argument("--style", default="Neutral")
    parser.add_argument("--target-sample-rate", type=int, default=48000)
    parser.add_argument("--noise-db", type=float, default=-42.0)
    parser.add_argument("--min-silence-sec", type=float, default=0.35)
    parser.add_argument("--pad-sec", type=float, default=0.08)
    parser.add_argument("--min-segment-sec", type=float, default=0.7)
    parser.add_argument("--max-segment-sec", type=float, default=14.0)
    parser.add_argument(
        "--fallback-segment-sec",
        type=float,
        default=8.0,
        help="Fixed split size used when silence detection finds no usable boundaries.",
    )
    parser.add_argument(
        "--pair-mode",
        choices=["strict", "truncate"],
        default="strict",
        help=(
            "strict fails when transcript unit count differs from segment count. "
            "truncate writes paired rows up to the shorter count and records the mismatch."
        ),
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing segment wavs.")
    return parser.parse_args()


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )


def ffprobe_duration(path: Path) -> float:
    proc = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ]
    )
    payload = json.loads(proc.stdout)
    return float(payload["format"]["duration"])


def detect_silences(path: Path, noise_db: float, min_silence_sec: float) -> list[tuple[float, float]]:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        str(path),
        "-af",
        f"silencedetect=noise={noise_db}dB:d={min_silence_sec}",
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(
        cmd,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    output = "\n".join([proc.stdout or "", proc.stderr or ""])
    silences: list[tuple[float, float]] = []
    current_start: float | None = None
    for line in output.splitlines():
        start_match = SILENCE_START_RE.search(line)
        if start_match:
            current_start = float(start_match.group("value"))
            continue
        end_match = SILENCE_END_RE.search(line)
        if end_match and current_start is not None:
            end = float(end_match.group("value"))
            if end > current_start:
                silences.append((current_start, end))
            current_start = None
    return silences


def split_long_range(start: float, end: float, max_segment_sec: float) -> Iterable[tuple[float, float]]:
    duration = end - start
    if duration <= max_segment_sec:
        yield start, end
        return
    count = max(1, int(duration // max_segment_sec) + int(duration % max_segment_sec > 0))
    step = duration / count
    for i in range(count):
        chunk_start = start + i * step
        chunk_end = end if i == count - 1 else start + (i + 1) * step
        yield chunk_start, chunk_end


def build_speech_ranges(
    duration: float,
    silences: list[tuple[float, float]],
    *,
    pad_sec: float,
    min_segment_sec: float,
    max_segment_sec: float,
    fallback_segment_sec: float,
) -> list[tuple[float, float]]:
    raw_ranges: list[tuple[float, float]] = []
    cursor = 0.0
    for silence_start, silence_end in silences:
        if silence_start > cursor:
            raw_ranges.append((cursor, silence_start))
        cursor = max(cursor, silence_end)
    if cursor < duration:
        raw_ranges.append((cursor, duration))

    if not raw_ranges:
        cursor = 0.0
        while cursor < duration:
            raw_ranges.append((cursor, min(duration, cursor + fallback_segment_sec)))
            cursor += fallback_segment_sec

    padded: list[tuple[float, float]] = []
    for start, end in raw_ranges:
        start = max(0.0, start - pad_sec)
        end = min(duration, end + pad_sec)
        if end - start < min_segment_sec:
            continue
        for chunk_start, chunk_end in split_long_range(start, end, max_segment_sec):
            if chunk_end - chunk_start >= min_segment_sec:
                padded.append((chunk_start, chunk_end))
    return padded


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text.replace("\ufeff", " ")).strip()
    return text


def split_plain_text(text: str) -> list[str]:
    lines = [clean_text(line) for line in text.splitlines() if clean_text(line)]
    if len(lines) > 1:
        return lines
    units = [clean_text(part) for part in SENTENCE_SPLIT_RE.split(text) if clean_text(part)]
    return units


def read_srt_or_vtt(path: Path) -> list[str]:
    units: list[str] = []
    current: list[str] = []
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line:
            if current:
                units.append(clean_text(" ".join(current)))
                current = []
            continue
        if line.isdigit() or "-->" in line or line.upper().startswith("WEBVTT"):
            continue
        current.append(line)
    if current:
        units.append(clean_text(" ".join(current)))
    return [unit for unit in units if unit]


def read_list(path: Path) -> list[str]:
    units: list[str] = []
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split("|", 3)
        units.append(clean_text(parts[3] if len(parts) == 4 else line))
    return [unit for unit in units if unit]


def read_delimited(path: Path, delimiter: str) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        has_header = csv.Sniffer().has_header(sample) if sample.strip() else False
        if has_header:
            reader = csv.DictReader(f, delimiter=delimiter)
            fieldnames = [name or "" for name in (reader.fieldnames or [])]
            text_field = next(
                (
                    name
                    for name in fieldnames
                    if name.casefold() in {"text", "transcript", "sentence", "line", "台詞", "セリフ"}
                ),
                fieldnames[-1] if fieldnames else None,
            )
            return [clean_text(row.get(text_field, "")) for row in reader if text_field and row.get(text_field)]
        reader = csv.reader(f, delimiter=delimiter)
        return [clean_text(row[-1]) for row in reader if row]


def read_transcript(path: Path) -> list[str]:
    suffix = path.suffix.casefold()
    if suffix in {".srt", ".vtt"}:
        return read_srt_or_vtt(path)
    if suffix == ".list":
        return read_list(path)
    if suffix == ".csv":
        return read_delimited(path, ",")
    if suffix == ".tsv":
        return read_delimited(path, "\t")
    text = path.read_text(encoding="utf-8-sig")
    return split_plain_text(text)


def export_segment(
    source: Path,
    output: Path,
    *,
    start: float,
    end: float,
    sample_rate: int,
    force: bool,
) -> None:
    if output.is_file() and not force:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-to",
            f"{end:.3f}",
            "-i",
            str(source),
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-sample_fmt",
            "s16",
            str(output),
        ],
        check=True,
    )


def write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    args = parse_args()
    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"audio dir not found: {audio_dir}")

    wavs = sorted(audio_dir.glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"no wav files found in: {audio_dir}")

    segments: list[Segment] = []
    next_index = 1
    for wav in wavs:
        duration = ffprobe_duration(wav)
        silences = detect_silences(wav, args.noise_db, args.min_silence_sec)
        ranges = build_speech_ranges(
            duration,
            silences,
            pad_sec=args.pad_sec,
            min_segment_sec=args.min_segment_sec,
            max_segment_sec=args.max_segment_sec,
            fallback_segment_sec=args.fallback_segment_sec,
        )
        for start, end in ranges:
            rel_name = f"{args.speaker}_{next_index:05d}.wav"
            output = raw_dir / rel_name
            export_segment(
                wav,
                output,
                start=start,
                end=end,
                sample_rate=args.target_sample_rate,
                force=args.force,
            )
            segments.append(
                Segment(
                    index=next_index,
                    source=str(wav.resolve()),
                    start=round(start, 3),
                    end=round(end, 3),
                    duration=round(end - start, 3),
                    output=str(output.resolve()),
                )
            )
            next_index += 1

    transcript_units: list[str] = []
    if args.transcript:
        transcript_path = Path(args.transcript)
        if not transcript_path.is_file():
            raise FileNotFoundError(f"transcript not found: {transcript_path}")
        transcript_units = read_transcript(transcript_path)
        if not transcript_units:
            raise ValueError(f"no transcript units found in: {transcript_path}")

        if len(transcript_units) != len(segments) and args.pair_mode == "strict":
            write_lines(output_dir / "transcript_units.txt", transcript_units)
            (output_dir / "segments.jsonl").write_text(
                "\n".join(json.dumps(asdict(row), ensure_ascii=False) for row in segments) + "\n",
                encoding="utf-8",
            )
            raise ValueError(
                "segment/transcript count mismatch: "
                f"segments={len(segments)} transcript_units={len(transcript_units)}. "
                "Adjust silence options, edit transcript lines, or rerun with --pair-mode truncate."
            )

        paired = min(len(transcript_units), len(segments))
        for i in range(paired):
            segments[i].text = transcript_units[i]

    (output_dir / "segments.jsonl").write_text(
        "\n".join(json.dumps(asdict(row), ensure_ascii=False) for row in segments) + "\n",
        encoding="utf-8",
    )

    with (output_dir / "segments.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["index", "output", "source", "start", "end", "duration", "text"])
        for row in segments:
            writer.writerow([row.index, row.output, row.source, row.start, row.end, row.duration, row.text or ""])

    paired_segments = [row for row in segments if row.text]
    if paired_segments:
        esd_lines = [
            f"{Path(row.output).name}|{args.style}|{args.language}|{row.text}"
            for row in paired_segments
        ]
        write_lines(output_dir / "esd.list", esd_lines)

        with (output_dir / "train.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            for row in paired_segments:
                payload = {
                    "audio": Path(row.output).resolve().as_posix(),
                    "text": row.text,
                    "speaker": args.speaker,
                }
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    summary = {
        "audio_dir": str(audio_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "speaker": args.speaker,
        "source_wavs": len(wavs),
        "segments": len(segments),
        "transcript_units": len(transcript_units),
        "paired": len(paired_segments),
        "unpaired_segments": len(segments) - len(paired_segments),
        "unpaired_transcript_units": max(0, len(transcript_units) - len(paired_segments)),
        "total_segment_duration_sec": round(sum(row.duration for row in segments), 3),
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
