#!/usr/bin/env python3
"""Extract stable, diverse keyframes from a static-scene video for Stage A."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MANIFEST_SCHEMA_VERSION = 1


class KeyframeError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    duration_sec: float
    width: int
    height: int
    codec: str
    source_fps: float | None
    file_size: int
    mtime_ns: int

    def fingerprint(self) -> dict[str, Any]:
        return {
            "path": str(self.path.resolve()),
            "file_size": self.file_size,
            "mtime_ns": self.mtime_ns,
        }


@dataclass
class Candidate:
    index: int
    timestamp_sec: float
    path: Path
    width: int
    height: int
    sharpness: float
    underexposed_ratio: float
    overexposed_ratio: float
    feature: np.ndarray = field(repr=False)
    quality_score: float = 0.0
    selection_score: float | None = None
    max_similarity: float | None = None
    nearest_time_gap_sec: float | None = None
    selected: bool = False
    selection_reason: str | None = None
    rejection_reason: str | None = None
    output_name: str | None = None


def stable_output_name(index: int) -> str:
    if index < 0:
        raise ValueError("output index must be non-negative")
    return f"{index:06d}.jpg"


def validate_parameters(args: argparse.Namespace) -> None:
    if args.target_count <= 0:
        raise ValueError("--target-count must be positive")
    if not math.isfinite(args.candidate_fps) or args.candidate_fps <= 0:
        raise ValueError("--candidate-fps must be a positive finite number")
    if not math.isfinite(args.min_time_gap_sec) or args.min_time_gap_sec < 0:
        raise ValueError("--min-time-gap-sec must be finite and non-negative")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be in [1,100]")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if not 0.0 <= args.duplicate_threshold <= 1.0:
        raise ValueError("--duplicate-threshold must be in [0,1]")
    if not 0.0 <= args.max_underexposed_ratio <= 1.0:
        raise ValueError("--max-underexposed-ratio must be in [0,1]")
    if not 0.0 <= args.max_overexposed_ratio <= 1.0:
        raise ValueError("--max-overexposed-ratio must be in [0,1]")
    if not 0 <= args.dark_threshold < args.bright_threshold <= 255:
        raise ValueError("exposure thresholds must satisfy 0 <= dark < bright <= 255")


def require_dependencies() -> dict[str, str]:
    resolved = {}
    for executable in ("ffprobe", "ffmpeg"):
        path = shutil.which(executable)
        if path is None:
            raise KeyframeError(
                f"required executable '{executable}' was not found in PATH; install FFmpeg first"
            )
        resolved[executable] = path
    return resolved


def _parse_rate(value: str | None) -> float | None:
    if not value or value in {"0/0", "N/A"}:
        return None
    numerator, separator, denominator = value.partition("/")
    try:
        rate = float(numerator) / float(denominator) if separator else float(value)
    except (ValueError, ZeroDivisionError):
        return None
    return rate if math.isfinite(rate) and rate > 0 else None


def probe_video(video: Path, ffprobe: str) -> VideoInfo:
    video = Path(video).expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f"video does not exist: {video}")
    command = [
        ffprobe,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "format=duration:stream=codec_name,width,height,avg_frame_rate,duration",
        "-of", "json",
        str(video),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise KeyframeError(f"ffprobe failed for {video}:\n{result.stderr.strip()}")
    try:
        payload = json.loads(result.stdout)
        stream = payload["streams"][0]
        duration = float(payload.get("format", {}).get("duration") or stream.get("duration"))
        width = int(stream["width"])
        height = int(stream["height"])
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise KeyframeError(f"ffprobe returned incomplete video metadata for {video}") from error
    if not math.isfinite(duration) or duration <= 0 or width <= 0 or height <= 0:
        raise KeyframeError(f"invalid video metadata: duration={duration}, size={width}x{height}")
    stat = video.stat()
    return VideoInfo(
        path=video,
        duration_sec=duration,
        width=width,
        height=height,
        codec=str(stream.get("codec_name") or "unknown"),
        source_fps=_parse_rate(stream.get("avg_frame_rate")),
        file_size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
    )


def extract_uniform_candidates(
    video: Path,
    candidate_fps: float,
    temporary_directory: Path,
    ffmpeg: str,
) -> list[Path]:
    pattern = temporary_directory / "candidate_%08d.jpg"
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(video),
        "-map", "0:v:0",
        "-vf", f"fps={candidate_fps:.12g}",
        "-q:v", "2",
        "-start_number", "0",
        str(pattern),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise KeyframeError(f"ffmpeg candidate extraction failed:\n{result.stderr.strip()}")
    candidates = sorted(temporary_directory.glob("candidate_*.jpg"))
    if not candidates:
        raise KeyframeError("ffmpeg produced no candidate frames")
    return candidates


def compute_visual_feature(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    feature = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    return np.ascontiguousarray(feature.reshape(-1) / 255.0)


def visual_similarity(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape:
        raise ValueError("visual features must have identical shapes")
    similarity = 1.0 - float(np.mean(np.abs(first - second)))
    return float(np.clip(similarity, 0.0, 1.0))


def analyze_candidates(
    paths: Sequence[Path],
    candidate_fps: float,
    dark_threshold: int,
    bright_threshold: int,
) -> list[Candidate]:
    candidates = []
    for index, path in enumerate(paths):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise KeyframeError(f"failed to decode candidate frame: {path}")
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        underexposed = float(np.mean(gray <= dark_threshold))
        overexposed = float(np.mean(gray >= bright_threshold))
        candidates.append(
            Candidate(
                index=index,
                timestamp_sec=index / candidate_fps,
                path=path,
                width=width,
                height=height,
                sharpness=sharpness,
                underexposed_ratio=underexposed,
                overexposed_ratio=overexposed,
                feature=compute_visual_feature(image),
            )
        )
    return candidates


def _assign_quality_scores(
    candidates: Sequence[Candidate],
    max_underexposed_ratio: float,
    max_overexposed_ratio: float,
) -> None:
    logged = np.log1p(np.array([candidate.sharpness for candidate in candidates], dtype=np.float64))
    low, high = np.percentile(logged, [10, 90])
    if high - low <= 1e-12:
        sharpness_scores = np.full_like(logged, 0.5)
    else:
        sharpness_scores = np.clip((logged - low) / (high - low), 0.0, 1.0)
    under_scale = max(max_underexposed_ratio, 1e-6)
    over_scale = max(max_overexposed_ratio, 1e-6)
    for candidate, sharpness_score in zip(candidates, sharpness_scores):
        exposure_penalty = max(
            candidate.underexposed_ratio / under_scale,
            candidate.overexposed_ratio / over_scale,
        )
        exposure_score = 1.0 - float(np.clip(exposure_penalty, 0.0, 1.0))
        candidate.quality_score = 0.75 * float(sharpness_score) + 0.25 * exposure_score


def select_keyframes(
    candidates: Sequence[Candidate],
    target_count: int,
    min_time_gap_sec: float,
    duplicate_threshold: float,
    max_underexposed_ratio: float,
    max_overexposed_ratio: float,
    seed: int,
) -> list[Candidate]:
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    if not candidates:
        return []
    _assign_quality_scores(candidates, max_underexposed_ratio, max_overexposed_ratio)
    rng = random.Random(seed)
    tie_breakers = {candidate.index: rng.random() * 1e-9 for candidate in candidates}
    selected: list[Candidate] = []
    permanent_rejections: dict[int, str] = {}
    ideal_gap = max(min_time_gap_sec, (candidates[-1].timestamp_sec + 1e-9) / target_count)

    while len(selected) < target_count:
        best: Candidate | None = None
        best_score = float("-inf")
        best_similarity = 0.0
        best_time_gap = float("inf")
        for candidate in candidates:
            if candidate.selected or candidate.index in permanent_rejections:
                continue
            if candidate.underexposed_ratio > max_underexposed_ratio:
                permanent_rejections[candidate.index] = "underexposed"
                continue
            if candidate.overexposed_ratio > max_overexposed_ratio:
                permanent_rejections[candidate.index] = "overexposed"
                continue

            if selected:
                nearest_gap = min(
                    abs(candidate.timestamp_sec - chosen.timestamp_sec) for chosen in selected
                )
                if nearest_gap + 1e-9 < min_time_gap_sec:
                    permanent_rejections[candidate.index] = "time_gap"
                    continue
                max_similarity = max(
                    visual_similarity(candidate.feature, chosen.feature) for chosen in selected
                )
                if max_similarity >= duplicate_threshold:
                    permanent_rejections[candidate.index] = "visual_duplicate"
                    continue
                novelty = 1.0 - max_similarity
                temporal_score = min(nearest_gap / ideal_gap, 1.0)
            else:
                nearest_gap = float("inf")
                max_similarity = 0.0
                novelty = 1.0
                temporal_score = 1.0

            score = (
                0.70 * candidate.quality_score
                + 0.20 * novelty
                + 0.10 * temporal_score
                + tie_breakers[candidate.index]
            )
            if score > best_score:
                best = candidate
                best_score = score
                best_similarity = max_similarity
                best_time_gap = nearest_gap

        if best is None:
            break
        best.selected = True
        best.selection_score = best_score
        best.max_similarity = best_similarity
        best.nearest_time_gap_sec = None if math.isinf(best_time_gap) else best_time_gap
        best.selection_reason = (
            "selected for sharpness/exposure quality, visual novelty, and temporal coverage"
        )
        selected.append(best)

    for candidate in candidates:
        if not candidate.selected:
            candidate.rejection_reason = permanent_rejections.get(
                candidate.index, "target_count_reached"
            )
    selected.sort(key=lambda candidate: (candidate.timestamp_sec, candidate.index))
    for output_index, candidate in enumerate(selected):
        candidate.output_name = stable_output_name(output_index)
    return selected


def _candidate_record(candidate: Candidate) -> dict[str, Any]:
    return {
        "candidate_index": candidate.index,
        "timestamp_sec": round(candidate.timestamp_sec, 9),
        "shape": [candidate.height, candidate.width, 3],
        "sharpness_laplacian_variance": candidate.sharpness,
        "underexposed_ratio": candidate.underexposed_ratio,
        "overexposed_ratio": candidate.overexposed_ratio,
        "quality_score": candidate.quality_score,
        "selection_score": candidate.selection_score,
        "max_similarity_to_selected": candidate.max_similarity,
        "nearest_selected_time_gap_sec": candidate.nearest_time_gap_sec,
        "selected": candidate.selected,
        "selection_reason": candidate.selection_reason,
        "rejection_reason": candidate.rejection_reason,
        "output_name": candidate.output_name,
    }


def build_manifest(
    video: VideoInfo,
    parameters: dict[str, Any],
    candidates: Sequence[Candidate],
    selected: Sequence[Candidate],
    dry_run: bool,
    status: str,
    written_outputs: Sequence[str] = (),
) -> dict[str, Any]:
    written = set(written_outputs)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": status,
        "dry_run": dry_run,
        "video": {
            **video.fingerprint(),
            "duration_sec": video.duration_sec,
            "width": video.width,
            "height": video.height,
            "codec": video.codec,
            "source_fps": video.source_fps,
        },
        "parameters": parameters,
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "keyframes": [
            {**_candidate_record(candidate), "written": candidate.output_name in written}
            for candidate in selected
        ],
        "candidates": [_candidate_record(candidate) for candidate in candidates],
    }


def build_selection_report(
    video: VideoInfo,
    candidates: Sequence[Candidate],
    selected: Sequence[Candidate],
    target_count: int,
) -> str:
    sharpness = np.array([candidate.sharpness for candidate in candidates], dtype=np.float64)
    rejection_counts = Counter(
        candidate.rejection_reason for candidate in candidates if candidate.rejection_reason
    )
    lines = [
        f"video: {video.path}",
        f"duration_sec: {video.duration_sec:.6f}",
        f"source_resolution: {video.width}x{video.height}",
        f"candidate_count: {len(candidates)}",
        f"target_count: {target_count}",
        f"selected_count: {len(selected)}",
        f"selection_shortfall: {max(0, target_count - len(selected))}",
        f"sharpness_min: {sharpness.min():.6f}",
        f"sharpness_mean: {sharpness.mean():.6f}",
        f"sharpness_median: {np.median(sharpness):.6f}",
        f"sharpness_max: {sharpness.max():.6f}",
        f"duplicate_rejected: {rejection_counts['visual_duplicate']}",
        f"underexposed_rejected: {rejection_counts['underexposed']}",
        f"overexposed_rejected: {rejection_counts['overexposed']}",
        f"time_gap_rejected: {rejection_counts['time_gap']}",
        f"target_count_rejected: {rejection_counts['target_count_reached']}",
    ]
    return "\n".join(lines) + "\n"


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_write_bytes(path, encoded)


def atomic_write_text(path: Path, payload: str) -> None:
    _atomic_write_bytes(path, payload.encode("utf-8"))


def atomic_write_jpeg(path: Path, image: Image.Image, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            image.save(handle, format="JPEG", quality=quality, subsampling=0)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def create_contact_sheet(
    selected: Sequence[Candidate],
    output_path: Path,
    jpeg_quality: int,
    columns: int = 10,
) -> None:
    if not selected:
        sheet = Image.new("RGB", (640, 96), "black")
        ImageDraw.Draw(sheet).text((16, 36), "No frames selected", fill="white")
        atomic_write_jpeg(output_path, sheet, jpeg_quality)
        return
    thumb_width, thumb_height, label_height = 192, 108, 22
    columns = max(1, min(columns, len(selected)))
    rows = math.ceil(len(selected) / columns)
    sheet = Image.new(
        "RGB", (columns * thumb_width, rows * (thumb_height + label_height)), "black"
    )
    draw = ImageDraw.Draw(sheet)
    for index, candidate in enumerate(selected):
        with Image.open(candidate.path) as opened:
            thumbnail = ImageOps.fit(
                opened.convert("RGB"),
                (thumb_width, thumb_height),
                method=Image.Resampling.LANCZOS,
            )
        x = (index % columns) * thumb_width
        y = (index // columns) * (thumb_height + label_height)
        sheet.paste(thumbnail, (x, y))
        draw.text(
            (x + 3, y + thumb_height + 4),
            f"{index:06d}  {candidate.timestamp_sec:.3f}s",
            fill="white",
        )
    atomic_write_jpeg(output_path, sheet, jpeg_quality)


def write_metadata_artifacts(
    metadata_directory: Path,
    manifest: dict[str, Any],
    report: str,
    selected: Sequence[Candidate],
    jpeg_quality: int,
) -> None:
    metadata_directory.mkdir(parents=True, exist_ok=True)
    atomic_write_json(metadata_directory / "keyframes.json", manifest)
    atomic_write_text(metadata_directory / "selection_report.txt", report)
    create_contact_sheet(
        selected, metadata_directory / "contact_sheet.jpg", jpeg_quality=jpeg_quality
    )


def _load_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise KeyframeError(f"cannot read existing manifest {path}: {error}") from error
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise KeyframeError(f"unsupported existing manifest schema: {path}")
    return manifest


def _existing_output_files(output: Path) -> list[Path]:
    if not output.exists():
        return []
    if not output.is_dir():
        raise KeyframeError(f"output exists but is not a directory: {output}")
    return sorted(path for path in output.iterdir() if path.is_file())


def validate_resume_state(
    output: Path,
    existing_manifest: dict[str, Any] | None,
    video: VideoInfo,
    parameters: dict[str, Any],
    selected: Sequence[Candidate],
) -> set[str]:
    files = _existing_output_files(output)
    if files and existing_manifest is None:
        raise KeyframeError("existing output images have no keyframes.json; refusing to mix data")
    expected = {candidate.output_name: candidate for candidate in selected}
    actual_names = {path.name for path in files}
    unexpected = sorted(actual_names - set(expected))
    if unexpected:
        raise KeyframeError(f"unexpected files in output directory: {unexpected}")

    if existing_manifest is not None and files:
        if existing_manifest.get("dry_run"):
            raise KeyframeError("existing images conflict with a dry-run manifest")
        if existing_manifest.get("video", {}).get("path") != str(video.path.resolve()):
            raise KeyframeError("existing images were generated from a different video")
        if existing_manifest.get("video", {}).get("file_size") != video.file_size:
            raise KeyframeError("existing video size differs from manifest")
        if existing_manifest.get("video", {}).get("mtime_ns") != video.mtime_ns:
            raise KeyframeError("existing video timestamp differs from manifest")
        if existing_manifest.get("parameters") != parameters:
            raise KeyframeError("existing image parameters differ from the current invocation")
        manifest_outputs = [frame.get("output_name") for frame in existing_manifest.get("keyframes", [])]
        if manifest_outputs != [candidate.output_name for candidate in selected]:
            raise KeyframeError("existing manifest selection differs from the current selection")
        if existing_manifest.get("status") == "complete" and actual_names != set(expected):
            raise KeyframeError("complete manifest does not match existing images")

    for name in sorted(actual_names):
        candidate = expected[name]
        path = output / name
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                size = image.size
        except (OSError, ValueError) as error:
            raise KeyframeError(f"existing output image is invalid: {path}") from error
        if size != (candidate.width, candidate.height):
            raise KeyframeError(
                f"existing output resolution mismatch for {path}: {size} != "
                f"{(candidate.width, candidate.height)}"
            )
    return actual_names


def _parameters_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "target_count": args.target_count,
        "candidate_fps": args.candidate_fps,
        "min_time_gap_sec": args.min_time_gap_sec,
        "jpeg_quality": args.jpeg_quality,
        "seed": args.seed,
        "duplicate_threshold": args.duplicate_threshold,
        "max_underexposed_ratio": args.max_underexposed_ratio,
        "max_overexposed_ratio": args.max_overexposed_ratio,
        "dark_threshold": args.dark_threshold,
        "bright_threshold": args.bright_threshold,
        "preserve_source_resolution": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=250)
    parser.add_argument("--candidate-fps", type=float, default=4.0)
    parser.add_argument("--min-time-gap-sec", type=float, default=0.25)
    parser.add_argument("--jpeg-quality", type=int, default=98)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duplicate-threshold", type=float, default=0.985)
    parser.add_argument("--max-underexposed-ratio", type=float, default=0.45)
    parser.add_argument("--max-overexposed-ratio", type=float, default=0.45)
    parser.add_argument("--dark-threshold", type=int, default=8)
    parser.add_argument("--bright-threshold", type=int, default=247)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_parameters(args)
    dependencies = require_dependencies()
    video = probe_video(args.video, dependencies["ffprobe"])
    output = args.output.expanduser().resolve()
    metadata_directory = output.parent
    manifest_path = metadata_directory / "keyframes.json"
    existing_manifest = _load_manifest(manifest_path)
    existing_files = _existing_output_files(output)
    if args.dry_run and existing_files:
        raise KeyframeError("--dry-run refuses to overwrite metadata for existing output images")
    if existing_files and existing_manifest is None:
        raise KeyframeError("existing output images have no keyframes.json; refusing to continue")

    parameters = _parameters_from_args(args)
    metadata_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".keyframe_candidates.", dir=metadata_directory
    ) as temporary_name:
        candidate_paths = extract_uniform_candidates(
            video.path,
            args.candidate_fps,
            Path(temporary_name),
            dependencies["ffmpeg"],
        )
        candidates = analyze_candidates(
            candidate_paths,
            args.candidate_fps,
            args.dark_threshold,
            args.bright_threshold,
        )
        candidate_sizes = {(candidate.width, candidate.height) for candidate in candidates}
        if len(candidate_sizes) != 1:
            raise KeyframeError(f"candidate resolution changes within the video: {candidate_sizes}")
        candidate_width, candidate_height = next(iter(candidate_sizes))
        if (candidate_width, candidate_height) != (video.width, video.height):
            # FFmpeg applies display-rotation metadata without scaling. Record the
            # actual orientation-correct output resolution used by COLMAP.
            video = replace(video, width=candidate_width, height=candidate_height)
        selected = select_keyframes(
            candidates,
            args.target_count,
            args.min_time_gap_sec,
            args.duplicate_threshold,
            args.max_underexposed_ratio,
            args.max_overexposed_ratio,
            args.seed,
        )
        report = build_selection_report(video, candidates, selected, args.target_count)

        if len(selected) != args.target_count:
            manifest = build_manifest(
                video,
                parameters,
                candidates,
                selected,
                dry_run=args.dry_run,
                status="selection_incomplete",
            )
            write_metadata_artifacts(
                metadata_directory, manifest, report, selected, args.jpeg_quality
            )
            raise KeyframeError(
                f"selected {len(selected)} of requested {args.target_count} frames; "
                "review selection_report.txt and adjust thresholds or target count"
            )

        if args.dry_run:
            manifest = build_manifest(
                video,
                parameters,
                candidates,
                selected,
                dry_run=True,
                status="complete",
            )
            write_metadata_artifacts(
                metadata_directory, manifest, report, selected, args.jpeg_quality
            )
            return manifest

        existing_names = validate_resume_state(
            output, existing_manifest, video, parameters, selected
        )
        output.mkdir(parents=True, exist_ok=True)
        written = set(existing_names)
        manifest = build_manifest(
            video,
            parameters,
            candidates,
            selected,
            dry_run=False,
            status="in_progress",
            written_outputs=sorted(written),
        )
        write_metadata_artifacts(
            metadata_directory, manifest, report, selected, args.jpeg_quality
        )
        for candidate in selected:
            assert candidate.output_name is not None
            destination = output / candidate.output_name
            if candidate.output_name not in written:
                with Image.open(candidate.path) as opened:
                    image = opened.convert("RGB")
                if image.size != (video.width, video.height):
                    raise KeyframeError(
                        f"candidate resolution changed: {image.size} != {(video.width, video.height)}"
                    )
                atomic_write_jpeg(destination, image, args.jpeg_quality)
                written.add(candidate.output_name)
                manifest = build_manifest(
                    video,
                    parameters,
                    candidates,
                    selected,
                    dry_run=False,
                    status="in_progress",
                    written_outputs=sorted(written),
                )
                atomic_write_json(manifest_path, manifest)

        manifest = build_manifest(
            video,
            parameters,
            candidates,
            selected,
            dry_run=False,
            status="complete",
            written_outputs=sorted(written),
        )
        atomic_write_json(manifest_path, manifest)
        return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = run(args)
    except (FileNotFoundError, ValueError, KeyframeError) as error:
        print(f"ERROR: {error}")
        return 2
    mode = "dry-run" if args.dry_run else "write"
    print(
        f"KEYFRAME_EXTRACTION=PASS mode={mode} candidates={manifest['candidate_count']} "
        f"selected={manifest['selected_count']}"
    )
    print(f"manifest={args.output.expanduser().resolve().parent / 'keyframes.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
