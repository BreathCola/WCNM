"""Strict review packaging for frozen DiffusionRenderer glass proposals.

This module does not alter proposal generation.  It validates and packages the
automatic drafts produced by the method frozen at commit a2ee732 for human
review.  None of its manifests are accepted by the training mask loader.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw

from utils.dr_mask_proposal import (
    METHOD_VERSION,
    PROPOSAL_FILES,
    ProposalAuditError,
    _atomic_image,
    _atomic_json,
    decode_normal,
    generate_view_proposal,
    sha256_file,
)


FROZEN_PROPOSAL_COMMIT = "a2ee73212246674b97993ac7affa0e139e44dd5b"
REVIEW_SCHEMA_VERSION = 1
MIN_PROPOSAL_AREA_RATIO = 0.02
MAX_PROPOSAL_AREA_RATIO = 0.60
RETAINED_REVIEW_FILES = (
    "rgb.png",
    "dr_normal.png",
    "dr_depth.png",
    "proposal_overlay.png",
    "uncertainty.png",
    "proposal_metadata.json",
)
BOUNDARY_CROP_NAMES = (
    "bottom_yellow_plate.png",
    "side_edge.png",
    "strong_reflection.png",
    "top_edge.png",
)
RISK_WEIGHTS = {
    "background_may_be_included": 5.0,
    "glass_edge_may_be_missing": 5.0,
    "low_confidence_region": 4.0,
    "reflection_may_be_misclassified": 2.0,
    "bird_may_be_included": 1.0,
}


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_exact(path: Path, mode: str, size: tuple[int, int]) -> np.ndarray:
    if not path.is_file():
        raise ProposalAuditError(f"missing review artifact: {path}")
    try:
        with Image.open(path) as opened:
            if opened.mode != mode:
                raise ProposalAuditError(
                    f"review artifact mode mismatch: {path}: {opened.mode} != {mode}"
                )
            if opened.size != size:
                raise ProposalAuditError(
                    f"review artifact size mismatch: {path}: {opened.size} != {size}"
                )
            values = np.asarray(opened, dtype=np.uint8)
    except (OSError, ValueError) as error:
        if isinstance(error, ProposalAuditError):
            raise
        raise ProposalAuditError(f"cannot parse review artifact {path}: {error}") from error
    return values


def _bbox_from_mask(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ProposalAuditError("proposal is empty")
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _risk_score(metadata: dict[str, Any]) -> tuple[float, list[str]]:
    active = sorted(
        name for name, payload in metadata["risks"].items() if bool(payload["flag"])
    )
    score = sum(RISK_WEIGHTS.get(name, 0.0) for name in active)
    score += 10.0 * float(metadata["uncertainty_fraction_gt_0_55"])
    return float(score), active


def validate_proposal_frames(
    audit: dict[str, Any],
    output_root: Path,
    expected_count: int = 111,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fail closed on any missing, mismatched, padded, or degenerate frame."""
    output_root = Path(output_root)
    proposal_root = output_root / "proposal_soft"
    source_size = tuple(int(value) for value in audit["source_resolution"])
    native_size = tuple(int(value) for value in audit["native_dr_resolution"])
    expected_entries = audit["entries"]
    expected_stems = [entry["stem"] for entry in expected_entries]
    if len(expected_stems) != expected_count or expected_stems != [
        f"{index:06d}" for index in range(expected_count)
    ]:
        raise ProposalAuditError(
            f"strict review packaging requires {expected_count} ordered real stems"
        )
    padding_slots = [int(item["raw_slot"]) for item in audit["padding_slots_excluded"]]
    expected_padding_slots = list(
        range(expected_count, expected_count + len(audit["padding_slots_excluded"]))
    )
    if padding_slots != expected_padding_slots:
        raise ProposalAuditError(f"unexpected padding proof: {padding_slots}")
    if not proposal_root.is_dir():
        raise ProposalAuditError(f"missing proposal root: {proposal_root}")
    actual_stems = sorted(path.name for path in proposal_root.iterdir() if path.is_dir())
    if actual_stems != expected_stems:
        raise ProposalAuditError("proposal stem set does not exactly match 111 real RGB stems")

    frames: list[dict[str, Any]] = []
    metadata_list: list[dict[str, Any]] = []
    for entry in expected_entries:
        stem = entry["stem"]
        if entry.get("is_padding") or int(entry["raw_slot"]) in padding_slots:
            raise ProposalAuditError(f"padding item entered proposal set: {stem}")
        directory = proposal_root / stem
        actual_files = {path.name for path in directory.iterdir() if path.is_file()}
        if actual_files != set(PROPOSAL_FILES):
            missing = sorted(set(PROPOSAL_FILES) - actual_files)
            unexpected = sorted(actual_files - set(PROPOSAL_FILES))
            raise ProposalAuditError(
                f"{stem}: incomplete proposal files; missing={missing}, unexpected={unexpected}"
            )
        try:
            metadata = json.loads((directory / "proposal_metadata.json").read_text("utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ProposalAuditError(f"{stem}: cannot parse proposal metadata: {error}") from error
        if metadata.get("stem") != stem or metadata.get("training_role") is not None:
            raise ProposalAuditError(f"{stem}: metadata stem/training-role mismatch")
        if tuple(metadata.get("source_size", [])) != source_size:
            raise ProposalAuditError(f"{stem}: metadata source size mismatch")
        if tuple(metadata.get("native_dr_size", [])) != native_size:
            raise ProposalAuditError(f"{stem}: metadata DR size mismatch")

        proposal = _read_exact(directory / "proposal_soft.png", "L", source_size)
        _read_exact(directory / "rgb.png", "RGB", source_size)
        _read_exact(directory / "dr_normal.png", "RGB", source_size)
        _read_exact(directory / "dr_depth.png", "RGB", source_size)
        _read_exact(directory / "proposal_overlay.png", "RGB", source_size)
        uncertainty = _read_exact(directory / "uncertainty.png", "L", source_size)
        if int(proposal.min()) != 0 or int(proposal.max()) != 255:
            raise ProposalAuditError(
                f"{stem}: proposal must contain both zero exterior and 255 high-confidence interior"
            )
        hard = proposal >= 128
        area_ratio = float(hard.mean())
        if not MIN_PROPOSAL_AREA_RATIO <= area_ratio <= MAX_PROPOSAL_AREA_RATIO:
            raise ProposalAuditError(
                f"{stem}: abnormal proposal area ratio {area_ratio:.6f} outside "
                f"[{MIN_PROPOSAL_AREA_RATIO}, {MAX_PROPOSAL_AREA_RATIO}]"
            )
        bbox = _bbox_from_mask(hard)
        if abs(area_ratio - float(metadata["area_ratio"])) > 0.002:
            raise ProposalAuditError(f"{stem}: proposal area disagrees with metadata")

        normal_path = Path(entry["artifacts"]["normal"]["path"])
        depth_path = Path(entry["artifacts"]["depth"]["path"])
        with Image.open(normal_path) as opened:
            normal_raw = np.asarray(opened, dtype=np.uint8)
        _, normal_valid = decode_normal(normal_raw)
        invalid_normal = int((~normal_valid).sum())
        if invalid_normal:
            raise ProposalAuditError(f"{stem}: invalid DR normal pixels={invalid_normal}")
        # Depth is an audited RGB uint8 visualization with no invalid sentinel.
        # Every decoded pixel is therefore valid; black remains a relative value.
        invalid_depth = 0
        risk_score, active_risks = _risk_score(metadata)
        uncertainty_ratio = float((uncertainty > 140).mean())
        frame = {
            "stem": stem,
            "frame_index": int(entry["frame_index"]),
            "raw_slot": int(entry["raw_slot"]),
            "is_padding": False,
            "inputs": {
                "rgb": {
                    "path": entry["rgb"]["path"],
                    "sha256": entry["rgb"]["sha256"],
                    "size": list(source_size),
                },
                "dr_normal": {
                    "path": str(normal_path),
                    "sha256": entry["artifacts"]["normal"]["sha256"],
                    "size": list(native_size),
                    "invalid_pixel_count": invalid_normal,
                },
                "dr_depth": {
                    "path": str(depth_path),
                    "sha256": entry["artifacts"]["depth"]["sha256"],
                    "size": list(native_size),
                    "invalid_pixel_count": invalid_depth,
                    "invalid_semantics": "none documented; black retained as relative depth",
                },
            },
            "outputs": {
                "proposal_soft": {
                    "path": str((directory / "proposal_soft.png").resolve()),
                    "sha256": sha256_file(directory / "proposal_soft.png"),
                    "size": list(source_size),
                    "mode": "L",
                    "dtype": "uint8",
                    "min": int(proposal.min()),
                    "max": int(proposal.max()),
                },
                "rgb": {
                    "path": str((directory / "rgb.png").resolve()),
                    "sha256": sha256_file(directory / "rgb.png"),
                    "size": list(source_size),
                },
                "dr_normal": {
                    "path": str((directory / "dr_normal.png").resolve()),
                    "sha256": sha256_file(directory / "dr_normal.png"),
                    "size": list(source_size),
                },
                "dr_depth": {
                    "path": str((directory / "dr_depth.png").resolve()),
                    "sha256": sha256_file(directory / "dr_depth.png"),
                    "size": list(source_size),
                },
                "overlay": {
                    "path": str((directory / "proposal_overlay.png").resolve()),
                    "sha256": sha256_file(directory / "proposal_overlay.png"),
                    "size": list(source_size),
                },
                "uncertainty": {
                    "path": str((directory / "uncertainty.png").resolve()),
                    "sha256": sha256_file(directory / "uncertainty.png"),
                    "size": list(source_size),
                },
            },
            "proposal_area_ratio": area_ratio,
            "bbox_xyxy": bbox,
            "bbox_width_ratio": (bbox[2] - bbox[0]) / source_size[0],
            "bbox_height_ratio": (bbox[3] - bbox[1]) / source_size[1],
            "uncertainty_ratio_gt_140": uncertainty_ratio,
            "metadata_uncertainty_fraction_gt_0_55": float(
                metadata["uncertainty_fraction_gt_0_55"]
            ),
            "risk_score": risk_score,
            "active_risks": active_risks,
            "risks": metadata["risks"],
        }
        frames.append(frame)
        metadata_list.append(metadata)
    return frames, metadata_list


def _quantile_summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "mean": float(array.mean()),
        "p01": float(np.quantile(array, 0.01)),
        "p05": float(np.quantile(array, 0.05)),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


def distribution_summary(frames: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "proposal_area_ratio": _quantile_summary(
            [frame["proposal_area_ratio"] for frame in frames]
        ),
        "uncertainty_ratio_gt_140": _quantile_summary(
            [frame["uncertainty_ratio_gt_140"] for frame in frames]
        ),
        "bbox_width_ratio": _quantile_summary(
            [frame["bbox_width_ratio"] for frame in frames]
        ),
        "bbox_height_ratio": _quantile_summary(
            [frame["bbox_height_ratio"] for frame in frames]
        ),
    }


def _histogram_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    values: Sequence[float],
    title: str,
) -> None:
    x0, y0, x1, y1 = box
    array = np.asarray(values, dtype=np.float64)
    counts, edges = np.histogram(array, bins=18)
    draw.rectangle(box, outline=(130, 130, 130), width=2)
    draw.text((x0 + 10, y0 + 8), title, fill=(245, 245, 245))
    chart_top = y0 + 38
    chart_bottom = y1 - 34
    chart_left = x0 + 34
    chart_right = x1 - 14
    maximum = max(int(counts.max()), 1)
    bar_width = (chart_right - chart_left) / len(counts)
    for index, count in enumerate(counts):
        left = int(chart_left + index * bar_width)
        right = max(left + 1, int(chart_left + (index + 1) * bar_width - 2))
        top = int(chart_bottom - (chart_bottom - chart_top) * count / maximum)
        draw.rectangle((left, top, right, chart_bottom), fill=(44, 190, 180))
    draw.text((chart_left, chart_bottom + 8), f"min {array.min():.4f}", fill=(210, 210, 210))
    label = f"p50 {np.quantile(array, 0.5):.4f}   max {array.max():.4f}"
    draw.text((chart_right - 180, chart_bottom + 8), label, fill=(210, 210, 210))


def make_distribution_plot(frames: Sequence[dict[str, Any]], output: Path) -> None:
    canvas = Image.new("RGB", (1500, 980), (22, 22, 24))
    draw = ImageDraw.Draw(canvas)
    panels = (
        ("proposal_area_ratio", "Proposal area ratio"),
        ("uncertainty_ratio_gt_140", "Uncertainty ratio (>140/255)"),
        ("bbox_width_ratio", "BBox width / image width"),
        ("bbox_height_ratio", "BBox height / image height"),
    )
    for index, (key, title) in enumerate(panels):
        column = index % 2
        row = index // 2
        x0 = 30 + column * 735
        y0 = 30 + row * 470
        values = [float(frame[key]) for frame in frames]
        _histogram_panel(draw, (x0, y0, x0 + 705, y0 + 430), values, title)
    _atomic_image(Path(output), canvas)


def _open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as opened:
        return opened.convert("RGB").copy()


def make_overlay_pages(
    proposal_root: Path,
    ordered_stems: Sequence[str],
    output_directory: Path,
    prefix: str,
    per_page: int = 12,
) -> list[str]:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    page_paths: list[str] = []
    columns = 3
    rows = math.ceil(per_page / columns)
    panel_size = (480, 270)
    header = 28
    margin = 8
    for page_index, start in enumerate(range(0, len(ordered_stems), per_page), 1):
        page_stems = ordered_stems[start : start + per_page]
        width = margin + columns * (panel_size[0] + margin)
        height = margin + rows * (panel_size[1] + header + margin)
        sheet = Image.new("RGB", (width, height), (18, 18, 18))
        draw = ImageDraw.Draw(sheet)
        for index, stem in enumerate(page_stems):
            column = index % columns
            row = index // columns
            x = margin + column * (panel_size[0] + margin)
            y = margin + row * (panel_size[1] + header + margin)
            overlay = _open_rgb(Path(proposal_root) / stem / "proposal_overlay.png")
            overlay.thumbnail(panel_size, Image.Resampling.LANCZOS)
            panel = Image.new("RGB", panel_size, (0, 0, 0))
            panel.paste(
                overlay,
                ((panel_size[0] - overlay.width) // 2, (panel_size[1] - overlay.height) // 2),
            )
            sheet.paste(panel, (x, y + header))
            draw.text((x + 4, y + 6), stem, fill=(240, 240, 240))
        path = output_directory / f"{prefix}_{page_index:02d}.png"
        _atomic_image(path, sheet)
        page_paths.append(str(path.resolve()))
    return page_paths


def _clamped_box(
    center_x: float,
    center_y: float,
    image_size: tuple[int, int],
    crop_size: tuple[int, int] = (1050, 590),
) -> tuple[int, int, int, int]:
    width, height = image_size
    crop_width = min(crop_size[0], width)
    crop_height = min(crop_size[1], height)
    x0 = int(round(center_x - crop_width / 2))
    y0 = int(round(center_y - crop_height / 2))
    x0 = min(max(x0, 0), width - crop_width)
    y0 = min(max(y0, 0), height - crop_height)
    return x0, y0, x0 + crop_width, y0 + crop_height


def _triptych_crop(
    rgb: Image.Image,
    overlay: Image.Image,
    uncertainty: Image.Image,
    box: tuple[int, int, int, int],
    label: str,
) -> Image.Image:
    panel_size = (480, 270)
    header = 30
    margin = 6
    result = Image.new("RGB", (3 * panel_size[0] + 4 * margin, panel_size[1] + header + 2 * margin), (18, 18, 18))
    draw = ImageDraw.Draw(result)
    sources = (rgb.crop(box), overlay.crop(box), uncertainty.crop(box).convert("RGB"))
    labels = ("RGB", "proposal overlay", "uncertainty")
    for index, (source, source_label) in enumerate(zip(sources, labels)):
        panel = source.resize(panel_size, Image.Resampling.LANCZOS)
        x = margin + index * (panel_size[0] + margin)
        result.paste(panel, (x, header + margin))
        draw.text((x + 3, 7), f"{label} | {source_label}", fill=(238, 238, 238))
    return result


def make_boundary_crops(
    proposal_root: Path,
    frames: Sequence[dict[str, Any]],
    output_root: Path,
) -> None:
    for frame in frames:
        source_size = tuple(frame["outputs"]["proposal_soft"]["size"])
        native_size = tuple(frame["inputs"]["dr_normal"]["size"])
        stem = frame["stem"]
        directory = Path(proposal_root) / stem
        rgb = _open_rgb(directory / "rgb.png")
        overlay = _open_rgb(directory / "proposal_overlay.png")
        with Image.open(directory / "uncertainty.png") as opened:
            uncertainty = opened.convert("L").copy()
        bbox = frame["bbox_xyxy"]
        x1, y1, x2, y2 = bbox
        center_x = 0.5 * (x1 + x2)

        uncertainty_small = np.asarray(
            uncertainty.resize(native_size, Image.Resampling.BILINEAR), dtype=np.float32
        ) / 255.0
        native_width, native_height = native_size
        scale_x = native_width / source_size[0]
        scale_y = native_height / source_size[1]
        sx1, sx2 = int(x1 * scale_x), max(int(x2 * scale_x), int(x1 * scale_x) + 1)
        sy1, sy2 = int(y1 * scale_y), max(int(y2 * scale_y), int(y1 * scale_y) + 1)
        strip = max(3, int((sx2 - sx1) * 0.08))
        left_score = float(
            uncertainty_small[sy1:sy2, sx1 : min(sx1 + strip, native_width)].mean()
        )
        right_score = float(uncertainty_small[sy1:sy2, max(sx2 - strip, 0) : sx2].mean())
        side = "left" if left_score >= right_score else "right"
        side_x = x1 if side == "left" else x2

        rgb_small = np.asarray(rgb.resize(native_size, Image.Resampling.BILINEAR), dtype=np.uint8)
        hsv = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2HSV).astype(np.float32)
        highlight = (hsv[..., 2] / 255.0) * (1.0 - 0.65 * hsv[..., 1] / 255.0)
        highlight = 0.75 * highlight + 0.25 * uncertainty_small
        valid = np.zeros((native_height, native_width), dtype=bool)
        valid[
            max(sy1, 0) : min(sy2, native_height),
            max(sx1, 0) : min(sx2, native_width),
        ] = True
        highlight[~valid] = -1.0
        reflection_y, reflection_x = np.unravel_index(np.argmax(highlight), highlight.shape)
        reflection_center = (
            (reflection_x + 0.5) / scale_x,
            (reflection_y + 0.5) / scale_y,
        )

        boxes = {
            "bottom_yellow_plate": _clamped_box(center_x, y2, source_size),
            "side_edge": _clamped_box(side_x, 0.5 * (y1 + y2), source_size),
            "strong_reflection": _clamped_box(*reflection_center, source_size),
            "top_edge": _clamped_box(center_x, y1, source_size),
        }
        crop_directory = Path(output_root) / stem
        for name, box in boxes.items():
            _atomic_image(
                crop_directory / f"{name}.png",
                _triptych_crop(rgb, overlay, uncertainty, box, name),
            )
        _atomic_json(
            crop_directory / "crop_metadata.json",
            {
                "schema_version": REVIEW_SCHEMA_VERSION,
                "stem": stem,
                "source_size": list(source_size),
                "side_selected": side,
                "side_uncertainty_left": left_score,
                "side_uncertainty_right": right_score,
                "boxes_xyxy": {name: list(box) for name, box in boxes.items()},
                "selection_role": "review navigation only; never changes proposal pixels",
            },
        )


def _robust_outlier(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    median = np.median(array)
    mad = np.median(np.abs(array - median))
    if mad <= 1e-12:
        return np.zeros(array.shape, dtype=bool)
    robust_z = 0.6745 * np.abs(array - median) / mad
    return robust_z > 3.5


def make_anomaly_list(frames: Sequence[dict[str, Any]], output_root: Path) -> list[dict[str, Any]]:
    metrics = (
        ("proposal_area_ratio", "area_ratio_outlier"),
        ("uncertainty_ratio_gt_140", "uncertainty_outlier"),
        ("bbox_width_ratio", "bbox_width_outlier"),
        ("bbox_height_ratio", "bbox_height_outlier"),
    )
    tags: dict[str, list[str]] = {frame["stem"]: [] for frame in frames}
    for key, label in metrics:
        flags = _robust_outlier([float(frame[key]) for frame in frames])
        for frame, flag in zip(frames, flags):
            if flag:
                tags[frame["stem"]].append(label)
    for frame in frames:
        for risk in (
            "background_may_be_included",
            "glass_edge_may_be_missing",
            "low_confidence_region",
        ):
            if risk in frame["active_risks"]:
                tags[frame["stem"]].append(risk)
    anomalies = [
        {
            "stem": frame["stem"],
            "tags": tags[frame["stem"]],
            "risk_score": frame["risk_score"],
            "proposal_area_ratio": frame["proposal_area_ratio"],
            "uncertainty_ratio_gt_140": frame["uncertainty_ratio_gt_140"],
            "action": "human review only; do not delete or rewrite automatically",
        }
        for frame in frames
        if tags[frame["stem"]]
    ]
    anomalies.sort(key=lambda item: (-item["risk_score"], -item["uncertainty_ratio_gt_140"], item["stem"]))
    _atomic_json(
        Path(output_root) / "anomalies.json",
        {
            "artifact": "automatic proposal review anomalies",
            "training_role": None,
            "count": len(anomalies),
            "automatic_action": None,
            "items": anomalies,
        },
    )
    lines = [
        "# Automatic proposal review anomalies",
        "",
        "These flags only prioritize human review. No proposal is deleted or rewritten.",
        "",
        "| Rank | Stem | Tags | Risk score | Area | Uncertainty |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for rank, item in enumerate(anomalies, 1):
        lines.append(
            f"| {rank} | {item['stem']} | {', '.join(item['tags'])} | "
            f"{item['risk_score']:.4f} | {item['proposal_area_ratio']:.4f} | "
            f"{item['uncertainty_ratio_gt_140']:.4f} |"
        )
    _atomic_text(Path(output_root) / "anomalies.md", "\n".join(lines) + "\n")
    return anomalies


def make_review_queue(
    proposal_root: Path,
    frames: Sequence[dict[str, Any]],
    output_root: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    ordered = sorted(
        frames,
        key=lambda frame: (
            -float(frame["risk_score"]),
            -float(frame["uncertainty_ratio_gt_140"]),
            frame["stem"],
        ),
    )
    queue = [
        {
            "rank": rank,
            "stem": frame["stem"],
            "risk_score": frame["risk_score"],
            "active_risks": frame["active_risks"],
            "uncertainty_ratio_gt_140": frame["uncertainty_ratio_gt_140"],
            "proposal_area_ratio": frame["proposal_area_ratio"],
            "proposal_directory": str((Path(proposal_root) / frame["stem"]).resolve()),
            "boundary_crop_directory": str(
                (Path(output_root).parent / "boundary_crops" / frame["stem"]).resolve()
            ),
        }
        for rank, frame in enumerate(ordered, 1)
    ]
    _atomic_json(
        Path(output_root) / "review_queue.json",
        {
            "schema_version": REVIEW_SCHEMA_VERSION,
            "artifact": "automatic proposal human review queue",
            "training_role": None,
            "sort": "weighted active risks descending, then uncertainty descending, then stem",
            "risk_weights": RISK_WEIGHTS,
            "count": len(queue),
            "items": queue,
        },
    )
    stream = io.StringIO()
    writer = csv.DictWriter(
        stream,
        fieldnames=(
            "rank", "stem", "risk_score", "active_risks",
            "uncertainty_ratio_gt_140", "proposal_area_ratio",
            "proposal_directory", "boundary_crop_directory",
        ),
    )
    writer.writeheader()
    for item in queue:
        row = dict(item)
        row["active_risks"] = ";".join(row["active_risks"])
        writer.writerow(row)
    _atomic_text(Path(output_root) / "review_queue.csv", stream.getvalue())
    pages = make_overlay_pages(
        proposal_root,
        [item["stem"] for item in queue],
        Path(output_root) / "contact_sheets",
        "review_priority_page",
    )
    return queue, pages


def build_review_package(
    audit: dict[str, Any],
    output_root: Path,
    expected_count: int = 111,
) -> dict[str, Any]:
    output_root = Path(output_root)
    frames, _ = validate_proposal_frames(audit, output_root, expected_count=expected_count)
    proposal_root = output_root / "proposal_soft"
    chronological_pages = make_overlay_pages(
        proposal_root,
        [frame["stem"] for frame in frames],
        output_root / "contact_sheets",
        "overlay_page",
    )
    make_boundary_crops(proposal_root, frames, output_root / "boundary_crops")
    queue, priority_pages = make_review_queue(
        proposal_root, frames, output_root / "review_queue"
    )
    summary = distribution_summary(frames)
    _atomic_json(output_root / "distribution_summary.json", summary)
    make_distribution_plot(frames, output_root / "distribution_plot.png")
    anomalies = make_anomaly_list(frames, output_root)

    manifest = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "artifact": "111-view automatic DR glass proposal review manifest",
        "status": "PASS",
        "training_role": None,
        "warning": "Automatic drafts only; not reviewed_soft and forbidden for L_spec.",
        "method": {
            "version": METHOD_VERSION,
            "frozen_reference_commit": FROZEN_PROPOSAL_COMMIT,
            "per_view_parameter_changes": False,
            "native_domain": list(audit["native_dr_resolution"]),
            "source_domain": list(audit["source_resolution"]),
            "boundary_lift": (
                "The frozen native soft signed-distance proposal and uncertainty/boundary "
                "fields are mapped to source size with cv2 INTER_LINEAR after strict "
                "manifest geometry audit. Hard preview is thresholded only after lift; "
                "the original source RGB is used for overlay and review crops, not for "
                "training or per-frame parameter tuning."
            ),
        },
        "completeness": {
            "expected_real_frames": expected_count,
            "validated_real_frames": len(frames),
            "ordered_stems": [frame["stem"] for frame in frames],
            "all_outputs_source_size": True,
            "all_proposals_mode": "L",
            "all_proposals_dtype": "uint8",
            "all_proposals_have_zero_and_255": True,
        },
        "padding_exclusion_proof": {
            "real_raw_slots": [frame["raw_slot"] for frame in frames],
            "excluded_padding_raw_slots": [
                int(item["raw_slot"]) for item in audit["padding_slots_excluded"]
            ],
            "intersection": [],
            "padding_proposal_count": 0,
        },
        "fail_closed_area_limits": [MIN_PROPOSAL_AREA_RATIO, MAX_PROPOSAL_AREA_RATIO],
        "distribution": summary,
        "review_queue_count": len(queue),
        "anomaly_count": len(anomalies),
        "chronological_contact_sheets": chronological_pages,
        "priority_contact_sheets": priority_pages,
        "reviewed_soft_created": False,
        "formal_training_manifest_created": False,
        "lambda_spec_authorized": False,
        "frames": frames,
    }
    _atomic_json(output_root / "review_manifest_111.json", manifest)
    return manifest


def generate_all_real_proposals(
    audit: dict[str, Any],
    output_root: Path,
    progress=None,
    expected_count: int = 111,
) -> list[dict[str, Any]]:
    """Run the frozen proposal method for all audited real frames in order."""
    entries = audit["entries"]
    if len(entries) != expected_count:
        raise ProposalAuditError(
            f"expected {expected_count} audited real entries, got {len(entries)}"
        )
    metadata = []
    for index, entry in enumerate(entries, 1):
        if entry.get("is_padding"):
            raise ProposalAuditError(f"padding record in real proposal loop: {entry}")
        metadata.append(
            generate_view_proposal(entry, Path(output_root) / "proposal_soft" / entry["stem"])
        )
        if progress is not None:
            progress(index, entry["stem"])
    return metadata
