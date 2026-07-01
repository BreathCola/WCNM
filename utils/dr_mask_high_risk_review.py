"""High-resolution, read-only review packs for selected DR mask proposals."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw

from utils.dr_mask_proposal import ProposalAuditError, _atomic_image, _atomic_json, sha256_file
from utils.dr_mask_review import _atomic_text, _clamped_box


REVIEW_PACK_SCHEMA_VERSION = 1
REVIEW_FIELDS = (
    "outer_boundary_ok",
    "base_included",
    "background_included",
    "edge_missing",
    "reflection_leak",
    "needs_manual_edit",
)
DEFAULT_GROUPS = {
    "background_risk": (
        "000041", "000012", "000040", "000039", "000013",
    ),
    "reflection_risk": (
        "000046", "000002", "000033", "000034", "000010",
        "000011", "000009", "000003", "000000", "000007",
        "000008", "000005", "000004", "000006", "000001",
    ),
    "highest_uncertainty": (
        "000025", "000088", "000087", "000086", "000024",
        "000090", "000092", "000091", "000026", "000093",
    ),
}
CROP_NAMES = (
    "top_edge",
    "bottom_edge_yellow_plate",
    "left_edge",
    "right_edge",
    "base_black_support",
    "strong_reflection",
)


def proposal_tree_digest(proposal_root: Path) -> tuple[str, int]:
    """Hash every proposal file and relative path without changing the tree."""
    proposal_root = Path(proposal_root)
    if not proposal_root.is_dir():
        raise ProposalAuditError(f"missing proposal tree: {proposal_root}")
    files = sorted(path for path in proposal_root.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(proposal_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest(), len(files)


def _validate_groups(groups: Mapping[str, Sequence[str]], available: set[str]) -> list[tuple[str, str]]:
    expected_order = ("background_risk", "reflection_risk", "highest_uncertainty")
    if tuple(groups) != expected_order:
        raise ProposalAuditError(f"review groups must be ordered as {expected_order}")
    ordered: list[tuple[str, str]] = []
    for group in expected_order:
        for stem in groups[group]:
            ordered.append((group, stem))
    stems = [stem for _, stem in ordered]
    if len(stems) != len(set(stems)):
        raise ProposalAuditError("high-risk review groups contain duplicate stems")
    missing = sorted(set(stems) - available)
    if missing:
        raise ProposalAuditError(f"high-risk stems absent from review manifest: {missing}")
    return ordered


def _read_source_rgb(path: Path, size: tuple[int, int]) -> Image.Image:
    try:
        with Image.open(path) as opened:
            if opened.mode != "RGB" or opened.size != size:
                raise ProposalAuditError(
                    f"source RGB mismatch: {path}: mode={opened.mode}, size={opened.size}"
                )
            return opened.copy()
    except OSError as error:
        raise ProposalAuditError(f"cannot parse source RGB {path}: {error}") from error


def _read_image(path: Path, mode: str, size: tuple[int, int]) -> Image.Image:
    try:
        with Image.open(path) as opened:
            if opened.mode != mode or opened.size != size:
                raise ProposalAuditError(
                    f"proposal review input mismatch: {path}: "
                    f"mode={opened.mode}, size={opened.size}"
                )
            return opened.copy()
    except OSError as error:
        raise ProposalAuditError(f"cannot parse proposal review input {path}: {error}") from error


def _boundary_overlay(rgb: Image.Image, soft: Image.Image) -> Image.Image:
    rgb_values = np.asarray(rgb, dtype=np.uint8).copy()
    hard = (np.asarray(soft, dtype=np.uint8) >= 128).astype(np.uint8)
    contours, _ = cv2.findContours(hard, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ProposalAuditError("proposal boundary is empty")
    cv2.drawContours(rgb_values, contours, -1, (255, 35, 210), 7, cv2.LINE_AA)
    return Image.fromarray(rgb_values)


def _uncertainty_overlay(rgb: Image.Image, uncertainty: Image.Image) -> Image.Image:
    base = np.asarray(rgb, dtype=np.float32)
    values = np.asarray(uncertainty, dtype=np.uint8)
    heat_bgr = cv2.applyColorMap(values, cv2.COLORMAP_TURBO)
    heat = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    alpha = (values.astype(np.float32) / 255.0)[..., None] * 0.72
    result = np.round(base * (1.0 - alpha) + heat * alpha).astype(np.uint8)
    return Image.fromarray(result)


def _shrink_only_panel(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    scale = min(size[0] / image.width, size[1] / image.height, 1.0)
    target = (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale))))
    if target != image.size:
        image = image.resize(target, Image.Resampling.LANCZOS)
    panel = Image.new("RGB", size, (8, 8, 8))
    converted = image.convert("RGB")
    panel.paste(converted, ((size[0] - converted.width) // 2, (size[1] - converted.height) // 2))
    return panel


def _draw_labeled_panel(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    image: Image.Image,
    label: str,
    column: int,
    row: int,
    panel_size: tuple[int, int],
    origin_y: int,
    direct_pixels: bool = False,
) -> None:
    margin = 12
    label_height = 28
    x = margin + column * (panel_size[0] + margin)
    y = origin_y + row * (panel_size[1] + label_height + margin)
    if direct_pixels:
        if image.width > panel_size[0] or image.height > panel_size[1]:
            raise ProposalAuditError(
                f"review crop does not fit 1:1 panel: {image.size} > {panel_size}"
            )
        panel = Image.new("RGB", panel_size, (8, 8, 8))
        converted = image.convert("RGB")
        panel.paste(
            converted,
            ((panel_size[0] - converted.width) // 2, (panel_size[1] - converted.height) // 2),
        )
    else:
        panel = _shrink_only_panel(image, panel_size)
    canvas.paste(panel, (x, y + label_height))
    draw.text((x + 4, y + 6), label, fill=(242, 242, 242))


def _reflection_box_from_existing_metadata(
    proposal_output_root: Path,
    stem: str,
    size: tuple[int, int],
) -> tuple[int, int, int, int]:
    path = proposal_output_root / "boundary_crops" / stem / "crop_metadata.json"
    try:
        payload = json.loads(path.read_text("utf-8"))
        box = payload["boxes_xyxy"]["strong_reflection"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ProposalAuditError(f"cannot read strong-reflection crop metadata: {path}") from error
    if len(box) != 4:
        raise ProposalAuditError(f"invalid strong-reflection box: {box}")
    center_x = 0.5 * (float(box[0]) + float(box[2]))
    center_y = 0.5 * (float(box[1]) + float(box[3]))
    return _clamped_box(center_x, center_y, size, crop_size=(900, 506))


def _review_crop_boxes(
    frame: dict[str, Any],
    proposal_output_root: Path,
    source_size: tuple[int, int],
) -> dict[str, tuple[int, int, int, int]]:
    x1, y1, x2, y2 = [int(value) for value in frame["bbox_xyxy"]]
    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)
    bbox_height = y2 - y1
    crop_size = (900, 506)
    return {
        "top_edge": _clamped_box(center_x, y1, source_size, crop_size),
        "bottom_edge_yellow_plate": _clamped_box(center_x, y2, source_size, crop_size),
        "left_edge": _clamped_box(x1, center_y, source_size, crop_size),
        "right_edge": _clamped_box(x2, center_y, source_size, crop_size),
        "base_black_support": _clamped_box(
            center_x, y1 + 0.74 * bbox_height, source_size, crop_size
        ),
        "strong_reflection": _reflection_box_from_existing_metadata(
            proposal_output_root, frame["stem"], source_size
        ),
    }


def _write_view_pack(
    proposal_output_root: Path,
    output_root: Path,
    frame: dict[str, Any],
    group: str,
    order: int,
) -> dict[str, Any]:
    stem = frame["stem"]
    source_size = tuple(int(value) for value in frame["outputs"]["proposal_soft"]["size"])
    source_rgb_path = Path(frame["inputs"]["rgb"]["path"])
    proposal_directory = proposal_output_root / "proposal_soft" / stem
    rgb = _read_source_rgb(source_rgb_path, source_size)
    stored_rgb = _read_image(proposal_directory / "rgb.png", "RGB", source_size)
    if not np.array_equal(np.asarray(rgb), np.asarray(stored_rgb)):
        raise ProposalAuditError(f"{stem}: source RGB decode differs from audited review RGB")
    soft = _read_image(proposal_directory / "proposal_soft.png", "L", source_size)
    overlay = _read_image(proposal_directory / "proposal_overlay.png", "RGB", source_size)
    uncertainty = _read_image(proposal_directory / "uncertainty.png", "L", source_size)
    expected_proposal_sha = frame["outputs"]["proposal_soft"]["sha256"]
    if sha256_file(proposal_directory / "proposal_soft.png") != expected_proposal_sha:
        raise ProposalAuditError(f"{stem}: proposal SHA-256 changed before review packing")

    boundary = _boundary_overlay(rgb, soft)
    uncertainty_overlay = _uncertainty_overlay(rgb, uncertainty)
    crop_boxes = _review_crop_boxes(frame, proposal_output_root, source_size)
    view_root = output_root / "views" / stem
    crop_root = view_root / "crops"
    crop_records: dict[str, Any] = {}
    crop_images: dict[str, Image.Image] = {}
    expected_crop_size = (min(900, source_size[0]), min(506, source_size[1]))
    for name in CROP_NAMES:
        box = crop_boxes[name]
        rgb_crop = rgb.crop(box)
        boundary_crop = boundary.crop(box)
        if rgb_crop.size != expected_crop_size or boundary_crop.size != expected_crop_size:
            raise ProposalAuditError(
                f"{stem}/{name}: crop is not exact source pixels {expected_crop_size}"
            )
        rgb_path = crop_root / f"{name}_rgb.png"
        boundary_path = crop_root / f"{name}_boundary.png"
        _atomic_image(rgb_path, rgb_crop)
        _atomic_image(boundary_path, boundary_crop)
        crop_images[name] = boundary_crop
        crop_records[name] = {
            "box_xyxy_source": list(box),
            "source_pixel_size": list(expected_crop_size),
            "resampling": None,
            "rgb_path": str(rgb_path.resolve()),
            "rgb_sha256": sha256_file(rgb_path),
            "boundary_path": str(boundary_path.resolve()),
            "boundary_sha256": sha256_file(boundary_path),
        }

    panel_size = (900, 506)
    margin = 12
    label_height = 28
    header_height = 112
    rows = 4
    canvas_width = margin + 3 * (panel_size[0] + margin)
    canvas_height = header_height + rows * (panel_size[1] + label_height + margin) + margin
    canvas = Image.new("RGB", (canvas_width, canvas_height), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    active_risks = ", ".join(frame["active_risks"]) or "none"
    draw.text((16, 12), f"{order:02d} | {stem} | {group}", fill=(255, 255, 255))
    draw.text(
        (16, 38),
        f"area={frame['proposal_area_ratio']:.6f}  bbox={frame['bbox_xyxy']}  "
        f"uncertainty={frame['uncertainty_ratio_gt_140']:.6f}",
        fill=(222, 222, 222),
    )
    draw.text((16, 64), f"risks={active_risks}", fill=(255, 190, 90))
    draw.text(
        (16, 88),
        "Automatic proposal only; review pixels are read-only; crop panels are direct source pixels.",
        fill=(190, 210, 255),
    )
    overview = (
        (rgb, "Original RGB"),
        (soft.convert("RGB"), "Proposal soft mask"),
        (overlay, "Proposal overlay"),
        (boundary, "Mask boundary only on RGB"),
        (uncertainty_overlay, "Uncertainty overlay"),
        (uncertainty.convert("RGB"), "Uncertainty grayscale"),
    )
    for index, (image, label) in enumerate(overview):
        _draw_labeled_panel(
            canvas, draw, image, label, index % 3, index // 3,
            panel_size, header_height, direct_pixels=False,
        )
    crop_origin_y = header_height + 2 * (panel_size[1] + label_height + margin)
    for index, name in enumerate(CROP_NAMES):
        _draw_labeled_panel(
            canvas, draw, crop_images[name], f"1:1 source crop | {name}",
            index % 3, index // 3, panel_size, crop_origin_y, direct_pixels=True,
        )
    pack_path = view_root / "review_pack.png"
    _atomic_image(pack_path, canvas)
    metadata = {
        "schema_version": REVIEW_PACK_SCHEMA_VERSION,
        "artifact": "high-risk automatic proposal human review pack",
        "training_role": None,
        "stem": stem,
        "group": group,
        "order": order,
        "source_rgb": {
            "path": str(source_rgb_path.resolve()),
            "sha256": frame["inputs"]["rgb"]["sha256"],
            "size": list(source_size),
        },
        "proposal_soft": {
            "path": str((proposal_directory / "proposal_soft.png").resolve()),
            "sha256": expected_proposal_sha,
            "modified": False,
        },
        "proposal_area_ratio": frame["proposal_area_ratio"],
        "bbox_xyxy": frame["bbox_xyxy"],
        "uncertainty_ratio_gt_140": frame["uncertainty_ratio_gt_140"],
        "active_risks": frame["active_risks"],
        "crops": crop_records,
        "review_pack": {
            "path": str(pack_path.resolve()),
            "sha256": sha256_file(pack_path),
            "size": [canvas_width, canvas_height],
        },
        "reviewed_soft_created": False,
    }
    metadata_path = view_root / "review_pack_metadata.json"
    _atomic_json(metadata_path, metadata)
    metadata["metadata_path"] = str(metadata_path.resolve())
    metadata["metadata_sha256"] = sha256_file(metadata_path)
    return metadata


def _make_contact_sheet(
    proposal_output_root: Path,
    ordered: Sequence[tuple[str, str]],
    frames_by_stem: Mapping[str, dict[str, Any]],
    output: Path,
) -> None:
    columns = 5
    rows = math.ceil(len(ordered) / columns)
    panel_size = (480, 270)
    header = 44
    margin = 8
    width = margin + columns * (panel_size[0] + margin)
    height = margin + rows * (panel_size[1] + header + margin)
    canvas = Image.new("RGB", (width, height), (16, 16, 16))
    draw = ImageDraw.Draw(canvas)
    for index, (group, stem) in enumerate(ordered):
        frame = frames_by_stem[stem]
        path = proposal_output_root / "proposal_soft" / stem / "proposal_overlay.png"
        with Image.open(path) as opened:
            panel = opened.convert("RGB")
            panel.thumbnail(panel_size, Image.Resampling.LANCZOS)
        cell = Image.new("RGB", panel_size, (0, 0, 0))
        cell.paste(panel, ((panel_size[0] - panel.width) // 2, (panel_size[1] - panel.height) // 2))
        column = index % columns
        row = index // columns
        x = margin + column * (panel_size[0] + margin)
        y = margin + row * (panel_size[1] + header + margin)
        canvas.paste(cell, (x, y + header))
        draw.text((x + 3, y + 4), f"{index + 1:02d} {stem}", fill=(245, 245, 245))
        draw.text(
            (x + 3, y + 23),
            f"{group} | area {frame['proposal_area_ratio']:.3f}",
            fill=(245, 190, 95),
        )
    _atomic_image(Path(output), canvas)


def _write_checklists(
    output_root: Path,
    ordered: Sequence[tuple[str, str]],
) -> dict[str, str]:
    items = []
    for index, (group, stem) in enumerate(ordered, 1):
        item = {
            "order": index,
            "group": group,
            "stem": stem,
            "review_pack": str((output_root / "views" / stem / "review_pack.png").resolve()),
            **{field: None for field in REVIEW_FIELDS},
            "notes": None,
        }
        items.append(item)
    json_path = output_root / "review_checklist.json"
    _atomic_json(
        json_path,
        {
            "schema_version": REVIEW_PACK_SCHEMA_VERSION,
            "artifact": "human-fillable proposal review checklist",
            "training_role": None,
            "items": items,
        },
    )
    stream = io.StringIO()
    fields = ("order", "group", "stem", "review_pack", *REVIEW_FIELDS, "notes")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for item in items:
        writer.writerow({key: "" if value is None else value for key, value in item.items()})
    csv_path = output_root / "review_checklist.csv"
    _atomic_text(csv_path, stream.getvalue())
    lines = [
        "# High-risk proposal review checklist",
        "",
        "Fill every field manually. These automatic proposals are not reviewed masks.",
        "",
        "| # | Group | Stem | outer_boundary_ok | base_included | background_included | edge_missing | reflection_leak | needs_manual_edit | Notes |",
        "|---:|---|---|---|---|---|---|---|---|---|",
    ]
    for item in items:
        lines.append(
            f"| {item['order']} | {item['group']} | {item['stem']} |  |  |  |  |  |  |  |"
        )
    md_path = output_root / "review_checklist.md"
    _atomic_text(md_path, "\n".join(lines) + "\n")
    return {
        "json": str(json_path.resolve()),
        "csv": str(csv_path.resolve()),
        "markdown": str(md_path.resolve()),
    }


def generate_high_risk_review_pack(
    proposal_output_root: Path,
    output_root: Path,
    groups: Mapping[str, Sequence[str]] = DEFAULT_GROUPS,
) -> dict[str, Any]:
    proposal_output_root = Path(proposal_output_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    manifest_path = proposal_output_root / "review_manifest_111.json"
    try:
        proposal_manifest = json.loads(manifest_path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProposalAuditError(f"cannot parse 111-view review manifest: {error}") from error
    if proposal_manifest.get("status") != "PASS":
        raise ProposalAuditError("111-view proposal manifest is not PASS")
    frames = proposal_manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ProposalAuditError("111-view proposal manifest has no frames")
    frames_by_stem = {frame["stem"]: frame for frame in frames}
    if len(frames_by_stem) != len(frames):
        raise ProposalAuditError("111-view proposal manifest contains duplicate stems")
    ordered = _validate_groups(groups, set(frames_by_stem))
    if output_root.exists() and any(
        path for path in output_root.iterdir() if path.name != "run_state.json"
    ):
        raise ProposalAuditError(f"high-risk review output must be new or empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    if (proposal_output_root / "reviewed_soft").exists():
        raise ProposalAuditError("proposal source unexpectedly contains reviewed_soft")

    digest_before, source_file_count = proposal_tree_digest(
        proposal_output_root / "proposal_soft"
    )
    records = []
    for index, (group, stem) in enumerate(ordered, 1):
        records.append(
            _write_view_pack(
                proposal_output_root, output_root, frames_by_stem[stem], group, index
            )
        )
    contact_sheet = output_root / "high_risk_contact_sheet.png"
    _make_contact_sheet(proposal_output_root, ordered, frames_by_stem, contact_sheet)
    checklists = _write_checklists(output_root, ordered)
    digest_after, final_source_file_count = proposal_tree_digest(
        proposal_output_root / "proposal_soft"
    )
    if digest_after != digest_before or final_source_file_count != source_file_count:
        raise ProposalAuditError("proposal file tree changed during read-only review packing")
    if (output_root / "reviewed_soft").exists():
        raise ProposalAuditError("review packing must not create reviewed_soft")
    manifest = {
        "schema_version": REVIEW_PACK_SCHEMA_VERSION,
        "artifact": "high-risk automatic proposal human review package",
        "status": "PASS",
        "training_role": None,
        "source_proposal_root": str(proposal_output_root),
        "source_review_manifest": str(manifest_path.resolve()),
        "source_review_manifest_sha256": sha256_file(manifest_path),
        "proposal_tree_file_count": source_file_count,
        "proposal_tree_sha256_before": digest_before,
        "proposal_tree_sha256_after": digest_after,
        "proposal_files_modified": False,
        "ordered_groups": {name: list(stems) for name, stems in groups.items()},
        "view_count": len(records),
        "contact_sheet": {
            "path": str(contact_sheet.resolve()),
            "sha256": sha256_file(contact_sheet),
        },
        "checklists": checklists,
        "views": records,
        "reviewed_soft_created": False,
        "formal_training_manifest_created": False,
        "lambda_spec_authorized": False,
    }
    _atomic_json(output_root / "review_pack_manifest.json", manifest)
    return manifest
