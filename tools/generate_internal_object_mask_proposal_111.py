#!/usr/bin/env python3
"""Generate D-016b 111-view internal-object proposal from reviewed anchors.

This creates proposal/review artifacts only. It never writes formal reviewed
masks and never runs RT-GS training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.internal_object_mask import (
    EXPECTED_STEMS,
    INTERNAL_OBJECT_SEMANTICS_VERSION,
    PROPOSAL_ROLE,
    SCHEMA_VERSION,
    canonical_payload_sha256,
    object_occupancy_domains,
    sha256_file,
)
from utils.specular_mask import load_resized_formal_mask, validate_specular_mask_set


DEFAULT_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
FIXED_NINE = ("000000", "000014", "000028", "000042", "000055", "000069", "000083", "000097", "000110")
OBJECTS = {"bird": 1, "yellow_base_board": 2, "white_platform": 3}
FINAL_ROLES = ("bird", "internal_base", "internal_object_union")

ENGINEERING_GUARDS = {
    "significant_component_min_pixels": 64,
    "significant_component_min_area_ratio": 0.002,
    "distant_component_min_distance_px": 96,
    "forward_backward_iou_review": 0.75,
    "neighbor_iou_review": 0.70,
    "area_jump_high": 1.8,
    "area_jump_low": 0.55,
    "centroid_jump_fraction": 0.08,
    "raw_outside_glass_ratio_review": 0.20,
    "glass_iou_review": 0.75,
}

COLORS = {
    "bird": np.array([255, 35, 35], dtype=np.float32),
    "internal_base": np.array([35, 120, 255], dtype=np.float32),
    "internal_object_union": np.array([255, 225, 35], dtype=np.float32),
    "glass_hard": np.array([0, 220, 230], dtype=np.float32),
    "Mpos": np.array([255, 35, 35], dtype=np.float32),
    "Mignore": np.array([230, 45, 230], dtype=np.float32),
    "Mneg": np.array([0, 220, 230], dtype=np.float32),
}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _binary_png(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.asarray(mask, dtype=bool).astype(np.uint8) * 255), mode="L").save(path)


def _load_mask(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode != "L":
            raise ValueError(f"mask must be L mode: {path}")
        if size is not None and image.size != size:
            raise ValueError(f"mask size mismatch: {path}: {image.size} != {size}")
        values = np.asarray(image)
    if values.dtype != np.uint8:
        raise ValueError(f"mask dtype must be uint8: {path}")
    return values >= 128


def _mask_sha(path: Path) -> str:
    return sha256_file(path)


def _git_head(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def _repo_head() -> str:
    return _git_head(ROOT)


def default_grounded_sam2_project() -> Path:
    return Path(os.environ.get("GROUNDED_SAM2_ROOT", ROOT.parent / "Grounded-SAM-2"))


def ensure_new_output_dir(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"BLOCKED_BY_EXISTING_OUTPUT: {output}")
    output.mkdir(parents=True)


def _bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _centroid(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    return [float(xs.mean()), float(ys.mean())]


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    union = int((a | b).sum())
    return float((a & b).sum() / union) if union else 1.0


def significant_component_stats(
    mask: np.ndarray,
    reference: np.ndarray | None = None,
    guards: dict | None = None,
) -> dict:
    guards = {**ENGINEERING_GUARDS, **(guards or {})}
    binary = np.asarray(mask, dtype=bool)
    total = int(binary.sum())
    labels_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), 8
    )
    threshold = max(
        int(guards["significant_component_min_pixels"]),
        int(math.ceil(total * float(guards["significant_component_min_area_ratio"]))),
    )
    components = []
    significant = []
    distant = []
    ref_points = None
    if reference is not None and np.asarray(reference, dtype=bool).any():
        ref_points = np.column_stack(np.where(np.asarray(reference, dtype=bool)))
    for label in range(1, labels_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        comp = labels == label
        centroid = [float(centroids[label][0]), float(centroids[label][1])]
        distance = None
        if ref_points is not None and comp.any():
            points = np.column_stack(np.where(comp))
            step = max(1, int(max(len(points), len(ref_points)) / 2048))
            delta = points[::step, None, :] - ref_points[None, ::step, :]
            distance = float(np.sqrt((delta * delta).sum(axis=-1)).min())
        row = {"area": area, "area_ratio": float(area / max(total, 1)), "centroid_xy": centroid, "distance_to_reference_px": distance}
        components.append(row)
        if area >= threshold:
            significant.append(row)
            if distance is not None and distance > float(guards["distant_component_min_distance_px"]):
                distant.append(row)
    largest = max((row["area"] for row in components), default=0)
    secondary = max(total - largest, 0)
    return {
        "raw_component_count": int(max(labels_count - 1, 0)),
        "significant_component_count": len(significant),
        "largest_component_ratio": float(largest / max(total, 1)),
        "secondary_component_total_ratio": float(secondary / max(total, 1)),
        "distant_significant_component_count": len(distant),
        "distant_significant_component_area_ratio": float(sum(row["area"] for row in distant) / max(total, 1)),
        "component_area_threshold": threshold,
        "components": components,
    }


def choose_bidirectional_candidate(
    forward: np.ndarray,
    backward: np.ndarray,
    *,
    anchor_distance_left: int,
    anchor_distance_right: int,
    guards: dict | None = None,
) -> tuple[np.ndarray, dict]:
    guards = {**ENGINEERING_GUARDS, **(guards or {})}
    iou = mask_iou(forward, backward)
    if anchor_distance_left <= anchor_distance_right:
        selected = np.asarray(forward, dtype=bool)
        reason = "forward_nearer_or_tie"
    else:
        selected = np.asarray(backward, dtype=bool)
        reason = "backward_nearer"
    flags = []
    if iou < float(guards["forward_backward_iou_review"]):
        flags.append("low_forward_backward_iou")
    return selected, {
        "selection_score": iou,
        "selection_reason": reason,
        "forward_backward_iou": iou,
        "review_flags": flags,
    }


def _overlay(rgb: np.ndarray, mask: np.ndarray, color: str, alpha: float = 0.45) -> Image.Image:
    out = rgb.astype(np.float32).copy()
    m = np.asarray(mask, dtype=bool)
    out[m] = (1.0 - alpha) * out[m] + alpha * COLORS[color]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def _panel(image: Image.Image, title: str, width: int = 360) -> Image.Image:
    font = ImageFont.load_default()
    img = image.convert("RGB").copy()
    img.thumbnail((width, int(width * img.height / max(img.width, 1))))
    canvas = Image.new("RGB", (width, img.height + 24), (18, 18, 18))
    canvas.paste(img, ((width - img.width) // 2, 24))
    ImageDraw.Draw(canvas).text((6, 6), title, fill=(255, 255, 255), font=font)
    return canvas


def _hstack(images: list[Image.Image]) -> Image.Image:
    canvas = Image.new("RGB", (sum(i.width for i in images), max(i.height for i in images)), (0, 0, 0))
    x = 0
    for image in images:
        canvas.paste(image, (x, 0))
        x += image.width
    return canvas


def _vstack(images: list[Image.Image]) -> Image.Image:
    canvas = Image.new("RGB", (max(i.width for i in images), sum(i.height for i in images)), (0, 0, 0))
    y = 0
    for image in images:
        canvas.paste(image, (0, y))
        y += image.height
    return canvas


def _line_plot(values: list[float], title: str, path: Path, width: int = 900, height: int = 240) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), title, fill=(0, 0, 0))
    if values:
        vmin, vmax = min(values), max(values)
        if abs(vmax - vmin) < 1e-12:
            vmax = vmin + 1.0
        pts = []
        for i, value in enumerate(values):
            x = 30 + i * (width - 60) / max(len(values) - 1, 1)
            y = height - 30 - (value - vmin) * (height - 70) / (vmax - vmin)
            pts.append((x, y))
        if len(pts) > 1:
            draw.line(pts, fill=(40, 80, 220), width=2)
        for x, y in pts:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(220, 40, 40))
    canvas.save(path)


def _domains(union: np.ndarray, glass: np.ndarray) -> dict[str, np.ndarray]:
    domains = object_occupancy_domains(
        torch.from_numpy(union[..., None].astype(np.float32)),
        torch.from_numpy(glass[..., None].astype(np.float32)),
        torch.from_numpy(glass[..., None].astype(np.float32)),
        erode_px=3,
        dilate_px=3,
    )
    return {key: (value[..., 0].detach().cpu().numpy() >= 0.5) for key, value in domains.items()}


def make_fixed_nine_human_review(fixed_root: Path, output_root: Path, repo_head: str) -> dict:
    manifest_path = fixed_root / "proposal_manifest.json"
    selection_path = fixed_root / "candidate_selection.json"
    summary_path = fixed_root / "proposal_summary.json"
    template_path = fixed_root / "review_selection_template.json"
    for path in (manifest_path, selection_path, summary_path, template_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = _json(manifest_path)
    selection = _json(selection_path)
    if manifest.get("internal_object_semantics_version") != INTERNAL_OBJECT_SEMANTICS_VERSION:
        raise ValueError("fixed-nine proposal semantic version mismatch")
    accepted = {}
    for stem in FIXED_NINE:
        auto = selection[stem]["auto_suggestion"]
        masks = {
            "bird": fixed_root / "processed" / "bird" / f"{stem}.png",
            "yellow_base_board": fixed_root / "raw" / "yellow_base_board" / f"{stem}.png",
            "white_platform": fixed_root / "raw" / "white_platform" / f"{stem}.png",
            "internal_base": fixed_root / "processed" / "internal_base" / f"{stem}.png",
            "internal_object_union": fixed_root / "processed" / "internal_object_union" / f"{stem}.png",
        }
        accepted[stem] = {
            "status": "accepted",
            "bird_candidate_id": auto["bird_candidate_id"],
            "yellow_base_board_candidate_id": auto["yellow_base_board_candidate_id"],
            "white_platform_candidate_id": auto["white_platform_candidate_id"],
            "connected_fixture_candidate_id": None,
            "bird_ok": True,
            "yellow_base_board_ok": True,
            "white_platform_ok": True,
            "isolated_pole_included": False,
            "glass_included": False,
            "outside_ground_included": False,
            "reflection_included": False,
            "automated_warnings_preserved": selection[stem].get("quality_guards", []),
            "mask_sha256": {role: _mask_sha(path) for role, path in masks.items()},
            "mask_paths": {role: str(path) for role, path in masks.items()},
        }
    payload = {
        "schema": "rtgs_d016b_fixed_nine_human_review_v3",
        "semantic_version": INTERNAL_OBJECT_SEMANTICS_VERSION,
        "review_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repository_head": repo_head,
        "review_authorization": {
            "type": "explicit_user_approval",
            "source": "current_codex_task_prompt",
        },
        "proposal_manifest_sha256": sha256_file(manifest_path),
        "candidate_selection_sha256": sha256_file(selection_path),
        "proposal_summary_sha256": sha256_file(summary_path),
        "accepted_stems": list(FIXED_NINE),
        "entries": accepted,
    }
    payload["payload_sha256"] = canonical_payload_sha256(payload)
    _write_json(output_root / "fixed_nine_human_review_v3.json", payload)
    return payload


def _import_video_predictor(project: Path):
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    from sam2.build_sam import build_sam2_video_predictor
    return build_sam2_video_predictor


def _propagate_interval(
    predictor,
    image_root: Path,
    start_idx: int,
    end_idx: int,
    masks: dict[str, np.ndarray],
    *,
    reverse: bool,
) -> dict[int, dict[str, np.ndarray]]:
    state = predictor.init_state(
        video_path=str(image_root),
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
        async_loading_frames=False,
    )
    anchor_idx = end_idx if reverse else start_idx
    predictor.reset_state(state)
    for name, obj_id in OBJECTS.items():
        predictor.add_new_mask(state, anchor_idx, obj_id, masks[name])
    out = {}
    with torch.inference_mode():
        for frame_idx, obj_ids, logits in predictor.propagate_in_video(
            state,
            start_frame_idx=anchor_idx,
            max_frame_num_to_track=abs(end_idx - start_idx),
            reverse=reverse,
        ):
            if frame_idx < start_idx or frame_idx > end_idx:
                continue
            frame = {}
            for i, obj_id in enumerate(obj_ids):
                name = {value: key for key, value in OBJECTS.items()}[int(obj_id)]
                frame[name] = (logits[i] > 0.0)[0].detach().cpu().numpy().astype(bool)
            out[int(frame_idx)] = frame
    return out


def _copy_anchor_masks(review: dict, fixed_root: Path, output: Path) -> dict[str, dict[str, np.ndarray]]:
    anchors = {}
    for stem in FIXED_NINE:
        entry = review["entries"][stem]
        masks = {
            "bird": _load_mask(Path(entry["mask_paths"]["bird"])),
            "yellow_base_board": _load_mask(Path(entry["mask_paths"]["yellow_base_board"])),
            "white_platform": _load_mask(Path(entry["mask_paths"]["white_platform"])),
        }
        anchors[stem] = masks
        for name, mask in masks.items():
            dst = output / "anchors" / name / f"{stem}.png"
            _binary_png(dst, mask)
            if sha256_file(dst) != entry["mask_sha256"][name]:
                raise ValueError(f"anchor hash changed while copying {stem} {name}")
    return anchors


def _frame_rgb(image_root: Path, stem: str) -> tuple[np.ndarray, tuple[int, int]]:
    with Image.open(image_root / f"{stem}.jpg") as image:
        rgb = np.asarray(image.convert("RGB"))
        return rgb, image.size


def _write_frame_review_page(path: Path, rgb: np.ndarray, masks: dict, domains: dict, metrics: dict) -> None:
    panels = [
        _panel(Image.fromarray(rgb), f"{metrics['stem']} original"),
        _panel(_overlay(rgb, masks["glass_hard"], "glass_hard"), "glass"),
        _panel(_overlay(rgb, masks["forward_bird"], "bird"), "forward bird"),
        _panel(_overlay(rgb, masks["backward_bird"], "bird"), "backward bird"),
        _panel(_overlay(rgb, masks["bird"], "bird"), "selected bird"),
        _panel(_overlay(rgb, masks["forward_yellow_base_board"], "internal_base"), "forward yellow"),
        _panel(_overlay(rgb, masks["backward_yellow_base_board"], "internal_base"), "backward yellow"),
        _panel(_overlay(rgb, masks["yellow_base_board"], "internal_base"), "selected yellow"),
        _panel(_overlay(rgb, masks["forward_white_platform"], "internal_base"), "forward white"),
        _panel(_overlay(rgb, masks["backward_white_platform"], "internal_base"), "backward white"),
        _panel(_overlay(rgb, masks["white_platform"], "internal_base"), "selected white"),
        _panel(_overlay(rgb, masks["internal_base"], "internal_base"), "internal_base"),
        _panel(_overlay(rgb, masks["internal_object_union"], "internal_object_union"), "union"),
        _panel(_overlay(rgb, domains["Mpos"], "Mpos"), "Mpos preview"),
        _panel(_overlay(rgb, domains["Mignore"], "Mignore"), "Mignore preview"),
        _panel(_overlay(rgb, domains["Mneg"], "Mneg"), "Mneg preview"),
    ]
    rows = [_hstack(panels[i:i + 4]) for i in range(0, len(panels), 4)]
    page = _vstack(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    page.save(path)


def _make_contact_pages(rows: list[dict], image_root: Path, output: Path, page_size: int = 24) -> None:
    for page_idx in range(0, len(rows), page_size):
        chunk = rows[page_idx:page_idx + page_size]
        overlay_rows, binary_rows, domain_rows = [], [], []
        for row in chunk:
            stem = row["stem"]
            rgb, _ = _frame_rgb(image_root, stem)
            bird = _load_mask(output / "processed" / "bird" / f"{stem}.png")
            base = _load_mask(output / "processed" / "internal_base" / f"{stem}.png")
            union = _load_mask(output / "processed" / "internal_object_union" / f"{stem}.png")
            glass = _load_mask(output / "processed" / "glass_hard" / f"{stem}.png")
            mpos = _load_mask(output / "domain_previews" / "Mpos" / f"{stem}.png")
            mignore = _load_mask(output / "domain_previews" / "Mignore" / f"{stem}.png")
            mneg = _load_mask(output / "domain_previews" / "Mneg" / f"{stem}.png")
            status = "anchor_accepted" if stem in FIXED_NINE else row["frame_status"]
            overlay_rows.append(_hstack([
                _panel(Image.fromarray(rgb), f"{stem} {status}", 240),
                _panel(_overlay(rgb, bird, "bird"), "Bird", 240),
                _panel(_overlay(rgb, base, "internal_base"), "Internal Base", 240),
                _panel(_overlay(rgb, union, "internal_object_union"), "Union", 240),
                _panel(_overlay(rgb, glass, "glass_hard"), "Glass", 240),
            ]))
            binary_rows.append(_hstack([
                _panel(Image.fromarray((bird.astype(np.uint8) * 255), mode="L").convert("RGB"), f"{stem} Bird", 240),
                _panel(Image.fromarray((base.astype(np.uint8) * 255), mode="L").convert("RGB"), "Base", 240),
                _panel(Image.fromarray((union.astype(np.uint8) * 255), mode="L").convert("RGB"), "Union", 240),
                _panel(Image.fromarray((glass.astype(np.uint8) * 255), mode="L").convert("RGB"), "Glass", 240),
            ]))
            domain_rows.append(_hstack([
                _panel(Image.fromarray(rgb), f"{stem} {status}", 240),
                _panel(_overlay(rgb, mpos, "Mpos"), "Mpos", 240),
                _panel(_overlay(rgb, mignore, "Mignore"), "Mignore", 240),
                _panel(_overlay(rgb, mneg, "Mneg"), "Mneg", 240),
            ]))
        suffix = f"{page_idx // page_size:03d}"
        _vstack(overlay_rows).save(output / "contact_sheets" / f"all_111_overlay_contact_sheet_{suffix}.png")
        _vstack(binary_rows).save(output / "contact_sheets" / f"all_111_binary_contact_sheet_{suffix}.png")
        _vstack(domain_rows).save(output / "contact_sheets" / f"all_111_domain_contact_sheet_{suffix}.png")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default=str(ROOT / "data/TiHuBird"))
    parser.add_argument("--images", default="images")
    parser.add_argument("--glass-mask-manifest", default="specular_masks_reviewed_v1/manifest.json")
    parser.add_argument("--fixed-nine-root", default=str(ROOT / "output/stage_d_tihubird_internal_object_mask_proposal_fixednine_probe_v3"))
    parser.add_argument("--output", default=str(ROOT / "output/stage_d_tihubird_internal_object_mask_proposal_111_v3"))
    default_gsam2 = default_grounded_sam2_project()
    parser.add_argument("--grounded-sam2-project", default=str(default_gsam2))
    parser.add_argument("--sam2-checkpoint", default=str(default_gsam2 / "checkpoints/sam2.1_hiera_large.pt"))
    parser.add_argument("--sam2-config", default=DEFAULT_SAM2_CONFIG)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260718)
    args = parser.parse_args()

    scene = Path(args.scene).resolve()
    image_root = (scene / args.images).resolve()
    output = Path(args.output).resolve()
    fixed_root = Path(args.fixed_nine_root).resolve()
    ensure_new_output_dir(output)
    for directory in (
        "anchors/bird", "anchors/yellow_base_board", "anchors/white_platform",
        "propagation/forward/bird", "propagation/forward/yellow_base_board", "propagation/forward/white_platform",
        "propagation/backward/bird", "propagation/backward/yellow_base_board", "propagation/backward/white_platform",
        "selected_raw/bird", "selected_raw/yellow_base_board", "selected_raw/white_platform",
        "processed/bird", "processed/internal_base", "processed/internal_object_union", "processed/internal_ignore_preview", "processed/glass_hard",
        "overlays", "domain_previews/Mpos", "domain_previews/Mignore", "domain_previews/Mneg",
        "review_pages", "high_risk_review_pages", "contact_sheets", "temporal_metrics_plots",
    ):
        (output / directory).mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    image_files = sorted(path for path in image_root.iterdir() if path.suffix.lower() in (".jpg", ".jpeg", ".png"))
    stems = [path.stem for path in image_files]
    if stems != EXPECTED_STEMS:
        raise ValueError("111-view proposal requires exactly stems 000000--000110")
    glass_manifest = validate_specular_mask_set(scene, args.images, args.glass_mask_manifest)
    repo_head = _repo_head()
    review = make_fixed_nine_human_review(fixed_root, output, repo_head)
    anchors = _copy_anchor_masks(review, fixed_root, output)

    project = Path(args.grounded_sam2_project).resolve()
    build_predictor = _import_video_predictor(project)
    predictor = build_predictor(args.sam2_config, args.sam2_checkpoint, device=args.device)
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    forward = {stem: {} for stem in EXPECTED_STEMS}
    backward = {stem: {} for stem in EXPECTED_STEMS}
    intervals = []
    anchor_indices = [int(stem) for stem in FIXED_NINE]
    for left, right in zip(anchor_indices[:-1], anchor_indices[1:]):
        left_stem, right_stem = f"{left:06d}", f"{right:06d}"
        intervals.append({"left": left_stem, "right": right_stem, "range": [left, right]})
        fwd = _propagate_interval(predictor, image_root, left, right, anchors[left_stem], reverse=False)
        bwd = _propagate_interval(predictor, image_root, left, right, anchors[right_stem], reverse=True)
        for frame_idx in range(left, right + 1):
            stem = f"{frame_idx:06d}"
            for name in OBJECTS:
                forward[stem][name] = fwd.get(frame_idx, {}).get(name, np.zeros_like(anchors[left_stem][name]))
                backward[stem][name] = bwd.get(frame_idx, {}).get(name, np.zeros_like(anchors[right_stem][name]))
                _binary_png(output / "propagation" / "forward" / name / f"{stem}.png", forward[stem][name])
                _binary_png(output / "propagation" / "backward" / name / f"{stem}.png", backward[stem][name])

    rows = []
    previous_union = None
    previous_centroid = None
    previous_bbox = None
    previous_area = None
    review_queue = []
    raw_masks = {}
    for frame_idx, stem in enumerate(EXPECTED_STEMS):
        rgb, size = _frame_rgb(image_root, stem)
        glass_values, _, _ = load_resized_formal_mask(glass_manifest, stem, size, size)
        glass = glass_values >= 0.5
        selected = {}
        object_metrics = {}
        if stem in FIXED_NINE:
            selected = {name: anchors[stem][name] for name in OBJECTS}
            for name in OBJECTS:
                object_metrics[name] = {
                    "selection_score": 1.0,
                    "selection_reason": "anchor_exact_reuse",
                    "forward_backward_iou": 1.0,
                    "review_flags": [],
                }
        else:
            left = max(idx for idx in anchor_indices if idx < frame_idx)
            right = min(idx for idx in anchor_indices if idx > frame_idx)
            for name in OBJECTS:
                selected[name], object_metrics[name] = choose_bidirectional_candidate(
                    forward[stem][name],
                    backward[stem][name],
                    anchor_distance_left=frame_idx - left,
                    anchor_distance_right=right - frame_idx,
                )
        bird_raw = selected["bird"]
        yellow_raw = selected["yellow_base_board"]
        white_raw = selected["white_platform"]
        base_raw = yellow_raw | white_raw
        union_raw = bird_raw | base_raw
        bird = bird_raw & glass
        base = base_raw & glass
        union = union_raw & glass
        domains = _domains(union, glass)
        for name, mask in selected.items():
            _binary_png(output / "selected_raw" / name / f"{stem}.png", mask)
        for role, mask in {
            "bird": bird,
            "internal_base": base,
            "internal_object_union": union,
            "internal_ignore_preview": domains["Mignore"],
            "glass_hard": glass,
        }.items():
            _binary_png(output / "processed" / role / f"{stem}.png", mask)
        for role in ("Mpos", "Mignore", "Mneg"):
            _binary_png(output / "domain_previews" / role / f"{stem}.png", domains[role])
        if stem in FIXED_NINE:
            if sha256_file(output / "processed" / "bird" / f"{stem}.png") != review["entries"][stem]["mask_sha256"]["bird"]:
                raise ValueError(f"anchor bird parity failed: {stem}")
            if sha256_file(output / "processed" / "internal_base" / f"{stem}.png") != review["entries"][stem]["mask_sha256"]["internal_base"]:
                raise ValueError(f"anchor internal_base parity failed: {stem}")
            if sha256_file(output / "processed" / "internal_object_union" / f"{stem}.png") != review["entries"][stem]["mask_sha256"]["internal_object_union"]:
                raise ValueError(f"anchor union parity failed: {stem}")
        raw_outside = float(((union_raw & ~glass).sum()) / max(union_raw.sum(), 1))
        comp = significant_component_stats(union, reference=bird | base)
        flags = []
        for name, meta in object_metrics.items():
            flags.extend(f"{name}:{flag}" for flag in meta["review_flags"])
        if raw_outside > ENGINEERING_GUARDS["raw_outside_glass_ratio_review"]:
            flags.append("raw_outside_glass_ratio_high")
        if comp["distant_significant_component_count"] > 0:
            flags.append("distant_significant_component")
        neighbor_iou = mask_iou(previous_union, union) if previous_union is not None else 1.0
        centroid = _centroid(union)
        bbox = _bbox(union)
        centroid_jump = 0.0
        if previous_centroid and centroid:
            centroid_jump = math.dist(previous_centroid, centroid) / max(size)
            if centroid_jump > ENGINEERING_GUARDS["centroid_jump_fraction"]:
                flags.append("centroid_jump")
        bbox_displacement = 0.0
        if previous_bbox and bbox:
            bbox_displacement = max(abs(a - b) for a, b in zip(previous_bbox, bbox)) / max(size)
        area_ratio_previous = 1.0
        if previous_area:
            area_ratio_previous = float(union.sum() / max(previous_area, 1))
            if area_ratio_previous > ENGINEERING_GUARDS["area_jump_high"]:
                flags.append("area_jump_high")
            if area_ratio_previous < ENGINEERING_GUARDS["area_jump_low"]:
                flags.append("area_jump_low")
        if previous_union is not None and neighbor_iou < ENGINEERING_GUARDS["neighbor_iou_review"]:
            flags.append("low_previous_neighbor_iou")
        status = "anchor_accepted" if stem in FIXED_NINE else ("review_required" if flags else "auto_candidate_ready")
        if flags and stem not in FIXED_NINE:
            review_queue.append({"stem": stem, "status": status, "reasons": ";".join(sorted(set(flags)))})
        row = {
            "stem": stem,
            "frame_status": status,
            "bird_area": int(bird.sum()),
            "internal_base_area": int(base.sum()),
            "internal_object_union_area": int(union.sum()),
            "union_glass_ratio": float(union.sum() / max(glass.sum(), 1)),
            "raw_outside_glass_ratio": raw_outside,
            "neighbor_iou_previous": neighbor_iou,
            "area_ratio_previous": area_ratio_previous,
            "centroid_jump_fraction": centroid_jump,
            "bbox_displacement_fraction": bbox_displacement,
            "raw_component_count": comp["raw_component_count"],
            "significant_component_count": comp["significant_component_count"],
            "largest_component_ratio": comp["largest_component_ratio"],
            "secondary_component_total_ratio": comp["secondary_component_total_ratio"],
            "distant_significant_component_count": comp["distant_significant_component_count"],
            "distant_significant_component_area_ratio": comp["distant_significant_component_area_ratio"],
            "bird_forward_backward_iou": object_metrics["bird"]["forward_backward_iou"],
            "yellow_forward_backward_iou": object_metrics["yellow_base_board"]["forward_backward_iou"],
            "white_forward_backward_iou": object_metrics["white_platform"]["forward_backward_iou"],
            "review_flags": ";".join(sorted(set(flags))),
            "bird_sha256": sha256_file(output / "processed" / "bird" / f"{stem}.png"),
            "internal_base_sha256": sha256_file(output / "processed" / "internal_base" / f"{stem}.png"),
            "internal_object_union_sha256": sha256_file(output / "processed" / "internal_object_union" / f"{stem}.png"),
            "glass_sha256": glass_manifest["entries"][stem]["sha256"],
            "rgb_sha256": sha256_file(image_root / f"{stem}.jpg"),
        }
        rows.append(row)
        raw_masks[stem] = {
            "forward_bird": forward[stem].get("bird", bird_raw),
            "backward_bird": backward[stem].get("bird", bird_raw),
            "forward_yellow_base_board": forward[stem].get("yellow_base_board", yellow_raw),
            "backward_yellow_base_board": backward[stem].get("yellow_base_board", yellow_raw),
            "forward_white_platform": forward[stem].get("white_platform", white_raw),
            "backward_white_platform": backward[stem].get("white_platform", white_raw),
            "bird": bird,
            "yellow_base_board": yellow_raw & glass,
            "white_platform": white_raw & glass,
            "internal_base": base,
            "internal_object_union": union,
            "glass_hard": glass,
        }
        overlay = Image.fromarray(rgb)
        overlay = _overlay(np.asarray(overlay), glass, "glass_hard", 0.22)
        overlay = _overlay(np.asarray(overlay), base, "internal_base", 0.42)
        overlay = _overlay(np.asarray(overlay), bird, "bird", 0.48)
        overlay.save(output / "overlays" / f"{stem}_overlay.png")
        previous_union = union
        previous_centroid = centroid
        previous_bbox = bbox
        previous_area = int(union.sum())
        if status == "review_required":
            _write_frame_review_page(
                output / "high_risk_review_pages" / f"{stem}_review.png",
                rgb,
                raw_masks[stem],
                domains,
                row,
            )
        if stem in FIXED_NINE or status == "review_required":
            _write_frame_review_page(
                output / "review_pages" / f"{stem}_review.png",
                rgb,
                raw_masks[stem],
                domains,
                row,
            )

    with (output / "per_frame_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "review_queue.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("stem", "status", "reasons"))
        writer.writeheader()
        writer.writerows(review_queue)
    _make_contact_pages(rows, image_root, output)
    for key in (
        "bird_area", "internal_base_area", "internal_object_union_area",
        "neighbor_iou_previous", "area_ratio_previous",
        "bird_forward_backward_iou", "yellow_forward_backward_iou",
        "white_forward_backward_iou", "centroid_jump_fraction",
        "bbox_displacement_fraction", "significant_component_count",
        "raw_outside_glass_ratio",
    ):
        _line_plot([float(row[key]) for row in rows], key, output / "temporal_metrics_plots" / f"{key}.png")

    summary = {
        "artifact_role": "stage_d_internal_object_mask_proposal_111_v3",
        "human_status": "proposal_requires_review",
        "internal_object_semantics_version": INTERNAL_OBJECT_SEMANTICS_VERSION,
        "count": len(rows),
        "status_counts": {status: sum(1 for row in rows if row["frame_status"] == status) for status in ("anchor_accepted", "auto_candidate_ready", "review_required", "manual_edit_required")},
        "review_queue_count": len(review_queue),
        "engineering_review_guards": ENGINEERING_GUARDS,
        "anchor_stems": list(FIXED_NINE),
    }
    propagation = {
        "schema": "rtgs_d016b_sam2_anchor_propagation_v1",
        "sam2": {
            "conda_env": "grounded-sam2",
            "project_path": str(project),
            "project_git_head": _git_head(project),
            "config": args.sam2_config,
            "checkpoint": str(Path(args.sam2_checkpoint).resolve()),
            "checkpoint_sha256": sha256_file(args.sam2_checkpoint),
            "device": args.device,
            "seed": args.seed,
        },
        "object_ids": OBJECTS,
        "intervals": intervals,
        "forward_backward_method": "SAM2VideoPredictor add_new_mask anchors; propagate_in_video forward and reverse per interval",
        "fixed_nine_human_review_sha256": sha256_file(output / "fixed_nine_human_review_v3.json"),
    }
    _write_json(output / "propagation_manifest.json", propagation)
    proposal = {
        "schema_version": SCHEMA_VERSION,
        "role": PROPOSAL_ROLE,
        "proposal_kind": "stage_d_internal_object_mask_proposal_111_v3",
        "human_status": "proposal_requires_review",
        "internal_object_semantics_version": INTERNAL_OBJECT_SEMANTICS_VERSION,
        "count": 111,
        "ordered_stems": EXPECTED_STEMS,
        "formal_roles": list(FINAL_ROLES),
        "helper_object_ids": OBJECTS,
        "summary": summary,
        "propagation_manifest_sha256": sha256_file(output / "propagation_manifest.json"),
        "fixed_nine_human_review_sha256": sha256_file(output / "fixed_nine_human_review_v3.json"),
        "glass_manifest": {
            "path": glass_manifest["manifest_path"],
            "manifest_file_sha256": glass_manifest["manifest_file_sha256"],
            "manifest_payload_sha256": glass_manifest["manifest_payload_sha256"],
            "aggregate_sha256": glass_manifest["aggregate_sha256"],
        },
        "entries": rows,
    }
    proposal["manifest_payload_sha256"] = canonical_payload_sha256(proposal)
    _write_json(output / "proposal_manifest.json", proposal)
    _write_json(output / "proposal_summary.json", summary)
    _write_json(output / "review_selection_template.json", {
        row["stem"]: {
            "status": "accept|reject|manual_edit_required",
            "bird_ok": None,
            "internal_base_ok": None,
            "isolated_pole_included": None,
            "glass_included": None,
            "outside_ground_included": None,
            "reflection_included": None,
            "notes": "",
        }
        for row in rows
    })
    (output / "README.md").write_text(
        "# D-016b 111-view internal-object proposal\n\n"
        "This is a proposal requiring review. It is not formal, not reviewed, not accepted, and not training-ready.\n"
        "Training-time domains still require glass_hard AND valid_two_hit. No RT-GS training was run.\n",
        encoding="utf-8",
    )
    if len(rows) != 111 or {row["stem"] for row in rows} != set(EXPECTED_STEMS):
        raise ValueError("111-view proposal completeness failure")
    for row in rows:
        stem = row["stem"]
        bird = _load_mask(output / "processed" / "bird" / f"{stem}.png")
        base = _load_mask(output / "processed" / "internal_base" / f"{stem}.png")
        union = _load_mask(output / "processed" / "internal_object_union" / f"{stem}.png")
        glass = _load_mask(output / "processed" / "glass_hard" / f"{stem}.png")
        if not np.array_equal(union, bird | base):
            raise ValueError(f"union equation failed: {stem}")
        if (union & ~glass).any():
            raise ValueError(f"processed union outside glass: {stem}")
    print(json.dumps({"output": str(output), "summary": summary["status_counts"], "review_queue_count": len(review_queue)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
