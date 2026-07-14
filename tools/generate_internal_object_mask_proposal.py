#!/usr/bin/env python3
"""Generate D-016 fixed-nine v3 Grounded-SAM2 candidate review proposals.

The output is a proposal review artifact only. It is not formal reviewed
training supervision and cannot be promoted without explicit human review.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.internal_object_mask import (
    BIRD_PROMPTS,
    CONNECTED_FIXTURE_PROMPTS,
    DEFAULT_CANDIDATE_GUARDS,
    EXPECTED_STEMS,
    INTERNAL_OBJECT_SEMANTICS_VERSION,
    PROPOSAL_ROLE,
    SCHEMA_VERSION,
    WHITE_PLATFORM_PROMPTS,
    YELLOW_BASE_BOARD_PROMPTS,
    candidate_metrics,
    canonical_payload_sha256,
    mask_component_summary,
    object_occupancy_domains,
    score_candidate,
    sha256_file,
)
from utils.specular_mask import load_resized_formal_mask, validate_specular_mask_set


DEFAULT_GSAM2 = Path("/home/hanglee/桌面/Grounded-SAM-2")
DEFAULT_SAM2_CHECKPOINT = DEFAULT_GSAM2 / "checkpoints/sam2.1_hiera_large.pt"
DEFAULT_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEFAULT_GDINO_CONFIG = DEFAULT_GSAM2 / "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
DEFAULT_GDINO_CHECKPOINT = DEFAULT_GSAM2 / "gdino_checkpoints/groundingdino_swint_ogc.pth"
FIXED_NINE = ("000000", "000014", "000028", "000042", "000055", "000069", "000083", "000097", "000110")
CLASSES = ("bird", "yellow_base_board", "white_platform", "connected_fixture")
COLORS = {
    "bird": np.array([255, 35, 35], dtype=np.float32),
    "internal_base": np.array([35, 120, 255], dtype=np.float32),
    "internal_object_union": np.array([255, 225, 35], dtype=np.float32),
    "glass_hard": np.array([0, 220, 230], dtype=np.float32),
    "internal_ignore_preview": np.array([230, 45, 230], dtype=np.float32),
}


def _import_grounded_sam2(project: Path):
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    from torchvision.ops import box_convert
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
    return box_convert, build_sam2, SAM2ImagePredictor, load_model, load_image, predict


def _git_head(path: Path) -> str | None:
    if not (path / ".git").is_dir():
        return None
    import subprocess
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def _binary_png(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.asarray(mask, dtype=bool).astype(np.uint8) * 255), mode="L").save(path)


def _overlay(rgb: np.ndarray, mask: np.ndarray, color_name: str, alpha: float = 0.45) -> Image.Image:
    out = rgb.astype(np.float32).copy()
    mask = np.asarray(mask, dtype=bool)
    out[mask] = (1.0 - alpha) * out[mask] + alpha * COLORS[color_name]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def _panel(image: Image.Image, title: str, width: int = 420) -> Image.Image:
    font = ImageFont.load_default()
    img = image.convert("RGB").copy()
    img.thumbnail((width, int(width * img.height / max(img.width, 1))))
    canvas = Image.new("RGB", (width, img.height + 24), (18, 18, 18))
    canvas.paste(img, ((width - img.width) // 2, 24))
    ImageDraw.Draw(canvas).text((6, 6), title, fill=(255, 255, 255), font=font)
    return canvas


def _hstack(images: list[Image.Image]) -> Image.Image:
    width = sum(image.width for image in images)
    height = max(image.height for image in images)
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    x = 0
    for image in images:
        canvas.paste(image, (x, 0))
        x += image.width
    return canvas


def _vstack(images: list[Image.Image]) -> Image.Image:
    width = max(image.width for image in images)
    height = sum(image.height for image in images)
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    y = 0
    for image in images:
        canvas.paste(image, (0, y))
        y += image.height
    return canvas


def _class_prompts(cls: str) -> tuple[str, ...]:
    if cls == "bird":
        return BIRD_PROMPTS
    if cls == "yellow_base_board":
        return YELLOW_BASE_BOARD_PROMPTS
    if cls == "white_platform":
        return WHITE_PLATFORM_PROMPTS
    if cls == "connected_fixture":
        return CONNECTED_FIXTURE_PROMPTS
    raise ValueError(cls)


def _candidate_id(stem: str, cls: str, index: int) -> str:
    return f"{stem}_{cls}_{index:03d}"


def _predict_candidates(
    stem: str,
    cls: str,
    img_path: Path,
    prompts: tuple[str, ...],
    grounding_model,
    sam2_predictor,
    box_convert,
    load_image,
    predict,
    device: str,
    box_threshold: float,
    text_threshold: float,
    glass_mask: np.ndarray,
    output: Path,
    guards: dict[str, float],
    *,
    bird_mask: np.ndarray | None = None,
    yellow_base_board_mask: np.ndarray | None = None,
    white_platform_mask: np.ndarray | None = None,
) -> list[dict]:
    image_source, image = load_image(str(img_path))
    h, w, _ = image_source.shape
    sam2_predictor.set_image(image_source)
    candidates: list[dict] = []
    for prompt in prompts:
        text = prompt.lower().strip()
        if not text.endswith("."):
            text += "."
        boxes, confidences, labels = predict(
            model=grounding_model,
            image=image,
            caption=text,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device,
        )
        if boxes.numel() == 0:
            continue
        boxes_xyxy = box_convert(
            boxes=boxes * torch.tensor([w, h, w, h], dtype=boxes.dtype, device=boxes.device),
            in_fmt="cxcywh",
            out_fmt="xyxy",
        ).detach().cpu().numpy()
        masks, scores, _ = sam2_predictor.predict(
            point_coords=None, point_labels=None, box=boxes_xyxy, multimask_output=False,
        )
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        for box_index, mask in enumerate(masks.astype(bool)):
            cid = _candidate_id(stem, cls, len(candidates))
            xyxy = [float(v) for v in boxes_xyxy[box_index].tolist()]
            box_w, box_h = max(0.0, xyxy[2] - xyxy[0]), max(0.0, xyxy[3] - xyxy[1])
            metrics = candidate_metrics(
                mask, glass_mask, bird_mask=bird_mask,
                yellow_base_board_mask=yellow_base_board_mask,
                white_platform_mask=white_platform_mask,
            )
            confidence = float(confidences[box_index].detach().cpu())
            scored = score_candidate(cls, confidence, metrics, guards)
            record = {
                "candidate_id": cid,
                "stem": stem,
                "candidate_class": cls,
                "prompt": prompt,
                "grounding_label": str(labels[box_index]),
                "grounding_confidence": confidence,
                "sam_score": float(np.asarray(scores).reshape(-1)[box_index]) if np.asarray(scores).size else None,
                "box_xyxy": xyxy,
                "box_width": box_w,
                "box_height": box_h,
                "box_area": box_w * box_h,
                "box_image_ratio": float((box_w * box_h) / max(w * h, 1)),
                "selected": False,
                "rejected": bool(scored["reject_reasons"]),
                "unresolved": False,
                "selection_status": "candidate_rejected" if scored["reject_reasons"] else "candidate_viable",
                "selection_reason": None,
                "review_flags": list(scored["reject_reasons"]),
                **metrics,
                **scored,
            }
            mask_path = output / "candidates" / cls / f"{cid}.png"
            json_path = output / "candidates" / cls / f"{cid}.json"
            _binary_png(mask_path, mask)
            json_path.write_text(json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
            record["mask_path"] = str(mask_path)
            record["metadata_path"] = str(json_path)
            candidates.append(record)
    candidates.sort(key=lambda row: (-float(row["candidate_score"]), row["candidate_id"]))
    return candidates


def _select_one(candidates: list[dict]) -> dict | None:
    viable = [row for row in candidates if not row["reject_reasons"]]
    if not viable:
        for row in candidates:
            row["unresolved"] = True
        return None
    selected = viable[0]
    selected["selected"] = True
    selected["selection_status"] = "selected"
    selected["selection_reason"] = "highest_scoring_viable_candidate"
    return selected


def _load_candidate_mask(record: dict | None, shape: tuple[int, int]) -> np.ndarray:
    if not record:
        return np.zeros(shape, dtype=bool)
    with Image.open(record["mask_path"]) as image:
        return np.asarray(image) >= 128


def _domains(object_union: np.ndarray, glass: np.ndarray) -> dict[str, np.ndarray]:
    domains = object_occupancy_domains(
        torch.from_numpy(object_union[..., None].astype(np.float32)),
        torch.from_numpy(glass[..., None].astype(np.float32)),
        torch.from_numpy(glass[..., None].astype(np.float32)),
        erode_px=3,
        dilate_px=3,
    )
    return {key: (value[..., 0].detach().cpu().numpy() >= 0.5) for key, value in domains.items()}


def _quality_flags(final_masks: dict[str, np.ndarray], guards: dict[str, float]) -> list[str]:
    glass = final_masks["glass_hard"]
    base = final_masks["internal_base"]
    union = final_masks["internal_object_union"]
    flags = []
    base_ratio = float(base.sum() / max(glass.sum(), 1))
    union_ratio = float(union.sum() / max(glass.sum(), 1))
    base_components = mask_component_summary(base)["connected_components"]
    union_components = mask_component_summary(union)["connected_components"]
    if base_ratio < guards["internal_base_min_glass_ratio"]:
        flags.append("internal_base_too_small_possible_missing_yellow_board")
    if base_ratio > guards["internal_base_max_glass_ratio"]:
        flags.append("internal_base_too_large_possible_glass_or_ground")
    if union_ratio > guards["internal_object_union_max_glass_ratio"]:
        flags.append("internal_object_union_too_large")
    if base_components > guards["internal_base_max_components"]:
        flags.append("internal_base_too_many_components")
    if union_components > guards["internal_object_union_max_components"]:
        flags.append("internal_object_union_too_many_components")
    return flags


def _status_from(selected: dict | None, candidates: list[dict]) -> str:
    if selected:
        return "auto_candidate_ready"
    if candidates:
        return "unresolved"
    return "manual_edit_required"


def _write_review_page(
    output: Path,
    stem: str,
    rgb: np.ndarray,
    final_masks: dict[str, np.ndarray],
    candidates: dict[str, list[dict]],
    status: dict,
) -> Image.Image:
    domains = _domains(final_masks["internal_object_union"], final_masks["glass_hard"])
    panels = [
        _panel(Image.fromarray(rgb), f"{stem} Original"),
        _panel(_overlay(rgb, final_masks["glass_hard"], "glass_hard"), "glass_hard overlay"),
        _panel(_overlay(rgb, final_masks["bird"], "bird"), f"bird: {status['bird_status']}"),
        _panel(_overlay(rgb, final_masks["internal_base"], "internal_base"), f"internal_base: {status['internal_base_status']}"),
        _panel(_overlay(rgb, final_masks["internal_object_union"], "internal_object_union"), f"union: {status['frame_status']}"),
        _panel(_overlay(rgb, domains["Mpos"], "bird"), "Mpos preview"),
        _panel(_overlay(rgb, domains["Mignore"], "internal_ignore_preview"), "Mignore preview"),
        _panel(_overlay(rgb, domains["Mneg"], "glass_hard"), "Mneg preview"),
    ]
    candidate_panels = []
    for cls in CLASSES:
        for row in candidates.get(cls, [])[:6]:
            mask = _load_candidate_mask(row, rgb.shape[:2])
            color = "bird" if cls == "bird" else "internal_base"
            title = f"{cls} {row['candidate_id']} score={row['candidate_score']:.2f} {row['selection_status']}"
            candidate_panels.append(_panel(_overlay(rgb, mask, color), title))
    rows = [_hstack(panels[:4]), _hstack(panels[4:])]
    for index in range(0, len(candidate_panels), 3):
        rows.append(_hstack(candidate_panels[index:index + 3]))
    page = _vstack(rows)
    page.save(output / "review_pages" / f"{stem}_candidate_review.png")
    return page


def _write_fixed_sheet(images: list[Image.Image], path: Path) -> None:
    if images:
        _vstack(images).save(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default=str(ROOT / "data/TiHuBird"))
    parser.add_argument("--images", default="images")
    parser.add_argument("--glass-mask-manifest", default="specular_masks_reviewed_v1/manifest.json")
    parser.add_argument("--output", default=str(ROOT / "output/stage_d_tihubird_internal_object_mask_proposal_fixednine_probe_v3"))
    parser.add_argument("--grounded-sam2-project", default=str(DEFAULT_GSAM2))
    parser.add_argument("--sam2-checkpoint", default=str(DEFAULT_SAM2_CHECKPOINT))
    parser.add_argument("--sam2-config", default=DEFAULT_SAM2_CONFIG)
    parser.add_argument("--gdino-config", default=str(DEFAULT_GDINO_CONFIG))
    parser.add_argument("--gdino-checkpoint", default=str(DEFAULT_GDINO_CHECKPOINT))
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--stems", nargs="*", default=list(FIXED_NINE))
    parser.add_argument("--enable-connected-fixture", action="store_true")
    args = parser.parse_args()

    scene = Path(args.scene).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"BLOCKED_BY_EXISTING_OUTPUT: {output}")
    stems = list(args.stems)
    if stems != list(FIXED_NINE):
        raise ValueError("D-016 fixed-nine v3 proposal generation is fixed-nine only")
    unexpected = sorted(set(stems) - set(EXPECTED_STEMS))
    if unexpected:
        raise ValueError(f"unexpected stems: {unexpected}")
    for directory in (
        "candidates/bird", "candidates/yellow_base_board", "candidates/white_platform", "candidates/connected_fixture",
        "raw/bird", "raw/yellow_base_board", "raw/white_platform", "raw/connected_fixture",
        "processed/bird", "processed/internal_base", "processed/internal_object_union",
        "processed/internal_ignore_preview", "processed/glass_hard",
        "overlays", "review_pages", "contact_sheets",
    ):
        (output / directory).mkdir(parents=True, exist_ok=True)

    project = Path(args.grounded_sam2_project).resolve()
    for path in (project, Path(args.sam2_checkpoint), Path(args.gdino_config), Path(args.gdino_checkpoint)):
        if not path.exists():
            raise FileNotFoundError(path)
    os.chdir(project)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    box_convert, build_sam2, SAM2ImagePredictor, load_model, load_image, predict = _import_grounded_sam2(project)
    sam2_model = build_sam2(args.sam2_config, args.sam2_checkpoint, device=args.device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    grounding_model = load_model(args.gdino_config, args.gdino_checkpoint, device=args.device)
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    glass_manifest = validate_specular_mask_set(scene, args.images, args.glass_mask_manifest)
    guards = dict(DEFAULT_CANDIDATE_GUARDS)
    rows, fixed_binary, fixed_overlay, review_pages, ignore_pages = [], [], [], [], []
    selection_json, review_template, review_queue = {}, {}, []

    for stem in stems:
        rgb_path = scene / args.images / f"{stem}.jpg"
        with Image.open(rgb_path) as rgb_image:
            size = rgb_image.size
            rgb = np.asarray(rgb_image.convert("RGB"))
        glass_values, _, _ = load_resized_formal_mask(glass_manifest, stem, size, size)
        glass = glass_values >= 0.5
        candidates: dict[str, list[dict]] = {}
        candidates["bird"] = _predict_candidates(
            stem, "bird", rgb_path, BIRD_PROMPTS, grounding_model, sam2_predictor,
            box_convert, load_image, predict, args.device, args.box_threshold,
            args.text_threshold, glass, output, guards,
        )
        selected_bird = _select_one(candidates["bird"])
        bird_raw = _load_candidate_mask(selected_bird, glass.shape)
        candidates["yellow_base_board"] = _predict_candidates(
            stem, "yellow_base_board", rgb_path, YELLOW_BASE_BOARD_PROMPTS, grounding_model,
            sam2_predictor, box_convert, load_image, predict, args.device,
            args.box_threshold, args.text_threshold, glass, output, guards,
            bird_mask=bird_raw,
        )
        selected_yellow = _select_one(candidates["yellow_base_board"])
        yellow_raw = _load_candidate_mask(selected_yellow, glass.shape)
        candidates["white_platform"] = _predict_candidates(
            stem, "white_platform", rgb_path, WHITE_PLATFORM_PROMPTS, grounding_model,
            sam2_predictor, box_convert, load_image, predict, args.device,
            args.box_threshold, args.text_threshold, glass, output, guards,
            bird_mask=bird_raw, yellow_base_board_mask=yellow_raw,
        )
        selected_white = _select_one(candidates["white_platform"])
        white_raw = _load_candidate_mask(selected_white, glass.shape)
        if args.enable_connected_fixture:
            candidates["connected_fixture"] = _predict_candidates(
                stem, "connected_fixture", rgb_path, CONNECTED_FIXTURE_PROMPTS, grounding_model,
                sam2_predictor, box_convert, load_image, predict, args.device,
                args.box_threshold, args.text_threshold, glass, output, guards,
                bird_mask=bird_raw, yellow_base_board_mask=yellow_raw, white_platform_mask=white_raw,
            )
            selected_fixture = _select_one(candidates["connected_fixture"])
        else:
            candidates["connected_fixture"] = []
            selected_fixture = None
        fixture_raw = _load_candidate_mask(selected_fixture, glass.shape)
        internal_base_raw = yellow_raw | white_raw | fixture_raw
        union_raw = bird_raw | internal_base_raw
        final_masks = {
            "bird": bird_raw & glass,
            "yellow_base_board": yellow_raw & glass,
            "white_platform": white_raw & glass,
            "connected_fixture": fixture_raw & glass,
            "internal_base": internal_base_raw & glass,
            "internal_object_union": union_raw & glass,
            "glass_hard": glass,
        }
        domains = _domains(final_masks["internal_object_union"], glass)
        for role in ("bird", "yellow_base_board", "white_platform", "connected_fixture"):
            _binary_png(output / "raw" / role / f"{stem}.png", final_masks[role] if role in final_masks else np.zeros_like(glass))
        for role in ("bird", "internal_base", "internal_object_union", "glass_hard"):
            _binary_png(output / "processed" / role / f"{stem}.png", final_masks[role])
        _binary_png(output / "processed" / "internal_ignore_preview" / f"{stem}.png", domains["Mignore"])

        bird_status = _status_from(selected_bird, candidates["bird"])
        yellow_status = _status_from(selected_yellow, candidates["yellow_base_board"])
        white_status = _status_from(selected_white, candidates["white_platform"])
        fixture_status = "disabled_empty" if not args.enable_connected_fixture else _status_from(selected_fixture, candidates["connected_fixture"])
        quality_flags = _quality_flags(final_masks, guards)
        internal_base_status = (
            "auto_candidate_ready"
            if selected_yellow and selected_white and not quality_flags else "unresolved"
        )
        frame_status = (
            "auto_candidate_ready"
            if bird_status == "auto_candidate_ready" and internal_base_status == "auto_candidate_ready"
            else "unresolved"
        )
        if quality_flags:
            frame_status = "unresolved"
        if frame_status != "auto_candidate_ready":
            review_queue.append({"stem": stem, "status": frame_status, "reason": ";".join(quality_flags) or "unresolved_candidate_selection"})
        raw_outside = float(((union_raw & ~glass).sum()) / max(union_raw.sum(), 1))
        base_summary = mask_component_summary(final_masks["internal_base"])
        union_summary = mask_component_summary(final_masks["internal_object_union"])
        row = {
            "stem": stem,
            "bird_candidate_count": len(candidates["bird"]),
            "bird_selected_candidate_id": selected_bird["candidate_id"] if selected_bird else None,
            "bird_status": bird_status,
            "bird_image_ratio": candidate_metrics(final_masks["bird"], glass)["mask_image_ratio"],
            "bird_glass_ratio": candidate_metrics(final_masks["bird"], glass)["mask_glass_ratio"],
            "bird_connected_components": mask_component_summary(final_masks["bird"])["connected_components"],
            "bird_reflection_risk": "review_required_if_duplicate_component",
            "yellow_base_board_candidate_count": len(candidates["yellow_base_board"]),
            "yellow_base_board_selected_id": selected_yellow["candidate_id"] if selected_yellow else None,
            "yellow_base_board_status": yellow_status,
            "yellow_base_board_glass_ratio": candidate_metrics(final_masks["yellow_base_board"], glass)["mask_glass_ratio"],
            "white_platform_candidate_count": len(candidates["white_platform"]),
            "white_platform_selected_id": selected_white["candidate_id"] if selected_white else None,
            "white_platform_status": white_status,
            "white_platform_glass_ratio": candidate_metrics(final_masks["white_platform"], glass)["mask_glass_ratio"],
            "connected_fixture_candidate_count": len(candidates["connected_fixture"]),
            "connected_fixture_selected_id": selected_fixture["candidate_id"] if selected_fixture else None,
            "connected_fixture_status": fixture_status,
            "fixture_connects_bird": bool(selected_fixture and selected_fixture.get("bird_distance_px", 1e9) <= 8.0),
            "fixture_connects_base": bool(selected_fixture and selected_fixture.get("forms_bird_to_base_connection")),
            "fixture_independent_pole_risk": "rejected_or_disabled",
            "internal_base_status": internal_base_status,
            "internal_base_image_ratio": candidate_metrics(final_masks["internal_base"], glass)["mask_image_ratio"],
            "internal_base_glass_ratio": candidate_metrics(final_masks["internal_base"], glass)["mask_glass_ratio"],
            "internal_base_connected_components": base_summary["connected_components"],
            "internal_base_bbox": base_summary["bbox_xyxy"],
            "internal_base_centroid": base_summary["centroid_xy"],
            "union_image_ratio": candidate_metrics(final_masks["internal_object_union"], glass)["mask_image_ratio"],
            "union_glass_ratio": candidate_metrics(final_masks["internal_object_union"], glass)["mask_glass_ratio"],
            "union_connected_components": union_summary["connected_components"],
            "raw_outside_glass_ratio": raw_outside,
            "clipped_removed_ratio": raw_outside,
            "Mpos_area": int(domains["Mpos"].sum()),
            "Mignore_area": int(domains["Mignore"].sum()),
            "Mneg_area": int(domains["Mneg"].sum()),
            "quality_guards": ";".join(quality_flags),
            "review_flags": ";".join(quality_flags),
            "frame_status": frame_status,
        }
        rows.append(row)
        status = {**row, "quality_flags": quality_flags}
        selection_json[stem] = {
            "status": frame_status,
            "auto_suggestion": {
                "bird_candidate_id": row["bird_selected_candidate_id"],
                "yellow_base_board_candidate_id": row["yellow_base_board_selected_id"],
                "white_platform_candidate_id": row["white_platform_selected_id"],
                "connected_fixture_candidate_id": row["connected_fixture_selected_id"],
            },
            "candidates": candidates,
            "quality_guards": quality_flags,
            "ratios": {
                "internal_base_glass_ratio": row["internal_base_glass_ratio"],
                "internal_object_union_glass_ratio": row["union_glass_ratio"],
            },
        }
        review_template[stem] = {
            "bird_candidate_id": None,
            "yellow_base_board_candidate_id": None,
            "white_platform_candidate_id": None,
            "connected_fixture_candidate_id": None,
            "status": "accept|reject|manual_edit_required",
            "bird_ok": None,
            "yellow_base_board_ok": None,
            "white_platform_ok": None,
            "isolated_pole_included": None,
            "glass_included": None,
            "outside_ground_included": None,
            "reflection_included": None,
            "ignore_region_ok": None,
            "notes": "",
        }
        binary = _hstack([
            _panel(Image.fromarray(rgb), f"{stem} Original"),
            _panel(Image.fromarray((final_masks["bird"].astype(np.uint8) * 255), mode="L").convert("RGB"), "Bird"),
            _panel(Image.fromarray((final_masks["internal_base"].astype(np.uint8) * 255), mode="L").convert("RGB"), "Internal Base"),
            _panel(Image.fromarray((final_masks["internal_object_union"].astype(np.uint8) * 255), mode="L").convert("RGB"), "Internal Object Union"),
            _panel(Image.fromarray((glass.astype(np.uint8) * 255), mode="L").convert("RGB"), "Glass Hard"),
        ])
        overlay = _hstack([
            _panel(Image.fromarray(rgb), f"{stem} Original"),
            _panel(_overlay(rgb, final_masks["bird"], "bird"), "Bird"),
            _panel(_overlay(rgb, final_masks["internal_base"], "internal_base"), "Internal Base"),
            _panel(_overlay(rgb, final_masks["internal_object_union"], "internal_object_union"), "Internal Object Union"),
            _panel(_overlay(rgb, glass, "glass_hard"), "Glass Hard"),
        ])
        ignore = _hstack([
            _panel(Image.fromarray(rgb), f"{stem} Original"),
            _panel(_overlay(rgb, domains["Mpos"], "bird"), "Mpos"),
            _panel(_overlay(rgb, domains["Mignore"], "internal_ignore_preview"), "Mignore"),
            _panel(_overlay(rgb, domains["Mneg"], "glass_hard"), "Mneg"),
        ])
        binary.save(output / "overlays" / f"{stem}_binary_comparison.png")
        overlay.save(output / "overlays" / f"{stem}_overlay_comparison.png")
        ignore.save(output / "overlays" / f"{stem}_ignore_domain_review.png")
        fixed_binary.append(binary)
        fixed_overlay.append(overlay)
        ignore_pages.append(ignore)
        review_pages.append(_write_review_page(output, stem, rgb, final_masks, candidates, status))

    _write_fixed_sheet(fixed_binary, output / "contact_sheets" / "fixed_nine_binary_comparison.png")
    _write_fixed_sheet(fixed_overlay, output / "contact_sheets" / "fixed_nine_overlay_comparison.png")
    _write_fixed_sheet(review_pages, output / "contact_sheets" / "fixed_nine_candidate_review.png")
    _write_fixed_sheet(ignore_pages, output / "contact_sheets" / "fixed_nine_ignore_domain_review.png")
    with (output / "candidate_selection.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output / "candidate_selection.json").write_text(json.dumps(selection_json, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "review_selection_template.json").write_text(json.dumps(review_template, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    with (output / "review_queue.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("stem", "status", "reason"))
        writer.writeheader()
        writer.writerows(review_queue)

    grounded_sam2 = {
        "conda_env": "grounded-sam2",
        "project_path": str(project),
        "git_head": _git_head(project),
        "sam2_config": args.sam2_config,
        "sam2_checkpoint": str(Path(args.sam2_checkpoint).resolve()),
        "sam2_checkpoint_sha256": sha256_file(args.sam2_checkpoint),
        "grounding_dino_config": str(Path(args.gdino_config).resolve()),
        "grounding_dino_checkpoint": str(Path(args.gdino_checkpoint).resolve()),
        "grounding_dino_checkpoint_sha256": sha256_file(args.gdino_checkpoint),
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "seed": args.seed,
        "device": args.device,
        "input_image_sha256": {stem: sha256_file(scene / args.images / f"{stem}.jpg") for stem in stems},
        "glass_manifest": glass_manifest,
        "method": "per-frame independent grounding; no video propagation; fixed-nine only",
        "video_propagation": None,
    }
    summary = {
        "artifact_role": "Grounded-SAM2 fixed-nine v3 proposal review artifact",
        "internal_object_semantics_version": INTERNAL_OBJECT_SEMANTICS_VERSION,
        "status_counts": {
            status: sum(1 for row in rows if row["frame_status"] == status)
            for status in ("auto_candidate_ready", "unresolved", "manual_edit_required")
        },
        "fixed_nine": list(stems),
        "engineering_review_guards": guards,
        "connected_fixture": {
            "enabled": bool(args.enable_connected_fixture),
            "default": "disabled_empty_to_avoid_independent_pole_or_rail_selection",
        },
        "rows": rows,
        "review_queue_count": len(review_queue),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "role": PROPOSAL_ROLE,
        "internal_object_semantics_version": INTERNAL_OBJECT_SEMANTICS_VERSION,
        "legacy_aliases": {
            "base": "legacy_not_auto_promoted_to_internal_base",
            "bird_support": "legacy_not_auto_promoted_to_internal_base",
        },
        "human_status": "proposal_requires_review",
        "count": len(stems),
        "ordered_stems": list(stems),
        "formal_roles": ["bird", "internal_base", "internal_object_union"],
        "helper_candidate_classes": list(CLASSES),
        "prompts": {
            "bird": list(BIRD_PROMPTS),
            "yellow_base_board": list(YELLOW_BASE_BOARD_PROMPTS),
            "white_platform": list(WHITE_PLATFORM_PROMPTS),
            "connected_fixture": list(CONNECTED_FIXTURE_PROMPTS),
        },
        "candidate_selection_policy": "score_individual_candidates_no_unconditional_or_v3",
        "grounded_sam2": grounded_sam2,
        "summary": summary,
    }
    manifest["manifest_payload_sha256"] = canonical_payload_sha256(manifest)
    (output / "proposal_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "proposal_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# D-016 fixed-nine candidate review v3\n\n"
        "Artifact role: Grounded-SAM2 fixed-nine v3 proposal review artifact.\n\n"
        "Formal v3 classes are bird, internal_base, and internal_object_union.\n"
        "internal_base includes the yellow rectangular base board, the white platform,\n"
        "and only human-confirmed connected fixtures. It is not glass_hard, not the\n"
        "transparent glass floor, not the whole display case, not outside ground,\n"
        "not independent rails or poles, and not bird.\n\n"
        "This is not a formal reviewed mask, not accepted supervision, and not training-ready.\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "summary": summary["status_counts"], "review_queue_count": len(review_queue)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
