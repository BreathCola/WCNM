#!/usr/bin/env python3
"""Generate D-016 fixed-nine Grounded-SAM2 candidate review proposals.

The output is a proposal review artifact only.  It is not a formal reviewed
mask set and is never training supervision.
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
    DEFAULT_CANDIDATE_GUARDS,
    EXPECTED_STEMS,
    INTERNAL_OBJECT_SEMANTICS_VERSION,
    PROPOSAL_ROLE,
    SCHEMA_VERSION,
    SUPPORT_MOUNT_PROMPTS,
    SUPPORT_PLINTH_PROMPTS,
    candidate_metrics,
    canonical_payload_sha256,
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
CLASSES = ("bird", "support_plinth", "support_mount")
COLORS = {
    "bird": np.array([255, 35, 35], dtype=np.float32),
    "bird_support": np.array([35, 120, 255], dtype=np.float32),
    "union": np.array([255, 225, 35], dtype=np.float32),
    "glass_hard": np.array([0, 220, 230], dtype=np.float32),
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
    color = COLORS[color_name]
    mask = np.asarray(mask, dtype=bool)
    out[mask] = (1.0 - alpha) * out[mask] + alpha * color
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
    if cls == "support_plinth":
        return SUPPORT_PLINTH_PROMPTS
    if cls == "support_mount":
        return SUPPORT_MOUNT_PROMPTS
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
    bird_mask: np.ndarray | None,
    guards: dict[str, float],
    output: Path,
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
        masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes_xyxy,
            multimask_output=False,
        )
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        for box_index, mask in enumerate(masks.astype(bool)):
            cid = _candidate_id(stem, cls, len(candidates))
            confidence = float(confidences[box_index].detach().cpu())
            xyxy = [float(v) for v in boxes_xyxy[box_index].tolist()]
            box_w, box_h = max(0.0, xyxy[2] - xyxy[0]), max(0.0, xyxy[3] - xyxy[1])
            metrics = candidate_metrics(mask, glass_mask, bird_mask=bird_mask)
            scored = score_candidate(cls, confidence, metrics, guards)
            record = {
                "candidate_id": cid,
                "stem": stem,
                "class": cls,
                "prompt": prompt,
                "grounding_label": str(labels[box_index]),
                "grounding_confidence": confidence,
                "sam_score": float(np.asarray(scores).reshape(-1)[box_index]) if np.asarray(scores).size else None,
                "box_xyxy": xyxy,
                "box_width": box_w,
                "box_height": box_h,
                "box_area": box_w * box_h,
                "box_image_ratio": float((box_w * box_h) / max(w * h, 1)),
                **metrics,
                **scored,
                "selected": False,
                "selection_status": "candidate_rejected" if scored["reject_reasons"] else "candidate_viable",
                "selection_reason": None,
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


def _write_review_page(
    output: Path,
    stem: str,
    rgb: np.ndarray,
    glass: np.ndarray,
    final_masks: dict[str, np.ndarray],
    candidates: dict[str, list[dict]],
    status: dict,
) -> Image.Image:
    panels = [
        _panel(Image.fromarray(rgb), f"{stem} Original"),
        _panel(_overlay(rgb, glass, "glass_hard"), "glass_hard overlay"),
        _panel(_overlay(rgb, final_masks["bird"], "bird"), f"final bird: {status['bird_status']}"),
        _panel(_overlay(rgb, final_masks["bird_support"], "bird_support"), f"final bird_support: {status['bird_support_status']}"),
        _panel(_overlay(rgb, final_masks["union"], "union"), f"union: {status['frame_status']}"),
    ]
    candidate_panels = []
    for cls in CLASSES:
        for row in candidates[cls][:4]:
            mask = _load_candidate_mask(row, rgb.shape[:2])
            title = f"{cls} {row['candidate_id']} score={row['candidate_score']:.2f} {row['selection_status']}"
            candidate_panels.append(_panel(_overlay(rgb, mask, "bird" if cls == "bird" else "bird_support"), title))
    rows = [_hstack(panels)]
    for index in range(0, len(candidate_panels), 3):
        rows.append(_hstack(candidate_panels[index:index + 3]))
    page = _vstack(rows)
    page.save(output / "review_pages" / f"{stem}_candidate_review.png")
    return page


def _write_fixed_sheet(images: list[Image.Image], path: Path) -> None:
    if images:
        _vstack(images).save(path)


def _status_from(selected: dict | None, candidates: list[dict]) -> str:
    if selected:
        return "auto_candidate_ready"
    if candidates:
        return "unresolved"
    return "manual_edit_required"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default=str(ROOT / "data/TiHuBird"))
    parser.add_argument("--images", default="images")
    parser.add_argument("--glass-mask-manifest", default="specular_masks_reviewed_v1/manifest.json")
    parser.add_argument("--output", default=str(ROOT / "output/stage_d_tihubird_internal_object_mask_proposal_fixednine_probe_v2"))
    parser.add_argument("--grounded-sam2-project", default=str(DEFAULT_GSAM2))
    parser.add_argument("--sam2-checkpoint", default=str(DEFAULT_SAM2_CHECKPOINT))
    parser.add_argument("--sam2-config", default=DEFAULT_SAM2_CONFIG)
    parser.add_argument("--gdino-config", default=str(DEFAULT_GDINO_CONFIG))
    parser.add_argument("--gdino-checkpoint", default=str(DEFAULT_GDINO_CHECKPOINT))
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--stems", nargs="*", default=list(FIXED_NINE))
    args = parser.parse_args()

    scene = Path(args.scene).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite proposal output: {output}")
    stems = list(args.stems)
    if stems != list(FIXED_NINE):
        raise ValueError("D-016a proposal generation is fixed-nine only")
    unexpected = sorted(set(stems) - set(EXPECTED_STEMS))
    if unexpected:
        raise ValueError(f"unexpected stems: {unexpected}")
    for directory in (
        "candidates/bird", "candidates/support_plinth", "candidates/support_mount",
        "raw/bird", "raw/support_plinth", "raw/support_mount",
        "processed/bird", "processed/bird_support", "processed/union", "processed/glass_hard",
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
    rows = []
    selection_json = {}
    fixed_binary, fixed_overlay, review_pages = [], [], []
    review_template = {}
    review_queue = []
    guards = dict(DEFAULT_CANDIDATE_GUARDS)

    for stem in stems:
        rgb_path = scene / args.images / f"{stem}.jpg"
        with Image.open(rgb_path) as rgb_image:
            size = rgb_image.size
            rgb = np.asarray(rgb_image.convert("RGB"))
        glass_values, glass_sha, _ = load_resized_formal_mask(glass_manifest, stem, size, size)
        glass = glass_values >= 0.5
        _binary_png(output / "processed" / "glass_hard" / f"{stem}.png", glass)
        candidates: dict[str, list[dict]] = {}
        bird_candidates = _predict_candidates(
            stem, "bird", rgb_path, _class_prompts("bird"), grounding_model,
            sam2_predictor, box_convert, load_image, predict, args.device,
            args.box_threshold, args.text_threshold, glass, None, guards, output,
        )
        selected_bird = _select_one(bird_candidates)
        bird_mask_raw = _load_candidate_mask(selected_bird, glass.shape)
        candidates["bird"] = bird_candidates
        support_masks_for_relation = bird_mask_raw if selected_bird else None
        for cls in ("support_plinth", "support_mount"):
            candidates[cls] = _predict_candidates(
                stem, cls, rgb_path, _class_prompts(cls), grounding_model,
                sam2_predictor, box_convert, load_image, predict, args.device,
                args.box_threshold, args.text_threshold, glass, support_masks_for_relation,
                guards, output,
            )
        selected_plinth = _select_one(candidates["support_plinth"])
        selected_mount = _select_one(candidates["support_mount"])
        plinth_raw = _load_candidate_mask(selected_plinth, glass.shape)
        mount_raw = _load_candidate_mask(selected_mount, glass.shape)
        bird_support_raw = plinth_raw | mount_raw
        union_raw = bird_mask_raw | bird_support_raw
        final_masks = {
            "bird": bird_mask_raw & glass,
            "bird_support": bird_support_raw & glass,
            "union": union_raw & glass,
            "support_plinth": plinth_raw & glass,
            "support_mount": mount_raw & glass,
        }
        _binary_png(output / "raw" / "bird" / f"{stem}.png", bird_mask_raw)
        _binary_png(output / "raw" / "support_plinth" / f"{stem}.png", plinth_raw)
        _binary_png(output / "raw" / "support_mount" / f"{stem}.png", mount_raw)
        for role in ("bird", "bird_support", "union"):
            _binary_png(output / "processed" / role / f"{stem}.png", final_masks[role])
        bird_status = _status_from(selected_bird, bird_candidates)
        support_status = (
            "auto_candidate_ready"
            if selected_plinth or selected_mount else (
                "unresolved" if candidates["support_plinth"] or candidates["support_mount"]
                else "manual_edit_required"
            )
        )
        union_ratio = float(final_masks["union"].sum() / max(glass.sum(), 1))
        support_ratio = float(final_masks["bird_support"].sum() / max(glass.sum(), 1))
        bird_ratio = float(final_masks["bird"].sum() / max(glass.sum(), 1))
        union_guard = [] if union_ratio <= guards["union_max_glass_ratio"] else ["union_mask_glass_ratio_too_large"]
        frame_status = (
            "auto_candidate_ready"
            if bird_status == support_status == "auto_candidate_ready" and not union_guard
            else "unresolved"
        )
        status = {
            "bird_status": bird_status,
            "bird_support_status": support_status,
            "frame_status": frame_status,
            "quality_guards": union_guard,
        }
        if frame_status != "auto_candidate_ready":
            review_queue.append({"stem": stem, "status": frame_status, "reason": ";".join(union_guard) or "unresolved_candidate_selection"})
        selection_json[stem] = {
            "status": frame_status,
            "bird_candidate_id": selected_bird["candidate_id"] if selected_bird else None,
            "support_plinth_candidate_id": selected_plinth["candidate_id"] if selected_plinth else None,
            "support_mount_candidate_id": selected_mount["candidate_id"] if selected_mount else None,
            "bird_candidates": bird_candidates,
            "support_plinth_candidates": candidates["support_plinth"],
            "support_mount_candidates": candidates["support_mount"],
            "quality_guards": status["quality_guards"],
            "ratios": {
                "bird_glass_ratio": bird_ratio,
                "bird_support_glass_ratio": support_ratio,
                "union_glass_ratio": union_ratio,
            },
        }
        review_template[stem] = {
            "bird_candidate_id": None,
            "support_plinth_candidate_id": None,
            "support_mount_candidate_id": None,
            "status": "accept|reject|manual_edit_required",
            "notes": "",
        }
        rows.append({
            "stem": stem,
            "bird_candidate_count": len(bird_candidates),
            "bird_candidate_id": selection_json[stem]["bird_candidate_id"],
            "bird_status": bird_status,
            "support_plinth_candidate_count": len(candidates["support_plinth"]),
            "support_mount_candidate_count": len(candidates["support_mount"]),
            "support_plinth_candidate_id": selection_json[stem]["support_plinth_candidate_id"],
            "support_mount_candidate_id": selection_json[stem]["support_mount_candidate_id"],
            "bird_support_status": support_status,
            "frame_status": frame_status,
            "bird_glass_ratio": bird_ratio,
            "bird_support_glass_ratio": support_ratio,
            "union_glass_ratio": union_ratio,
            "raw_outside_glass_ratio": float(((union_raw & ~glass).sum()) / max(union_raw.sum(), 1)),
            "raw_clipped_removed_ratio": float(((union_raw & ~glass).sum()) / max(union_raw.sum(), 1)),
            "union_connected_components": candidate_metrics(union_raw, glass)["connected_components"],
            "union_glass_iou": candidate_metrics(union_raw, glass)["glass_iou"],
            "quality_guards": ";".join(status["quality_guards"]),
        })
        binary = _hstack([
            _panel(Image.fromarray(rgb), f"{stem} Original"),
            _panel(Image.fromarray((final_masks["bird"].astype(np.uint8) * 255), mode="L").convert("RGB"), "Bird"),
            _panel(Image.fromarray((final_masks["bird_support"].astype(np.uint8) * 255), mode="L").convert("RGB"), "Bird Support"),
            _panel(Image.fromarray((final_masks["union"].astype(np.uint8) * 255), mode="L").convert("RGB"), "Internal Object Union"),
            _panel(Image.fromarray((glass.astype(np.uint8) * 255), mode="L").convert("RGB"), "Glass Hard"),
        ])
        overlay = _hstack([
            _panel(Image.fromarray(rgb), f"{stem} Original"),
            _panel(_overlay(rgb, final_masks["bird"], "bird"), "Bird"),
            _panel(_overlay(rgb, final_masks["bird_support"], "bird_support"), "Bird Support"),
            _panel(_overlay(rgb, final_masks["union"], "union"), "Internal Object Union"),
            _panel(_overlay(rgb, glass, "glass_hard"), "Glass Hard"),
        ])
        binary.save(output / "overlays" / f"{stem}_binary_comparison.png")
        overlay.save(output / "overlays" / f"{stem}_overlay_comparison.png")
        fixed_binary.append(binary)
        fixed_overlay.append(overlay)
        review_pages.append(_write_review_page(output, stem, rgb, glass, final_masks, candidates, status))

    _write_fixed_sheet(fixed_binary, output / "contact_sheets" / "fixed_nine_binary_comparison.png")
    _write_fixed_sheet(fixed_overlay, output / "contact_sheets" / "fixed_nine_overlay_comparison.png")
    _write_fixed_sheet(review_pages, output / "contact_sheets" / "fixed_nine_candidate_review.png")
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
        "method": "per-frame independent grounding; no video propagation claimed",
        "video_propagation": None,
    }
    summary = {
        "artifact_role": "Grounded-SAM2 fixed-nine proposal review artifact",
        "status_counts": {
            status: sum(1 for row in rows if row["frame_status"] == status)
            for status in ("auto_candidate_ready", "unresolved", "manual_edit_required")
        },
        "fixed_nine": list(stems),
        "guards": guards,
        "rows": rows,
        "review_queue_count": len(review_queue),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "role": PROPOSAL_ROLE,
        "internal_object_semantics_version": INTERNAL_OBJECT_SEMANTICS_VERSION,
        "legacy_aliases": {"base": "bird_support"},
        "human_status": "proposal_requires_review",
        "count": len(stems),
        "ordered_stems": list(stems),
        "prompts": {
            "bird": list(BIRD_PROMPTS),
            "support_plinth": list(SUPPORT_PLINTH_PROMPTS),
            "support_mount": list(SUPPORT_MOUNT_PROMPTS),
        },
        "candidate_selection_policy": "score_individual_candidates_no_unconditional_or_v2",
        "grounded_sam2": grounded_sam2,
        "summary": summary,
    }
    manifest["manifest_payload_sha256"] = canonical_payload_sha256(manifest)
    (output / "proposal_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "proposal_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# D-016 fixed-nine candidate review v2\n\n"
        "Artifact role: Grounded-SAM2 fixed-nine proposal review artifact.\n\n"
        "This is not a formal reviewed mask, not accepted supervision, and not training-ready.\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "summary": summary["status_counts"], "review_queue_count": len(review_queue)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
