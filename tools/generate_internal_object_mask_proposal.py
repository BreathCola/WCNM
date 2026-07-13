#!/usr/bin/env python3
"""Generate D-016 Grounded-SAM2 internal-object mask proposals.

This script is intended to be run inside the local `grounded-sam2` conda
environment.  It uses local project/config/checkpoint paths only.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.internal_object_mask import (
    BASE_PROMPTS,
    BIRD_PROMPTS,
    EXPECTED_STEMS,
    MASK_ROLES,
    proposal_view_metadata,
    sha256_file,
    write_proposal_manifest,
)
from utils.specular_mask import validate_specular_mask_set, load_resized_formal_mask


DEFAULT_GSAM2 = Path("/home/hanglee/桌面/Grounded-SAM-2")
DEFAULT_SAM2_CHECKPOINT = DEFAULT_GSAM2 / "checkpoints/sam2.1_hiera_large.pt"
DEFAULT_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEFAULT_GDINO_CONFIG = DEFAULT_GSAM2 / "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
DEFAULT_GDINO_CHECKPOINT = DEFAULT_GSAM2 / "gdino_checkpoints/groundingdino_swint_ogc.pth"


def _import_grounded_sam2(project: Path):
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    from torchvision.ops import box_convert
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
    return box_convert, build_sam2, SAM2ImagePredictor, load_model, load_image, predict


def _git_head(path: Path) -> str | None:
    head = path / ".git/HEAD"
    if not head.is_file():
        return None
    import subprocess
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def _binary_png(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


def _overlay(rgb: np.ndarray, bird: np.ndarray, base: np.ndarray) -> np.ndarray:
    out = rgb.copy().astype(np.float32)
    color = np.zeros_like(out)
    color[bird] = (255, 30, 30)
    color[base] = (30, 150, 255)
    mask = bird | base
    out[mask] = 0.55 * out[mask] + 0.45 * color[mask]
    return np.clip(out, 0, 255).astype(np.uint8)


def _predict_union(
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
) -> np.ndarray:
    image_source, image = load_image(str(img_path))
    h, w, _ = image_source.shape
    union = np.zeros((h, w), dtype=bool)
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
        boxes = boxes * torch.tensor([w, h, w, h], dtype=boxes.dtype, device=boxes.device)
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").detach().cpu().numpy()
        sam2_predictor.set_image(image_source)
        masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        if masks.size:
            union |= masks.astype(bool).any(axis=0)
    return union


def _write_contact_sheet(output: Path, stems: list[str], columns: int = 5) -> None:
    thumbs = []
    for stem in stems:
        path = output / "overlays" / f"{stem}.png"
        if path.is_file():
            with Image.open(path) as image:
                thumb = image.convert("RGB")
                thumb.thumbnail((320, 180))
                canvas = Image.new("RGB", (320, 180), (0, 0, 0))
                canvas.paste(thumb, ((320 - thumb.width) // 2, (180 - thumb.height) // 2))
                thumbs.append(canvas)
    if not thumbs:
        return
    rows = int(np.ceil(len(thumbs) / columns))
    sheet = Image.new("RGB", (columns * 320, rows * 180), (20, 20, 20))
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % columns) * 320, (index // columns) * 180))
    (output / "contact_sheets").mkdir(parents=True, exist_ok=True)
    sheet.save(output / "contact_sheets" / "chronological_overlays.png")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default=str(ROOT / "data/TiHuBird"))
    parser.add_argument("--images", default="images")
    parser.add_argument("--glass-mask-manifest", default="specular_masks_reviewed_v1/manifest.json")
    parser.add_argument("--output", default=str(ROOT / "output/stage_d_tihubird_internal_object_mask_proposal_v1"))
    parser.add_argument("--grounded-sam2-project", default=str(DEFAULT_GSAM2))
    parser.add_argument("--sam2-checkpoint", default=str(DEFAULT_SAM2_CHECKPOINT))
    parser.add_argument("--sam2-config", default=DEFAULT_SAM2_CONFIG)
    parser.add_argument("--gdino-config", default=str(DEFAULT_GDINO_CONFIG))
    parser.add_argument("--gdino-checkpoint", default=str(DEFAULT_GDINO_CHECKPOINT))
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--stems", nargs="*", default=EXPECTED_STEMS)
    args = parser.parse_args()

    scene = Path(args.scene).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite proposal output: {output}")
    for directory in (
        "raw/bird", "raw/base", "raw/union", "processed/bird",
        "processed/base", "processed/union", "processed/glass_hard",
        "overlays", "contact_sheets", "per_view_metadata",
    ):
        (output / directory).mkdir(parents=True, exist_ok=True)
    stems = list(args.stems)
    unexpected = sorted(set(stems) - set(EXPECTED_STEMS))
    if unexpected:
        raise ValueError(f"unexpected stems: {unexpected}")
    full_111 = stems == EXPECTED_STEMS

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

    glass = validate_specular_mask_set(scene, args.images, args.glass_mask_manifest)
    entries = []
    for stem in stems:
        rgb_path = scene / args.images / f"{stem}.jpg"
        with Image.open(rgb_path) as rgb_image:
            size = rgb_image.size
            rgb = np.asarray(rgb_image.convert("RGB"))
        glass_values, _, _ = load_resized_formal_mask(glass, stem, size, size)
        glass_path = output / "processed" / "glass_hard" / f"{stem}.png"
        _binary_png(glass_path, glass_values >= 0.5)
        bird = _predict_union(
            rgb_path, BIRD_PROMPTS, grounding_model, sam2_predictor, box_convert,
            load_image, predict, args.device, args.box_threshold, args.text_threshold,
        )
        base = _predict_union(
            rgb_path, BASE_PROMPTS, grounding_model, sam2_predictor, box_convert,
            load_image, predict, args.device, args.box_threshold, args.text_threshold,
        )
        raw_root = output / "raw"
        processed_root = output / "processed"
        _binary_png(raw_root / "bird" / f"{stem}.png", bird)
        _binary_png(raw_root / "base" / f"{stem}.png", base)
        _binary_png(raw_root / "union" / f"{stem}.png", bird | base)
        glass_bool = glass_values >= 0.5
        clipped_bird = bird & glass_bool
        clipped_base = base & glass_bool
        clipped_union = (bird | base) & glass_bool
        for role, mask in (("bird", clipped_bird), ("base", clipped_base), ("union", clipped_union)):
            _binary_png(processed_root / role / f"{stem}.png", mask)
        Image.fromarray(_overlay(rgb, clipped_bird, clipped_base)).save(output / "overlays" / f"{stem}.png")
        metadata = proposal_view_metadata(
            stem,
            rgb_path,
            raw_root / "bird" / f"{stem}.png",
            raw_root / "base" / f"{stem}.png",
            glass_path,
        )
        (output / "per_view_metadata").mkdir(parents=True, exist_ok=True)
        (output / "per_view_metadata" / f"{stem}.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        entries.append(metadata)
    _write_contact_sheet(output, stems)
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
        "method": "per-frame independent grounding; no video propagation claimed",
        "full_111": full_111,
    }
    if full_111:
        write_proposal_manifest(output, entries, grounded_sam2, "grounded-sam2-independent-frame-v1")
    else:
        (output / "manifest.partial.json").write_text(
            json.dumps({"role": "partial_fixed_view_probe", "entries": entries, "grounded_sam2": grounded_sam2}, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
