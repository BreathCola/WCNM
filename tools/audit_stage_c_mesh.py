#!/usr/bin/env python3
"""Render and audit D-only geometry for the Stage C glass-mesh gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gaussian_renderer.surfel_renderer import render
from geometry.stage_c_audit import (
    audit_view,
    classify_mesh_audit,
    global_depth_scale,
    make_mask_variants,
    voxel_consistency,
)
from scene import Scene
from scene.diffuse_surfel_model import DiffuseSurfelModel
from utils.specular_mask import validate_specular_mask_set


SCHEMA = "rtgs_stage_c_mesh_audit_v1"
EXPECTED_CHECKPOINT_SHA256 = "050500d607e1910ca088049ae73619949ad183e23c85354a8408bb29571fbe84"
REPRESENTATIVE_STEMS = {
    "000000", "000010", "000020", "000030", "000039", "000040",
    "000041", "000050", "000060", "000075", "000090", "000110",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def load_diffuse(
    checkpoint_path: Path, candidate_ks_min: float | None
) -> tuple[DiffuseSurfelModel, dict, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if (
        checkpoint.get("format") != "rtgs_stage_b"
        or checkpoint.get("global_iteration") != 15000
        or checkpoint.get("reflection_iteration") != 12000
        or checkpoint.get("config", {}).get("resolution") != 8
    ):
        raise ValueError("Stage C audit requires the selected 3k-A global-15000 checkpoint")
    state = checkpoint["diffuse"]
    selection = torch.ones(state["xyz"].shape[0], dtype=torch.bool)
    if candidate_ks_min is not None:
        selection = torch.sigmoid(state["ks_raw"].detach().cpu().reshape(-1)) >= candidate_ks_min
        if not selection.any():
            raise ValueError("candidate_ks_min selected no Diffuse surfels")
    model = DiffuseSurfelModel(checkpoint["config"].get("roughness_min", 0.03))
    mapping = {
        "_xyz": "xyz", "_rotation": "rotation", "_scaling": "scaling_2d",
        "_opacity": "opacity_raw", "_base_color": "base_color_raw",
        "_roughness": "roughness_raw", "_f0": "f0_raw", "_ks": "ks_raw",
    }
    for attribute, key in mapping.items():
        setattr(
            model, attribute,
            state[key].detach().cpu()[selection].to(device="cuda", dtype=torch.float32),
        )
    model._exposure = state["exposure"].detach().to(device="cuda", dtype=torch.float32)
    model.exposure_mapping = dict(state["exposure_mapping"])
    return model, checkpoint, {
        "candidate_ks_min": candidate_ks_min,
        "source_diffuse_count": int(state["xyz"].shape[0]),
        "selected_diffuse_count": int(selection.sum()),
        "selected_fraction": float(selection.float().mean()),
    }


def scene_args(args, manifest: Path, validated_manifest: dict) -> SimpleNamespace:
    return SimpleNamespace(
        model_path=str(args.output), source_path=str(args.source), images="images", depths="",
        eval=False, train_test_exp=False, white_background=False, resolution=args.resolution,
        data_device="cpu", normal_priors="__stage_c_no_normal_prior__",
        normal_prior_space="camera", specular_masks=str(manifest),
        _validated_specular_mask_manifest=validated_manifest,
    )


def colorize_depth(depth: np.ndarray, valid: np.ndarray, lo: float, hi: float) -> np.ndarray:
    normalized = np.clip((depth - lo) / max(hi - lo, 1e-8), 0, 1)
    image = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image[~valid] = 0
    return image


def write_source_page(
    path: Path, rgb_path: Path, depth: np.ndarray, alpha: np.ndarray,
    normal: np.ndarray, masks, depth_scale: tuple[float, float]
) -> None:
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    h, w = rgb.shape[:2]
    size = (w // 2, h // 2)
    hard = cv2.resize(masks.hard.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    overlay = rgb.copy()
    overlay[hard] = (0.55 * overlay[hard] + 0.45 * np.array([255, 32, 32])).astype(np.uint8)
    valid = np.isfinite(depth) & (depth > 0) & (alpha > 1e-4)
    depth_rgb = colorize_depth(depth, valid, *depth_scale)
    normal_rgb = np.clip((normal + 1.0) * 127.5, 0, 255).astype(np.uint8)
    alpha_rgb = np.repeat(np.clip(alpha[..., None] * 255, 0, 255).astype(np.uint8), 3, axis=2)
    panels = [overlay, depth_rgb, normal_rgb, alpha_rgb]
    panels = [cv2.resize(panel, size, interpolation=cv2.INTER_LINEAR) for panel in panels]
    page = np.concatenate((np.concatenate(panels[:2], axis=1), np.concatenate(panels[2:], axis=1)), axis=0)
    Image.fromarray(page).save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--mask-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resolution", type=int, default=8)
    parser.add_argument("--erode-pixels", type=int, default=2)
    parser.add_argument("--candidate-ks-min", type=float)
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.source = args.source.resolve()
    args.mask_manifest = args.mask_manifest.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite Stage C audit output: {args.output}")
    if args.resolution != 8:
        raise ValueError("the selected geometry source was trained at resolution=8")
    if args.candidate_ks_min is not None and not 0.0 < args.candidate_ks_min < 1.0:
        raise ValueError("candidate_ks_min must lie in (0, 1)")
    checkpoint_hash = sha256_file(args.checkpoint)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("selected Stage C checkpoint hash mismatch")
    manifest = json.loads(args.mask_manifest.read_text(encoding="utf-8"))
    entries = {entry["stem"]: entry for entry in manifest["entries"]}
    if len(entries) != 111:
        raise ValueError("formal mask manifest must contain exactly 111 views")
    validated_manifest = validate_specular_mask_set(
        args.source, "images", args.mask_manifest
    )

    args.output.mkdir(parents=True)
    (args.output / "raw_views").mkdir()
    (args.output / "source_resolution_pages").mkdir()

    diffuse, checkpoint, selection = load_diffuse(args.checkpoint, args.candidate_ks_min)
    scene = Scene(
        scene_args(args, args.mask_manifest, validated_manifest), diffuse,
        shuffle=False, initialize_model=False, write_metadata=False,
    )
    cameras = scene.getTrainCameras()
    if len(cameras) != 111:
        raise ValueError(f"expected 111 cameras, found {len(cameras)}")
    pipe = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
    background = torch.zeros(3, dtype=torch.float32, device="cuda")

    rows, samples, points_by_view, rendered = [], [], [], {}
    with torch.no_grad():
        for camera in cameras:
            stem = Path(camera.image_name).stem
            if stem not in entries:
                raise ValueError(f"camera {stem} missing from formal mask manifest")
            package = render(camera, diffuse, pipe, background)
            depth = package["depth"][..., 0].detach().cpu().numpy().astype(np.float32)
            alpha = package["alpha"][..., 0].detach().cpu().numpy().astype(np.float32)
            normal = package["normal"].detach().cpu().numpy().astype(np.float32)
            position = package["position"].detach().cpu().numpy().astype(np.float32)
            soft = camera.specular_mask[0].detach().cpu().numpy().astype(np.float32)
            masks = make_mask_variants(soft, erode_pixels=args.erode_pixels)
            row = {"stem": stem, **audit_view(depth, alpha, normal, masks)}
            rows.append(row)
            valid = masks.eroded & np.isfinite(depth) & (depth > 0) & (alpha > 1e-4)
            samples.append(depth[valid][::8])
            points_by_view.append(position[valid][::4])
            np.savez_compressed(
                args.output / "raw_views" / f"{stem}.npz",
                depth=depth, alpha=alpha.astype(np.float16), normal=normal.astype(np.float16),
                mask_soft=masks.soft.astype(np.float16), mask_hard=masks.hard,
                mask_eroded=masks.eroded, camera_center=camera.camera_center.detach().cpu().numpy(),
                world_view_transform=camera.world_view_transform.detach().cpu().numpy(),
                full_proj_transform=camera.full_proj_transform.detach().cpu().numpy(),
            )
            if stem in REPRESENTATIVE_STEMS:
                rendered[stem] = (depth, alpha, normal, masks)
            del package

    scale = global_depth_scale(samples)
    for stem, values in rendered.items():
        write_source_page(
            args.output / "source_resolution_pages" / f"{stem}.png",
            args.source / entries[stem]["rgb_path"], *values, scale,
        )
    voxel = voxel_consistency(points_by_view)
    verdict, failures = classify_mesh_audit(rows, voxel)
    summary = {
        "schema": SCHEMA, "verdict": verdict, "failures": failures,
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": checkpoint_hash,
        "global_iteration": checkpoint["global_iteration"],
        "reflection_local_iteration": checkpoint["reflection_iteration"],
        "geometry_source": "Diffuse state only; Reflection state was not rendered or traced",
        "diffuse_selection": selection,
        "geometry_resolution": [cameras[0].image_width, cameras[0].image_height],
        "source_resolution": entries[next(iter(entries))]["size"],
        "mask_manifest": str(args.mask_manifest.resolve()),
        "mask_aggregate_sha256": manifest["aggregate_mask_sha256"],
        "mask_policy": {
            "soft": "formal reviewed mask; L_spec only",
            "hard": "soft >= 0.5; Stage C two-hit ray domain",
            "eroded": f"hard eroded {args.erode_pixels} geometry pixels; mesh/depth validation",
            "reflection": "unchanged full valid-D-surface domain",
        },
        "depth_display_scale": {"p01": scale[0], "p99": scale[1], "shared_by_all_pages": True},
        "per_view": rows, "voxel_consistency": voxel,
        "representative_source_resolution_pages": sorted(rendered),
    }
    atomic_json(args.output / "mesh_audit.json", summary)
    print(json.dumps({"verdict": verdict, "failures": failures, "output": str(args.output)}, indent=2))
    return 0 if verdict == "STAGE_C_MESH_AUDIT_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
