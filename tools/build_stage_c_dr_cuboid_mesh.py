#!/usr/bin/env python3
"""Build an independently versioned DR-guided, mask-constrained glass cuboid."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from plyfile import PlyData

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometry.dr_cuboid import (
    cuboid_vertices_faces, decode_c03_normal, fit_depth_calibration,
    fit_orthogonal_axes, intersect_cuboid_near, mask_metrics,
    optimize_cuboid_bounds, projected_mask,
)
from geometry.tsdf_fusion import (
    camera_intrinsics_from_transforms, mesh_topology, write_mesh_ply,
)


REPRESENTATIVE = {
    "000000", "000039", "000040", "000041", "000053", "000063", "000075", "000110",
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


def load_vertices(path: Path) -> np.ndarray:
    vertex = PlyData.read(path)["vertex"].data
    return np.stack([vertex[name] for name in ("x", "y", "z")], axis=1).astype(np.float64)


def aggregate(rows: list[dict], key: str) -> dict:
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    return {
        "minimum": float(values.min()), "median": float(np.median(values)),
        "mean": float(values.mean()), "maximum": float(values.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-d-audit", required=True, type=Path)
    parser.add_argument("--dr-raw", required=True, type=Path)
    parser.add_argument("--mask-manifest", required=True, type=Path)
    parser.add_argument("--source-mesh", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    for name in ("full_d_audit", "dr_raw", "mask_manifest", "source_mesh", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite DR cuboid output: {args.output}")
    raw_paths = sorted((args.full_d_audit / "raw_views").glob("*.npz"))
    if len(raw_paths) != 111:
        raise ValueError("full-D audit must contain exactly 111 raw views")
    dr_manifest_path = args.dr_raw / "manifest.json"
    dr_manifest = json.loads(dr_manifest_path.read_text(encoding="utf-8"))
    dr_entries = {
        Path(row["rtgs_input_file"]).stem: row
        for row in dr_manifest["frames_and_padding"] if row.get("frame_kind") == "real"
    }
    mask_manifest = json.loads(args.mask_manifest.read_text(encoding="utf-8"))
    mask_entries = {row["stem"]: row for row in mask_manifest["entries"]}
    if len(dr_entries) != 111 or len(mask_entries) != 111:
        raise ValueError("DR and mask manifests must each map 111 real views")

    normals, views, depth_records = [], [], {}
    for path in raw_paths:
        stem = path.stem
        with np.load(path, allow_pickle=False) as archive:
            raw = {key: archive[key] for key in archive.files}
        depth = raw["depth"].astype(np.float64)
        alpha = raw["alpha"].astype(np.float64)
        hard = raw["mask_hard"].astype(bool)
        eroded = raw["mask_eroded"].astype(bool)
        height, width = depth.shape
        intrinsics, rotation, camera_center = camera_intrinsics_from_transforms(
            raw["world_view_transform"], raw["full_proj_transform"], width, height
        )
        entry = dr_entries[stem]
        depth_path = args.dr_raw / entry["raw_prior_files"]["depth"]
        normal_path = args.dr_raw / entry["raw_prior_files"]["normal"]
        with Image.open(depth_path) as opened:
            resized_depth = np.asarray(
                opened.convert("RGB").resize((width, height), Image.Resampling.BILINEAR),
                dtype=np.float32,
            ).mean(axis=-1) / 255.0
        with Image.open(normal_path) as opened:
            resized_normal_rgb = np.asarray(
                opened.convert("RGB").resize((width, height), Image.Resampling.BILINEAR),
                dtype=np.uint8,
            )
        camera_normal = decode_c03_normal(resized_normal_rgb)
        world_normal = camera_normal @ rotation.T
        selected = world_normal[eroded][::4]
        normals.append(selected)
        calibration_support = (
            (~hard) & np.isfinite(depth) & (depth > 0) & (alpha > 1e-3)
        )
        calibration = fit_depth_calibration(resized_depth, depth, calibration_support)
        calibrated_depth = calibration.apply(resized_depth)
        views.append({
            "stem": stem, "mask": hard.astype(np.uint8), "intrinsics": intrinsics,
            "rotation": rotation, "camera_center": camera_center,
        })
        depth_records[stem] = {
            "calibration": calibration, "calibrated_depth": calibrated_depth,
            "mask_eroded": eroded, "intrinsics": intrinsics,
            "rotation": rotation, "camera_center": camera_center,
        }

    axes, axis_report = fit_orthogonal_axes(np.concatenate(normals, axis=0))
    source_vertices = load_vertices(args.source_mesh)
    source_local = source_vertices @ axes
    initial_lower = np.quantile(source_local, 0.02, axis=0)
    initial_upper = np.quantile(source_local, 0.98, axis=0)
    lower, upper, optimization = optimize_cuboid_bounds(
        axes, initial_lower, initial_upper, views
    )
    vertices, faces = cuboid_vertices_faces(axes, lower, upper)
    topology = mesh_topology(vertices, faces)

    geometry_rows = []
    for view, mask_row in zip(views, optimization["per_view"]):
        stem = view["stem"]
        record = depth_records[stem]
        eroded = record["mask_eroded"]
        ys, xs = np.nonzero(eroded)
        ys, xs = ys[::8], xs[::8]
        pixels = np.stack((xs, ys, np.ones_like(xs)), axis=1).astype(np.float64)
        camera_direction = pixels @ np.linalg.inv(record["intrinsics"]).T
        world_direction = camera_direction @ record["rotation"].T
        origin = np.broadcast_to(record["camera_center"], world_direction.shape)
        near, hit = intersect_cuboid_near(axes, lower, upper, origin, world_direction)
        dr_depth = record["calibrated_depth"][ys, xs]
        comparable = hit & np.isfinite(dr_depth) & (dr_depth > 0.5) & (dr_depth < 10.0)
        absolute = np.abs(near[comparable] - dr_depth[comparable])
        relative = absolute / np.maximum(dr_depth[comparable], 1e-6)
        calibration = record["calibration"]
        geometry_rows.append({
            "stem": stem, **mask_row, "calibration_mode": calibration.mode,
            "calibration_slope": calibration.slope,
            "calibration_intercept": calibration.intercept,
            "calibration_r2": calibration.r2,
            "depth_comparable_count": int(comparable.sum()),
            "cuboid_vs_calibrated_dr_depth_abs_p50": float(np.median(absolute)),
            "cuboid_vs_calibrated_dr_depth_abs_p95": float(np.quantile(absolute, 0.95)),
            "cuboid_vs_calibrated_dr_depth_rel_p50": float(np.median(relative)),
        })

    source_rows = []
    debug_payload = {}
    for view in views:
        stem = view["stem"]
        entry = mask_entries[stem]
        source_width, source_height = entry["size"]
        with np.load(
            args.full_d_audit / "raw_views" / f"{stem}.npz", allow_pickle=False
        ) as archive:
            intrinsics, rotation, center = camera_intrinsics_from_transforms(
                archive["world_view_transform"], archive["full_proj_transform"],
                source_width, source_height,
            )
        with Image.open(args.mask_manifest.parent / entry["mask_path"]) as opened:
            target = np.asarray(opened.convert("L"), dtype=np.uint8) >= 128
        prediction = projected_mask(
            vertices, intrinsics, rotation, center, (source_height, source_width)
        ).astype(bool)
        source_rows.append({"stem": stem, **mask_metrics(prediction, target)})
        if stem in REPRESENTATIVE:
            debug_payload[stem] = (prediction, target, entry)

    args.output.mkdir(parents=True)
    debug_dir = args.output / "source_resolution_mask_fit"
    debug_dir.mkdir()
    for stem, (prediction, target, entry) in debug_payload.items():
        with Image.open(args.mask_manifest.parent.parent / entry["rgb_path"]) as opened:
            image = np.asarray(opened.convert("RGB"), dtype=np.uint8).copy()
        target_edge = cv2.morphologyEx(target.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)).astype(bool)
        predicted_edge = cv2.morphologyEx(prediction.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)).astype(bool)
        image[target_edge] = [255, 32, 32]
        image[predicted_edge] = [32, 255, 32]
        image[target_edge & predicted_edge] = [255, 255, 32]
        Image.fromarray(image).save(debug_dir / f"{stem}.png")

    write_mesh_ply(args.output / "glass_mesh.ply", vertices, faces)
    source_summary = {key: aggregate(source_rows, key) for key in ("recall", "precision", "iou")}
    geometry_summary = {
        "calibration_r2": aggregate(geometry_rows, "calibration_r2"),
        "cuboid_vs_calibrated_dr_depth_abs_p50": aggregate(
            geometry_rows, "cuboid_vs_calibrated_dr_depth_abs_p50"
        ),
        "cuboid_vs_calibrated_dr_depth_rel_p50": aggregate(
            geometry_rows, "cuboid_vs_calibrated_dr_depth_rel_p50"
        ),
        "direct_depth_calibration_views": int(sum(
            row["calibration_mode"] == "depth" for row in geometry_rows
        )),
        "inverse_depth_calibration_views": int(sum(
            row["calibration_mode"] == "inverse_depth" for row in geometry_rows
        )),
    }
    ready = bool(
        topology["watertight"]
        and source_summary["recall"]["minimum"] >= 0.90
        and source_summary["iou"]["mean"] >= 0.93
        and geometry_summary["calibration_r2"]["minimum"] >= 0.30
    )
    report = {
        "schema": "rtgs_stage_c_dr_cuboid_mesh_v1",
        "method": "DR-normal axes + D/TSDF metric initialization + formal-mask fit + calibrated DR-depth audit",
        "checkpoint_sha256": json.loads(
            (args.source_mesh.parent / "mesh_metadata.json").read_text(encoding="utf-8")
        )["checkpoint_sha256"],
        "dr_manifest": str(dr_manifest_path), "dr_manifest_sha256": sha256_file(dr_manifest_path),
        "mask_manifest": str(args.mask_manifest), "mask_manifest_sha256": sha256_file(args.mask_manifest),
        "source_mesh": str(args.source_mesh), "source_mesh_sha256": sha256_file(args.source_mesh),
        "axes_columns": axes.tolist(), "axis_fit": axis_report,
        "initial_lower": initial_lower.tolist(), "initial_upper": initial_upper.tolist(),
        "fitted_lower": lower.tolist(), "fitted_upper": upper.tolist(),
        "optimization": {key: value for key, value in optimization.items() if key != "per_view"},
        "topology": topology, "geometry_resolution_per_view": geometry_rows,
        "source_resolution_per_view": source_rows,
        "source_resolution_summary": source_summary,
        "dr_depth_summary": geometry_summary,
        "two_hit_ready": ready,
        "acceptance_gate": {
            "watertight": True, "source_recall_minimum": 0.90,
            "source_iou_mean": 0.93, "dr_depth_calibration_r2_minimum": 0.30,
        },
    }
    atomic_json(args.output / "mesh_metadata.json", report)
    print(json.dumps({
        "two_hit_ready": ready, "source_resolution_summary": source_summary,
        "dr_depth_summary": geometry_summary, "topology": topology,
    }, indent=2))
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
