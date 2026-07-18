#!/usr/bin/env python3
"""Build a Tao-owned COLMAP/DR/reviewed-mask cuboid and audit two-hit cache.

This tool never reads a training checkpoint.  It is intentionally impossible
to run before a complete formally reviewed glass-mask manifest exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometry.dr_cuboid import (
    cuboid_vertices_faces, fit_depth_calibration, mask_metrics,
    optimize_cuboid_bounds, projected_mask,
)
from geometry.tao_cuboid import (
    GEOMETRY_SCHEMA, RUNTIME_INTERSECTION_SCHEMA, intersect_cuboid_numpy,
    select_normal_axis_convention,
)
from geometry.tsdf_fusion import mesh_topology, write_mesh_ply
from scene.colmap_loader import (
    qvec2rotmat, read_extrinsics_binary, read_intrinsics_binary,
)
from utils.dr_mask_proposal import audit_dr_artifacts, sha256_file


RELEASE_ID = "stage_c_tao_geometry_release_v1"
FORMAL_MASK_SCHEMA = "rtgs_tao_reviewed_glass_masks_v1"


def _atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_points3d(path: Path) -> dict[int, np.ndarray]:
    points = {}
    with path.open("rb") as handle:
        count = struct.unpack("<Q", handle.read(8))[0]
        for _ in range(count):
            point_id, x, y, z, _r, _g, _b, _error = struct.unpack(
                "<QdddBBBd", handle.read(43)
            )
            track_length = struct.unpack("<Q", handle.read(8))[0]
            handle.seek(8 * track_length, 1)
            points[int(point_id)] = np.array([x, y, z], dtype=np.float64)
    return points


def _intrinsics(camera) -> np.ndarray:
    if camera.model != "PINHOLE" or len(camera.params) != 4:
        raise ValueError("Tao geometry requires one undistorted PINHOLE camera")
    fx, fy, cx, cy = camera.params
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def _camera_record(image, camera) -> dict:
    world_to_camera = qvec2rotmat(image.qvec)
    center = -world_to_camera.T @ image.tvec
    return {
        "stem": Path(image.name).stem,
        "width": int(camera.width), "height": int(camera.height),
        "intrinsics": _intrinsics(camera),
        "world_to_camera": world_to_camera,
        "projection_rotation": world_to_camera.T,
        "camera_center": center,
        "xys": np.asarray(image.xys, dtype=np.float64),
        "point3D_ids": np.asarray(image.point3D_ids, dtype=np.int64),
    }


def _validate_formal_masks(scene: Path, manifest_path: Path, stems: list[str]) -> dict:
    manifest_path = manifest_path.expanduser().resolve()
    if not manifest_path.is_file():
        raise RuntimeError("BLOCKED_BY_GLASS_MASK_IDENTITY: formal manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "schema": FORMAL_MASK_SCHEMA,
        "artifact_role": "formal_reviewed_glass_masks",
        "human_status": "accepted",
        "scene": scene.name,
        "count": len(stems),
        "ordered_stems": stems,
    }
    for key, value in required.items():
        if manifest.get(key) != value:
            raise RuntimeError(
                f"BLOCKED_BY_GLASS_MASK_IDENTITY: {key}={manifest.get(key)!r} != {value!r}"
            )
    entries = manifest.get("entries")
    if not isinstance(entries, list) or [row.get("stem") for row in entries] != stems:
        raise RuntimeError("BLOCKED_BY_GLASS_MASK_IDENTITY: mask entries are incomplete")
    by_stem = {}
    for row in entries:
        stem = row["stem"]
        rgb_path = scene / "images" / f"{stem}.jpg"
        if not rgb_path.is_file() or row.get("rgb_sha256") != sha256_file(rgb_path):
            raise RuntimeError(f"BLOCKED_BY_GLASS_MASK_IDENTITY: {stem} RGB hash mismatch")
        role_paths = {}
        for role in ("mask_soft", "mask_hard", "mask_eroded"):
            relative = row.get(f"{role}_path")
            expected_hash = row.get(f"{role}_sha256")
            path = manifest_path.parent / str(relative)
            if not path.is_file() or expected_hash != sha256_file(path):
                raise RuntimeError(f"BLOCKED_BY_GLASS_MASK_IDENTITY: {stem} {role} mismatch")
            with Image.open(path) as opened:
                if opened.mode != "L" or opened.size != (2320, 2032):
                    raise RuntimeError(
                        f"BLOCKED_BY_GLASS_MASK_IDENTITY: {stem} {role} mode/size mismatch"
                    )
            role_paths[role] = path
        by_stem[stem] = {"record": row, **role_paths}
    return {
        "path": manifest_path, "sha256": sha256_file(manifest_path),
        "manifest": manifest, "by_stem": by_stem,
    }


def _aggregate(rows: list[dict], key: str) -> dict:
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    return {
        "minimum": float(values.min()), "mean": float(values.mean()),
        "median": float(np.median(values)), "maximum": float(values.max()),
    }


def _camera_rays(view: dict) -> tuple[np.ndarray, np.ndarray]:
    height, width = view["height"], view["width"]
    yy, xx = np.mgrid[:height, :width]
    pixels = np.stack((xx, yy, np.ones_like(xx)), axis=-1).reshape(-1, 3)
    camera_direction = pixels @ np.linalg.inv(view["intrinsics"]).T
    world_direction = camera_direction @ view["world_to_camera"]
    world_direction /= np.linalg.norm(world_direction, axis=1, keepdims=True)
    origins = np.broadcast_to(view["camera_center"], world_direction.shape)
    return origins, world_direction


def _display_depth(values: np.ndarray, valid: np.ndarray) -> Image.Image:
    display = np.zeros(values.shape, dtype=np.uint8)
    if np.any(valid):
        lo, hi = np.quantile(values[valid], (0.01, 0.99))
        normalized = np.clip((values - lo) / max(hi - lo, 1e-8), 0, 1)
        display[valid] = np.round(normalized[valid] * 255).astype(np.uint8)
    return Image.fromarray(display, mode="L")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=ROOT / "data/Tao")
    parser.add_argument("--dr-raw", type=Path, default=ROOT / "output/stage_a_tao_dr_raw_112_v1")
    parser.add_argument("--reviewed-mask-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "output/stage_c_tao_geometry_release_v1")
    parser.add_argument("--fit-max-dimension", type=int, default=512)
    parser.add_argument("--minimum-mean-iou", type=float, default=0.90)
    parser.add_argument("--minimum-view-iou", type=float, default=0.75)
    parser.add_argument("--minimum-mean-two-hit-coverage", type=float, default=0.95)
    parser.add_argument("--minimum-view-two-hit-coverage", type=float, default=0.85)
    parser.add_argument("--minimum-normal-alignment-p50", type=float, default=0.75)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scene = args.scene.expanduser().resolve()
    raw_root = args.dr_raw.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing Tao geometry output: {output}")
    audit = audit_dr_artifacts(scene, raw_root, expected_count=None, images="images")
    stems = [row["stem"] for row in audit["entries"]]
    masks = _validate_formal_masks(scene, args.reviewed_mask_manifest, stems)
    sparse = scene / "sparse/0"
    cameras = read_intrinsics_binary(str(sparse / "cameras.bin"))
    images = read_extrinsics_binary(str(sparse / "images.bin"))
    points = _read_points3d(sparse / "points3D.bin")
    by_name = {Path(image.name).stem: image for image in images.values()}
    if sorted(by_name) != stems:
        raise ValueError("COLMAP registrations do not match formal mask stems")
    views = [_camera_record(by_name[stem], cameras[by_name[stem].camera_id]) for stem in stems]
    dr_by_stem = {row["stem"]: row for row in audit["entries"]}
    normal_samples = []
    masked_sparse = []
    depth_calibration_rows = []
    for view in views:
        stem = view["stem"]
        with Image.open(masks["by_stem"][stem]["mask_hard"]) as opened:
            hard = np.asarray(opened, dtype=np.uint8) >= 128
        normal_path = Path(dr_by_stem[stem]["artifacts"]["normal"]["path"])
        depth_path = Path(dr_by_stem[stem]["artifacts"]["depth"]["path"])
        with Image.open(normal_path) as opened:
            normal_rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8)
        with Image.open(depth_path) as opened:
            dr_depth = np.asarray(opened.convert("RGB"), dtype=np.float64).mean(axis=-1) / 255.0
        native_size = (normal_rgb.shape[1], normal_rgb.shape[0])
        native_mask = cv2.resize(hard.astype(np.uint8), native_size, interpolation=cv2.INTER_NEAREST) > 0
        # Row-vector camera normals transform to world by ``n_camera @ R``;
        # select_normal_axis_convention receives camera-to-world and applies its
        # transpose, hence the explicit transpose here.
        normal_samples.append((normal_rgb, view["world_to_camera"].T, native_mask))
        ids = view["point3D_ids"]
        valid_obs = np.array([int(point_id) in points for point_id in ids], dtype=bool)
        xys = view["xys"][valid_obs]
        xyz = np.stack([points[int(point_id)] for point_id in ids[valid_obs]], axis=0)
        xi = np.clip(np.rint(xys[:, 0]).astype(int), 0, view["width"] - 1)
        yi = np.clip(np.rint(xys[:, 1]).astype(int), 0, view["height"] - 1)
        inside = hard[yi, xi]
        masked_sparse.append(xyz[inside])
        sx = np.clip(np.rint(xys[:, 0] * native_size[0] / view["width"]).astype(int), 0, native_size[0] - 1)
        sy = np.clip(np.rint(xys[:, 1] * native_size[1] / view["height"]).astype(int), 0, native_size[1] - 1)
        relative = dr_depth[sy, sx]
        metric = np.linalg.norm(xyz - view["camera_center"][None], axis=1)
        support = np.ones(relative.shape, dtype=bool)
        calibration = fit_depth_calibration(relative, metric, support)
        depth_calibration_rows.append({
            "stem": stem, "projectable_sparse_points": int(relative.size),
            "mode": calibration.mode, "slope": calibration.slope,
            "intercept": calibration.intercept, "r2": calibration.r2,
        })
    mapping, axes, normal_report = select_normal_axis_convention(normal_samples)
    if normal_report["selected_fit"]["alignment_p50"] < args.minimum_normal_alignment_p50:
        raise RuntimeError("BLOCKED_BY_GEOMETRY_IDENTITY: DR normal-axis fit is below threshold")
    sparse_points = np.concatenate(masked_sparse, axis=0)
    local = sparse_points @ axes
    lower = np.quantile(local, 0.02, axis=0)
    upper = np.quantile(local, 0.98, axis=0)
    expansion = np.maximum((upper - lower) * 0.15, 1e-4)
    lower -= expansion
    upper += expansion
    fit_views = []
    for view in views:
        stem = view["stem"]
        with Image.open(masks["by_stem"][stem]["mask_hard"]) as opened:
            hard = np.asarray(opened, dtype=np.uint8) >= 128
        scale = min(1.0, args.fit_max_dimension / max(view["width"], view["height"]))
        fit_width = max(32, int(round(view["width"] * scale)))
        fit_height = max(32, int(round(view["height"] * scale)))
        fit_mask = cv2.resize(hard.astype(np.uint8), (fit_width, fit_height), interpolation=cv2.INTER_NEAREST)
        fit_k = view["intrinsics"].copy()
        fit_k[0] *= fit_width / view["width"]
        fit_k[1] *= fit_height / view["height"]
        fit_views.append({
            "stem": stem, "mask": fit_mask, "intrinsics": fit_k,
            "rotation": view["projection_rotation"], "camera_center": view["camera_center"],
        })
    lower, upper, optimization = optimize_cuboid_bounds(axes, lower, upper, fit_views)
    vertices, faces = cuboid_vertices_faces(axes, lower, upper)
    full_rows = []
    projected_by_stem = {}
    for view in views:
        stem = view["stem"]
        with Image.open(masks["by_stem"][stem]["mask_hard"]) as opened:
            hard = np.asarray(opened, dtype=np.uint8) >= 128
        predicted = projected_mask(
            vertices, view["intrinsics"], view["projection_rotation"],
            view["camera_center"], hard.shape,
        )
        projected_by_stem[stem] = predicted
        full_rows.append({"stem": stem, **mask_metrics(predicted, hard)})
    output.mkdir(parents=True)
    cache_root = output / "audit_two_hit_cache"
    debug_root = output / "fixed_view_review"
    cache_root.mkdir()
    debug_root.mkdir()
    coverage_rows = []
    fixed = {stems[index] for index in np.linspace(0, len(stems) - 1, 9, dtype=int)}
    for view in views:
        stem = view["stem"]
        with Image.open(masks["by_stem"][stem]["mask_hard"]) as opened:
            hard = np.asarray(opened, dtype=np.uint8) >= 128
        origins, directions = _camera_rays(view)
        hits = intersect_cuboid_numpy(axes, lower, upper, origins, directions)
        shape = (view["height"], view["width"])
        cache = {name: values.reshape(shape + values.shape[1:]) for name, values in hits.items()}
        valid = cache["valid_two_hit"] & hard
        hard_count = int(hard.sum())
        coverage = float(valid.sum() / max(hard_count, 1))
        finite = bool(all(np.isfinite(cache[name][valid]).all() for name in (
            "t_near", "t_far", "front_position", "back_position", "front_normal", "back_normal"
        )))
        ordered = bool(np.all(cache["t_far"][valid] > cache["t_near"][valid]))
        crossing_ok = bool(np.all(cache["crossing_count"][valid] == 2))
        coverage_rows.append({
            "stem": stem, "hard_pixels": hard_count, "valid_two_hit_pixels": int(valid.sum()),
            "hard_mask_two_hit_coverage": coverage, "finite": finite,
            "t_far_gt_t_near": ordered, "crossing_count_two": crossing_ok,
        })
        np.savez_compressed(
            cache_root / f"{stem}.npz", schema=np.array(RUNTIME_INTERSECTION_SCHEMA),
            axes=axes.astype(np.float64), lower=lower.astype(np.float64), upper=upper.astype(np.float64),
            **cache,
        )
        if stem in fixed:
            _display_depth(cache["t_near"], valid).save(debug_root / f"{stem}_near.png")
            _display_depth(cache["t_far"], valid).save(debug_root / f"{stem}_far.png")
            Image.fromarray(valid.astype(np.uint8) * 255).save(debug_root / f"{stem}_valid.png")
            rgb_path = scene / "images" / f"{stem}.jpg"
            with Image.open(rgb_path) as opened:
                review = np.asarray(opened.convert("RGB"), dtype=np.uint8).copy()
            predicted = projected_by_stem[stem]
            overlap = hard & predicted
            false_positive = predicted & ~hard
            false_negative = hard & ~predicted
            tint = review.astype(np.float32)
            tint[overlap] = 0.55 * tint[overlap] + 0.45 * np.array([0, 255, 100])
            tint[false_positive] = 0.45 * tint[false_positive] + 0.55 * np.array([255, 40, 40])
            tint[false_negative] = 0.45 * tint[false_negative] + 0.55 * np.array([40, 80, 255])
            Image.fromarray(np.round(tint).astype(np.uint8)).save(
                debug_root / f"{stem}_silhouette_overlay.png"
            )
    topology = mesh_topology(vertices, faces)
    write_mesh_ply(output / "glass_cuboid.ply", vertices, faces)
    thresholds = {
        "minimum_mean_iou": args.minimum_mean_iou,
        "minimum_view_iou": args.minimum_view_iou,
        "minimum_mean_two_hit_coverage": args.minimum_mean_two_hit_coverage,
        "minimum_view_two_hit_coverage": args.minimum_view_two_hit_coverage,
        "minimum_normal_alignment_p50": args.minimum_normal_alignment_p50,
        "finite_required": 1.0, "t_far_gt_t_near_required": 1.0,
        "cuboid_crossing_count_required": 2,
        "fixed_view_count": len(fixed),
        "fixed_view_products_per_view": 4,
    }
    iou = _aggregate(full_rows, "iou")
    coverage = _aggregate(coverage_rows, "hard_mask_two_hit_coverage")
    gates = {
        "mean_iou": iou["mean"] >= args.minimum_mean_iou,
        "minimum_iou": iou["minimum"] >= args.minimum_view_iou,
        "mean_two_hit_coverage": coverage["mean"] >= args.minimum_mean_two_hit_coverage,
        "minimum_two_hit_coverage": coverage["minimum"] >= args.minimum_view_two_hit_coverage,
        "finite_100_percent": all(row["finite"] for row in coverage_rows),
        "t_far_gt_t_near_100_percent": all(row["t_far_gt_t_near"] for row in coverage_rows),
        "crossing_count": all(row["crossing_count_two"] for row in coverage_rows),
        "watertight_cuboid": (
            topology["boundary_edge_count"] == 0
            and topology["nonmanifold_edge_count"] == 0
        ),
        "fixed_view_review_products": len(list(debug_root.glob("*.png"))) == 4 * len(fixed),
    }
    metadata = {
        "schema": GEOMETRY_SCHEMA,
        "scene": scene.name, "release_id": RELEASE_ID,
        "inputs": {
            "colmap": {name: sha256_file(sparse / name) for name in ("cameras.bin", "images.bin", "points3D.bin")},
            "sparse_point_count": len(points),
            "reviewed_mask_manifest": str(masks["path"]),
            "reviewed_mask_manifest_sha256": masks["sha256"],
            "dr_manifest_sha256": audit["raw_manifest_sha256"],
            "rgb_count": len(stems),
            "training_checkpoint_used": False,
        },
        "normal_axis_search": normal_report,
        "selected_normal_mapping": mapping.tolist(),
        "axes": axes.tolist(), "lower": lower.tolist(), "upper": upper.tolist(),
        "depth_calibration": depth_calibration_rows,
        "sparse_initialization_point_count": int(sparse_points.shape[0]),
        "silhouette_optimization": optimization,
        "source_resolution_mask_metrics": full_rows,
        "source_resolution_iou": iou,
        "two_hit_metrics": coverage_rows,
        "two_hit_coverage": coverage,
        "topology": topology, "thresholds_predeclared": thresholds,
        "gates": gates,
        "runtime_intersection": RUNTIME_INTERSECTION_SCHEMA,
        "audit_cache_role": "parity_and_review_only_not_required_for_novel_views",
    }
    passed = all(gates.values())
    metadata["verdict"] = "TAO_GEOMETRY_RELEASE_VALID" if passed else "BLOCKED_BY_GEOMETRY_IDENTITY"
    _atomic_json(output / "geometry_metadata.json", metadata)
    if not passed:
        print(json.dumps({"output": str(output), "verdict": metadata["verdict"], "gates": gates}, indent=2))
        return 2
    files = sorted(path for path in output.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    file_rows = []
    for path in files:
        relative = path.relative_to(output).as_posix()
        value = sha256_file(path)
        digest.update(relative.encode()); digest.update(b"\0"); digest.update(value.encode()); digest.update(b"\n")
        file_rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": value})
    release = {
        "schema": "rtgs_tao_geometry_release_manifest_v1",
        "geometry_release_id": RELEASE_ID,
        "scene": scene.name, "aggregate_sha256": digest.hexdigest(),
        "runtime_intersection": RUNTIME_INTERSECTION_SCHEMA,
        "novel_view_requires_cache": False,
        "formal_reviewed_mask_manifest_sha256": masks["sha256"],
        "files": file_rows,
        "verdict": "TAO_GEOMETRY_RELEASE_VALID",
    }
    _atomic_json(output / "release_manifest.json", release)
    print(json.dumps({"output": str(output), "verdict": release["verdict"], "aggregate_sha256": release["aggregate_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
