#!/usr/bin/env python3
"""Build a review-only Tao glass-mask proposal from DR cues and COLMAP.

The command defaults to plan-only.  ``--execute`` writes a fresh proposal, but
never a formal mask, geometry release, cache, checkpoint, PLY, or training run.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import struct
import sys
import traceback
from typing import Any

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometry.tao_dr_cuboid_bootstrap import (
    BOOTSTRAP_SCHEMA,
    CUBOID_EDGES,
    CUBOID_FACES,
    DEPTH_CALIBRATION_SCHEMA,
    EXPECTED_COLMAP_SHA256,
    EXPECTED_DR_MANIFEST_SHA256,
    PREDECLARED_THRESHOLDS,
    camera_focus_point,
    cuboid_vertices,
    fit_relative_depth_calibration,
    geometry_parameter_sha256,
    optimize_review_cuboid,
    rasterize_projected_cuboid,
    search_normal_axis_convention,
    validate_colmap_identity,
    validate_dr_manifest_identity,
    validate_tao_stems,
)
from scene.colmap_loader import qvec2rotmat, read_extrinsics_binary, read_intrinsics_binary
from utils.dr_mask_proposal import audit_dr_artifacts
from utils.tao_dr_mask_repair import (
    METRICS_SCHEMA,
    PROPOSAL_SCHEMA,
    REVIEW_TEMPLATE_SCHEMA,
    aggregate_quality_metrics,
    atomic_image,
    atomic_json,
    compute_native_cues,
    confidence_and_risks,
    face_visibility,
    initialize_bounds_from_calibrated_views,
    interior_likelihood,
    make_contact_sheets,
    make_review_page,
    mapped_world_normal,
    old_new_difference,
    projected_geometry_overlay,
    sha256_file,
    silhouette_boundary,
    soft_mask_from_projection,
    tint_mask,
    tree_manifest,
    write_metrics_csv,
    write_review_queue,
)


OUTPUT_NAME = "stage_b_tao_glass_mask_dr_geometry_repair_proposal_v2"
OLD_PROPOSAL_SCHEMA = "rtgs_glass_mask_proposal_v1"
EXPECTED_OLD_PROPOSAL_MANIFEST_SHA256 = (
    "5b80d2e5708ca0603797aa511403660dcfe2981e4bcbae32dfbcfd0b07fb4cd0"
)
FIXED_REPRESENTATIVE_INDICES = (0, 14, 28, 42, 56, 70, 84, 98, 111)


def _read_points(path: Path) -> dict[int, np.ndarray]:
    points: dict[int, np.ndarray] = {}
    with Path(path).open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError("COLMAP points3D header is truncated")
        count = struct.unpack("<Q", raw)[0]
        for _ in range(count):
            fixed = handle.read(43)
            if len(fixed) != 43:
                raise ValueError("COLMAP points3D record is truncated")
            point_id, x, y, z, _r, _g, _b, _error = struct.unpack("<QdddBBBd", fixed)
            track = handle.read(8)
            if len(track) != 8:
                raise ValueError("COLMAP point track is truncated")
            track_length = struct.unpack("<Q", track)[0]
            handle.seek(8 * track_length, 1)
            points[int(point_id)] = np.array([x, y, z], dtype=np.float64)
    if len(points) != count or not np.isfinite(np.stack(list(points.values()))).all():
        raise ValueError("COLMAP sparse points are incomplete or non-finite")
    return points


def _intrinsics(camera) -> np.ndarray:
    if camera.model != "PINHOLE" or len(camera.params) != 4:
        raise ValueError("Tao DR mask repair requires the undistorted PINHOLE camera")
    fx, fy, cx, cy = camera.params
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def _view_record(image, camera, native_size: tuple[int, int]) -> dict[str, Any]:
    rotation = qvec2rotmat(image.qvec)
    center = -rotation.T @ image.tvec
    source_intrinsics = _intrinsics(camera)
    native_intrinsics = source_intrinsics.copy()
    native_intrinsics[0] *= native_size[0] / int(camera.width)
    native_intrinsics[1] *= native_size[1] / int(camera.height)
    forward = np.array([0.0, 0.0, 1.0]) @ rotation
    return {
        "stem": Path(image.name).stem,
        "source_size": (int(camera.width), int(camera.height)),
        "native_size": native_size,
        "source_intrinsics": source_intrinsics,
        "intrinsics": native_intrinsics,
        "world_to_camera": rotation,
        "camera_center": center,
        "camera_forward": forward / np.linalg.norm(forward),
        "xys": np.asarray(image.xys, dtype=np.float64),
        "point3D_ids": np.asarray(image.point3D_ids, dtype=np.int64),
    }


def _read_rgb(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    with Image.open(path) as opened:
        if opened.mode != "RGB":
            raise ValueError(f"expected RGB image: {path}")
        if size is not None and opened.size != size:
            raise ValueError(f"image size mismatch: {path}: {opened.size} != {size}")
        return np.asarray(opened, dtype=np.uint8)


def _nearest(values: np.ndarray, xy: np.ndarray) -> np.ndarray:
    height, width = values.shape[:2]
    x = np.clip(np.rint(xy[:, 0]).astype(int), 0, width - 1)
    y = np.clip(np.rint(xy[:, 1]).astype(int), 0, height - 1)
    return values[y, x]


def _depth_calibration_for_view(
    view: dict[str, Any],
    relative_depth: np.ndarray,
    points: dict[int, np.ndarray],
):
    valid_ids = np.array([int(value) in points for value in view["point3D_ids"]], dtype=bool)
    xys = view["xys"][valid_ids]
    ids = view["point3D_ids"][valid_ids]
    if ids.size:
        xyz = np.stack([points[int(value)] for value in ids], axis=0)
        camera = (xyz - view["camera_center"][None]) @ view["world_to_camera"].T
        source_width, source_height = view["source_size"]
        native_width, native_height = view["native_size"]
        xy_native = xys * np.array([native_width / source_width, native_height / source_height])
        inside = (
            (xy_native[:, 0] >= 0) & (xy_native[:, 0] < native_width)
            & (xy_native[:, 1] >= 0) & (xy_native[:, 1] < native_height)
            & np.isfinite(camera[:, 2]) & (camera[:, 2] > 1e-6)
        )
        relative = _nearest(relative_depth, xy_native[inside])
        metric = camera[inside, 2]
    else:
        relative = np.empty(0); metric = np.empty(0)
    return fit_relative_depth_calibration(
        relative, metric, projectable_point_count=int(relative.size)
    )


def _plan(scene: Path, dr_raw: Path, old_proposal: Path, output: Path) -> dict[str, Any]:
    return {
        "schema": "rtgs_tao_dr_geometry_mask_repair_bootstrap_plan_v2",
        "task": "TAO-DR-MASK-REPAIR-001",
        "scene": "Tao",
        "schema_scope": "review-only",
        "training_eligible": False,
        "promotion_performed": False,
        "formal_geometry_release": False,
        "human_status": "proposal_requires_review",
        "execute_required": True,
        "inputs": {
            "rgb": str((scene / "images").resolve()),
            "colmap": str((scene / "sparse/0").resolve()),
            "dr_raw": str(dr_raw.resolve()),
            "expected_dr_manifest_sha256": EXPECTED_DR_MANIFEST_SHA256,
            "old_proposal": str(old_proposal.resolve()),
            "old_proposal_role": "after-fit comparison and risk analysis only",
        },
        "forbidden_inputs": [
            "Tao training checkpoints/optimizer state/PLY/training cache",
            "assets from any other scene",
            "semantic/internal-object masks",
            "old proposal as fit supervision, silhouette target, or fallback",
        ],
        "method": {
            "normal": "joint 48-way signed-permutation search and world-space orthogonal axes",
            "depth": "per-view relative-only depth or inverse-depth affine calibration on COLMAP observations",
            "cues": [
                "normal discontinuity", "relative-depth discontinuity", "RGB edge",
                "basecolor edge", "diffuse-albedo edge", "planar-normal consistency",
                "highlight/reflection uncertainty",
            ],
            "initialization": "calibrated DR dense support + robust focus-local Tao sparse range",
            "objective": "continuous DR cue likelihood sampled by one projected 3-D cuboid",
            "fallback": None,
            "soft_mask": "signed distance field from the new projected hard silhouette",
        },
        "predeclared_thresholds": dict(PREDECLARED_THRESHOLDS),
        "output": str(output.resolve()),
        "stop_boundary": "human review; no formal mask, geometry release, or training",
    }


def _validate_old_comparison(root: Path, stems: list[str], source_size: tuple[int, int]) -> dict[str, Any]:
    manifest_path = root / "proposal_manifest.json"
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 != EXPECTED_OLD_PROPOSAL_MANIFEST_SHA256:
        raise RuntimeError("old comparison proposal manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "schema": OLD_PROPOSAL_SCHEMA,
        "artifact_role": "glass_mask_proposal_for_human_review",
        "human_status": "proposal_requires_review",
        "training_eligible": False,
        "promotion_performed": False,
        "count": len(stems),
        "ordered_stems": stems,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(f"old proposal comparison identity mismatch: {key}")
    masks = {}
    for stem in stems:
        path = root / "mask_hard" / f"{stem}.png"
        with Image.open(path) as opened:
            if opened.mode != "L" or opened.size != source_size:
                raise ValueError(f"old comparison mask mode/size mismatch: {stem}")
            masks[stem] = np.asarray(opened, dtype=np.uint8) >= 128
    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha256,
        "masks": masks,
        "role": "comparison_only_loaded_after_cuboid_fit",
    }


def _blocked_manifest(plan: dict[str, Any], error: BaseException) -> dict[str, Any]:
    return {
        "schema": PROPOSAL_SCHEMA,
        "schema_scope": "review-only",
        "training_eligible": False,
        "promotion_performed": False,
        "formal_geometry_release": False,
        "human_status": "proposal_requires_review",
        "verdict": "TAO_DR_GEOMETRY_MASK_REPAIR_BLOCKED",
        "failure_mode": type(error).__name__,
        "failure": str(error),
        "fallback_performed": False,
        "plan": plan,
    }


def _write_blocked(staging: Path, output: Path, plan: dict[str, Any], error: BaseException) -> None:
    atomic_json(staging / "failure_diagnostic.json", {
        "error_type": type(error).__name__, "error": str(error),
        "traceback": traceback.format_exc(), "fallback_performed": False,
    })
    atomic_json(staging / "proposal_manifest.json", _blocked_manifest(plan, error))
    os.replace(staging, output)


def _run(args) -> int:
    scene = args.scene.expanduser().resolve()
    dr_raw = args.dr_raw.expanduser().resolve()
    old_root = args.old_proposal.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if scene.name != "Tao":
        raise ValueError("this review bootstrap accepts only the Tao scene")
    if output.exists():
        raise FileExistsError(f"refusing existing review output: {output}")
    plan = _plan(scene, dr_raw, old_root, output)
    if not args.execute:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0
    staging = output.parent / f".{output.name}.building-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"refusing existing staging output: {staging}")
    staging.mkdir(parents=True)
    # This write happens before any calibration/axis search/cuboid evaluation.
    atomic_json(staging / "bootstrap_plan.json", plan)
    try:
        audit = audit_dr_artifacts(scene, dr_raw, expected_count=112, images="images")
        validate_dr_manifest_identity(audit["raw_manifest_sha256"])
        stems = validate_tao_stems([row["stem"] for row in audit["entries"]])
        native_size = tuple(audit["native_dr_resolution"])
        source_size = tuple(audit["source_resolution"])
        sparse_root = scene / "sparse/0"
        colmap_hashes = validate_colmap_identity({
            name: sha256_file(sparse_root / name)
            for name in ("cameras.bin", "images.bin", "points3D.bin")
        })
        cameras = read_intrinsics_binary(str(sparse_root / "cameras.bin"))
        images = read_extrinsics_binary(str(sparse_root / "images.bin"))
        points = _read_points(sparse_root / "points3D.bin")
        by_stem = {Path(image.name).stem: image for image in images.values()}
        if sorted(by_stem) != stems or len(cameras) != 1:
            raise RuntimeError("Tao COLMAP/RGB identity is not exactly 112 aligned views")
        dr_by_stem = {row["stem"]: row for row in audit["entries"]}
        views = []
        calibration_rows = []
        normal_samples = []
        for index, stem in enumerate(stems):
            image = by_stem[stem]
            view = _view_record(image, cameras[image.camera_id], native_size)
            artifacts = dr_by_stem[stem]["artifacts"]
            stored_rgb = _read_rgb(Path(artifacts["rgb"]["path"]), native_size)
            raw_normal = _read_rgb(Path(artifacts["normal"]["path"]), native_size)
            raw_depth = _read_rgb(Path(artifacts["depth"]["path"]), native_size)
            basecolor = _read_rgb(Path(artifacts["basecolor"]["path"]), native_size)
            diffuse = _read_rgb(Path(artifacts["diffuse_albedo"]["path"]), native_size)
            cues = compute_native_cues(stored_rgb, raw_normal, raw_depth, basecolor, diffuse)
            calibration = _depth_calibration_for_view(view, cues["relative_depth"], points)
            calibrated_depth = calibration.apply(cues["relative_depth"])
            view.update({
                "stored_rgb": stored_rgb, "raw_normal": raw_normal,
                "raw_depth": raw_depth, "basecolor": basecolor,
                "diffuse_albedo": diffuse, "cues": cues,
                "fused_boundary": cues["fused_boundary"],
                "boundary_alignment_likelihood": cv2.dilate(
                    cues["fused_boundary"], np.ones((11, 11), np.uint8), iterations=1
                ),
                "calibration": calibration,
                "calibrated_depth": calibrated_depth.astype(np.float32),
                "depth_weight": calibration.trust_weight,
            })
            calibration_row = {"stem": stem, **calibration.as_dict()}
            calibration_rows.append(calibration_row)
            selection = cues["planar_normal_consistency"] >= float(
                PREDECLARED_THRESHOLDS["normal_planar_selection_minimum"]
            )
            normal_samples.append({
                "raw_normal": raw_normal, "world_to_camera": view["world_to_camera"],
                "selection": selection,
            })
            views.append(view)
            if (index + 1) % 16 == 0 or index + 1 == len(stems):
                print(f"AUDIT_CALIBRATION {index + 1}/112", flush=True)
        trusted = [row for row in calibration_rows if row["trusted"]]
        depth_report = {
            "schema": DEPTH_CALIBRATION_SCHEMA,
            "contract": "per-view relative-only; no raw cross-view comparison and no shared scale",
            "thresholds_predeclared": {
                key: PREDECLARED_THRESHOLDS[key] for key in PREDECLARED_THRESHOLDS
                if "depth" in key
            },
            "trusted_view_count": len(trusted),
            "trusted_view_fraction": len(trusted) / len(stems),
            "per_view": calibration_rows,
        }
        atomic_json(staging / "depth_calibration.json", depth_report)
        if len(trusted) < int(PREDECLARED_THRESHOLDS["minimum_trusted_depth_views"]):
            raise RuntimeError("insufficient trustworthy per-view DR depth calibrations")

        mapping, axes, normal_report = search_normal_axis_convention(normal_samples)
        atomic_json(staging / "normal_axis_search.json", normal_report)
        for view in views:
            view["world_normal"] = mapped_world_normal(
                view["raw_normal"], mapping, view["world_to_camera"]
            )
            view["interior_likelihood"] = interior_likelihood(
                view["calibrated_depth"], view["cues"]["planar_normal_consistency"],
                view["intrinsics"], view["cues"]["normal_unit_unmapped"],
            )

        focus_report = camera_focus_point(
            np.stack([view["camera_center"] for view in views]),
            np.stack([view["camera_forward"] for view in views]),
        )
        focus = np.asarray(focus_report.pop("focus"), dtype=np.float64)
        focus_report["focus"] = focus.tolist()
        # Sparse range evidence must itself agree with the new DR enclosure
        # likelihood.  Merely taking every museum point near the optical focus
        # over-expands the initializer and is not a robust target-object range.
        sparse_evidence = []
        support_quantile = float(
            PREDECLARED_THRESHOLDS["initial_support_likelihood_quantile"]
        )
        for view in views:
            valid_ids = np.array([
                int(value) in points for value in view["point3D_ids"]
            ], dtype=bool)
            ids = view["point3D_ids"][valid_ids]
            if not ids.size:
                continue
            xy = view["xys"][valid_ids] * np.array([
                native_size[0] / source_size[0], native_size[1] / source_size[1]
            ])
            score = _nearest(view["interior_likelihood"], xy)
            threshold = float(np.quantile(view["interior_likelihood"], support_quantile))
            selected = ids[score >= threshold]
            sparse_evidence.extend(points[int(value)] for value in selected)
        if len(sparse_evidence) < 100:
            raise RuntimeError("DR-supported Tao sparse geometry is insufficient for initialization")
        sparse_values = np.stack(sparse_evidence, axis=0)
        initial_lower, initial_upper, initialization = initialize_bounds_from_calibrated_views(
            views, sparse_values, axes, focus, focus_report["camera_radius_median"]
        )
        initialization["dr_supported_sparse_observation_count_before_focus_filter"] = len(
            sparse_evidence
        )
        lower, upper, fit = optimize_review_cuboid(
            axes, initial_lower, initial_upper, views
        )
        vertices = cuboid_vertices(axes, lower, upper)
        parameter_hash = geometry_parameter_sha256(axes, lower, upper)
        extent = upper - lower
        scene_scale = float(focus_report["camera_radius_median"])

        # Fail before reading old proposal pixels or writing any mask when the
        # fitted DR/COLMAP geometry itself misses a predeclared gate.
        preliminary_rows = []
        for native in fit["per_view"]:
            row = {
                "stem": native["stem"],
                "source_size": list(native_size),
                "new_area_ratio": float(native["area_ratio"]),
                "old_area_ratio": 0.0,
                "old_new_area_change": 0.0,
                "boundary_alignment": float(native["boundary_alignment"]),
                "normal_face_alignment": float(native["normal_face_alignment"]),
                "depth_relative_residual": float(native["depth_relative_residual"]),
                "depth_weight": float(native["depth_weight"]),
                "border_touching": bool(native["border_touching"]),
                "projected_vertices": native["projected_vertices"],
            }
            confidence, _risks = confidence_and_risks(row)
            row["confidence"] = confidence
            row["low_confidence"] = confidence < float(
                PREDECLARED_THRESHOLDS["low_confidence_threshold"]
            )
            preliminary_rows.append(row)
        preliminary_quality = aggregate_quality_metrics(preliminary_rows)
        trusted_depth_residual = [
            row["depth_relative_residual"] for row in preliminary_rows
            if row["depth_weight"] > 0
        ]
        axis_error = float(np.max(np.abs(axes.T @ axes - np.eye(3))))
        preliminary_gates = {
            "all_views_projectable": all(row["projectable"] for row in fit["per_view"]),
            "normal_alignment_p50": normal_report["selected_fit"]["alignment_p50"] >= float(PREDECLARED_THRESHOLDS["minimum_normal_alignment_p50"]),
            "axes_orthogonal": axis_error <= float(PREDECLARED_THRESHOLDS["maximum_axis_orthogonality_error"]),
            "extents_positive_finite": bool(np.all(extent > 0) and np.isfinite(vertices).all()),
            "extents_scene_bounded": bool(np.all(extent / scene_scale >= float(PREDECLARED_THRESHOLDS["minimum_extent_scene_fraction"])) and np.all(extent / scene_scale <= float(PREDECLARED_THRESHOLDS["maximum_extent_scene_fraction"]))),
            "projected_area_range": all(float(PREDECLARED_THRESHOLDS["minimum_projected_area_ratio"]) <= row["new_area_ratio"] <= float(PREDECLARED_THRESHOLDS["maximum_projected_area_ratio"]) for row in preliminary_rows),
            "mean_boundary_alignment": preliminary_quality["boundary_alignment"]["mean"] >= float(PREDECLARED_THRESHOLDS["minimum_mean_boundary_alignment"]),
            "minimum_boundary_alignment": preliminary_quality["boundary_alignment"]["minimum"] >= float(PREDECLARED_THRESHOLDS["minimum_view_boundary_alignment"]),
            "mean_normal_face_alignment": preliminary_quality["normal_face_alignment"]["mean"] >= float(PREDECLARED_THRESHOLDS["minimum_mean_normal_face_alignment"]),
            "trusted_depth_views": len(trusted) >= int(PREDECLARED_THRESHOLDS["minimum_trusted_depth_views"]),
            "calibrated_depth_residual": float(np.mean(trusted_depth_residual)) <= float(PREDECLARED_THRESHOLDS["maximum_mean_calibrated_depth_relative_residual"]),
            "border_touching_views": preliminary_quality["border_touching_view_count"] <= int(PREDECLARED_THRESHOLDS["maximum_border_touching_views"]),
            "adjacent_area_continuity": preliminary_quality["adjacent_log_area_delta_p95"] <= float(PREDECLARED_THRESHOLDS["maximum_adjacent_log_area_delta_p95"]),
            "adjacent_corner_continuity": preliminary_quality["adjacent_corner_displacement_p95"] <= float(PREDECLARED_THRESHOLDS["maximum_adjacent_corner_displacement_p95"]),
            "low_confidence_view_count": preliminary_quality["low_confidence_view_count"] <= int(PREDECLARED_THRESHOLDS["maximum_low_confidence_views"]),
            "old_proposal_excluded": True,
            "fallback_forbidden": True,
        }
        preliminary_bootstrap = {
            "schema": BOOTSTRAP_SCHEMA,
            "schema_scope": "review-only", "formal_geometry_release": False,
            "training_eligible": False, "scene": "Tao",
            "normal_mapping": mapping.tolist(), "axes": axes.tolist(),
            "initial_lower": initial_lower.tolist(), "initial_upper": initial_upper.tolist(),
            "lower": lower.tolist(), "upper": upper.tolist(), "extents": extent.tolist(),
            "vertices": vertices.tolist(), "geometry_parameter_sha256": parameter_hash,
            "focus": focus_report, "initialization": initialization,
            "optimization": fit["optimization"],
            "pre_artifact_quality": preliminary_quality,
            "pre_artifact_gates": preliminary_gates,
            "old_proposal_used_in_fit": False, "fallback_available": False,
        }
        atomic_json(staging / "bootstrap_geometry.json", preliminary_bootstrap)
        if not all(preliminary_gates.values()):
            failed = [key for key, value in preliminary_gates.items() if not value]
            raise RuntimeError(
                "predeclared DR/COLMAP geometry gates failed before mask generation: "
                + ", ".join(failed)
            )

        old = _validate_old_comparison(old_root, stems, source_size)
        directories = (
            "mask_soft", "mask_hard", "mask_eroded", "silhouette_boundary",
            "overlays", "cue_overlays", "old_vs_new", "review_pages",
            "fixed_view_geometry_review", "geometry_overlays", "contact_sheets",
        )
        for name in directories:
            (staging / name).mkdir(exist_ok=True)
        evaluation = {row["stem"]: row for row in fit["per_view"]}
        metric_rows = []
        erode_kernel = np.ones((
            2 * int(PREDECLARED_THRESHOLDS["erode_pixels"]) + 1,
            2 * int(PREDECLARED_THRESHOLDS["erode_pixels"]) + 1,
        ), np.uint8)
        fixed_stems = [f"{index:06d}" for index in FIXED_REPRESENTATIVE_INDICES]
        for index, view in enumerate(views):
            stem = view["stem"]
            rgb = _read_rgb(scene / "images" / f"{stem}.jpg", source_size)
            hard, source_hull, source_uv = rasterize_projected_cuboid(
                vertices, view["source_intrinsics"], view["world_to_camera"],
                view["camera_center"], (source_size[1], source_size[0]),
            )
            if not np.any(hard):
                raise RuntimeError(f"new cuboid projection is empty for {stem}")
            soft = soft_mask_from_projection(
                hard, float(PREDECLARED_THRESHOLDS["soft_mask_feather_pixels"])
            )
            eroded = cv2.erode(hard, erode_kernel, iterations=1)
            boundary = silhouette_boundary(hard)
            overlay = tint_mask(rgb, hard)
            geometry_overlay, geometry_uv, geometry_hull = projected_geometry_overlay(
                overlay, vertices, view
            )
            old_hard = old["masks"][stem]
            old_overlay = tint_mask(rgb, old_hard, color=(255, 150, 30))
            difference = old_new_difference(old_hard, hard)
            cue_heat = cv2.applyColorMap(
                np.rint(np.clip(view["fused_boundary"], 0, 1) * 255).astype(np.uint8),
                cv2.COLORMAP_TURBO,
            )[..., ::-1]
            cue_overlay = np.clip(
                0.55 * view["stored_rgb"].astype(np.float32)
                + 0.45 * cue_heat.astype(np.float32), 0, 255,
            ).astype(np.uint8)
            native = evaluation[stem]
            row = {
                "stem": stem,
                "source_size": list(source_size),
                "geometry_parameter_sha256": parameter_hash,
                "new_mask_source": "one fixed review-only DR/COLMAP cuboid projection",
                "old_proposal_used_in_fit": False,
                "fallback_used": False,
                "new_area_ratio": float(hard.mean()),
                "old_area_ratio": float(old_hard.mean()),
                "old_new_area_change": float(hard.mean() - old_hard.mean()),
                "old_new_iou_comparison_only": float(
                    np.count_nonzero(hard.astype(bool) & old_hard)
                    / max(np.count_nonzero(hard.astype(bool) | old_hard), 1)
                ),
                "boundary_alignment": float(native["boundary_alignment"]),
                "normal_face_alignment": float(native["normal_face_alignment"]),
                "depth_relative_residual": float(native["depth_relative_residual"]),
                "depth_weight": float(native["depth_weight"]),
                "depth_calibration_mode": view["calibration"].mode,
                "depth_calibration_r2": float(view["calibration"].r2),
                "border_touching": bool(native["border_touching"]),
                "projected_vertices": source_uv.tolist(),
                "projected_hull": source_hull.tolist(),
                "projected_edges": [[int(a), int(b)] for a, b in CUBOID_EDGES],
                "per_face_visibility": face_visibility(vertices, view["camera_center"]),
                "review_page": f"review_pages/{stem}.png",
            }
            confidence, risks = confidence_and_risks(row)
            row["confidence"] = confidence
            row["risk_flags"] = risks
            row["low_confidence"] = confidence < float(
                PREDECLARED_THRESHOLDS["low_confidence_threshold"]
            )
            atomic_image(staging / "mask_soft" / f"{stem}.png", Image.fromarray(soft))
            atomic_image(staging / "mask_hard" / f"{stem}.png", Image.fromarray(hard * 255))
            atomic_image(staging / "mask_eroded" / f"{stem}.png", Image.fromarray(eroded * 255))
            atomic_image(staging / "silhouette_boundary" / f"{stem}.png", Image.fromarray(boundary))
            atomic_image(staging / "overlays" / f"{stem}.png", Image.fromarray(overlay))
            atomic_image(staging / "geometry_overlays" / f"{stem}.png", Image.fromarray(geometry_overlay))
            atomic_image(staging / "cue_overlays" / f"{stem}.png", Image.fromarray(cue_overlay))
            atomic_image(staging / "old_vs_new" / f"{stem}.png", Image.fromarray(difference))
            page = make_review_page(
                stem=stem, source_rgb=rgb, dr_normal=view["raw_normal"],
                dr_depth=view["raw_depth"], dr_basecolor=view["basecolor"],
                fused_boundary=view["fused_boundary"], geometry_overlay=geometry_overlay,
                new_overlay=overlay, old_overlay=old_overlay, difference=difference,
                metric=row,
            )
            atomic_image(staging / "review_pages" / f"{stem}.png", page)
            if stem in fixed_stems:
                atomic_image(staging / "fixed_view_geometry_review" / f"{stem}.png", page)
            metric_rows.append(row)
            if (index + 1) % 16 == 0 or index + 1 == len(views):
                print(f"PROJECT_REVIEW {index + 1}/112", flush=True)

        quality = aggregate_quality_metrics(metric_rows)
        depth_trusted_residual = [
            row["depth_relative_residual"] for row in metric_rows if row["depth_weight"] > 0
        ]
        axis_error = float(np.max(np.abs(axes.T @ axes - np.eye(3))))
        face_edge_counts = {tuple(sorted(edge)): 0 for edge in CUBOID_EDGES.tolist()}
        for face in CUBOID_FACES.tolist():
            for offset in range(4):
                edge = tuple(sorted((face[offset], face[(offset + 1) % 4])))
                face_edge_counts[edge] += 1
        watertight = (
            len(face_edge_counts) == 12
            and all(count == 2 for count in face_edge_counts.values())
        )
        gates = {
            "stems_112_complete": len(metric_rows) == 112 and [row["stem"] for row in metric_rows] == stems,
            "dr_colmap_rgb_identity": (
                audit["raw_manifest_sha256"] == EXPECTED_DR_MANIFEST_SHA256
                and colmap_hashes == EXPECTED_COLMAP_SHA256
            ),
            "normal_alignment_p50": normal_report["selected_fit"]["alignment_p50"] >= float(PREDECLARED_THRESHOLDS["minimum_normal_alignment_p50"]),
            "axes_orthogonal": axis_error <= float(PREDECLARED_THRESHOLDS["maximum_axis_orthogonality_error"]),
            "extents_positive": bool(np.all(extent > 0)),
            "extents_scene_bounded": bool(np.all(extent / scene_scale >= float(PREDECLARED_THRESHOLDS["minimum_extent_scene_fraction"])) and np.all(extent / scene_scale <= float(PREDECLARED_THRESHOLDS["maximum_extent_scene_fraction"]))),
            "finite_vertices_planes": bool(np.isfinite(vertices).all() and np.isfinite(lower).all() and np.isfinite(upper).all()),
            "watertight_cuboid": bool(watertight),
            "all_views_projectable_nonempty": all(row["new_area_ratio"] > 0 for row in metric_rows),
            "no_near_full_fallback": all(row["new_area_ratio"] < float(PREDECLARED_THRESHOLDS["near_full_image_ratio"]) and not row["fallback_used"] for row in metric_rows),
            "projected_area_range": all(float(PREDECLARED_THRESHOLDS["minimum_projected_area_ratio"]) <= row["new_area_ratio"] <= float(PREDECLARED_THRESHOLDS["maximum_projected_area_ratio"]) for row in metric_rows),
            "mean_boundary_alignment": quality["boundary_alignment"]["mean"] >= float(PREDECLARED_THRESHOLDS["minimum_mean_boundary_alignment"]),
            "minimum_boundary_alignment": quality["boundary_alignment"]["minimum"] >= float(PREDECLARED_THRESHOLDS["minimum_view_boundary_alignment"]),
            "mean_normal_face_alignment": quality["normal_face_alignment"]["mean"] >= float(PREDECLARED_THRESHOLDS["minimum_mean_normal_face_alignment"]),
            "trusted_depth_views": len(trusted) >= int(PREDECLARED_THRESHOLDS["minimum_trusted_depth_views"]),
            "calibrated_depth_residual": float(np.mean(depth_trusted_residual)) <= float(PREDECLARED_THRESHOLDS["maximum_mean_calibrated_depth_relative_residual"]),
            "border_touching_views": quality["border_touching_view_count"] <= int(PREDECLARED_THRESHOLDS["maximum_border_touching_views"]),
            "adjacent_area_continuity": quality["adjacent_log_area_delta_p95"] <= float(PREDECLARED_THRESHOLDS["maximum_adjacent_log_area_delta_p95"]),
            "adjacent_corner_continuity": quality["adjacent_corner_displacement_p95"] <= float(PREDECLARED_THRESHOLDS["maximum_adjacent_corner_displacement_p95"]),
            "low_confidence_view_count": quality["low_confidence_view_count"] <= int(PREDECLARED_THRESHOLDS["maximum_low_confidence_views"]),
            "one_fixed_multiview_parameter_hash": len({row["geometry_parameter_sha256"] for row in metric_rows}) == 1,
            "soft_masks_from_projection_distance_field": True,
            "old_proposal_excluded_from_fit": fit["optimization"]["old_proposal_used_in_objective"] is False,
        }

        metrics_payload = {"schema": METRICS_SCHEMA, "per_frame": metric_rows, "aggregate": quality}
        atomic_json(staging / "per_frame_metrics.json", metrics_payload)
        write_metrics_csv(staging / "per_frame_metrics.csv", metric_rows)
        write_review_queue(staging / "review_queue.csv", metric_rows)
        review_template = {
            "schema": REVIEW_TEMPLATE_SCHEMA,
            "human_status": "proposal_requires_review",
            "training_eligible": False,
            "entries": [{
                "stem": stem, "accept_complete_glass_projection": None,
                "background_included": None, "glass_edge_missing": None,
                "ceramic_misclassified_as_glass": None, "needs_manual_edit": None,
                "notes": "",
            } for stem in stems],
        }
        atomic_json(staging / "review_template.json", review_template)
        contacts = make_contact_sheets(staging, metric_rows, fixed_stems=fixed_stems)
        bootstrap = {
            "schema": BOOTSTRAP_SCHEMA,
            "schema_scope": "review-only",
            "formal_geometry_release": False,
            "training_eligible": False,
            "scene": "Tao",
            "method": "multi-view DR cue likelihood + Tao COLMAP metric bootstrap",
            "old_proposal_used_in_fit": False,
            "fallback_available": False,
            "focus": focus_report,
            "normal_mapping": mapping.tolist(), "axes": axes.tolist(),
            "initial_lower": initial_lower.tolist(), "initial_upper": initial_upper.tolist(),
            "lower": lower.tolist(), "upper": upper.tolist(), "extents": extent.tolist(),
            "vertices": vertices.tolist(),
            "planes": [
                {"axis": index, "side": side, "offset": float(value)}
                for index in range(3) for side, value in (("lower", lower[index]), ("upper", upper[index]))
            ],
            "geometry_parameter_sha256": parameter_hash,
            "initialization": initialization,
            "optimization": fit["optimization"],
            "quality": quality, "gates": gates,
        }
        atomic_json(staging / "bootstrap_geometry.json", bootstrap)
        row58 = next(row for row in metric_rows if row["stem"] == "000058")
        special58 = {
            "stem": "000058", "old_area_ratio": row58["old_area_ratio"],
            "new_area_ratio": row58["new_area_ratio"],
            "new_mask_source": row58["new_mask_source"],
            "inherited_old_fallback": False,
            "large_area_or_background_risk": any(
                risk in row58["risk_flags"] for risk in (
                    "projected_area_background_or_near_full_risk",
                    "projected_silhouette_touches_border",
                )
            ),
            "review_page": row58["review_page"],
        }
        atomic_json(staging / "frame_000058_report.json", special58)
        passed = all(gates.values())
        files, aggregate = tree_manifest(staging, exclude=("proposal_manifest.json",))
        manifest = {
            "schema": PROPOSAL_SCHEMA,
            "schema_scope": "review-only",
            "artifact_role": "tao_dr_geometry_glass_mask_proposal_for_human_review",
            "training_eligible": False,
            "promotion_performed": False,
            "formal_geometry_release": False,
            "human_status": "proposal_requires_review",
            "scene": "Tao", "count": 112, "ordered_stems": stems,
            "mask_semantics": "complete projected glass enclosure; visible ceramic, stand and background through glass are not holes",
            "input_identity": {
                "dr_manifest_sha256": audit["raw_manifest_sha256"],
                "dr_real_frames": audit["real_frame_count"],
                "dr_padding_frames_excluded": audit["padding_frame_count"],
                "colmap": colmap_hashes,
                "sparse_point_count": len(points),
                "old_proposal_manifest_sha256": old["manifest_sha256"],
                "old_proposal_role": old["role"],
            },
            "bootstrap_geometry_schema": BOOTSTRAP_SCHEMA,
            "geometry_parameter_sha256": parameter_hash,
            "predeclared_thresholds": dict(PREDECLARED_THRESHOLDS),
            "gates": gates, "quality": quality, "frame_000058": special58,
            "contact_sheets": contacts,
            "files_before_manifest": files,
            "aggregate_before_manifest_sha256": aggregate,
            "formal_mask_created": False, "formal_geometry_created": False,
            "training_started": False, "optimizer_updated": False,
            "checkpoint_created": False, "ply_created": False,
            "verdict": "TAO_GLASS_MASK_REVIEW_REQUIRED" if passed else "TAO_DR_GEOMETRY_MASK_REPAIR_BLOCKED",
        }
        atomic_json(staging / "proposal_manifest.json", manifest)
        os.replace(staging, output)
        print(json.dumps({
            "output": str(output), "verdict": manifest["verdict"],
            "geometry_parameter_sha256": parameter_hash,
            "gates": gates, "frame_000058": special58,
        }, indent=2, ensure_ascii=False))
        return 0 if passed else 2
    except Exception as error:
        if staging.exists():
            _write_blocked(staging, output, plan, error)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=ROOT / "data/Tao")
    parser.add_argument("--dr-raw", type=Path, default=ROOT / "output/stage_a_tao_dr_raw_112_v1")
    parser.add_argument("--old-proposal", type=Path, default=ROOT / "output/stage_b_tao_glass_mask_proposal_112_v1")
    parser.add_argument("--output", type=Path, default=ROOT / f"output/{OUTPUT_NAME}")
    parser.add_argument("--execute", action="store_true", help="write the review-only proposal")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
