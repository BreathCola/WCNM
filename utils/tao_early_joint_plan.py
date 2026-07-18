"""Pure, checkpoint-free planning for Tao global-0 layered early-joint training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
from typing import Any

import numpy as np

from utils.dr_mask_proposal import sha256_file


PLAN_SCHEMA = "rtgs_tao_layered_early_joint_plan_v1"
FORMAL_MASK_SCHEMA = "rtgs_tao_reviewed_glass_masks_v1"
GEOMETRY_RELEASE_SCHEMA = "rtgs_tao_geometry_release_manifest_v1"
GEOMETRY_RELEASE_ID = "stage_c_tao_geometry_release_v1"
OUTPUT_NAME = "stage_d_tao_layered_early_joint_g00000_g03000_v1"


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"{label}: missing {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label}: cannot parse {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label}: manifest must be an object")
    return value


def read_colmap_points(path: Path) -> np.ndarray:
    """Read only XYZ from a COLMAP points3D.bin, without PLY/checkpoint code."""
    rows = []
    with Path(path).open("rb") as handle:
        count_raw = handle.read(8)
        if len(count_raw) != 8:
            raise ValueError("COLMAP points3D header is truncated")
        count = struct.unpack("<Q", count_raw)[0]
        for _ in range(count):
            fixed = handle.read(43)
            if len(fixed) != 43:
                raise ValueError("COLMAP points3D record is truncated")
            _point_id, x, y, z, _r, _g, _b, _error = struct.unpack("<QdddBBBd", fixed)
            track_raw = handle.read(8)
            if len(track_raw) != 8:
                raise ValueError("COLMAP points3D track is truncated")
            track_length = struct.unpack("<Q", track_raw)[0]
            handle.seek(8 * track_length, 1)
            rows.append((x, y, z))
    values = np.asarray(rows, dtype=np.float64)
    if values.shape != (count, 3) or not np.isfinite(values).all():
        raise ValueError("COLMAP points3D XYZ is incomplete/non-finite")
    return values


def _array_hash(label: str, values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(label.encode("utf-8")); digest.update(b"\0")
    digest.update(str(array.shape).encode("ascii")); digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def support_safe_global0_initialization(
    points: np.ndarray,
    axes: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    reflection_count: int = 4096,
    transmittance_count: int = 4096,
    reflection_seed: int = 1101,
    transmittance_seed: int = 1102,
    support_sigma: float = 3.0,
    scale_fraction: float = 0.002,
) -> dict[str, Any]:
    """Construct deterministic R/T positions whose complete 3-sigma support is owned."""
    points = np.asarray(points, dtype=np.float64)
    axes = np.asarray(axes, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("D initialization points must be finite Nx3")
    if axes.shape != (3, 3) or not np.allclose(axes.T @ axes, np.eye(3), atol=1e-5):
        raise ValueError("geometry axes must be orthonormal columns")
    if lower.shape != (3,) or upper.shape != (3,) or np.any(upper <= lower):
        raise ValueError("geometry bounds are invalid")
    if reflection_count <= 0 or transmittance_count <= 0:
        raise ValueError("R/T initialization counts must be positive")
    extent = upper - lower
    initial_scale = float(max(extent.max() * scale_fraction, 1e-6))
    support_margin = float(support_sigma * np.sqrt(3.0) * initial_scale)
    inside_lower = lower + support_margin
    inside_upper = upper - support_margin
    if np.any(inside_upper <= inside_lower):
        raise ValueError("cuboid is too thin for declared complete-support margin")

    local = points @ axes
    d_inside = np.all((local > lower) & (local < upper), axis=1)
    t_eligible = np.flatnonzero(np.all(
        (local > inside_lower[None]) & (local < inside_upper[None]), axis=1
    ))
    t_rng = np.random.default_rng(transmittance_seed)
    if t_eligible.size > transmittance_count:
        selected_indices = np.sort(t_rng.choice(t_eligible, transmittance_count, replace=False))
    else:
        selected_indices = t_eligible
    selected_local = local[selected_indices]
    fill_count = transmittance_count - selected_local.shape[0]
    fill_local = t_rng.uniform(inside_lower, inside_upper, size=(fill_count, 3))
    t_local = np.concatenate((selected_local, fill_local), axis=0)
    t_world = t_local @ axes.T

    scene_lo = points.min(axis=0)
    scene_hi = points.max(axis=0)
    scene_pad = np.maximum((scene_hi - scene_lo) * 0.10, extent.max() * 0.05)
    r_rng = np.random.default_rng(reflection_seed)
    outside_rows = []
    attempts = 0
    maximum_attempts = reflection_count * 500
    while sum(row.shape[0] for row in outside_rows) < reflection_count and attempts < maximum_attempts:
        batch_count = min(max(reflection_count * 2, 1024), maximum_attempts - attempts)
        candidate = r_rng.uniform(scene_lo - scene_pad, scene_hi + scene_pad, size=(batch_count, 3))
        candidate_local = candidate @ axes
        safe = np.any(
            (candidate_local < (lower - support_margin)[None])
            | (candidate_local > (upper + support_margin)[None]),
            axis=1,
        )
        outside_rows.append(candidate[safe])
        attempts += batch_count
    if not outside_rows:
        raise RuntimeError("could not sample strict-outside reflection initialization")
    r_world = np.concatenate(outside_rows, axis=0)[:reflection_count]
    r_local = r_world @ axes
    r_safe = np.any(
        (r_local < (lower - support_margin)[None])
        | (r_local > (upper + support_margin)[None]), axis=1,
    )
    t_safe = np.all(
        (t_local > inside_lower[None]) & (t_local < inside_upper[None]), axis=1,
    )
    if not r_safe.all() or not t_safe.all():
        raise AssertionError("support-safe R/T initialization invariant failed")

    return {
        "D": {
            "source": "Tao COLMAP points3D.bin only",
            "count": int(points.shape[0]),
            "position_sha256": _array_hash("D_COLMAP", points),
            "strict_inside_center_count": int(d_inside.sum()),
            "strict_inside_direct_policy": "excluded from formal D-direct queries",
        },
        "D_interface": {
            "source": "frozen analytic Tao cuboid geometry",
            "radiance_field": False,
            "namespace": "D/material",
            "parameters": ["position", "normal", "coverage", "base_tint", "ks", "f0", "roughness"],
        },
        "R": {
            "source": "fresh deterministic strict-outside fill",
            "count": int(r_world.shape[0]), "seed": int(reflection_seed),
            "position_sha256": _array_hash("R_FRESH", r_world),
            "initial_scale": initial_scale, "initial_opacity": 0.05,
            "support_sigma": float(support_sigma), "support_margin": support_margin,
            "complete_support_strict_outside": bool(r_safe.all()),
        },
        "T": {
            "source": "Tao strict-inside COLMAP points, then deterministic support-safe fill",
            "count": int(t_world.shape[0]), "seed": int(transmittance_seed),
            "eligible_colmap_count": int(t_eligible.size),
            "selected_colmap_count": int(selected_indices.size),
            "deterministic_fill_count": int(fill_count),
            "selected_colmap_index_sha256": _array_hash("T_COLMAP_INDICES", selected_indices[:, None]),
            "position_sha256": _array_hash("T_INSIDE", t_world),
            "initial_scale": initial_scale, "initial_opacity": 0.05,
            "support_sigma": float(support_sigma), "support_margin": support_margin,
            "complete_support_strict_inside": bool(t_safe.all()),
            "semantic_mask_used": False,
        },
    }


def _validate_mask_and_geometry(
    scene: Path, mask_path: Path, geometry_path: Path, stems: list[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    mask = _read_json(mask_path, "BLOCKED_BY_GLASS_MASK_IDENTITY")
    expected_mask = {
        "schema": FORMAL_MASK_SCHEMA, "artifact_role": "formal_reviewed_glass_masks",
        "human_status": "accepted", "scene": "Tao", "count": len(stems),
        "ordered_stems": stems,
    }
    for key, expected in expected_mask.items():
        if mask.get(key) != expected:
            raise RuntimeError(
                f"BLOCKED_BY_GLASS_MASK_IDENTITY: {key}={mask.get(key)!r} != {expected!r}"
            )
    geometry = _read_json(geometry_path, "BLOCKED_BY_GEOMETRY_IDENTITY")
    expected_geometry = {
        "schema": GEOMETRY_RELEASE_SCHEMA, "geometry_release_id": GEOMETRY_RELEASE_ID,
        "scene": "Tao", "verdict": "TAO_GEOMETRY_RELEASE_VALID",
        "novel_view_requires_cache": False,
    }
    for key, expected in expected_geometry.items():
        if geometry.get(key) != expected:
            raise RuntimeError(
                f"BLOCKED_BY_GEOMETRY_IDENTITY: {key}={geometry.get(key)!r} != {expected!r}"
            )
    mask_hash = sha256_file(mask_path)
    if geometry.get("formal_reviewed_mask_manifest_sha256") != mask_hash:
        raise RuntimeError("BLOCKED_BY_GEOMETRY_IDENTITY: geometry/mask identity mismatch")
    metadata_path = geometry_path.parent / "geometry_metadata.json"
    metadata = _read_json(metadata_path, "BLOCKED_BY_GEOMETRY_IDENTITY")
    file_hashes = {
        row.get("path"): row.get("sha256") for row in geometry.get("files", [])
        if isinstance(row, dict)
    }
    if file_hashes.get("geometry_metadata.json") != sha256_file(metadata_path):
        raise RuntimeError("BLOCKED_BY_GEOMETRY_IDENTITY: geometry metadata hash mismatch")
    if metadata.get("scene") != "Tao" or metadata.get("release_id") != GEOMETRY_RELEASE_ID:
        raise RuntimeError("BLOCKED_BY_GEOMETRY_IDENTITY: metadata is not Tao-owned")
    if metadata.get("inputs", {}).get("training_checkpoint_used") is not False:
        raise RuntimeError("BLOCKED_BY_GEOMETRY_IDENTITY: checkpoint-derived geometry is forbidden")
    return mask, geometry, metadata


def build_tao_early_joint_plan(
    *, scene: Path, reviewed_mask_manifest: Path, geometry_release: Path,
    output: Path, execute: bool,
) -> dict[str, Any]:
    scene = Path(scene).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if scene.name != "Tao":
        raise ValueError("the v1 operator accepts the independent Tao scene only")
    if output.exists():
        raise FileExistsError(f"refusing existing Tao training output: {output}")
    images = sorted((scene / "images").glob("*"))
    images = [path for path in images if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    stems = [path.stem for path in images]
    if stems != [f"{index:06d}" for index in range(112)]:
        raise ValueError("Tao RGB identity must be exactly stems 000000--000111")
    normal_manifest_path = scene / "normal_priors/manifest.json"
    normal_manifest = _read_json(normal_manifest_path, "BLOCKED_BY_TAO_SCENE_IDENTITY")
    normal_rows = normal_manifest.get("files")
    if not isinstance(normal_rows, list) or len(normal_rows) != 112:
        raise RuntimeError("BLOCKED_BY_TAO_SCENE_IDENTITY: StableNormal manifest is not 112-view")
    normal_stems = [Path(str(row.get("output_file", ""))).stem for row in normal_rows]
    if normal_stems != stems:
        raise RuntimeError("BLOCKED_BY_TAO_SCENE_IDENTITY: StableNormal stems differ from RGB")
    for row in normal_rows:
        path = normal_manifest_path.parent / str(row.get("output_file", ""))
        if not path.is_file() or row.get("dtype") != "float32" \
                or row.get("layout") != "HWC" or row.get("space") != "camera" \
                or row.get("shape") != [2032, 2320, 3]:
            raise RuntimeError(
                f"BLOCKED_BY_TAO_SCENE_IDENTITY: invalid StableNormal prior {path.name}"
            )
    mask_path = Path(reviewed_mask_manifest).expanduser().resolve()
    geometry_path = Path(geometry_release).expanduser().resolve()
    _mask, geometry, metadata = _validate_mask_and_geometry(
        scene, mask_path, geometry_path, stems,
    )
    points_path = scene / "sparse/0/points3D.bin"
    points = read_colmap_points(points_path)
    if points.shape[0] != 66_187:
        raise ValueError(f"Tao COLMAP point count changed: {points.shape[0]} != 66187")
    initialization = support_safe_global0_initialization(
        points, np.asarray(metadata["axes"]), np.asarray(metadata["lower"]),
        np.asarray(metadata["upper"]),
    )
    nodes = [0, 100, 250, 500, 1000, 2000, 3000]
    plan: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "execute_requested": bool(execute),
        "scene": {"name": "Tao", "path": str(scene), "stems": stems, "count": 112},
        "global_start": 0, "global_final": 3000,
        "checkpoint_source": None, "resume_source": None,
        "checkpoint_source_allowed": False,
        "legacy_tao_output_used": False, "tihubird_input_used": False,
        "output": str(output),
        "output_policy": "fresh_only_fail_closed",
        "identities": {
            "reviewed_mask_manifest": str(mask_path),
            "reviewed_mask_manifest_sha256": sha256_file(mask_path),
            "geometry_release": str(geometry_path),
            "geometry_release_id": geometry["geometry_release_id"],
            "geometry_release_aggregate_sha256": geometry["aggregate_sha256"],
            "colmap_points3D_sha256": sha256_file(points_path),
            "stable_normal_manifest": str(normal_manifest_path),
            "stable_normal_manifest_sha256": sha256_file(normal_manifest_path),
            "stable_normal_count": len(normal_rows),
            "stable_normal_space": "camera",
        },
        "initialization": initialization,
        "formal_renderer": {
            "schema": "rtgs_tao_layered_renderer_v1",
            "trace_count_per_training_step": 4,
            "trace_order": ["R_front", "T_direct", "R_back_from_T", "Cout"],
            "ownership": {
                "R_front": "strict-outside R", "T_direct": "strict-inside T",
                "R_back_from_T": "same strict-inside T", "Cout": "strict-outside D",
            },
            "recursive_bounces": 0, "back_internal_reflection_bounces": 1,
            "snell_refraction": False, "dispersion": False, "caustics": False,
            "diagnostics_policy": "unfiltered/ownership-class traces only under no_grad review nodes",
        },
        "full_frame_rgb_training": True,
        "glass_mask_use": "transparent path and interface supervision only; never RGB crop",
        "semantic_internal_object_mask": False,
        "ain_positive_push": False,
        "losses": [
            "full_frame_l1", "full_frame_dssim", "D_monocular_normal",
            "interface_regularizers_only",
        ],
        "parameter_groups": {
            "D_scene": {"trainable": ["position", "scale", "rotation", "opacity", "base_color", "normal", "material"], "frozen": []},
            "D_interface": {"trainable": ["base_tint", "ks", "f0", "roughness"], "frozen": ["position", "normal", "coverage", "topology"]},
            "R": {"trainable": ["color", "opacity", "position", "scale", "rotation"], "ownership": "complete 3-sigma strict-outside"},
            "T": {"trainable": ["color", "opacity", "position", "scale", "rotation"], "ownership": "complete 3-sigma strict-inside"},
        },
        "schedule": [
            {
                "global": [1, 250], "all_branches_in_final": True,
                "learning_rates": {"D": "baseline_3dgs", "D_interface_material": 1e-4, "R_color_opacity": 2.5e-3, "T_color_opacity": 2.5e-3, "R_T_geometry": 0.0},
                "topology": {"D_densify_prune": False, "R_densify_prune": False, "T_densify_prune": False},
            },
            {
                "global": [251, 1000], "all_branches_in_final": True,
                "learning_rates": {"D": "baseline_3dgs", "D_interface_material": 5e-5, "R_T_position": 1e-6, "R_T_scale": 5e-5, "R_T_rotation": 1e-4, "R_T_color_opacity": 1e-3},
                "topology": {"D_densify_prune": True, "D_topology_nodes": [500, 750, 1000], "R_densify_prune": False, "T_densify_prune": False},
            },
            {
                "global": [1001, 3000], "all_branches_in_final": True,
                "learning_rates": {"D": "bounded_baseline_decay", "D_interface_material": 2.5e-5, "R_T_position": 5e-7, "R_T_scale": 2.5e-5, "R_T_rotation": 5e-5, "R_T_color_opacity": 5e-4},
                "topology": {"D_densify_prune": False, "R_densify_prune": False, "T_densify_prune": False},
                "bounded_joint_refinement": True,
            },
        ],
        "review_and_checkpoint_nodes": nodes,
        "paths": {
            "checkpoints": {str(node): str(output / "checkpoints" / f"global_{node:05d}.pth") for node in nodes},
            "reviews": {str(node): str(output / "review" / f"global_{node:05d}") for node in nodes},
            "layered_exports": str(output / "layered_exports"),
        },
        "resource_policy": {
            "estimated_gpu_memory": "measure at global 0 dry review; reduce ray_chunk_size before changing model counts",
            "estimated_time": "record wall-clock projections after nodes 100 and 250; no unverified duration claim",
            "oom_policy": "fail node, preserve prior checkpoint, reduce trace chunk only",
            "maximum_global": 3000,
        },
        "training_executed": False,
    }
    return plan
