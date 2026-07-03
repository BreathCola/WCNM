#!/usr/bin/env python3
"""Create and freeze the TiHuBird Stage C geometry release v1."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometry.geometry_release import (
    RELEASE_SCHEMA, aggregate_assets, camera_identity_sha256, sha256_file,
)
from geometry.two_hit import CACHE_SCHEMA, load_two_hit_cache


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def freeze_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        os.chmod(path, 0o555 if path.is_dir() else 0o444)
    os.chmod(root, 0o555)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-output", required=True, type=Path)
    parser.add_argument("--cache-output", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--mask-manifest", required=True, type=Path)
    parser.add_argument("--camera-root", required=True, type=Path)
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--manifest-output", required=True, type=Path)
    args = parser.parse_args()
    for key, value in vars(args).items():
        setattr(args, key, value.resolve())
    if args.release_root.exists() or args.manifest_output.exists():
        raise FileExistsError("geometry release identity already exists")
    code_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    mesh_metadata_source = args.mesh_output / "mesh_metadata.json"
    cache_metadata_source = args.cache_output / "cache_metadata.json"
    mesh_metadata = json.loads(mesh_metadata_source.read_text(encoding="utf-8"))
    cache_metadata = json.loads(cache_metadata_source.read_text(encoding="utf-8"))
    checkpoint_hash = sha256_file(args.checkpoint)
    if mesh_metadata["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("mesh/checkpoint source mismatch")
    if cache_metadata["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("cache/checkpoint source mismatch")
    source_mesh = args.mesh_output / "glass_mesh.ply"
    mesh_hash = sha256_file(source_mesh)
    if cache_metadata["mesh_sha256"] != mesh_hash:
        raise ValueError("mesh/cache source mismatch")

    args.release_root.mkdir(parents=True)
    copy_file(source_mesh, args.release_root / "glass_mesh.ply")
    copy_file(mesh_metadata_source, args.release_root / "mesh_metadata.json")
    copy_file(cache_metadata_source, args.release_root / "cache_metadata.json")
    for name in ("near_depth.png", "far_depth.png", "two_hit_valid.png", "back_surface_points.ply"):
        copy_file(args.cache_output / name, args.release_root / name)
    shutil.copytree(args.cache_output / "mesh_hits", args.release_root / "mesh_hits")

    mask_manifest = json.loads(args.mask_manifest.read_text(encoding="utf-8"))
    masks = {row["stem"]: row for row in mask_manifest["entries"]}
    cache_entries = []
    for cache_path in sorted((args.release_root / "mesh_hits").glob("*.npz")):
        cache = load_two_hit_cache(
            cache_path, expected_mesh_sha256=mesh_hash,
            expected_checkpoint_sha256=checkpoint_hash,
        )
        stem = str(cache["stem"].item())
        mask = masks[stem]
        camera_path = args.camera_root / f"{stem}.npz"
        with np.load(camera_path, allow_pickle=False) as archive:
            camera_hash = camera_identity_sha256(stem, archive)
        image_path = args.mask_manifest.parent.parent / mask["rgb_path"]
        mask_path = args.mask_manifest.parent / mask["mask_path"]
        cache_entries.append({
            "stem": stem, "relative_path": f"mesh_hits/{cache_path.name}",
            "sha256": sha256_file(cache_path),
            "source_image_path": str(image_path.resolve()),
            "source_image_sha256": sha256_file(image_path),
            "mask_path": str(mask_path.resolve()), "mask_sha256": sha256_file(mask_path),
            "camera_identity_sha256": camera_hash,
        })
    if len(cache_entries) != 111:
        raise ValueError("release creation requires exactly 111 caches")

    assets = []
    for path in sorted(item for item in args.release_root.rglob("*") if item.is_file()):
        assets.append({
            "relative_path": path.relative_to(args.release_root).as_posix(),
            "sha256": sha256_file(path), "size_bytes": path.stat().st_size,
        })
    manifest = {
        "schema": RELEASE_SCHEMA,
        "geometry_release_id": "stage_c_geometry_release_v1",
        "experiment_identity": "C03-r8 Tier 2 onset study",
        "geometry_source_branch": "3k-A", "geometry_source_global": 15000,
        "geometry_source_checkpoint_path": str(args.checkpoint),
        "geometry_source_checkpoint_sha256": checkpoint_hash,
        "code_commit_before_release": code_commit,
        "formal_mask_manifest_path": str(args.mask_manifest),
        "formal_mask_manifest_sha256": sha256_file(args.mask_manifest),
        "formal_mask_aggregate_sha256": mask_manifest["aggregate_mask_sha256"],
        "camera_identity_root": str(args.camera_root),
        "dr_provenance": {
            "manifest": mesh_metadata["dr_manifest"],
            "manifest_sha256": mesh_metadata["dr_manifest_sha256"],
            "normal_axis_fit": mesh_metadata["axis_fit"],
            "depth_metric_alignment": mesh_metadata["dr_depth_summary"],
        },
        "cuboid_six_plane_prior": {
            "axes_columns": mesh_metadata["axes_columns"],
            "fitted_lower": mesh_metadata["fitted_lower"],
            "fitted_upper": mesh_metadata["fitted_upper"],
            "assumption": "data-constrained six-plane glass-enclosure proxy for TiHuBird only",
        },
        "plane_fitting_and_mask_optimization": {
            "initial_lower": mesh_metadata["initial_lower"],
            "initial_upper": mesh_metadata["initial_upper"],
            "optimization": mesh_metadata["optimization"],
            "source_resolution_summary": mesh_metadata["source_resolution_summary"],
            "method": mesh_metadata["method"],
        },
        "mesh_generation_and_cleanup": {
            "vertex_count": mesh_metadata["topology"]["vertex_count"],
            "face_count": mesh_metadata["topology"]["face_count"],
            "watertight": mesh_metadata["topology"]["watertight"],
            "source_mesh": mesh_metadata["source_mesh"],
            "source_mesh_sha256": mesh_metadata["source_mesh_sha256"],
        },
        "release_root": str(args.release_root),
        "glass_mesh_relative_path": "glass_mesh.ply",
        "glass_mesh_sha256": mesh_hash,
        "mesh_metadata_sha256": sha256_file(args.release_root / "mesh_metadata.json"),
        "cache_schema": CACHE_SCHEMA, "cache_count": len(cache_entries),
        "cache_entries": cache_entries, "assets": assets,
        "aggregate_sha256": aggregate_assets(assets),
        "creation_time_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "This mesh is a data-constrained six-plane glass-enclosure proxy.",
            "It applies only to the current approximately cuboid TiHuBird scene.",
            "It is not a general transparent-object mesh extraction method.",
            "Raw DiffusionRenderer depth is relative and is used only after metric alignment.",
        ],
        "stage_d_contract": {
            "read_only": True, "runtime_mesh_generation": False,
            "runtime_cache_generation": False,
            "checkpoint_fields": ["geometry_release_id", "geometry_release_aggregate_sha256"],
        },
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.manifest_output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    os.replace(temporary, args.manifest_output)
    freeze_tree(args.release_root)
    freeze_tree(args.mesh_output)
    freeze_tree(args.cache_output)
    print(json.dumps({
        "geometry_release_id": manifest["geometry_release_id"],
        "aggregate_sha256": manifest["aggregate_sha256"],
        "manifest": str(args.manifest_output), "release_root": str(args.release_root),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
