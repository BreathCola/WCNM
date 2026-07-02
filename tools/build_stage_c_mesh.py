#!/usr/bin/env python3
"""Fuse the accepted Stage C audit maps and extract a fixed glass mesh."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometry.tsdf_fusion import (
    extract_mesh, extract_outer_shell_mesh, fuse_tsdf, largest_face_component,
    make_volume_bounds, mesh_topology, write_mesh_ply,
)


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-resolution", type=int, default=128)
    parser.add_argument("--minimum-weight", type=int, default=2)
    parser.add_argument("--margin-fraction", type=float, default=0.03)
    parser.add_argument(
        "--cleanup", choices=("none", "outer-shell"), default="none",
        help="optional deterministic occupancy cleanup before mesh extraction",
    )
    parser.add_argument("--closing-iterations", type=int, default=2)
    args = parser.parse_args()
    args.audit = args.audit.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite Stage C mesh output: {args.output}")
    audit = json.loads((args.audit / "mesh_audit.json").read_text(encoding="utf-8"))
    if audit.get("verdict") != "STAGE_C_MESH_AUDIT_PASS":
        raise ValueError("mesh extraction requires STAGE_C_MESH_AUDIT_PASS")
    selection = audit.get("diffuse_selection") or {}
    if selection.get("candidate_ks_min") != 0.9:
        raise ValueError("mesh extraction requires the audited ks>=0.9 D candidate")
    raw_paths = sorted((args.audit / "raw_views").glob("*.npz"))
    if len(raw_paths) != 111:
        raise ValueError("mesh audit must contain exactly 111 raw views")
    lo, hi, voxel_size, dims = make_volume_bounds(
        audit["voxel_consistency"]["robust_bounds_min"],
        audit["voxel_consistency"]["robust_bounds_max"],
        args.max_resolution,
        args.margin_fraction,
    )
    views = []
    for path in raw_paths:
        with np.load(path) as data:
            views.append({key: data[key] for key in (
                "depth", "mask_eroded", "world_view_transform", "full_proj_transform"
            )})
    volume = fuse_tsdf(views, lo, dims, voxel_size)
    cleanup = None
    if args.cleanup == "outer-shell":
        vertices, faces, _, cleanup = extract_outer_shell_mesh(
            volume, args.minimum_weight, args.closing_iterations
        )
    else:
        vertices, faces, _ = extract_mesh(volume, args.minimum_weight)
    vertices, faces, components = largest_face_component(vertices, faces)
    topology = mesh_topology(vertices, faces)
    args.output.mkdir(parents=True)
    write_mesh_ply(args.output / "glass_mesh.ply", vertices, faces)
    np.savez_compressed(
        args.output / "tsdf_volume.npz", tsdf=volume.values.astype(np.float16),
        weight=volume.weights, origin=volume.origin, voxel_size=volume.voxel_size,
    )
    report = {
        "schema": "rtgs_stage_c_glass_mesh_v1", "audit": str(args.audit),
        "checkpoint_sha256": audit["checkpoint_sha256"], "diffuse_selection": selection,
        "bounds_min": lo.tolist(), "bounds_max": hi.tolist(), "grid_dims": dims.tolist(),
        "voxel_size": voxel_size, "minimum_weight": args.minimum_weight,
        "margin_fraction": args.margin_fraction,
        "cleanup": cleanup,
        "known_voxel_fraction": float(np.count_nonzero(volume.weights) / volume.weights.size),
        "negative_voxel_fraction": float(np.count_nonzero(volume.values < 0) / volume.values.size),
        "components": components, "topology": topology,
        "two_hit_ready": bool(
            topology["watertight"] and components["largest_face_fraction"] >= 0.70
        ),
    }
    atomic_json(args.output / "mesh_metadata.json", report)
    print(json.dumps(report, indent=2))
    return 0 if report["two_hit_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
