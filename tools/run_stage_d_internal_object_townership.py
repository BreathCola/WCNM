#!/usr/bin/env python3
"""Plan or execute the D-016 internal-object T ownership pilot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometry.geometry_release import validate_geometry_release
from stage_d_training import (
    FORMAL_RELEASE_SHA256,
    FORMAL_SOURCE_SHA256,
    INTERNAL_OBJECT_ENDPOINT,
    INTERNAL_OBJECT_NODES,
    INTERNAL_OBJECT_OUTPUT_NAME,
)
from utils.internal_object_mask import validate_internal_object_mask_set
from utils.specular_mask import validate_specular_mask_set


SOURCE = ROOT / "output/tier2_c03_r8_oneshot_v4_rstart_g03000_to_g15000/chkpnt15000.pth"
RELEASE = ROOT / "geometry_releases/stage_c_geometry_release_v1.json"
GLASS_MASK = "specular_masks_reviewed_v1/manifest.json"
INTERNAL_MASK = "internal_object_masks_reviewed_v1/manifest.json"
OUTPUT = ROOT / "output" / INTERNAL_OBJECT_OUTPUT_NAME


def sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def training_command(args) -> list[str]:
    return [
        sys.executable, str(ROOT / "train.py"),
        "-s", str(ROOT / "data/TiHuBird"),
        "-m", str(OUTPUT),
        "--images", "images",
        "--model_type", "surfel",
        "--stage", "stage_d",
        "--experiment", "TiHuBird C03-r8 Stage D D-016 internal-object T ownership pilot",
        "--resolution", "8",
        "--normal_priors", "diffrender_priors_candidates/axis_smoke/C03/normal",
        "--normal_prior_space", "camera",
        "--specular_masks", GLASS_MASK,
        "--internal_object_masks", args.internal_object_masks,
        "--geometry_release_manifest", str(RELEASE),
        "--transmittance_init_mode", "transferred_d_inside",
        "--transmittance_init_count", "4096",
        "--transmittance_init_seed", "20260703",
        "--transmittance_compose", "alpha_over",
        "--transparent_path_mode", "cuboid_front_v1",
        "--transparent_direct_mode", "off",
        "--transparent_reflection_mode", "off",
        "--cout_ownership_mode", "support_safe_outside",
        "--ray_background", "scene",
        "--ray_chunk_size", "2048",
        "--iterations", str(INTERNAL_OBJECT_ENDPOINT),
        "--start_checkpoint", str(SOURCE),
        "--lambda_spec", "0.2",
        "--specular_k0", "0.9",
        "--lambda_depth", "0.2",
        "--stage_d_depth_start_iteration", "40000",
        "--stage_d_internal_object_pilot",
        "--stage_d_phase_a_end_iteration", str(INTERNAL_OBJECT_ENDPOINT),
        "--transparent_interface_margin", "0.05",
        "--transparent_interface_margin_mode", "exclude",
        "--transfer_min_views", "3",
        "--transfer_min_total_weight", "0.01",
        "--transfer_depth_margin", "0.05",
        "--object_mask_min_views", str(args.object_mask_min_views),
        "--object_mask_min_support_ratio", str(args.object_mask_min_support_ratio),
        "--object_mask_boundary_ignore_px", str(args.object_mask_boundary_ignore_px),
        "--object_mask_erode_px", str(args.object_mask_erode_px),
        "--object_mask_dilate_px", str(args.object_mask_dilate_px),
        "--object_alpha_floor", str(args.object_alpha_floor),
        "--lambda_object_positive", str(args.lambda_object_positive),
        "--lambda_object_negative", str(args.lambda_object_negative),
        "--stage_d_cache_parity_atol", "2e-5",
        "--stage_d_cache_parity_mean_atol", "2e-6",
        "--disable_viewer",
        "--quiet",
        "--save_iterations", *[str(node) for node in INTERNAL_OBJECT_NODES],
        "--checkpoint_iterations", *[str(node) for node in INTERNAL_OBJECT_NODES],
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--internal-object-masks", default=INTERNAL_MASK)
    parser.add_argument("--object-mask-min-views", type=int, default=3)
    parser.add_argument("--object-mask-min-support-ratio", type=float, default=0.6)
    parser.add_argument("--object-mask-boundary-ignore-px", type=int, default=5)
    parser.add_argument("--object-mask-erode-px", type=int, default=3)
    parser.add_argument("--object-mask-dilate-px", type=int, default=3)
    parser.add_argument("--object-alpha-floor", type=float, default=0.35)
    parser.add_argument("--lambda-object-positive", type=float, default=0.05)
    parser.add_argument("--lambda-object-negative", type=float, default=0.05)
    args = parser.parse_args()

    release = validate_geometry_release(RELEASE)
    glass = validate_specular_mask_set(ROOT / "data/TiHuBird", "images", GLASS_MASK)
    internal = validate_internal_object_mask_set(
        ROOT / "data/TiHuBird", "images", args.internal_object_masks,
    )
    source_sha = sha256_file(SOURCE)
    if source_sha != FORMAL_SOURCE_SHA256:
        raise RuntimeError("immutable Stage-B source hash mismatch")
    if release.validation.get("aggregate_sha256") != FORMAL_RELEASE_SHA256:
        raise RuntimeError("immutable Stage-C release hash mismatch")
    if OUTPUT.exists():
        raise FileExistsError(f"refusing existing D-016 output: {OUTPUT}")
    command = training_command(args)
    plan = {
        "schema": "rtgs_stage_d_internal_object_operator_plan_v1",
        "execute": bool(args.execute),
        "output": str(OUTPUT),
        "source": str(SOURCE),
        "source_sha256": source_sha,
        "release": str(RELEASE),
        "release_aggregate_sha256": release.validation.get("aggregate_sha256"),
        "glass_mask_manifest_sha256": glass["manifest_file_sha256"],
        "internal_object_manifest_sha256": internal["manifest_file_sha256"],
        "command": command,
        "note": "Default mode is plan-only; --execute launches the bounded T-only pilot.",
    }
    print(json.dumps(plan, indent=2, sort_keys=True))
    if not args.execute:
        return 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
