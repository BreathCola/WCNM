#!/usr/bin/env python3
"""Rebind a review-only mask package to a revalidated, pixel-identical DR tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.dr_mask_proposal import audit_dr_artifacts, sha256_file
from utils.glass_mask_proposal import _atomic_json, tree_digest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    args = parser.parse_args(argv)
    scene = args.scene.expanduser().resolve()
    raw_root = args.raw_root.expanduser().resolve()
    package = args.package.expanduser().resolve()
    manifest_path = package / "proposal_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("scene") != "Tao" or manifest.get("count") != 112:
        raise ValueError("refusing to revalidate a different/non-Tao proposal")
    if manifest.get("training_eligible") is not False \
            or manifest.get("promotion_performed") is not False:
        raise ValueError("proposal package training/promotion boundary changed")
    audit = audit_dr_artifacts(scene, raw_root, expected_count=112, images="images")
    rows = manifest.get("per_frame_metrics", [])
    if [row.get("stem") for row in rows] != [entry["stem"] for entry in audit["entries"]]:
        raise ValueError("proposal/DR stem identity changed")
    for row, entry in zip(rows, audit["entries"]):
        stem = row["stem"]
        expected = {
            "source_rgb_sha256": entry["rgb"]["sha256"],
            "mask_soft_sha256": sha256_file(package / "mask_soft" / f"{stem}.png"),
            "mask_hard_sha256": sha256_file(package / "mask_hard" / f"{stem}.png"),
            "mask_eroded_sha256": sha256_file(package / "mask_eroded" / f"{stem}.png"),
            "overlay_sha256": sha256_file(package / "overlays" / f"{stem}.png"),
        }
        for key, value in expected.items():
            if row.get(key) != value:
                raise ValueError(f"{stem}: proposal artifact identity changed for {key}")
    _atomic_json(package / "dr_input_audit.json", audit)
    manifest["raw_manifest_sha256"] = audit["raw_manifest_sha256"]
    manifest["generation_validation_sha256"] = audit["generation_validation_sha256"]
    manifest["tree_before_manifest"] = tree_digest(
        package, exclude=("proposal_manifest.json",)
    )
    manifest["package_revalidation"] = {
        "status": "PASS", "pixel_artifacts_mutated": False,
        "reason": "bind enriched DR mode/shape/dtype contract",
    }
    _atomic_json(manifest_path, manifest)
    print(json.dumps({
        "count": manifest["count"], "human_status": manifest["human_status"],
        "training_eligible": manifest["training_eligible"],
        "raw_manifest_sha256": manifest["raw_manifest_sha256"],
        "tree_aggregate_sha256": manifest["tree_before_manifest"]["aggregate_sha256"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
