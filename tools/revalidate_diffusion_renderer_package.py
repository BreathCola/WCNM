#!/usr/bin/env python3
"""Revalidate an immutable packaged DR tree and atomically enrich its manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.diffusion_renderer_raw import atomic_json, validate_and_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    args = parser.parse_args(argv)
    scene = args.scene.expanduser().resolve()
    package = args.package.expanduser().resolve()
    existing = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    if existing.get("scene") != scene.name or existing.get("real_frame_count") != 112:
        raise ValueError("refusing to revalidate a different/non-Tao DR package")
    manifest, validation = validate_and_manifest(
        scene=scene, images=str(existing["images_directory"]),
        raw_root=package / "artifacts", group_name=scene.name,
        target_hw=tuple(existing["effective_config"]["inference_res"]),
        frames_per_chunk=int(existing["frames_per_chunk"]),
        generation_identity=existing["generation_identity"],
        effective_config=existing["effective_config"], repository_root=ROOT,
    )
    for record in manifest["frames_and_padding"]:
        record["diffusion_renderer_rgb_file"] = "artifacts/" + record["diffusion_renderer_rgb_file"]
        record["raw_prior_files"] = {
            kind: "artifacts/" + value for kind, value in record["raw_prior_files"].items()
        }
    validation["revalidation_role"] = "packaged_tree_audit_no_raw_artifact_mutation"
    atomic_json(package / "manifest.json", manifest)
    atomic_json(package / "validation_summary.json", validation)
    print(json.dumps({
        "artifact_contracts": manifest["artifact_contracts"],
        "real_frames": manifest["real_frame_count"],
        "padding_frames": manifest["padding_frame_count"],
        "validation": validation["validation"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
