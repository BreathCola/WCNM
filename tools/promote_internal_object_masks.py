#!/usr/bin/env python3
"""Promote reviewed D-016 proposal masks to a formal immutable manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.internal_object_mask import (
    EXPECTED_STEMS,
    MASK_INTERPOLATION,
    MASK_ROLES,
    INTERNAL_OBJECT_SEMANTICS_VERSION,
    PROPOSAL_ROLE,
    REVIEWED_ROLE,
    SCHEMA_VERSION,
    canonical_payload_sha256,
    inspect_l_mask,
    sha256_file,
    validate_internal_object_mask_set,
)


def _approved_stems(path: Path) -> list[str]:
    values = [
        line.strip().split(",")[0]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lower().startswith("stem")
    ]
    return values


def _make_readonly(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        mode = path.stat().st_mode
        if path.is_file():
            path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        elif path.is_dir():
            path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    root.chmod(root.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default=str(ROOT / "data/TiHuBird"))
    parser.add_argument("--images", default="images")
    parser.add_argument("--proposal", required=True)
    parser.add_argument("--approved-stems-file", required=True)
    parser.add_argument("--output", default=str(ROOT / "data/TiHuBird/internal_object_masks_reviewed_v1"))
    parser.add_argument("--approval-note", required=True)
    args = parser.parse_args()

    scene = Path(args.scene).resolve()
    proposal = Path(args.proposal).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite formal mask directory: {output}")
    manifest_path = proposal / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("role") != PROPOSAL_ROLE:
        raise ValueError("promotion source must be an internal-object proposal manifest")
    if payload.get("internal_object_semantics_version") != INTERNAL_OBJECT_SEMANTICS_VERSION:
        raise ValueError("proposal semantic version is not promotable")
    if payload.get("ordered_stems") != EXPECTED_STEMS:
        raise ValueError("proposal does not contain exactly 111 ordered stems")
    approved = _approved_stems(Path(args.approved_stems_file))
    if approved != EXPECTED_STEMS:
        raise ValueError("promotion requires explicit approval of stems 000000--000110")

    output.mkdir(parents=True)
    for role in MASK_ROLES:
        (output / role).mkdir()
    aggregate = hashlib.sha256()
    entries = []
    for stem in EXPECTED_STEMS:
        rgb = scene / args.images / f"{stem}.jpg"
        if not rgb.is_file():
            raise FileNotFoundError(rgb)
        entry = {
            "stem": stem,
            "human_status": "accepted",
            "rgb_path": f"{args.images}/{rgb.name}",
            "rgb_sha256": sha256_file(rgb),
            "source": "grounded_sam2_proposal_human_reviewed",
            "approval_note": args.approval_note,
        }
        with Image.open(rgb) as image:
            size = image.size
        for role in MASK_ROLES:
            src = proposal / "processed" / role / f"{stem}.png"
            dst = output / role / f"{stem}.png"
            inspect_l_mask(src, size)
            shutil.copy2(src, dst)
            mask_sha = sha256_file(dst)
            entry[f"{role}_mask_path"] = f"{role}/{stem}.png"
            entry[f"{role}_mask_sha256"] = mask_sha
            aggregate.update(f"{stem} {role} {mask_sha}\n".encode("utf-8"))
        entries.append(entry)
    formal = {
        "schema_version": SCHEMA_VERSION,
        "role": REVIEWED_ROLE,
        "human_status": "accepted",
        "count": 111,
        "ordered_stems": EXPECTED_STEMS,
        "mask_interpolation": MASK_INTERPOLATION,
        "internal_object_semantics_version": INTERNAL_OBJECT_SEMANTICS_VERSION,
        "legacy_aliases": {
            "base": "legacy_not_auto_promoted_to_internal_base",
            "bird_support": "legacy_not_auto_promoted_to_internal_base",
        },
        "source_proposal_manifest": str(manifest_path),
        "source_proposal_manifest_sha256": sha256_file(manifest_path),
        "provenance": payload.get("grounded_sam2", {}),
        "approval": {
            "approved_stems_file": str(Path(args.approved_stems_file).resolve()),
            "approved_stems_file_sha256": sha256_file(args.approved_stems_file),
            "note": args.approval_note,
        },
        "aggregate_mask_sha256": aggregate.hexdigest(),
        "entries": entries,
    }
    formal["manifest_payload_sha256"] = canonical_payload_sha256(formal)
    (output / "manifest.json").write_text(
        json.dumps(formal, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    validate_internal_object_mask_set(scene, args.images, output / "manifest.json")
    _make_readonly(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
