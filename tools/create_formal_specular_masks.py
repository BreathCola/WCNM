#!/usr/bin/env python3
"""Create the versioned TiHuBird reviewed-v1 soft-mask archive exactly once."""

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.specular_mask import (
    FORMAL_ROLE,
    FORMAL_SCHEMA_VERSION,
    MASK_INTERPOLATION,
    canonical_payload_sha256,
    inspect_mask,
    sha256_file,
    validate_specular_mask_set,
)


REPAIR_STEMS = {"000039", "000040", "000041"}


def _copy_exact(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    if temporary.read_bytes() != source.read_bytes():
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"byte-copy verification failed: {source}")
    os.replace(temporary, target)


def create_archive(scene, proposal_root, repair_root, destination, read_only=True):
    scene = Path(scene).resolve()
    proposal_root = Path(proposal_root).resolve()
    repair_root = Path(repair_root).resolve()
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"formal archive destination must not already exist: {destination}")
    expected = [f"{index:06d}" for index in range(111)]
    images = sorted((scene / "images").glob("*.jpg"))
    if [path.stem for path in images] != expected:
        raise ValueError("TiHuBird RGB set must be exactly 000000--000110")

    destination.mkdir(parents=True)
    entries = []
    aggregate = hashlib.sha256()
    try:
        for stem, rgb_path in zip(expected, images):
            if stem in REPAIR_STEMS:
                source_kind = "repair_candidate_v1"
                source_path = repair_root / "repair_candidates" / f"{stem}.png"
            else:
                source_kind = "proposal_v1"
                source_path = proposal_root / "proposal_soft" / stem / "proposal_soft.png"
            if not source_path.is_file():
                raise FileNotFoundError(f"missing accepted mask source: {source_path}")
            target = destination / f"{stem}.png"
            _copy_exact(source_path, target)
            with Image.open(rgb_path) as rgb:
                size = rgb.size
            facts = inspect_mask(target, size)
            digest = sha256_file(target)
            if digest != sha256_file(source_path):
                raise RuntimeError(f"source/target hash mismatch for {stem}")
            aggregate.update(f"{stem} {digest}\n".encode("utf-8"))
            entries.append({
                "stem": stem,
                "rgb_path": f"images/{rgb_path.name}",
                "rgb_sha256": sha256_file(rgb_path),
                "mask_path": target.name,
                "mask_sha256": digest,
                "source": source_kind,
                "source_reference": str(source_path.relative_to(scene.parent.parent)),
                "source_sha256": digest,
                "size": facts["size"],
                "mode": facts["mode"],
                "dtype": facts["dtype"],
                "area_ratio": facts["area_ratio"],
                "bbox_xyxy": facts["bbox_xyxy"],
                "human_status": "accepted",
            })
        payload = {
            "schema_version": FORMAL_SCHEMA_VERSION,
            "role": FORMAL_ROLE,
            "version": "reviewed_v1",
            "human_status": "accepted",
            "count": 111,
            "ordered_stems": expected,
            "mask_interpolation": MASK_INTERPOLATION,
            "aggregate_mask_sha256": aggregate.hexdigest(),
            "padding_exclusion_proof": {
                "excluded_stems": [f"{index:06d}" for index in range(111, 120)],
                "mixed_count": 0,
                "reason": "DR batch padding slots are not real TiHuBird RGB stems",
            },
            "entries": entries,
        }
        payload["manifest_payload_sha256"] = canonical_payload_sha256(payload)
        manifest = destination / "manifest.json"
        manifest.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        validated = validate_specular_mask_set(scene, "images", manifest)
        if read_only:
            for path in destination.iterdir():
                path.chmod(0o444)
            destination.chmod(0o555)
        return validated
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--proposal_root", required=True)
    parser.add_argument("--repair_root", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    result = create_archive(args.scene, args.proposal_root, args.repair_root, args.destination)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
