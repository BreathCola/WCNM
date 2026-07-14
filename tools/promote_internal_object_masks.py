#!/usr/bin/env python3
"""Promote the D-016c reviewed 111-view proposal to a formal local release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.internal_object_mask import (
    EXPECTED_STEMS,
    INTERNAL_OBJECT_SEMANTICS_VERSION,
    MASK_INTERPOLATION,
    MASK_ROLES,
    PROPOSAL_ROLE,
    REVIEWED_ROLE,
    SCHEMA_VERSION,
    canonical_payload_sha256,
    inspect_l_mask,
    sha256_file,
    validate_internal_object_mask_set,
)
from utils.specular_mask import load_resized_formal_mask, validate_specular_mask_set


DEFAULT_PROPOSAL = ROOT / "output/stage_d_tihubird_internal_object_mask_proposal_111_v3"
DEFAULT_OUTPUT = ROOT / "data/TiHuBird/internal_object_masks_reviewed_v3"
ACCEPTED_WITH_WARNING = ("000012", "000048", "000049", "000050", "000051", "000052", "000053")
WARNING_NOTES = {
    "000012": "yellow-base forward/backward IoU was low; user verified the final visible mask is correct.",
    "000048": "raw propagation leaked outside glass; processed final mask is clipped correctly inside glass_hard.",
    "000049": "raw propagation leaked outside glass; processed final mask is clipped correctly inside glass_hard.",
    "000050": "raw propagation leaked outside glass; processed final mask is clipped correctly inside glass_hard.",
    "000051": "raw propagation leaked outside glass; processed final mask is clipped correctly inside glass_hard.",
    "000052": "raw propagation leaked outside glass; processed final mask is clipped correctly inside glass_hard.",
    "000053": "raw propagation leaked outside glass; processed final mask is clipped correctly inside glass_hard.",
}
FIXED_NINE = ("000000", "000014", "000028", "000042", "000055", "000069", "000083", "000097", "000110")


class PromotionError(RuntimeError):
    """Raised when the D-016c promotion contract is not satisfied."""


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PromotionError(f"missing required JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _git_head() -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()


def _git_branch() -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), "branch", "--show-current"], text=True).strip()


def _make_readonly(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        mode = path.stat().st_mode
        if path.is_file():
            path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        elif path.is_dir():
            path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    root.chmod(root.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def _ensure_writable_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite formal internal-object release: {path}")


def _binary_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    facts = inspect_l_mask(path, size)
    if facts["min"] not in (0, 255) or facts["max"] not in (0, 255):
        raise PromotionError(f"mask is not binary 0/255: {path}")
    with Image.open(path) as image:
        return np.asarray(image, dtype=np.uint8) >= 128


def _processed_hash(proposal: Path) -> str:
    digest = hashlib.sha256()
    for role in (*MASK_ROLES, "glass_hard"):
        for stem in EXPECTED_STEMS:
            path = proposal / "processed" / role / f"{stem}.png"
            digest.update(f"{role}/{stem}.png ".encode("utf-8"))
            digest.update(sha256_file(path).encode("ascii"))
            digest.update(b"\n")
    return digest.hexdigest()


def _read_metrics(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise PromotionError(f"missing required CSV: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if [row.get("stem") for row in rows] != EXPECTED_STEMS:
        raise PromotionError("per_frame_metrics.csv must contain exactly stems 000000--000110")
    for row in rows:
        for key, value in row.items():
            if value in (None, ""):
                continue
            if key.endswith("_sha256") or key in {"stem", "frame_status", "review_flags"}:
                continue
            try:
                number = float(value)
            except ValueError as exc:
                raise PromotionError(f"non-numeric metric {key} for {row['stem']}: {value}") from exc
            if not math.isfinite(number):
                raise PromotionError(f"non-finite metric {key} for {row['stem']}: {value}")
    return {row["stem"]: row for row in rows}


def _check_hash_value(name: str, value: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise PromotionError(f"{name} is not a lowercase SHA-256 digest")


def audit_proposal(
    *,
    scene: Path = ROOT / "data/TiHuBird",
    images: str = "images",
    proposal: Path = DEFAULT_PROPOSAL,
) -> dict[str, Any]:
    """Validate the reviewed proposal without writing a formal release."""
    scene = Path(scene).resolve()
    proposal = Path(proposal).resolve()
    image_root = scene / images
    specular_root = scene / "specular_masks_reviewed_v1"
    specular_manifest = validate_specular_mask_set(scene, images, "specular_masks_reviewed_v1/manifest.json")

    proposal_manifest_path = proposal / "proposal_manifest.json"
    proposal_manifest = _json(proposal_manifest_path)
    proposal_summary = _json(proposal / "proposal_summary.json")
    generation_identity = _json(proposal / "generation_code_identity.json")
    fixed_nine_review = _json(proposal / "fixed_nine_human_review_v3.json")
    metrics = _read_metrics(proposal / "per_frame_metrics.csv")

    if proposal_manifest.get("role") != PROPOSAL_ROLE:
        raise PromotionError("source proposal role mismatch")
    if proposal_manifest.get("human_status") != "proposal_requires_review":
        raise PromotionError("source proposal must remain proposal_requires_review")
    if proposal_manifest.get("internal_object_semantics_version") != INTERNAL_OBJECT_SEMANTICS_VERSION:
        raise PromotionError("source proposal semantic version mismatch")
    if proposal_manifest.get("count") != 111 or proposal_manifest.get("ordered_stems") != EXPECTED_STEMS:
        raise PromotionError("source proposal must contain exactly stems 000000--000110")
    if proposal_manifest.get("manifest_payload_sha256") != canonical_payload_sha256(proposal_manifest):
        raise PromotionError("source proposal payload hash mismatch")
    proposal_manifest_file_sha = sha256_file(proposal_manifest_path)
    if generation_identity.get("proposal_manifest_sha256") != proposal_manifest_file_sha:
        raise PromotionError("generation_code_identity proposal manifest hash mismatch")
    if proposal_summary.get("artifact_role") != "stage_d_internal_object_mask_proposal_111_v3":
        raise PromotionError("proposal_summary artifact role mismatch")
    if proposal_summary.get("status_counts", {}).get("manual_edit_required") != 0:
        raise PromotionError("manual_edit_required frames cannot be promoted")
    if fixed_nine_review.get("review_authorization", {}).get("type") != "explicit_user_approval":
        raise PromotionError("fixed-nine anchor review authorization missing")
    if fixed_nine_review.get("accepted_stems") != list(FIXED_NINE):
        raise PromotionError("fixed-nine anchor stems mismatch")

    for role in (*MASK_ROLES, "glass_hard"):
        role_root = proposal / "processed" / role
        if sorted(path.stem for path in role_root.glob("*.png")) != EXPECTED_STEMS:
            raise PromotionError(f"processed/{role} must contain exactly 111 PNG stems")

    rgb_hashes: dict[str, str] = {}
    glass_hashes: dict[str, str] = {}
    glass_hard_hashes: dict[str, str] = {}
    mask_hashes: dict[str, dict[str, str]] = {role: {} for role in MASK_ROLES}
    entries: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    anchor_mismatches: list[str] = []

    for stem in EXPECTED_STEMS:
        rgb = image_root / f"{stem}.jpg"
        if not rgb.is_file():
            raise PromotionError(f"missing RGB image: {rgb}")
        with Image.open(rgb) as image:
            if image.mode != "RGB":
                raise PromotionError(f"RGB image must be mode RGB: {rgb}")
            size = image.size
        rgb_sha = sha256_file(rgb)
        if metrics[stem].get("rgb_sha256") != rgb_sha:
            raise PromotionError(f"RGB hash mismatch for {stem}")
        rgb_hashes[stem] = rgb_sha

        glass = specular_root / f"{stem}.png"
        if not glass.is_file():
            raise PromotionError(f"missing reviewed glass mask: {glass}")
        glass_sha = sha256_file(glass)
        if metrics[stem].get("glass_sha256") != glass_sha:
            raise PromotionError(f"glass hash mismatch for {stem}")
        if proposal_manifest["entries"][int(stem)]["glass_sha256"] != glass_sha:
            raise PromotionError(f"proposal manifest glass hash mismatch for {stem}")
        proposal_glass = proposal / "processed/glass_hard" / f"{stem}.png"
        proposal_glass_sha = sha256_file(proposal_glass)
        glass_mask = _binary_mask(proposal_glass, size)
        derived_glass_values, derived_glass_sha, _ = load_resized_formal_mask(
            specular_manifest, stem, size, size,
        )
        if derived_glass_sha != glass_sha or not np.array_equal(glass_mask, derived_glass_values >= 0.5):
            raise PromotionError(f"processed glass_hard does not match formal glass threshold for {stem}")
        glass_hashes[stem] = glass_sha
        glass_hard_hashes[stem] = proposal_glass_sha

        native: dict[str, np.ndarray] = {}
        entry = {
            "stem": stem,
            "human_status": "accepted_with_warning" if stem in ACCEPTED_WITH_WARNING else "accepted",
            "rgb_path": f"{images}/{stem}.jpg",
            "rgb_sha256": rgb_sha,
            "glass_mask_path": f"specular_masks_reviewed_v1/{stem}.png",
            "glass_mask_sha256": glass_sha,
            "glass_hard_mask_path": f"glass_hard/{stem}.png",
            "glass_hard_mask_sha256": proposal_glass_sha,
            "source_processed_paths": {},
            "source_processed_sha256": {},
        }
        entry["source_processed_paths"]["glass_hard"] = str(proposal_glass)
        entry["source_processed_sha256"]["glass_hard"] = proposal_glass_sha
        for role in MASK_ROLES:
            src = proposal / "processed" / role / f"{stem}.png"
            mask_sha = sha256_file(src)
            if proposal_manifest["entries"][int(stem)][f"{role}_sha256"] != mask_sha:
                raise PromotionError(f"proposal manifest {role} hash mismatch for {stem}")
            if metrics[stem].get(f"{role}_sha256") != mask_sha:
                raise PromotionError(f"per-frame metrics {role} hash mismatch for {stem}")
            native[role] = _binary_mask(src, size)
            mask_hashes[role][stem] = mask_sha
            entry["source_processed_paths"][role] = str(src)
            entry["source_processed_sha256"][role] = mask_sha
            entry[f"{role}_mask_path"] = f"{role}/{stem}.png"
            entry[f"{role}_mask_sha256"] = mask_sha
            aggregate.update(f"{stem} {role} {mask_sha}\n".encode("utf-8"))
        if not np.array_equal(native["internal_object_union"], native["bird"] | native["internal_base"]):
            raise PromotionError(f"union equation failed for {stem}")
        if (native["bird"] & ~native["internal_object_union"]).any():
            raise PromotionError(f"bird subset failed for {stem}")
        if (native["internal_base"] & ~native["internal_object_union"]).any():
            raise PromotionError(f"internal_base subset failed for {stem}")
        if (native["internal_object_union"] & ~glass_mask).any():
            raise PromotionError(f"union subset glass_hard failed for {stem}")
        if stem in FIXED_NINE:
            anchor = fixed_nine_review["entries"][stem]["mask_sha256"]
            for role in MASK_ROLES:
                if anchor[role] != mask_hashes[role][stem]:
                    anchor_mismatches.append(f"{stem}:{role}")
        entries.append(entry)
    if anchor_mismatches:
        raise PromotionError("fixed-nine anchor hash mismatch: " + ", ".join(anchor_mismatches))

    source_generation_code_identity = {
        "path": str((proposal / "generation_code_identity.json").resolve()),
        "sha256": sha256_file(proposal / "generation_code_identity.json"),
        "payload": generation_identity,
    }
    return {
        "status": "PASS",
        "proposal_path": str(proposal),
        "proposal_processed_hash": _processed_hash(proposal),
        "source_proposal_manifest_sha256": proposal_manifest_file_sha,
        "source_proposal_payload_hash": proposal_manifest["manifest_payload_sha256"],
        "source_generation_code_identity": source_generation_code_identity,
        "proposal_summary_sha256": sha256_file(proposal / "proposal_summary.json"),
        "per_frame_metrics_sha256": sha256_file(proposal / "per_frame_metrics.csv"),
        "review_queue_sha256": sha256_file(proposal / "review_queue.csv"),
        "fixed_nine_human_review_sha256": sha256_file(proposal / "fixed_nine_human_review_v3.json"),
        "proposal_status_counts": proposal_summary.get("status_counts", {}),
        "proposal_review_queue_count": proposal_summary.get("review_queue_count"),
        "rgb_hashes": rgb_hashes,
        "glass_hashes": glass_hashes,
        "glass_hard_hashes": glass_hard_hashes,
        "bird_hashes": mask_hashes["bird"],
        "internal_base_hashes": mask_hashes["internal_base"],
        "internal_object_union_hashes": mask_hashes["internal_object_union"],
        "aggregate_mask_sha256": aggregate.hexdigest(),
        "entries": entries,
    }


def promote_reviewed_release(
    *,
    scene: Path = ROOT / "data/TiHuBird",
    images: str = "images",
    proposal: Path = DEFAULT_PROPOSAL,
    output: Path = DEFAULT_OUTPUT,
    dry_run: bool = False,
) -> dict[str, Any]:
    audit = audit_proposal(scene=scene, images=images, proposal=proposal)
    output = Path(output).resolve()
    if dry_run:
        return {"mode": "dry_run", **audit}
    _ensure_writable_parent(output)

    output.mkdir()
    try:
        for role in MASK_ROLES:
            (output / role).mkdir()
        (output / "glass_hard").mkdir()
        (output / "internal_ignore").mkdir()
        for entry in audit["entries"]:
            stem = entry["stem"]
            src_glass = Path(entry["source_processed_paths"]["glass_hard"])
            dst_glass = output / "glass_hard" / f"{stem}.png"
            shutil.copy2(src_glass, dst_glass)
            if sha256_file(src_glass) != sha256_file(dst_glass) or sha256_file(dst_glass) != entry["glass_hard_mask_sha256"]:
                raise PromotionError(f"byte-copy hash verification failed for {stem} glass_hard")
            for role in MASK_ROLES:
                src = Path(entry["source_processed_paths"][role])
                dst = output / role / f"{stem}.png"
                before = sha256_file(src)
                shutil.copy2(src, dst)
                after = sha256_file(dst)
                if before != after or after != entry[f"{role}_mask_sha256"]:
                    raise PromotionError(f"byte-copy hash verification failed for {stem} {role}")

        review_record = {
            "schema_version": SCHEMA_VERSION,
            "artifact_role": "stage_d_internal_object_human_review_record",
            "human_status": "accepted",
            "accepted_count": 111,
            "rejected_count": 0,
            "manual_edit_required_count": 0,
            "accepted_stems": EXPECTED_STEMS,
            "accepted_with_warning": list(ACCEPTED_WITH_WARNING),
            "accepted_with_warning_notes": WARNING_NOTES,
            "review_authorization": {
                "type": "explicit_user_approval",
                "source": "current_codex_task_prompt",
            },
            "automated_review_flags_provenance": {
                "review_queue_sha256": audit["review_queue_sha256"],
                "proposal_status_counts": audit["proposal_status_counts"],
                "proposal_review_queue_count": audit["proposal_review_queue_count"],
                "disposition": "All automatic warnings were manually overridden to accepted by explicit user approval.",
            },
            "processed_mask_pixels_modified": False,
        }
        review_record["canonical_payload_sha256"] = canonical_payload_sha256(review_record)

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "artifact_role": REVIEWED_ROLE,
            "role": REVIEWED_ROLE,
            "human_status": "accepted",
            "accepted_count": 111,
            "count": 111,
            "ordered_stems": EXPECTED_STEMS,
            "accepted_stems": EXPECTED_STEMS,
            "accepted_with_warning": list(ACCEPTED_WITH_WARNING),
            "accepted_with_warning_notes": WARNING_NOTES,
            "rejected_count": 0,
            "manual_edit_required_count": 0,
            "mask_interpolation": MASK_INTERPOLATION,
            "internal_object_semantics_version": INTERNAL_OBJECT_SEMANTICS_VERSION,
            "legacy_aliases": {
                "base": "legacy_not_auto_promoted_to_internal_base",
                "bird_support": "legacy_not_auto_promoted_to_internal_base",
            },
            "source_proposal_path": audit["proposal_path"],
            "source_proposal_manifest_sha256": audit["source_proposal_manifest_sha256"],
            "source_proposal_payload_hash": audit["source_proposal_payload_hash"],
            "source_proposal_processed_hash": audit["proposal_processed_hash"],
            "source_generation_code_identity": audit["source_generation_code_identity"],
            "proposal_summary_sha256": audit["proposal_summary_sha256"],
            "per_frame_metrics_sha256": audit["per_frame_metrics_sha256"],
            "review_queue_sha256": audit["review_queue_sha256"],
            "fixed_nine_human_review_sha256": audit["fixed_nine_human_review_sha256"],
            "rgb_hashes": audit["rgb_hashes"],
            "glass_hashes": audit["glass_hashes"],
            "glass_hard_hashes": audit["glass_hard_hashes"],
            "bird_hashes": audit["bird_hashes"],
            "internal_base_hashes": audit["internal_base_hashes"],
            "internal_object_union_hashes": audit["internal_object_union_hashes"],
            "aggregate_mask_sha256": audit["aggregate_mask_sha256"],
            "repository_commit_identity": {
                "head": _git_head(),
                "branch": _git_branch(),
                "data_release_tracked_by_git": False,
                "note": "data/TiHuBird is ignored by this repository; this release is a local immutable data artifact.",
            },
            "review_authorization": {
                "type": "explicit_user_approval",
                "source": "current_codex_task_prompt",
            },
            "immutable_contract": {
                "write_bits_removed_after_creation": True,
                "overwrite_policy": "fail_closed_if_output_directory_exists",
                "copy_policy": "byte_copy_from_source_processed_masks_no_reencoding",
            },
            "human_review_record_path": "human_review_record.json",
            "human_review_record_sha256_pending": True,
            "entries": audit["entries"],
        }
        _write_json(output / "human_review_record.json", review_record)
        manifest["human_review_record_sha256"] = sha256_file(output / "human_review_record.json")
        manifest.pop("human_review_record_sha256_pending")
        manifest["canonical_payload_sha256"] = canonical_payload_sha256(manifest)
        manifest["manifest_payload_sha256"] = manifest["canonical_payload_sha256"]
        _write_json(output / "manifest.json", manifest)
        (output / "README.md").write_text(
            "# TiHuBird Internal-Object Masks Reviewed V3\n\n"
            "Formal Stage D D-016c local reviewed release.\n\n"
            "- Role: `stage_d_internal_object_masks_reviewed`\n"
            "- Human status: `accepted`\n"
            "- Semantic version: `tihubird_bird_and_internal_base_v3`\n"
            "- Accepted stems: `000000` through `000110` (111/111)\n"
            "- Source proposal: `output/stage_d_tihubird_internal_object_mask_proposal_111_v3`\n"
            "- Mask files are byte copies of the proposal `processed/` PNGs.\n"
            "- `glass_hard/` is included as the reviewed hard glass domain used for release validation.\n"
            "- The proposal remains provenance only and is not a training input.\n"
            "- `internal_ignore/` is intentionally empty for this release.\n"
            "- This directory is a local data artifact because `data/` is ignored by Git.\n",
            encoding="utf-8",
        )
        validated = validate_internal_object_mask_set(scene, images, output / "manifest.json")
        _make_readonly(output)
    except Exception:
        if output.exists():
            shutil.rmtree(output, ignore_errors=True)
        raise

    after_source_hash = _processed_hash(Path(proposal).resolve())
    if after_source_hash != audit["proposal_processed_hash"]:
        raise PromotionError("source proposal processed hash changed during promotion")
    return {
        "mode": "promote",
        "formal_release_path": str(output),
        "validated": validated,
        "source_proposal_processed_hash_before": audit["proposal_processed_hash"],
        "source_proposal_processed_hash_after": after_source_hash,
        "aggregate_mask_sha256": audit["aggregate_mask_sha256"],
        "manifest_sha256": sha256_file(output / "manifest.json"),
        "human_review_record_sha256": sha256_file(output / "human_review_record.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default=str(ROOT / "data/TiHuBird"))
    parser.add_argument("--images", default="images")
    parser.add_argument("--proposal", default=str(DEFAULT_PROPOSAL))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = promote_reviewed_release(
        scene=Path(args.scene),
        images=args.images,
        proposal=Path(args.proposal),
        output=Path(args.output),
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
