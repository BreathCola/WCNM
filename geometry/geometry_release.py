"""CPU-only validation and loading for immutable Stage C geometry releases."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from geometry.two_hit import CACHE_SCHEMA, load_two_hit_cache


RELEASE_SCHEMA = "rtgs_stage_c_geometry_release_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate_assets(entries: list[dict]) -> str:
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda row: row["relative_path"]):
        digest.update(entry["relative_path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(entry["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def camera_identity_sha256(stem: str, archive) -> str:
    digest = hashlib.sha256(stem.encode("ascii"))
    depth = np.asarray(archive["depth"])
    digest.update(np.asarray(depth.shape, dtype=np.int64).tobytes())
    for key in ("world_view_transform", "full_proj_transform", "camera_center"):
        value = np.ascontiguousarray(archive[key], dtype=np.float32)
        digest.update(key.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _require_read_only(path: Path) -> None:
    if path.stat().st_mode & 0o222:
        raise ValueError(f"geometry release asset is writable: {path}")


def load_release_manifest(path: Path) -> dict:
    manifest_path = Path(path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != RELEASE_SCHEMA:
        raise ValueError("unsupported Stage C geometry release schema")
    if manifest.get("geometry_release_id") != "stage_c_geometry_release_v1":
        raise ValueError("unsupported Stage C geometry release identity")
    manifest["_manifest_path"] = str(manifest_path)
    return manifest


def validate_geometry_release(path: Path) -> dict:
    manifest = load_release_manifest(path)
    release_root = Path(manifest["release_root"]).resolve()
    if not release_root.is_dir():
        raise FileNotFoundError(f"geometry release root is missing: {release_root}")
    _require_read_only(release_root)
    assets = manifest.get("assets")
    if not isinstance(assets, list) or not assets:
        raise ValueError("geometry release assets are missing")
    for entry in assets:
        relative = Path(entry["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("geometry release asset path escapes release root")
        asset = release_root / relative
        if not asset.is_file():
            raise FileNotFoundError(f"geometry release asset is missing: {asset}")
        _require_read_only(asset)
        if sha256_file(asset) != entry["sha256"]:
            raise ValueError(f"geometry release asset hash mismatch: {asset}")
    aggregate = aggregate_assets(assets)
    if aggregate != manifest.get("aggregate_sha256"):
        raise ValueError("geometry release aggregate SHA-256 mismatch")

    checkpoint = Path(manifest["geometry_source_checkpoint_path"])
    if sha256_file(checkpoint) != manifest["geometry_source_checkpoint_sha256"]:
        raise ValueError("geometry source checkpoint hash mismatch")
    mask_manifest_path = Path(manifest["formal_mask_manifest_path"])
    if sha256_file(mask_manifest_path) != manifest["formal_mask_manifest_sha256"]:
        raise ValueError("formal mask manifest hash mismatch")
    mask_manifest = json.loads(mask_manifest_path.read_text(encoding="utf-8"))
    masks = {entry["stem"]: entry for entry in mask_manifest["entries"]}
    camera_root = Path(manifest["camera_identity_root"])
    cache_metadata = json.loads((release_root / "cache_metadata.json").read_text(encoding="utf-8"))
    cache_rows = {row["stem"]: row for row in cache_metadata["per_view"]}
    cache_entries = manifest.get("cache_entries", [])
    if len(cache_entries) != 111 or len(cache_rows) != 111 or len(masks) != 111:
        raise ValueError("geometry release requires exactly 111 aligned views")

    hard_fractions, eroded_fractions, ordered_fractions = [], [], []
    for entry in cache_entries:
        stem = entry["stem"]
        if stem not in masks or stem not in cache_rows:
            raise ValueError(f"unaligned geometry release view: {stem}")
        image_path = Path(entry["source_image_path"])
        mask_path = Path(entry["mask_path"])
        if sha256_file(image_path) != entry["source_image_sha256"]:
            raise ValueError(f"source image hash mismatch: {stem}")
        if sha256_file(mask_path) != entry["mask_sha256"]:
            raise ValueError(f"mask hash mismatch: {stem}")
        camera_path = camera_root / f"{stem}.npz"
        with np.load(camera_path, allow_pickle=False) as camera_archive:
            if camera_identity_sha256(stem, camera_archive) != entry["camera_identity_sha256"]:
                raise ValueError(f"camera identity mismatch: {stem}")
            hard = np.asarray(camera_archive["mask_hard"], dtype=bool)
            eroded = np.asarray(camera_archive["mask_eroded"], dtype=bool)
        cache = load_two_hit_cache(
            release_root / entry["relative_path"],
            expected_mesh_sha256=manifest["glass_mesh_sha256"],
            expected_checkpoint_sha256=manifest["geometry_source_checkpoint_sha256"],
        )
        valid = np.asarray(cache["valid_two_hit"], dtype=bool)
        if valid.shape != hard.shape:
            raise ValueError(f"cache/mask shape mismatch: {stem}")
        hard_fraction = float(valid.sum() / max(hard.sum(), 1))
        eroded_fraction = float((valid & eroded).sum() / max(eroded.sum(), 1))
        ordered_fraction = float(np.mean(cache["t_far"][valid] > cache["t_near"][valid]))
        expected = cache_rows[stem]
        for actual, key in (
            (hard_fraction, "valid_fraction_hard"),
            (eroded_fraction, "valid_fraction_eroded"),
            (ordered_fraction, "far_gt_near_fraction"),
        ):
            if not np.isclose(actual, expected[key], rtol=0, atol=1e-12):
                raise ValueError(f"cache coverage metadata mismatch for {stem}: {key}")
        hard_fractions.append(hard_fraction)
        eroded_fractions.append(eroded_fraction)
        ordered_fractions.append(ordered_fraction)

    aggregate_expected = cache_metadata["aggregate"]
    measured = {
        "valid_fraction_hard_mean": float(np.mean(hard_fractions)),
        "valid_fraction_hard_min": float(np.min(hard_fractions)),
        "valid_fraction_eroded_mean": float(np.mean(eroded_fractions)),
        "far_gt_near_fraction": float(np.mean(ordered_fractions)),
    }
    for key, value in measured.items():
        if not np.isclose(value, aggregate_expected[key], rtol=0, atol=1e-12):
            raise ValueError(f"release aggregate coverage mismatch: {key}")
    return {
        "verdict": "STAGE_C_GEOMETRY_RELEASE_VALID",
        "geometry_release_id": manifest["geometry_release_id"],
        "aggregate_sha256": aggregate,
        "cache_count": len(cache_entries),
        "coverage": measured,
        "runtime_generation_required": False,
    }


class GeometryRelease:
    """Validated read-only access to one frozen geometry release."""

    def __init__(self, manifest_path: Path):
        self.validation = validate_geometry_release(manifest_path)
        self.manifest = load_release_manifest(manifest_path)
        self.root = Path(self.manifest["release_root"])
        self._entries = {row["stem"]: row for row in self.manifest["cache_entries"]}

    def load_view(self, stem: str) -> dict:
        if stem not in self._entries:
            raise KeyError(f"view is absent from frozen geometry release: {stem}")
        return load_two_hit_cache(
            self.root / self._entries[stem]["relative_path"],
            expected_mesh_sha256=self.manifest["glass_mesh_sha256"],
            expected_checkpoint_sha256=self.manifest["geometry_source_checkpoint_sha256"],
        )
