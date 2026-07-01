"""Fail-closed formal Stage B soft-mask manifests and camera loading."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


FORMAL_ROLE = "stage_b_formal_reviewed_specular_soft_masks"
FORMAL_SCHEMA_VERSION = 1
MASK_INTERPOLATION = "opencv.INTER_LINEAR"
ALLOWED_SOURCES = {"proposal_v1", "repair_candidate_v1"}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_payload_sha256(payload):
    clean = dict(payload)
    clean.pop("manifest_payload_sha256", None)
    encoded = json.dumps(clean, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_manifest(source_path, manifest_path):
    path = Path(manifest_path)
    if not path.is_absolute():
        path = Path(source_path) / path
    return path.resolve()


def _bbox(values):
    ys, xs = np.where(values > 0)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def inspect_mask(path, expected_size=None):
    path = Path(path)
    try:
        with Image.open(path) as image:
            mode = image.mode
            size = image.size
            values = np.asarray(image)
    except Exception as exc:
        raise ValueError(f"cannot parse specular soft mask: {path}") from exc
    if mode != "L" or values.dtype != np.uint8 or values.ndim != 2:
        raise ValueError(f"formal specular soft mask must be mode L uint8: {path}")
    if expected_size is not None and tuple(size) != tuple(expected_size):
        raise ValueError(f"specular soft mask size mismatch: {path}: {size} != {expected_size}")
    minimum, maximum = int(values.min()), int(values.max())
    if minimum == maximum:
        raise ValueError(f"formal specular soft mask cannot be all-zero or all-one/constant: {path}")
    if minimum != 0 or maximum != 255:
        raise ValueError(f"formal specular soft mask must retain both 0 and 255: {path}")
    return {
        "size": [int(size[0]), int(size[1])],
        "mode": mode,
        "dtype": str(values.dtype),
        "min": minimum,
        "max": maximum,
        "area_ratio": float(values.astype(np.float64).sum() / (255.0 * values.size)),
        "bbox_xyxy": _bbox(values),
    }


def validate_specular_mask_set(source_path, images_directory, manifest_path):
    """Validate a complete immutable formal manifest, every RGB, and every mask.

    The third argument is deliberately a JSON manifest path, never a mask
    directory. Proposal and repair trees therefore cannot be loaded by training.
    """
    source = Path(source_path).resolve()
    image_root = (source / images_directory).resolve()
    manifest_file = _resolve_manifest(source, manifest_path)
    if not manifest_file.is_file() or manifest_file.suffix.lower() != ".json":
        raise FileNotFoundError(f"formal specular mask manifest does not exist: {manifest_file}")
    try:
        payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"cannot parse formal specular mask manifest: {manifest_file}") from exc

    if payload.get("schema_version") != FORMAL_SCHEMA_VERSION or payload.get("role") != FORMAL_ROLE:
        raise ValueError("--specular_masks accepts only a formal reviewed-mask manifest")
    if payload.get("human_status") != "accepted" or payload.get("count") != 111:
        raise ValueError("formal specular mask manifest must record accepted 111/111 masks")
    if payload.get("mask_interpolation") != MASK_INTERPOLATION:
        raise ValueError(f"formal manifest must declare {MASK_INTERPOLATION} resizing")
    expected_stems = [f"{index:06d}" for index in range(111)]
    if payload.get("ordered_stems") != expected_stems:
        raise ValueError("formal manifest stems must be exactly 000000--000110")
    padding = payload.get("padding_exclusion_proof", {})
    if padding.get("excluded_stems") != [f"{index:06d}" for index in range(111, 120)] or padding.get("mixed_count") != 0:
        raise ValueError("formal manifest lacks the 111--119 padding exclusion proof")
    if payload.get("manifest_payload_sha256") != canonical_payload_sha256(payload):
        raise ValueError("formal specular mask manifest payload hash mismatch")

    image_files = sorted(path for path in image_root.iterdir() if path.is_file())
    images = {path.stem: path for path in image_files}
    if sorted(images) != expected_stems or len(images) != 111:
        raise ValueError("scene RGB stems must be exactly 000000--000110 for this formal manifest")
    entries = payload.get("entries")
    if not isinstance(entries, list) or len(entries) != 111:
        raise ValueError("formal manifest must contain exactly 111 entries")
    if [entry.get("stem") for entry in entries] != expected_stems:
        raise ValueError("formal manifest entries are missing, duplicated, extra, or out of order")

    manifest_root = manifest_file.parent
    expected_files = {f"{stem}.png" for stem in expected_stems} | {manifest_file.name}
    actual_files = {path.name for path in manifest_root.iterdir() if path.is_file()}
    if actual_files != expected_files:
        raise ValueError(f"formal mask directory contains missing or extra files: {sorted(actual_files ^ expected_files)}")

    aggregate = hashlib.sha256()
    runtime_entries = {}
    for entry, stem in zip(entries, expected_stems):
        if entry.get("source") not in ALLOWED_SOURCES or entry.get("human_status") != "accepted":
            raise ValueError(f"formal mask {stem} has unknown source or is not accepted")
        rgb_path = images[stem]
        if entry.get("rgb_path") != f"{images_directory}/{rgb_path.name}":
            raise ValueError(f"formal mask {stem} RGB path mismatch")
        rgb_sha = sha256_file(rgb_path)
        if entry.get("rgb_sha256") != rgb_sha:
            raise ValueError(f"formal mask {stem} RGB hash mismatch")
        mask_name = entry.get("mask_path")
        if mask_name != f"{stem}.png" or Path(mask_name).name != mask_name:
            raise ValueError(f"formal mask {stem} path must be a local stem-matched PNG")
        mask_path = manifest_root / mask_name
        mask_sha = sha256_file(mask_path)
        if entry.get("mask_sha256") != mask_sha or entry.get("source_sha256") != mask_sha:
            raise ValueError(f"formal mask {stem} mask/source hash mismatch")
        with Image.open(rgb_path) as rgb:
            rgb_size = rgb.size
        facts = inspect_mask(mask_path, rgb_size)
        for key in ("size", "mode", "dtype", "area_ratio", "bbox_xyxy"):
            expected = entry.get(key)
            actual = facts[key]
            if key == "area_ratio":
                if not isinstance(expected, (int, float)) or abs(float(expected) - actual) > 1e-12:
                    raise ValueError(f"formal mask {stem} area ratio mismatch")
            elif expected != actual:
                raise ValueError(f"formal mask {stem} {key} mismatch")
        aggregate.update(f"{stem} {mask_sha}\n".encode("utf-8"))
        runtime_entries[stem] = {
            "stem": stem,
            "path": str(mask_path),
            "sha256": mask_sha,
            "size": facts["size"],
        }
    if aggregate.hexdigest() != payload.get("aggregate_mask_sha256"):
        raise ValueError("formal mask aggregate hash mismatch")
    return {
        "role": FORMAL_ROLE,
        "count": 111,
        "aggregate_sha256": aggregate.hexdigest(),
        "manifest_payload_sha256": payload["manifest_payload_sha256"],
        "manifest_file_sha256": sha256_file(manifest_file),
        "manifest_path": str(manifest_file),
        "mask_interpolation": MASK_INTERPOLATION,
        "entries": runtime_entries,
    }


def load_resized_formal_mask(validated_manifest, stem, original_size, output_size):
    if validated_manifest is None or validated_manifest.get("role") != FORMAL_ROLE:
        raise ValueError("camera mask loading requires a validated formal manifest")
    entry = validated_manifest["entries"].get(stem)
    if entry is None:
        raise ValueError(f"formal manifest has no mask for camera {stem}")
    path = Path(entry["path"])
    if sha256_file(path) != entry["sha256"]:
        raise ValueError(f"formal mask changed after manifest validation: {path}")
    facts = inspect_mask(path, original_size)
    with Image.open(path) as image:
        values = np.asarray(image, dtype=np.float32) / 255.0
    resized = cv2.resize(values, tuple(output_size), interpolation=cv2.INTER_LINEAR)
    resized = np.clip(resized, 0.0, 1.0).astype(np.float32, copy=False)
    if not np.isfinite(resized).all():
        raise ValueError(f"resized formal mask is non-finite: {path}")
    return resized, entry["sha256"], facts
