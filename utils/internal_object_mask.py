"""Stage D internal-object mask proposal/review and object-occupancy losses.

Proposal masks are human-review artifacts only.  Training accepts only the
formal reviewed manifest validated here.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


PROPOSAL_ROLE = "stage_d_internal_object_mask_proposal"
REVIEWED_ROLE = "stage_d_formal_reviewed_internal_object_masks"
SCHEMA_VERSION = 1
MASK_INTERPOLATION = "opencv.INTER_LINEAR"
HARD_THRESHOLD = 0.5
EXPECTED_STEMS = [f"{index:06d}" for index in range(111)]
MASK_ROLES = ("bird", "base", "union")

BIRD_PROMPTS = (
    "a colorful bird figurine inside the transparent glass display case",
    "bird figurine",
    "decorative bird sculpture",
    "toy bird",
)
BASE_PROMPTS = (
    "the pedestal and supporting display base beneath the bird inside the transparent glass display case",
    "display pedestal",
    "supporting base",
    "bird stand",
)


class InternalObjectMaskError(RuntimeError):
    """Raised when internal-object mask provenance or payload is invalid."""


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_payload_sha256(payload: dict[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("manifest_payload_sha256", None)
    encoded = json.dumps(
        clean, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_image(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".png", dir=path.parent,
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        Image.fromarray(values).save(temporary, format="PNG")
        with Image.open(temporary) as reread:
            reread.load()
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def inspect_l_mask(path: Path | str, expected_size: tuple[int, int] | None = None) -> dict[str, Any]:
    path = Path(path)
    try:
        with Image.open(path) as image:
            mode = image.mode
            size = image.size
            values = np.asarray(image)
    except Exception as exc:
        raise InternalObjectMaskError(f"cannot parse internal-object mask: {path}") from exc
    if mode != "L" or values.dtype != np.uint8 or values.ndim != 2:
        raise InternalObjectMaskError(f"internal-object mask must be mode L uint8: {path}")
    if expected_size is not None and tuple(size) != tuple(expected_size):
        raise InternalObjectMaskError(
            f"internal-object mask size mismatch: {path}: {size} != {expected_size}"
        )
    binary = values >= 128
    area = int(binary.sum())
    pixel_count = int(binary.size)
    components = 0
    if area:
        components = int(cv2.connectedComponents(binary.astype(np.uint8), 8)[0] - 1)
    boundary = 0
    if area:
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(binary.astype(np.uint8), kernel, iterations=1).astype(bool)
        boundary = int((binary & ~eroded).sum())
    return {
        "size": [int(size[0]), int(size[1])],
        "mode": mode,
        "dtype": str(values.dtype),
        "min": int(values.min()),
        "max": int(values.max()),
        "area": area,
        "area_ratio": float(area / max(pixel_count, 1)),
        "bbox_xyxy": _bbox(values),
        "connected_components": components,
        "boundary_length_px": boundary,
    }


def _load_binary(path: Path | str, expected_size: tuple[int, int] | None = None) -> np.ndarray:
    facts = inspect_l_mask(path, expected_size)
    with Image.open(path) as image:
        values = np.asarray(image, dtype=np.uint8)
    if facts["min"] not in (0, 255) or facts["max"] not in (0, 255):
        raise InternalObjectMaskError(f"mask must be binary 0/255: {path}")
    return values >= 128


def clip_mask_to_glass(raw_mask: np.ndarray, glass_mask: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    raw = np.asarray(raw_mask, dtype=bool)
    glass = np.asarray(glass_mask, dtype=bool)
    if raw.shape != glass.shape:
        raise InternalObjectMaskError(f"raw/glass mask shape mismatch: {raw.shape} != {glass.shape}")
    clipped = raw & glass
    removed = raw & ~glass
    raw_area = int(raw.sum())
    clipped_area = int(clipped.sum())
    removed_area = int(removed.sum())
    return clipped, {
        "raw_area": raw_area,
        "clipped_area": clipped_area,
        "removed_outside_glass_area": removed_area,
        "removed_fraction": float(removed_area / max(raw_area, 1)),
        "empty_after_clip": clipped_area == 0,
        "full_after_clip": clipped_area == clipped.size,
    }


def proposal_view_metadata(
    stem: str,
    rgb_path: Path,
    raw_bird_path: Path,
    raw_base_path: Path,
    glass_mask_path: Path,
) -> dict[str, Any]:
    with Image.open(rgb_path) as rgb:
        if rgb.mode != "RGB":
            raise InternalObjectMaskError(f"RGB image must be mode RGB: {rgb_path}")
        size = rgb.size
    bird = _load_binary(raw_bird_path, size)
    base = _load_binary(raw_base_path, size)
    glass = _load_binary(glass_mask_path, size)
    raw_union = bird | base
    clipped_bird, bird_clip = clip_mask_to_glass(bird, glass)
    clipped_base, base_clip = clip_mask_to_glass(base, glass)
    clipped_union, union_clip = clip_mask_to_glass(raw_union, glass)
    return {
        "stem": stem,
        "rgb_path": str(rgb_path),
        "rgb_sha256": sha256_file(rgb_path),
        "raw": {
            "bird": inspect_l_mask(raw_bird_path, size),
            "base": inspect_l_mask(raw_base_path, size),
            "union": {
                "area": int(raw_union.sum()),
                "area_ratio": float(raw_union.mean()),
                "bbox_xyxy": _bbox(raw_union.astype(np.uint8)),
            },
        },
        "glass": {
            "path": str(glass_mask_path),
            "sha256": sha256_file(glass_mask_path),
            "area": int(glass.sum()),
            "area_ratio": float(glass.mean()),
        },
        "clipping": {
            "bird": bird_clip,
            "base": base_clip,
            "union": union_clip,
        },
        "processed": {
            "bird_area": int(clipped_bird.sum()),
            "base_area": int(clipped_base.sum()),
            "union_area": int(clipped_union.sum()),
        },
        "review_flags": _review_flags(union_clip, clipped_union, glass),
    }


def _review_flags(union_clip: dict[str, Any], clipped_union: np.ndarray, glass: np.ndarray) -> list[str]:
    flags: list[str] = []
    if union_clip["empty_after_clip"]:
        flags.append("empty_after_glass_clip")
    if union_clip["full_after_clip"]:
        flags.append("full_frame_after_glass_clip")
    if union_clip["removed_fraction"] > 0.25:
        flags.append("large_prediction_outside_glass")
    glass_area = int(glass.sum())
    if glass_area and float(clipped_union.sum() / glass_area) > 0.90:
        flags.append("object_fills_nearly_all_glass")
    return flags


def write_proposal_manifest(
    output_root: Path | str,
    entries: list[dict[str, Any]],
    grounded_sam2: dict[str, Any],
    method: str,
) -> dict[str, Any]:
    output_root = Path(output_root)
    if [entry.get("stem") for entry in entries] != EXPECTED_STEMS:
        raise InternalObjectMaskError("proposal manifest requires ordered stems 000000--000110")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "role": PROPOSAL_ROLE,
        "method": method,
        "count": len(entries),
        "ordered_stems": EXPECTED_STEMS,
        "prompts": {"bird": list(BIRD_PROMPTS), "base": list(BASE_PROMPTS)},
        "grounded_sam2": grounded_sam2,
        "human_status": "proposal_requires_review",
        "entries": entries,
    }
    payload["manifest_payload_sha256"] = canonical_payload_sha256(payload)
    _atomic_json(output_root / "manifest.json", payload)
    validation = validate_proposal_manifest(output_root / "manifest.json")
    _atomic_json(output_root / "validation.json", validation)
    review_path = output_root / "review_queue.csv"
    with review_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("stem", "flags", "removed_fraction", "union_area"))
        writer.writeheader()
        for entry in entries:
            writer.writerow({
                "stem": entry["stem"],
                "flags": ";".join(entry.get("review_flags", [])),
                "removed_fraction": entry.get("clipping", {}).get("union", {}).get("removed_fraction", 0.0),
                "union_area": entry.get("processed", {}).get("union_area", 0),
            })
    return payload


def validate_proposal_manifest(manifest_path: Path | str) -> dict[str, Any]:
    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("role") != PROPOSAL_ROLE:
        raise InternalObjectMaskError("not an internal-object proposal manifest")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise InternalObjectMaskError("unsupported internal-object proposal schema")
    if payload.get("ordered_stems") != EXPECTED_STEMS or payload.get("count") != 111:
        raise InternalObjectMaskError("proposal must contain exactly stems 000000--000110")
    if payload.get("manifest_payload_sha256") != canonical_payload_sha256(payload):
        raise InternalObjectMaskError("proposal payload hash mismatch")
    flags = {
        entry["stem"]: entry.get("review_flags", [])
        for entry in payload.get("entries", [])
        if entry.get("review_flags")
    }
    return {
        "validation": "REVIEW_REQUIRED" if flags else "PASS_REVIEW_STILL_REQUIRED",
        "role": PROPOSAL_ROLE,
        "count": 111,
        "manifest_file_sha256": sha256_file(path),
        "manifest_payload_sha256": payload["manifest_payload_sha256"],
        "flagged_view_count": len(flags),
        "flagged_views": flags,
    }


def validate_internal_object_mask_set(source_path, images_directory, manifest_path) -> dict[str, Any]:
    source = Path(source_path).resolve()
    image_root = (source / images_directory).resolve()
    manifest = Path(manifest_path)
    if not manifest.is_absolute():
        manifest = source / manifest
    manifest = manifest.resolve()
    if not manifest.is_file() or manifest.suffix.lower() != ".json":
        raise FileNotFoundError(f"formal internal-object mask manifest does not exist: {manifest}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("role") == PROPOSAL_ROLE:
        raise ValueError("proposal internal-object masks are not formal training supervision")
    if payload.get("role") != REVIEWED_ROLE or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("--internal_object_masks accepts only a formal reviewed manifest")
    if payload.get("human_status") != "accepted" or payload.get("count") != 111:
        raise ValueError("formal internal-object manifest must record accepted 111/111 masks")
    if payload.get("mask_interpolation") != MASK_INTERPOLATION:
        raise ValueError(f"formal internal-object manifest must declare {MASK_INTERPOLATION}")
    if payload.get("ordered_stems") != EXPECTED_STEMS:
        raise ValueError("formal internal-object manifest stems must be exactly 000000--000110")
    if payload.get("manifest_payload_sha256") != canonical_payload_sha256(payload):
        raise ValueError("formal internal-object manifest payload hash mismatch")
    images = {path.stem: path for path in sorted(image_root.iterdir()) if path.is_file()}
    if sorted(images) != EXPECTED_STEMS:
        raise ValueError("scene RGB stems must be exactly 000000--000110")
    entries = payload.get("entries")
    if not isinstance(entries, list) or [entry.get("stem") for entry in entries] != EXPECTED_STEMS:
        raise ValueError("formal internal-object entries are missing, duplicated, extra, or out of order")
    root = manifest.parent
    runtime_entries: dict[str, Any] = {}
    aggregate = hashlib.sha256()
    for entry in entries:
        stem = entry["stem"]
        rgb = images[stem]
        if entry.get("rgb_path") != f"{images_directory}/{rgb.name}":
            raise ValueError(f"internal-object {stem} RGB path mismatch")
        if entry.get("rgb_sha256") != sha256_file(rgb):
            raise ValueError(f"internal-object {stem} RGB hash mismatch")
        with Image.open(rgb) as image:
            size = image.size
        role_entries: dict[str, Any] = {}
        for role in MASK_ROLES:
            rel = entry.get(f"{role}_mask_path")
            if rel != f"{role}/{stem}.png":
                raise ValueError(f"internal-object {stem} {role} path mismatch")
            path = root / rel
            mask_sha = sha256_file(path)
            if entry.get(f"{role}_mask_sha256") != mask_sha:
                raise ValueError(f"internal-object {stem} {role} hash mismatch")
            facts = inspect_l_mask(path, size)
            role_entries[role] = {"path": str(path), "sha256": mask_sha, "size": facts["size"]}
            aggregate.update(f"{stem} {role} {mask_sha}\n".encode("utf-8"))
        runtime_entries[stem] = {"stem": stem, "rgb_sha256": entry["rgb_sha256"], **role_entries}
    if payload.get("aggregate_mask_sha256") != aggregate.hexdigest():
        raise ValueError("formal internal-object aggregate hash mismatch")
    return {
        "role": REVIEWED_ROLE,
        "count": 111,
        "manifest_path": str(manifest),
        "manifest_file_sha256": sha256_file(manifest),
        "manifest_payload_sha256": payload["manifest_payload_sha256"],
        "aggregate_sha256": aggregate.hexdigest(),
        "mask_interpolation": MASK_INTERPOLATION,
        "entries": runtime_entries,
        "provenance": payload.get("provenance", {}),
    }


def load_resized_internal_object_masks(
    validated_manifest: dict[str, Any],
    stem: str,
    original_size: tuple[int, int],
    output_size: tuple[int, int],
) -> dict[str, Any]:
    if validated_manifest is None or validated_manifest.get("role") != REVIEWED_ROLE:
        raise ValueError("internal-object loading requires a validated formal manifest")
    entry = validated_manifest["entries"].get(stem)
    if entry is None:
        raise ValueError(f"formal internal-object manifest has no mask for camera {stem}")
    loaded = {}
    native_binary = {}
    for role in MASK_ROLES:
        role_entry = entry[role]
        path = Path(role_entry["path"])
        if sha256_file(path) != role_entry["sha256"]:
            raise ValueError(f"internal-object mask changed after validation: {path}")
        inspect_l_mask(path, original_size)
        with Image.open(path) as image:
            raw_values = np.asarray(image, dtype=np.uint8)
        native_binary[role] = raw_values >= 128
        values = raw_values.astype(np.float32) / 255.0
        resized = cv2.resize(values, tuple(output_size), interpolation=cv2.INTER_LINEAR)
        resized = np.clip(resized, 0.0, 1.0).astype(np.float32, copy=False)
        if not np.isfinite(resized).all():
            raise ValueError(f"resized internal-object mask is non-finite: {path}")
        loaded[role] = torch.from_numpy(resized[None].copy())
        loaded[f"{role}_sha256"] = role_entry["sha256"]
    native_union = native_binary["bird"] | native_binary["base"]
    if not np.array_equal(native_binary["union"], native_union):
        raise ValueError(f"formal internal-object union is not bird|base for camera {stem}")
    return loaded


def _as_hwc1(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.ndim == 3 and value.shape[0] == 1:
        value = value.permute(1, 2, 0)
    elif value.ndim == 2:
        value = value[..., None]
    if value.ndim != 3 or value.shape[-1] != 1:
        raise ValueError(f"{name} must be HWC1, CHW1, or HW")
    return value.float()


def _morph(mask: torch.Tensor, pixels: int, op: str) -> torch.Tensor:
    if pixels <= 0:
        return mask
    chw = mask.permute(2, 0, 1).unsqueeze(0).float()
    kernel = int(2 * pixels + 1)
    if op == "dilate":
        out = F.max_pool2d(chw, kernel_size=kernel, stride=1, padding=pixels)
    elif op == "erode":
        out = 1.0 - F.max_pool2d(1.0 - chw, kernel_size=kernel, stride=1, padding=pixels)
    else:
        raise ValueError(f"unknown morphology op: {op}")
    return out.squeeze(0).permute(1, 2, 0)


def object_occupancy_domains(
    object_union: torch.Tensor,
    glass_hard: torch.Tensor,
    valid_two_hit: torch.Tensor,
    erode_px: int = 3,
    dilate_px: int = 3,
    threshold: float = HARD_THRESHOLD,
) -> dict[str, torch.Tensor]:
    obj = (_as_hwc1(object_union, "object_union") >= threshold).float()
    glass = (_as_hwc1(glass_hard, "glass_hard") >= threshold).float()
    valid = (_as_hwc1(valid_two_hit, "valid_two_hit") >= threshold).float()
    eroded = _morph(obj, int(erode_px), "erode")
    dilated = _morph(obj, int(dilate_px), "dilate")
    base = glass * valid
    mpos = (eroded * base).clamp(0, 1)
    mneg = ((1.0 - dilated) * base).clamp(0, 1)
    mignore = ((dilated - eroded).clamp(0, 1) * base).clamp(0, 1)
    return {"Mpos": mpos, "Mneg": mneg, "Mignore": mignore, "domain": base}


def object_occupancy_loss(
    inside_alpha: torch.Tensor,
    domains: dict[str, torch.Tensor],
    *,
    alpha_floor: float = 0.35,
    lambda_positive: float = 0.0,
    lambda_negative: float = 0.0,
) -> dict[str, torch.Tensor]:
    ain = _as_hwc1(inside_alpha, "inside_alpha")
    mpos = _as_hwc1(domains["Mpos"], "Mpos").to(device=ain.device, dtype=ain.dtype)
    mneg = _as_hwc1(domains["Mneg"], "Mneg").to(device=ain.device, dtype=ain.dtype)
    pos_denom = mpos.sum().clamp_min(1.0)
    neg_denom = mneg.sum().clamp_min(1.0)
    positive = (mpos * torch.relu(ain.new_tensor(float(alpha_floor)) - ain)).sum() / pos_denom
    negative = (mneg * ain).sum() / neg_denom
    total = float(lambda_positive) * positive + float(lambda_negative) * negative
    if not torch.isfinite(total):
        raise FloatingPointError("internal-object occupancy loss is NaN/Inf")
    return {"total": total, "positive": positive, "negative": negative}


def object_domain_metrics(package: dict[str, torch.Tensor], domains: dict[str, torch.Tensor]) -> dict[str, float]:
    ain = _as_hwc1(package["inside_alpha"], "inside_alpha").detach().float()
    cin = package["inside_color"].detach().float()
    cout = package["outside_color"].detach().float()
    mpos = _as_hwc1(domains["Mpos"], "Mpos").to(ain.device) > 0.5
    mneg = _as_hwc1(domains["Mneg"], "Mneg").to(ain.device) > 0.5

    def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> float:
        if value.shape[-1] != mask.shape[-1]:
            mask = mask.expand_as(value)
        selected = value[mask]
        return float(selected.mean()) if selected.numel() else 0.0

    return {
        "ain_mean_mpos": masked_mean(ain, mpos),
        "ain_mean_mneg": masked_mean(ain, mneg),
        "ain_positive_coverage_ge_0_35": masked_mean((ain >= 0.35).float(), mpos),
        "ain_negative_spill_ge_0_10": masked_mean((ain >= 0.10).float(), mneg),
        "cin_energy_mpos": masked_mean(cin.abs(), mpos),
        "cin_energy_mneg": masked_mean(cin.abs(), mneg),
        "cout_energy_mneg": masked_mean(cout.abs(), mneg),
    }
