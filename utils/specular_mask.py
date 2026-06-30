"""Strict Stage B manual soft-mask validation and hashing."""

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


def validate_specular_mask_set(source_path, images_directory, masks_directory):
    source = Path(source_path)
    image_root = source / images_directory
    mask_root = source / masks_directory
    if not mask_root.is_dir():
        raise FileNotFoundError(f"specular mask directory does not exist: {mask_root}")
    images = sorted(path for path in image_root.iterdir() if path.is_file())
    expected = {path.stem: path for path in images}
    masks = sorted(mask_root.glob("*.png"))
    actual = {path.stem: path for path in masks}
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    if missing or unexpected or len(actual) != len(expected):
        raise ValueError(
            f"specular mask set must match images exactly: images={len(expected)} masks={len(actual)} "
            f"missing={missing} unexpected={unexpected}"
        )
    entries = []
    aggregate = hashlib.sha256()
    for stem in sorted(expected):
        image = Image.open(expected[stem])
        mask_path = actual[stem]
        mask = Image.open(mask_path)
        if mask.mode not in ("1", "L", "I", "F"):
            raise ValueError(f"specular soft mask must be single-channel: {mask_path}")
        if mask.size != image.size:
            raise ValueError(f"specular soft mask size mismatch for {stem}: {mask.size} != {image.size}")
        values = np.asarray(mask.convert("F"), dtype=np.float32)
        if values.max(initial=0.0) > 1.0:
            values = values / 255.0
        if not np.isfinite(values).all() or values.min(initial=0.0) < 0.0 or values.max(initial=0.0) > 1.0:
            raise ValueError(f"specular soft mask must be finite in [0,1]: {mask_path}")
        digest = hashlib.sha256(mask_path.read_bytes()).hexdigest()
        aggregate.update(f"{stem} {digest}\n".encode("utf-8"))
        entries.append({"image_stem": stem, "mask": str(mask_path.resolve()), "sha256": digest})
    return {
        "role": "stage_b_specular_soft_mask",
        "count": len(entries),
        "aggregate_sha256": aggregate.hexdigest(),
        "entries": entries,
    }
