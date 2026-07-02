"""Camera-ray generation and fixed-mesh two-hit cache helpers."""

from __future__ import annotations

from typing import Optional

import numpy as np

from geometry.tsdf_fusion import camera_intrinsics_from_transforms


CACHE_SCHEMA = "rtgs_stage_c_mesh_hit_cache_v1"


def masked_camera_rays(view: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.asarray(view["mask_hard"], dtype=bool)
    height, width = mask.shape
    intrinsics, rotation, center = camera_intrinsics_from_transforms(
        view["world_view_transform"], view["full_proj_transform"], width, height
    )
    ys, xs = np.nonzero(mask)
    pixels = np.stack((xs, ys, np.ones_like(xs)), axis=1).astype(np.float64)
    camera_rays = pixels @ np.linalg.inv(intrinsics).T
    world_rays = camera_rays @ rotation.T
    world_rays /= np.maximum(np.linalg.norm(world_rays, axis=1, keepdims=True), 1e-12)
    origins = np.broadcast_to(center.astype(np.float32), world_rays.shape).copy()
    linear = ys.astype(np.int64) * width + xs.astype(np.int64)
    return origins, world_rays.astype(np.float32), linear


def scatter_two_hits(
    shape: tuple[int, int], linear: np.ndarray, near: np.ndarray, far: np.ndarray,
    hit_count: np.ndarray, directions: np.ndarray, origins: np.ndarray,
) -> dict:
    size = int(np.prod(shape))
    near_image = np.zeros(size, np.float32)
    far_image = np.zeros(size, np.float32)
    count_image = np.zeros(size, np.int16)
    back = np.zeros((size, 3), np.float32)
    valid = (hit_count >= 2) & np.isfinite(near) & np.isfinite(far) & (far > near + 1e-5)
    target = linear[valid]
    near_image[target] = near[valid]
    far_image[target] = far[valid]
    count_image[target] = np.minimum(hit_count[valid], np.iinfo(np.int16).max).astype(np.int16)
    back[target] = origins[valid] + far[valid, None] * directions[valid]
    return {
        "t_near": near_image.reshape(shape), "t_far": far_image.reshape(shape),
        "hit_count": count_image.reshape(shape), "valid_two_hit": (count_image > 0).reshape(shape),
        "back_position": back.reshape(shape + (3,)),
    }


def load_two_hit_cache(
    path,
    *,
    expected_mesh_sha256: Optional[str] = None,
    expected_checkpoint_sha256: Optional[str] = None,
) -> dict:
    """Load and strictly validate one immutable Stage C cache entry."""
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "schema", "stem", "mesh_sha256", "checkpoint_sha256", "t_near",
            "t_far", "hit_count", "valid_two_hit", "back_position",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"two-hit cache is missing fields: {sorted(missing)}")
        result = {key: archive[key] for key in required}
    if str(result["schema"].item()) != CACHE_SCHEMA:
        raise ValueError("unsupported two-hit cache schema")
    mesh_sha256 = str(result["mesh_sha256"].item())
    checkpoint_sha256 = str(result["checkpoint_sha256"].item())
    if expected_mesh_sha256 is not None and mesh_sha256 != expected_mesh_sha256:
        raise ValueError("two-hit cache mesh hash mismatch")
    if expected_checkpoint_sha256 is not None and checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError("two-hit cache checkpoint hash mismatch")
    near = np.asarray(result["t_near"])
    far = np.asarray(result["t_far"])
    hit_count = np.asarray(result["hit_count"])
    valid = np.asarray(result["valid_two_hit"], dtype=bool)
    back = np.asarray(result["back_position"])
    if (
        far.shape != near.shape or hit_count.shape != near.shape
        or valid.shape != near.shape or back.shape != near.shape + (3,)
    ):
        raise ValueError("two-hit cache arrays have incompatible shapes")
    if not np.isfinite(near).all() or not np.isfinite(far).all() or not np.isfinite(back).all():
        raise ValueError("two-hit cache contains non-finite values")
    if np.any(far[valid] <= near[valid]):
        raise ValueError("two-hit cache violates t_far > t_near")
    if not np.array_equal(valid, hit_count >= 2):
        raise ValueError("two-hit cache validity disagrees with hit count")
    return result
