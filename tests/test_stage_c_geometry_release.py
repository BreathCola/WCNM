import hashlib

import numpy as np
import pytest

from geometry.geometry_release import aggregate_assets, camera_identity_sha256


def test_asset_aggregate_is_order_independent_and_path_bound():
    first = [
        {"relative_path": "b", "sha256": "2" * 64},
        {"relative_path": "a", "sha256": "1" * 64},
    ]
    second = list(reversed(first))
    assert aggregate_assets(first) == aggregate_assets(second)
    changed = [dict(first[0]), dict(first[1])]
    changed[0]["relative_path"] = "c"
    assert aggregate_assets(changed) != aggregate_assets(first)


def test_camera_identity_binds_stem_shape_and_matrices(tmp_path):
    path = tmp_path / "camera.npz"
    payload = {
        "depth": np.ones((2, 3), np.float32),
        "world_view_transform": np.eye(4, dtype=np.float32),
        "full_proj_transform": np.eye(4, dtype=np.float32),
        "camera_center": np.zeros(3, np.float32),
    }
    np.savez(path, **payload)
    with np.load(path, allow_pickle=False) as archive:
        first = camera_identity_sha256("000000", archive)
        assert first != camera_identity_sha256("000001", archive)
    payload["camera_center"][0] = 1
    np.savez(path, **payload)
    with np.load(path, allow_pickle=False) as archive:
        assert first != camera_identity_sha256("000000", archive)
