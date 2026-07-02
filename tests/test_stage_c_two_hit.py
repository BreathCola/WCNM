import numpy as np
import pytest

from geometry.mesh_intersector import MeshIntersector
from geometry.two_hit import CACHE_SCHEMA, load_two_hit_cache


def test_mesh_bvh_returns_front_and_back_cube_hits():
    vertices=np.array([
        [-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
        [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1],
    ],np.float32)
    faces=np.array([
        [0,2,1],[0,3,2],[4,5,6],[4,6,7],
        [0,1,5],[0,5,4],[1,2,6],[1,6,5],
        [2,3,7],[2,7,6],[3,0,4],[3,4,7],
    ],np.int64)
    bvh=MeshIntersector(vertices,faces)
    origins=np.array([[0,0,-3],[3,0,0]],np.float32)
    directions=np.array([[0,0,1],[-1,0,0]],np.float32)
    near,far,count=bvh.intersect(origins,directions)
    assert np.allclose(near,[2,2],atol=1e-5)
    assert np.allclose(far,[4,4],atol=1e-5)
    assert np.all(count==2)


def test_two_hit_cache_load_validates_schema_hashes_and_depth_order(tmp_path):
    path = tmp_path / "000039.npz"
    payload = {
        "schema": np.array(CACHE_SCHEMA), "stem": np.array("000039"),
        "mesh_sha256": np.array("mesh"), "checkpoint_sha256": np.array("checkpoint"),
        "t_near": np.array([[1.0, 0.0]], np.float32),
        "t_far": np.array([[2.0, 0.0]], np.float32),
        "hit_count": np.array([[2, 0]], np.int16),
        "valid_two_hit": np.array([[True, False]]),
        "back_position": np.zeros((1, 2, 3), np.float32),
    }
    np.savez_compressed(path, **payload)
    loaded = load_two_hit_cache(
        path, expected_mesh_sha256="mesh", expected_checkpoint_sha256="checkpoint"
    )
    assert loaded["stem"].item() == "000039"
    with pytest.raises(ValueError, match="mesh hash mismatch"):
        load_two_hit_cache(path, expected_mesh_sha256="other")
    payload["t_far"] = np.array([[0.5, 0.0]], np.float32)
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="t_far > t_near"):
        load_two_hit_cache(path)
