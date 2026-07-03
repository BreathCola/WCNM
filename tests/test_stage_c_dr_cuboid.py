import numpy as np

from geometry.dr_cuboid import (
    cuboid_vertices_faces, decode_c03_normal, fit_depth_calibration,
    fit_orthogonal_axes, intersect_cuboid_near, mask_metrics,
)
from geometry.tsdf_fusion import mesh_topology


def test_c03_normal_decode_flips_x_and_normalizes():
    raw = np.array([[[255, 127, 127]]], np.uint8)
    normal = decode_c03_normal(raw)
    assert np.allclose(normal[0, 0], [-1, 0, 0], atol=0.01)


def test_depth_calibration_selects_metric_depth_mapping():
    x = np.linspace(0.05, 0.95, 1000).reshape(20, 50)
    z = 2.0 + 3.0 * x
    support = np.ones_like(x, dtype=bool)
    calibration = fit_depth_calibration(x, z, support)
    assert calibration.mode == "depth"
    assert calibration.r2 > 0.999
    assert np.allclose(calibration.apply(x), z, atol=1e-5)


def test_axis_fit_and_cuboid_topology():
    rng = np.random.default_rng(3)
    rotation = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], np.float64)
    normals = np.concatenate([
        np.broadcast_to(rotation[:, axis], (200, 3)) + rng.normal(0, 0.01, (200, 3))
        for axis in range(3)
    ])
    axes, report = fit_orthogonal_axes(normals, maximum_samples=1000)
    alignment = np.max(np.abs(rotation.T @ axes), axis=1)
    assert np.all(alignment > 0.99)
    assert report["alignment_p50"] > 0.99
    vertices, faces = cuboid_vertices_faces(axes, [-1, -2, -3], [1, 2, 3])
    assert mesh_topology(vertices, faces)["watertight"]


def test_mask_metrics_are_exact():
    target = np.zeros((4, 4), bool)
    target[1:3, 1:3] = True
    assert mask_metrics(target, target) == {"recall": 1.0, "precision": 1.0, "iou": 1.0}


def test_cuboid_near_intersection_uses_ray_parameter():
    origins = np.array([[0, 0, -3], [3, 0, 0], [3, 0, -3]], np.float64)
    directions = np.array([[0, 0, 1], [-1, 0, 0], [0, 0, 1]], np.float64)
    near, valid = intersect_cuboid_near(
        np.eye(3), np.full(3, -1.0), np.full(3, 1.0), origins, directions
    )
    assert np.array_equal(valid, [True, True, False])
    assert np.allclose(near[:2], 2.0)
