import numpy as np

from geometry.stage_c_audit import (
    audit_view,
    classify_mesh_audit,
    make_mask_variants,
    voxel_consistency,
)


def test_mask_variants_preserve_soft_and_erode_hard_support():
    soft = np.zeros((9, 9), np.float32)
    soft[1:8, 1:8] = 0.75
    masks = make_mask_variants(soft, erode_pixels=1)
    assert np.array_equal(masks.soft, soft)
    assert masks.hard.sum() == 49
    assert masks.eroded.sum() == 25


def test_audit_view_reports_finite_connected_geometry():
    soft = np.ones((8, 8), np.float32)
    masks = make_mask_variants(soft, erode_pixels=1)
    depth = np.full((8, 8), 2.0, np.float32)
    alpha = np.ones((8, 8), np.float32)
    normal = np.zeros((8, 8, 3), np.float32)
    normal[..., 2] = 1
    result = audit_view(depth, alpha, normal, masks)
    assert result["valid_eroded_fraction"] == 1.0
    assert result["largest_valid_component_fraction"] == 1.0


def test_voxel_consistency_and_pass_gate_for_dense_multiview_shell():
    grid = np.stack(np.meshgrid(
        np.linspace(-1, 1, 24), np.linspace(-1, 1, 24), np.linspace(-1, 1, 24),
        indexing="ij",
    ), axis=-1).reshape(-1, 3)
    voxel = voxel_consistency([grid.copy() for _ in range(111)], grid_resolution=24)
    row = {
        "valid_eroded_fraction": 1.0, "boundary_valid_fraction": 1.0,
        "largest_valid_component_fraction": 1.0,
    }
    verdict, failures = classify_mesh_audit([dict(row) for _ in range(111)], voxel)
    assert verdict == "STAGE_C_MESH_AUDIT_PASS"
    assert failures == []


def test_pass_gate_blocks_incomplete_view_set():
    voxel = {"support_ge2_fraction": 1.0, "largest_supported_component_fraction": 1.0}
    row = {
        "valid_eroded_fraction": 1.0, "boundary_valid_fraction": 1.0,
        "largest_valid_component_fraction": 1.0,
    }
    verdict, failures = classify_mesh_audit([row], voxel)
    assert verdict == "STAGE_C_MESH_AUDIT_BLOCKED"
    assert "all 111 views audited" in failures
