import inspect
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from geometry.geometry_release import load_release_manifest
from geometry.tao_dr_cuboid_bootstrap import (
    BOOTSTRAP_SCHEMA,
    CUBOID_EDGES,
    CUBOID_FACES,
    EXPECTED_COLMAP_SHA256,
    EXPECTED_DR_MANIFEST_SHA256,
    PREDECLARED_THRESHOLDS,
    cuboid_vertices,
    fit_relative_depth_calibration,
    geometry_parameter_sha256,
    optimize_review_cuboid,
    rasterize_projected_cuboid,
    search_normal_axis_convention,
    validate_colmap_identity,
    validate_dr_manifest_identity,
    validate_tao_stems,
)
from tools.build_tao_cuboid_geometry import _validate_formal_masks
from tools.build_tao_dr_glass_mask_repair_proposal import (
    EXPECTED_OLD_PROPOSAL_MANIFEST_SHA256,
    _validate_old_comparison,
    main as repair_main,
)
from utils.tao_dr_mask_repair import (
    PROPOSAL_SCHEMA,
    compute_native_cues,
    interior_likelihood,
    soft_mask_from_projection,
)


def _encode_normal(value):
    return np.clip(np.rint((np.asarray(value) + 1.0) * 127.5), 0, 255).astype(np.uint8)


def test_tao_input_identity_is_exactly_112_stems():
    stems = [f"{index:06d}" for index in range(112)]
    assert validate_tao_stems(stems) == stems
    with pytest.raises(ValueError, match="000000--000111"):
        validate_tao_stems(stems[:-1])
    with pytest.raises(ValueError, match="000000--000111"):
        validate_tao_stems(stems[:-1] + ["000112"])


def test_dr_manifest_mismatch_fails_closed():
    assert validate_dr_manifest_identity(EXPECTED_DR_MANIFEST_SHA256)
    with pytest.raises(RuntimeError, match="SHA-256"):
        validate_dr_manifest_identity("0" * 64)


def test_colmap_and_old_comparison_hash_mismatch_fail_closed(tmp_path):
    assert validate_colmap_identity(dict(EXPECTED_COLMAP_SHA256)) == EXPECTED_COLMAP_SHA256
    wrong = dict(EXPECTED_COLMAP_SHA256)
    wrong["images.bin"] = "0" * 64
    with pytest.raises(RuntimeError, match="COLMAP SHA-256"):
        validate_colmap_identity(wrong)

    old = tmp_path / "old"
    old.mkdir()
    (old / "proposal_manifest.json").write_text("{}\n", encoding="utf-8")
    assert EXPECTED_OLD_PROPOSAL_MANIFEST_SHA256 != "0" * 64
    with pytest.raises(RuntimeError, match="old comparison proposal manifest SHA-256"):
        _validate_old_comparison(old, [f"{index:06d}" for index in range(112)], (2320, 2032))


def test_relative_depth_direct_and_inverse_synthetic_recovery():
    x = np.linspace(0.05, 0.95, 1500)
    direct_z = 1.5 + 3.2 * x
    direct = fit_relative_depth_calibration(x, direct_z)
    assert direct.mode == "depth" and direct.trusted and direct.r2 > 0.999
    assert np.allclose(direct.apply(x), direct_z, atol=1e-5)
    inverse_z = 1.0 / (0.25 + 0.75 * x)
    inverse = fit_relative_depth_calibration(x, inverse_z)
    assert inverse.mode == "inverse_depth" and inverse.trusted and inverse.r2 > 0.999
    assert np.allclose(inverse.apply(x), inverse_z, atol=1e-5)


def test_relative_depth_remains_per_view_and_low_support_is_untrusted():
    calibration = fit_relative_depth_calibration(np.arange(20), np.arange(20) + 1)
    assert calibration.mode == "unavailable"
    assert not calibration.trusted and calibration.trust_weight == 0
    assert "per-view relative-only" in calibration.as_dict()["depth_contract"]
    assert "shared" not in inspect.signature(fit_relative_depth_calibration).parameters


def test_normal_signed_permutation_search_reports_every_candidate_and_joint_axes():
    rng = np.random.default_rng(12)
    rotations = []
    for angle in (0.0, 0.35, 0.8, 1.2):
        c, s = np.cos(angle), np.sin(angle)
        rotations.append(np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float64))
    true_mapping = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], np.float64)
    samples = []
    for rotation in rotations:
        world = np.concatenate([
            np.broadcast_to(axis, (120, 3)) + rng.normal(0, 0.01, (120, 3))
            for axis in np.eye(3)
        ])
        world /= np.linalg.norm(world, axis=1, keepdims=True)
        camera = world @ rotation.T
        raw = camera @ true_mapping
        raw /= np.linalg.norm(raw, axis=1, keepdims=True)
        raw = _encode_normal(raw).reshape(20, 18, 3)
        samples.append({
            "raw_normal": raw,
            "world_to_camera": rotation,
            "selection": np.ones((20, 18), bool),
        })
    mapping, axes, report = search_normal_axis_convention(samples, maximum_samples_per_view=96)
    assert mapping.shape == (3, 3)
    assert len(report["candidates"]) == 48
    assert report["selected_fit"]["alignment_p50"] > 0.98
    assert np.max(np.abs(axes.T @ axes - np.eye(3))) < 1e-6
    # Semantic signs are intentionally unidentifiable, but the selected axes
    # must be one common world triad across all camera rotations.
    assert np.min(np.max(np.abs(axes), axis=0)) > 0.98


def test_cuboid_projection_matches_synthetic_camera_and_is_one_fixed_geometry():
    axes = np.eye(3)
    lower = np.array([-1.0, -1.0, 4.0])
    upper = np.array([1.0, 1.0, 6.0])
    vertices = cuboid_vertices(axes, lower, upper)
    intrinsics = np.array([[100, 0, 64], [0, 100, 64], [0, 0, 1]], np.float64)
    mask, hull, uv = rasterize_projected_cuboid(
        vertices, intrinsics, np.eye(3), np.zeros(3), (128, 128)
    )
    assert mask.shape == (128, 128) and mask.sum() > 1000
    assert hull.shape[0] == 4 and np.isfinite(uv).all()
    assert mask[64, 64] == 1 and mask[0, 0] == 0
    geometry_hash = geometry_parameter_sha256(axes, lower, upper)
    projected_view_count = 0
    for center_x in np.linspace(-0.5, 0.5, 112):
        center = np.array([center_x, 0, 0])
        projected, _, _ = rasterize_projected_cuboid(
            vertices, intrinsics, np.eye(3), center, (128, 128)
        )
        assert projected.any()
        assert geometry_parameter_sha256(axes, lower, upper) == geometry_hash
        projected_view_count += 1
    assert projected_view_count == 112


def test_cuboid_topology_is_watertight_and_extents_positive():
    vertices = cuboid_vertices(np.eye(3), [-1, -2, -3], [1, 2, 3])
    assert vertices.shape == (8, 3)
    incidence = {tuple(sorted(edge)): 0 for edge in CUBOID_EDGES.tolist()}
    for face in CUBOID_FACES:
        for start, end in zip(face, np.roll(face, -1)):
            incidence[tuple(sorted((int(start), int(end))))] += 1
    assert set(incidence.values()) == {2}
    with pytest.raises(ValueError, match="positive"):
        cuboid_vertices(np.eye(3), [0, 0, 0], [1, 0, 1])


def test_soft_mask_is_new_projection_distance_field_not_old_pixels():
    hard = np.zeros((64, 64), np.uint8)
    cv2.rectangle(hard, (16, 12), (48, 52), 1, -1)
    soft = soft_mask_from_projection(hard, 5.0)
    assert soft.dtype == np.uint8 and soft[32, 32] == 255 and soft[0, 0] == 0
    assert np.any((soft > 0) & (soft < 255))
    old = np.zeros_like(hard); cv2.circle(old, (32, 32), 25, 1, -1)
    assert not np.array_equal(soft, old * 255)


def test_cues_include_all_declared_modalities_and_only_form_likelihoods():
    yy, xx = np.mgrid[:48, :64]
    rgb = np.stack((xx * 3, yy * 4, xx + yy), axis=-1).astype(np.uint8)
    normal = np.broadcast_to(_encode_normal([0, 0, 1]), rgb.shape).copy()
    depth = np.repeat(np.rint(xx[..., None] / 63 * 255).astype(np.uint8), 3, axis=2)
    cues = compute_native_cues(rgb, normal, depth, np.flip(rgb, axis=1), np.flip(rgb, axis=0))
    for key in (
        "normal_discontinuity", "relative_depth_discontinuity", "rgb_edge",
        "basecolor_edge", "diffuse_albedo_edge", "planar_normal_consistency",
        "reflection_uncertainty", "fused_boundary",
    ):
        assert cues[key].shape == (48, 64)
        assert np.isfinite(cues[key]).all()
    assert cues["fused_boundary"].dtype == np.float32


def test_interior_likelihood_separates_central_enclosure_from_near_floor():
    height, width = 80, 100
    depth = np.full((height, width), 5.0, np.float32)
    depth[20:65, 35:66] = 2.0
    depth[65:] = 2.0
    normal = np.zeros((height, width, 3), np.float32)
    normal[..., 2] = 1.0
    normal[65:] = np.array([0, 1, 0], np.float32)
    intrinsics = np.array([[80, 0, 50], [0, 80, 40], [0, 0, 1]], np.float64)
    likelihood = interior_likelihood(depth, np.ones_like(depth), intrinsics, normal)
    assert likelihood[40, 50] > likelihood[72, 50] + 0.20
    assert likelihood[40, 50] > likelihood[15, 10] + 0.20


def test_old_proposal_cannot_enter_cuboid_objective_or_fallback():
    parameters = inspect.signature(optimize_review_cuboid).parameters
    assert "old_proposal" not in parameters and "mask" not in parameters
    source = inspect.getsource(optimize_review_cuboid)
    assert '"old_proposal_used_in_objective": False' in source
    assert '"fallback_available": False' in source
    with pytest.raises(ValueError):
        optimize_review_cuboid(np.eye(3), [0, 0, 0], [0, 1, 1], [])


def test_review_only_schemas_are_incompatible_with_formal_loaders(tmp_path):
    proposal = tmp_path / "proposal.json"
    proposal.write_text(json.dumps({
        "schema": PROPOSAL_SCHEMA,
        "artifact_role": "tao_dr_geometry_glass_mask_proposal_for_human_review",
        "human_status": "proposal_requires_review",
        "training_eligible": False,
        "promotion_performed": False,
        "formal_geometry_release": False,
    }), encoding="utf-8")
    with pytest.raises(RuntimeError, match="BLOCKED_BY_GLASS_MASK_IDENTITY"):
        _validate_formal_masks(tmp_path / "Tao", proposal, ["000000"])
    bootstrap = tmp_path / "bootstrap.json"
    bootstrap.write_text(json.dumps({"schema": BOOTSTRAP_SCHEMA}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported Stage C geometry release schema"):
        load_release_manifest(bootstrap)


def test_existing_output_is_never_overwritten(tmp_path):
    output = tmp_path / "already_exists"
    output.mkdir()
    marker = output / "keep.txt"; marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing existing"):
        repair_main(["--output", str(output), "--execute"])
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_operator_has_no_training_checkpoint_ply_or_cross_scene_data_path():
    source_path = Path(inspect.getsourcefile(repair_main))
    source = source_path.read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "train.py" not in source
    assert "optimizer.step" not in source
    assert "write_mesh_ply" not in source
    assert "specular_masks_reviewed_v1" not in source
    assert "stage_c_tao_geometry_release_v1" not in source
    assert "output/stage_d" not in source
    assert "semantic_mask" not in source
    # The plan may mention forbidden concepts, but no CLI accepts such an input.
    parser_source = inspect.getsource(__import__(
        "tools.build_tao_dr_glass_mask_repair_proposal", fromlist=["build_parser"]
    ).build_parser)
    assert "checkpoint" not in parser_source.lower()
    assert "ply" not in parser_source.lower()


def test_frame_000058_contract_explicitly_forbids_inherited_fallback():
    source = Path(inspect.getsourcefile(repair_main)).read_text(encoding="utf-8")
    assert '"inherited_old_fallback": False' in source
    assert '"fallback_used": False' in source
    assert "review_only_relaxed_depth_fallback" not in source


def test_untrusted_geometry_stops_before_old_pixels_or_mask_writes():
    source = Path(inspect.getsourcefile(repair_main)).read_text(encoding="utf-8")
    gate = source.index("if not all(preliminary_gates.values())")
    old_read = source.index("old = _validate_old_comparison")
    mask_directory = source.index('"mask_soft", "mask_hard"')
    assert gate < old_read < mask_directory
    assert '"fallback_forbidden": True' in source


def test_predeclared_gates_are_fixed_before_execute_and_never_old_iou_based():
    assert PREDECLARED_THRESHOLDS["expected_view_count"] == 112
    assert PREDECLARED_THRESHOLDS["minimum_trusted_depth_views"] == 56
    assert PREDECLARED_THRESHOLDS["near_full_image_ratio"] < 1.0
    assert not any("iou" in key.lower() or key.lower().startswith("old_") for key in PREDECLARED_THRESHOLDS)
