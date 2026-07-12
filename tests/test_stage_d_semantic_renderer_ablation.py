import json
from pathlib import Path

import pytest
import torch

from raytracer.differentiable_raytrace import trace_candidates
from utils.semantic_renderer_ablation import (
    SCHEMA, apply_legacy_fallback, assert_outside_mask_bitwise_parity,
    black_pixel_attribution, make_back_face_candidate_filter,
    mask_boundary_diagnostics, overbright_diagnostics,
)
from geometry.cuboid_space import CuboidSpace, SUPPORT_INTERFACE, SUPPORT_STRICT_INSIDE
from tools.audit_stage_d_semantic_renderer_ablation import main as audit_main


def _space():
    return CuboidSpace(
        axes=torch.eye(3, dtype=torch.float64),
        lower=torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float64),
        upper=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64),
        interface_margin=0.1,
        epsilon=1e-6,
    )


def _package():
    final = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.4, 0.4, 0.4]],
         [[0.2, 0.2, 0.2], [0.3, 0.3, 0.3]]],
        dtype=torch.float32,
    )
    zeros3 = torch.zeros_like(final)
    zeros1 = torch.zeros((2, 2, 1), dtype=torch.float32)
    return {
        "final": final,
        "final_linear": final.clone(),
        "alpha": torch.ones((2, 2, 1), dtype=torch.float32),
        "diffuse_contribution": zeros3.clone(),
        "reflection_contribution": zeros3.clone(),
        "reflection_strict_outside_safe": torch.ones_like(final) * 0.2,
        "inside_contribution": zeros3.clone(),
        "cout_contribution": zeros3.clone(),
        "inside_color": zeros3.clone(),
        "inside_alpha": torch.tensor([[[0.9], [0.0]], [[0.0], [0.0]]]),
        "inside_hit_mask": zeros1.clone(),
        "conditional_inside_color": zeros3.clone(),
        "outside_color": zeros3.clone(),
        "outside_unfiltered": torch.ones_like(final) * 0.4,
        "two_hit_valid": torch.tensor([[[0.0], [1.0]], [[1.0], [0.0]]]),
    }


def test_candidate_hit_filter_defaults_to_original_trace_cpu():
    if not torch.cuda.is_available():
        pytest.skip("production candidate gather is CUDA-only")
    class Model:
        _xyz = torch.tensor([[0.0, 0.0, 1.0]], device="cuda")
        _rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda")
        _scaling = torch.log(torch.tensor([[0.5, 0.5]], device="cuda"))
        _opacity = torch.tensor([[2.0]], device="cuda")
        _color = torch.tensor([[0.2, 0.3, 0.4]], device="cuda")

        @property
        def get_xyz(self): return self._xyz
        @property
        def raytrace_scaling_raw(self): return self._scaling

    model = Model()
    origins = torch.tensor([[0.0, 0.0, 0.0]], device="cuda")
    directions = torch.tensor([[0.0, 0.0, 1.0]], device="cuda")
    candidates = torch.tensor([0], device="cuda")
    offsets = torch.tensor([0, 1], device="cuda")
    plain = trace_candidates(model, origins, directions, candidates, offsets, 3.0, 1e-4)
    kept = trace_candidates(
        model, origins, directions, candidates, offsets, 3.0, 1e-4,
        candidate_hit_filter=lambda ids, exact, points, distance, o, d: torch.ones_like(exact),
    )
    blocked = trace_candidates(
        model, origins, directions, candidates, offsets, 3.0, 1e-4,
        candidate_hit_filter=lambda ids, exact, points, distance, o, d: torch.zeros_like(exact),
    )
    for a, b in zip(plain, kept):
        assert torch.equal(a, b)
    assert float(blocked[1].sum()) == 0.0
    assert bool(blocked[3].any()) is False


def test_arm2_legacy_fallback_changes_only_hard_invalid_domain():
    arm0 = _package()
    legacy = {k: v.clone() if torch.is_tensor(v) else v for k, v in arm0.items()}
    legacy["final"] = torch.ones_like(arm0["final"])
    legacy["diffuse_contribution"] = torch.ones_like(arm0["final"]) * 0.5
    legacy["reflection_contribution"] = torch.ones_like(arm0["final"]) * 0.25
    mask = torch.tensor([[[1.0, 1.0], [0.0, 1.0]]])
    out = apply_legacy_fallback(arm0, legacy, mask)
    changed = out["final"] != arm0["final"]
    expected = torch.tensor(
        [[[True, True, True], [False, False, False]],
         [[False, False, False], [True, True, True]]]
    )
    assert torch.equal(changed, expected)
    assert out["fallback_validation"]["pixel_count"] == 2


def test_arm3_back_face_filter_rejects_strict_inside_candidate():
    cuboid = _space()
    candidate_classes = torch.tensor([SUPPORT_STRICT_INSIDE])
    back = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    filt = make_back_face_candidate_filter(cuboid, candidate_classes, back, 1e-5)
    ids = torch.tensor([[0]])
    exact = torch.tensor([[True]])
    points = torch.tensor([[[0.0, 0.0, 1.2]]], dtype=torch.float64)
    accepted = filt(ids, exact, points, torch.ones((1, 1)), points, points)
    assert accepted.tolist() == [[False]]


def test_arm3_back_face_filter_advances_across_chunks():
    cuboid = _space()
    candidate_classes = torch.tensor([SUPPORT_INTERFACE])
    back = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    filt = make_back_face_candidate_filter(cuboid, candidate_classes, back, 1e-5)
    ids = torch.tensor([[0]])
    exact = torch.tensor([[True]])
    outside_point = torch.tensor([[[0.0, 0.0, 1.2]]], dtype=torch.float64)
    inside_point = torch.tensor([[[0.0, 0.0, 0.9]]], dtype=torch.float64)
    accepted_first = filt(ids, exact, outside_point, torch.ones((1, 1)), outside_point, outside_point)
    accepted_second = filt(ids, exact, inside_point, torch.ones((1, 1)), inside_point, inside_point)
    assert accepted_first.tolist() == [[True]]
    assert accepted_second.tolist() == [[False]]


def test_black_boundary_and_overbright_metrics_are_float_tensor_based():
    package = _package()
    mask = torch.tensor([[[1.0, 1.0], [0.0, 1.0]]])
    black, maps = black_pixel_attribution(package, mask)
    boundary, boundary_maps = mask_boundary_diagnostics(mask, package["two_hit_valid"], package["final"])
    package["final_linear"][0, 1] = torch.tensor([1.2, 0.2, 0.2])
    package["reflection_contribution"][0, 1] = torch.tensor([0.9, 0.0, 0.0])
    over = overbright_diagnostics(package)
    assert black["near_black_count"] == 1
    assert "two_hit_invalid" in maps
    assert "1px" in boundary
    assert "mask_boundary" in boundary_maps
    assert over["pre_clamp_over_1_pixel_fraction"] > 0
    assert over["pre_clamp_max_rgb"] == 1.2000000476837158


def test_outside_mask_parity_rejects_changed_outside_pixels():
    arm0 = _package()
    candidate = {k: v.clone() if torch.is_tensor(v) else v for k, v in arm0.items()}
    mask = torch.tensor([[[1.0, 1.0], [0.0, 1.0]]])
    assert assert_outside_mask_bitwise_parity(arm0, candidate, mask)
    candidate["final"][1, 0, 0] += 0.1
    try:
        assert_outside_mask_bitwise_parity(arm0, candidate, mask)
    except RuntimeError as exc:
        assert "outside-mask bitwise parity" in str(exc)
    else:
        raise AssertionError("outside parity failure was not detected")


def test_audit_accepts_synthetic_zero_update_tree(tmp_path, monkeypatch):
    output = tmp_path / "out"
    stems = ("000000", "000012", "000039", "000040", "000041", "000053", "000063", "000083", "000110")
    arms = ("arm_0", "arm_1", "arm_2", "arm_3")
    rows = []
    for group in ("fresh_15000", "trained_20000"):
        (output / group).mkdir(parents=True)
        metadata = {
            "schema": SCHEMA,
            "source_checkpoint_sha256": "050500d607e1910ca088049ae73619949ad183e23c85354a8408bb29571fbe84",
            "zero_optimizer_update_proof": {
                "backward_called": False,
                "optimizer_step_called": False,
                "scheduler_step_called": False,
                "state_hash_unchanged": True,
            },
            "model_hash_before": {"t": "same"},
            "model_hash_after": {"t": "same"},
            "shared_t_snapshot_hashes": {stem: "same" for stem in stems},
        }
        (output / group / "group_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        for arm in arms:
            for stem in stems:
                path = output / group / arm / stem
                (path / "tensors").mkdir(parents=True)
                stats = {
                    "schema": SCHEMA,
                    "outside_mask_parity": {"final": True},
                    "black_attribution": {"near_black_threshold": 0.10, "near_black_fraction": 0.0},
                    "group": group, "arm": arm, "stem": stem,
                }
                (path / "stats.json").write_text(json.dumps(stats), encoding="utf-8")
                rows.append(stats)
    operator = {
        "schema": SCHEMA,
        "optimizer_updates": 0,
        "checkpoints_written": 0,
        "ply_written": 0,
        "release_aggregate_before": "4fedb22dc2f2e6415a3d3948ab26fba54df91ba06b66d951a09b3dc5f761188d",
        "release_aggregate_after": "4fedb22dc2f2e6415a3d3948ab26fba54df91ba06b66d951a09b3dc5f761188d",
        "release_unchanged": True,
        "group_metadata": {},
    }
    (output / "operator_record.json").write_text(json.dumps(operator), encoding="utf-8")
    (output / "semantic_renderer_ablation_aggregate.json").write_text(json.dumps(rows), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["audit", "--output", str(output)])
    assert audit_main() == 0
