from types import SimpleNamespace

import pytest
import torch

from gaussian_renderer.reflection_renderer import semantic_transparent_outside_only
from gaussian_renderer.transmittance_renderer import semantic_cout_outside_only
from geometry.cuboid_space import CuboidSpace, INSIDE
from raytracer.differentiable_raytrace import apply_surfel_filter
from scene.transmittance_surfel_model import TransmittanceSurfelModel
from stage_d_training import (
    FORMAL_RELEASE_ID, FORMAL_RELEASE_SHA256, FORMAL_SOURCE_SHA256,
    SEMANTIC_NODES, SEMANTIC_OUTPUT_NAME, _phase_for_iteration,
    _transmittance_topology_update_allowed, _validate_semantic_repair_contract,
)
from tools.run_stage_d_semantic_repair import training_command
from utils.semantic_repair import anti_veil_loss, smooth_ramp, spatial_frequency_energy


def cuboid():
    return CuboidSpace(
        axes=torch.eye(3), lower=torch.tensor([-1.0, -1.0, -1.0]),
        upper=torch.tensor([1.0, 1.0, 1.0]), interface_margin=0.1,
    )


def test_candidate_filter_excludes_inside_and_interface_without_changing_candidates():
    candidate_ids = torch.tensor([[0, 1, 2, -1], [2, 1, -1, -1]])
    valid = candidate_ids >= 0
    outside = torch.tensor([False, False, True])
    selected = apply_surfel_filter(candidate_ids, valid, outside)
    assert selected.tolist() == [
        [False, False, True, False], [True, False, False, False],
    ]
    assert torch.equal(apply_surfel_filter(candidate_ids, valid, None), valid)


def test_r_filter_changes_only_transparent_mask_rays():
    unfiltered = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    outside = torch.tensor([[10.0], [30.0]])
    transparent = torch.tensor([True, False, True, False])
    final = semantic_transparent_outside_only(unfiltered, outside, transparent)
    assert final.tolist() == [[10.0], [2.0], [30.0], [4.0]]


def test_cout_formal_component_is_exactly_outside_and_interface_is_not_aliased():
    components = {
        "inside": {"raw": torch.tensor([1.0])},
        "interface": {"raw": torch.tensor([2.0])},
        "outside": {"raw": torch.tensor([3.0])},
    }
    assert semantic_cout_outside_only(components) is components["outside"]
    with pytest.raises(ValueError, match="disjoint"):
        semantic_cout_outside_only({"inside": {}, "outside": {}})


def test_t_parameterization_is_strict_inside_and_checkpoint_metadata_is_versioned():
    model = TransmittanceSurfelModel()
    model.create_random_inside_cuboid(cuboid(), count=4096, seed=20260703)
    assert model.get_xyz.shape == (4096, 3)
    assert torch.all(cuboid().classify(model.get_xyz) == INSIDE)
    with torch.no_grad():
        model._xyz[0] = torch.tensor([1e6, -1e6, 1e6])
    model.assert_strictly_inside()
    state = model.capture()
    assert state["position_parameterization"] == "cuboid_inside_sigmoid_v1"
    assert state["cuboid_space"]["transparent_interface_margin_mode"] == "exclude"


def test_semantic_pilot_disables_all_t_topology_updates():
    opt = SimpleNamespace(
        stage_d_semantic_repair_pilot=True,
        transmittance_densify_from_iter=0,
        transmittance_densify_until_iter=10000,
        transmittance_densification_interval=1,
    )
    assert not _transmittance_topology_update_allowed(opt, 1)
    assert _phase_for_iteration(15001, semantic_repair=True) == "semantic_repair_cached_t_only"


def test_anti_veil_forward_gradients_and_no_dr_leak():
    t_opacity = torch.tensor([[[[4.0]]]], requires_grad=True)
    t_color = torch.full((1, 2, 2, 3), -5.0, requires_grad=True)
    d = torch.tensor(1.0, requires_grad=True)
    r = torch.tensor(1.0, requires_grad=True)
    alpha = torch.sigmoid(t_opacity).expand(1, 2, 2, 1)
    color = torch.sigmoid(t_color)
    cin = alpha * color
    gt = torch.full_like(color, 0.8)
    mask = torch.ones_like(alpha)
    losses = anti_veil_loss(cin, alpha, gt, mask)
    total = losses["black"] + losses["saturation"]
    total.backward()
    assert float(total) > 0
    assert t_opacity.grad is not None and torch.any(t_opacity.grad != 0)
    assert t_color.grad is not None and torch.any(t_color.grad != 0)
    assert d.grad is None and r.grad is None


def test_anti_veil_allows_localized_colorful_high_alpha_and_has_smooth_ramp():
    alpha = torch.full((1, 10, 10, 1), 0.05)
    alpha[:, :2] = 0.99
    colorful = torch.full((1, 10, 10, 3), 0.7)
    cin = alpha * colorful
    gt = torch.full_like(colorful, 0.8)
    loss = anti_veil_loss(cin, alpha, gt, torch.ones_like(alpha))
    assert float(loss["black"]) < 1e-3
    assert float(loss["saturation"]) < 1e-3
    assert smooth_ramp(0, 0, 200) == 0.0
    assert smooth_ramp(100, 0, 200) == pytest.approx(0.5)
    assert smooth_ramp(200, 0, 200) == 1.0


def test_spatial_frequency_metric_distinguishes_structure_from_constant_energy():
    constant = torch.ones((6, 6, 3)) * 0.5
    checker = constant.clone()
    checker[::2, ::2] = 0.0; checker[1::2, 1::2] = 0.0
    mask = torch.ones((6, 6, 1), dtype=torch.bool)
    flat = spatial_frequency_energy(constant, mask)
    structured = spatial_frequency_energy(checker, mask)
    assert flat == {"edge_l1": 0.0, "laplacian_l1": 0.0}
    assert structured["edge_l1"] > 0 and structured["laplacian_l1"] > 0


def semantic_contract(tmp_path):
    mask_hash = "056da740a6bb20e7b888be890ac39b597734d5e0487003b36384b57a9cb66551"
    mask_aggregate = "5" * 64
    dataset = SimpleNamespace(
        resolution=8, ray_chunk_size=2048,
        transmittance_init_mode="random_bbox", transmittance_init_count=4096,
        transmittance_init_seed=20260703,
        model_path=str(tmp_path / SEMANTIC_OUTPUT_NAME),
        _semantic_cuboid_space_metadata={"transparent_interface_margin_mode": "exclude"},
        _validated_specular_mask_manifest={
            "manifest_file_sha256": mask_hash, "aggregate_sha256": mask_aggregate,
        },
    )
    opt = SimpleNamespace(
        stage_d_semantic_repair_pilot=True, iterations=16000,
        stage_d_phase_a_end_iteration=16000,
        stage_d_depth_start_iteration=40000, lambda_spec=0.2, specular_k0=0.9,
    )
    release = SimpleNamespace(
        manifest={"geometry_release_id": FORMAL_RELEASE_ID},
        validation={"aggregate_sha256": FORMAL_RELEASE_SHA256},
    )
    source = {
        "sha256": FORMAL_SOURCE_SHA256, "global_iteration": 15000,
        "reflection_iteration": 12000,
        "stage_b_config": {
            "lambda_spec": 0.2, "specular_k0": 0.9,
            "specular_mask": {
                "manifest_file_sha256": mask_hash, "aggregate_sha256": mask_aggregate,
            },
        },
    }
    return dataset, opt, release, source


def test_semantic_contract_rejects_stage_d_resume_and_schedule_drift(tmp_path):
    dataset, opt, release, source = semantic_contract(tmp_path)
    _validate_semantic_repair_contract(
        dataset, opt, release, source, True, SEMANTIC_NODES, SEMANTIC_NODES,
    )
    with pytest.raises(ValueError, match="fresh_from_stage_b"):
        _validate_semantic_repair_contract(
            dataset, opt, release, source, False, SEMANTIC_NODES, SEMANTIC_NODES,
        )
    opt.iterations = 16001
    with pytest.raises(ValueError, match="endpoint"):
        _validate_semantic_repair_contract(
            dataset, opt, release, source, True, SEMANTIC_NODES, SEMANTIC_NODES,
        )


def test_operator_is_one_fixed_1000_step_command():
    command = training_command()
    assert "--stage_d_semantic_repair_pilot" in command
    assert "--stage_d_cached_twarmup" not in command
    assert "--stage_d_smoke" not in command
    assert command[command.index("--iterations") + 1] == "16000"
    assert command[command.index("--stage_d_phase_a_end_iteration") + 1] == "16000"
    assert command[command.index("--stage_d_depth_start_iteration") + 1] == "40000"
    checkpoint_index = command.index("--checkpoint_iterations")
    assert tuple(map(int, command[checkpoint_index + 1:])) == SEMANTIC_NODES
