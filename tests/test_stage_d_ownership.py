import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gaussian_renderer.reflection_renderer import (
    apply_cuboid_front_reflection_path, apply_transparent_contribution_mode,
    support_safe_outside_mask,
)
from gaussian_renderer.transmittance_renderer import cuboid_front_transmission_inputs
from geometry.cuboid_path import PATH_SCHEMA, build_cuboid_front_path
from geometry.cuboid_space import (
    CuboidSpace, SUPPORT_CROSSING, SUPPORT_INTERFACE,
    SUPPORT_STRICT_INSIDE, SUPPORT_STRICT_OUTSIDE,
)
from raytracer.differentiable_raytrace import trace_candidates
from scene.reflection_surfel_model import ReflectionSurfelModel
from scene.transmittance_surfel_model import TransmittanceSurfelModel
from tools.run_stage_d_ownership_ab import (
    CACHE, RETRYABLE_PREFLIGHT_COMMIT, archive_retryable_preflight_failure,
    common_training_contract, training_command,
)
from stage_d_training import _requires_cuboid_space
from utils.stage_d_static_cache import OWNERSHIP_CACHE_SCHEMA, renderer_contract


def space():
    return CuboidSpace(
        axes=torch.eye(3, dtype=torch.float64),
        lower=torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float64),
        upper=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64),
        interface_margin=0.1, epsilon=1e-6,
    )


def test_cuboid_front_path_is_independent_of_perturbed_d_gbuffer():
    cuboid = space()
    center = torch.tensor([0.0, 0.0, -3.0], dtype=torch.float64)
    back = torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float64)
    near = torch.tensor([[2.0]], dtype=torch.float64)
    far = torch.tensor([[4.0]], dtype=torch.float64)
    valid = torch.tensor([[True]])
    first = build_cuboid_front_path(center, back, near, far, valid, cuboid)
    # Deliberately unrelated D position/normal values are not accepted inputs to
    # the cuboid-front constructor, so perturbing them cannot alter the path.
    d_position = torch.tensor([[[99.0, -40.0, 12.0]]])
    d_normal = torch.tensor([[[0.3, 0.4, 0.5]]])
    d_position.add_(123.0); d_normal.mul_(-17.0)
    second = build_cuboid_front_path(center, back, near, far, valid, cuboid)
    assert first["schema"] == PATH_SCHEMA
    for key in ("front_position_selected", "front_normal_selected", "direction_selected"):
        assert torch.equal(first[key], second[key])
    assert torch.allclose(first["front_position_selected"], torch.tensor([[0., 0., -1.]], dtype=torch.float64))
    assert torch.allclose(first["front_normal_selected"], torch.tensor([[0., 0., -1.]], dtype=torch.float64))


def test_cuboid_front_path_fails_closed_on_invalid_frozen_geometry():
    cuboid = space(); center = torch.tensor([0., 0., -3.], dtype=torch.float64)
    with pytest.raises(FloatingPointError):
        build_cuboid_front_path(
            center, torch.tensor([[[0., 0., float("nan")]]], dtype=torch.float64),
            torch.tensor([[2.]], dtype=torch.float64),
            torch.tensor([[4.]], dtype=torch.float64), torch.tensor([[True]]), cuboid,
        )
    with pytest.raises(RuntimeError, match="back position/t_far"):
        build_cuboid_front_path(
            center, torch.tensor([[[0., 0., 1.]]], dtype=torch.float64),
            torch.tensor([[2.]], dtype=torch.float64),
            torch.tensor([[5.]], dtype=torch.float64), torch.tensor([[True]]), cuboid,
        )


def test_cuboid_front_r_and_t_inputs_ignore_d_and_preserve_nontransparent_rays():
    transparent = torch.tensor([False, True, False])
    legacy = {
        name: torch.arange(9, dtype=torch.float64).reshape(3, 3) + offset
        for offset, name in enumerate(("origins", "directions", "d_cam", "wo", "normal"))
    }
    front = torch.tensor([[0., 0., -1.]], dtype=torch.float64)
    normal = torch.tensor([[0., 0., -1.]], dtype=torch.float64)
    direction = torch.tensor([[0., 0., 1.]], dtype=torch.float64)
    first = apply_cuboid_front_reflection_path(
        legacy, transparent, front, normal, direction, 1e-3,
    )
    perturbed = {name: value.clone() for name, value in legacy.items()}
    perturbed["origins"][transparent] += 1000.0
    perturbed["normal"][transparent] *= -37.0
    second = apply_cuboid_front_reflection_path(
        perturbed, transparent, front, normal, direction, 1e-3,
    )
    for key in legacy:
        assert torch.equal(first[key][transparent], second[key][transparent])
        assert torch.equal(first[key][~transparent], legacy[key][~transparent])
    t_origin_a, t_distance_a = cuboid_front_transmission_inputs(
        front, direction, torch.tensor([[2.]], dtype=torch.float64), 1e-3,
    )
    # D position/normal are intentionally absent from this contract.
    t_origin_b, t_distance_b = cuboid_front_transmission_inputs(
        front, direction, torch.tensor([[2.]], dtype=torch.float64), 1e-3,
    )
    assert torch.equal(t_origin_a, t_origin_b)
    assert torch.equal(t_distance_a, t_distance_b)
    assert torch.allclose(t_distance_a, torch.tensor([[2.001]], dtype=torch.float64))


def test_support_classification_uses_complete_three_sigma_support_not_center():
    cuboid = space()
    points = torch.tensor([
        [0.0, 0.0, 0.0], [0.95, 0.0, 0.0],
        [3.0, 0.0, 0.0], [1.5, 0.0, 0.0],
    ], dtype=torch.float64)
    rotation = torch.tensor([[1., 0., 0., 0.]] * 4, dtype=torch.float64)
    scaling = torch.tensor([
        [0.1, 0.1], [0.01, 0.01], [0.1, 0.1], [0.3, 0.3],
    ], dtype=torch.float64)
    classes = cuboid.classify_support(points, rotation, scaling, sigma=3.0)
    assert classes.tolist() == [
        SUPPORT_STRICT_INSIDE, SUPPORT_INTERFACE,
        SUPPORT_STRICT_OUTSIDE, SUPPORT_CROSSING,
    ]
    # The last center is outside, but its finite support crosses the enclosure.
    assert int(cuboid.classify(points[-1:])[0]) == 2
    model = SimpleNamespace(
        get_xyz=points[-2:], get_rotation=rotation[-2:], get_scaling=scaling[-2:],
    )
    # The center-only view calls both points outside; support-safe mode admits
    # only the fully separated support and rejects the crossing support.
    assert support_safe_outside_mask(model, cuboid, sigma=3.0).tolist() == [True, False]


def test_transfer_evidence_aux_attributes_each_contribution_to_ray_and_depth():
    if not torch.cuda.is_available():
        pytest.skip("production candidate attribution is CUDA-only")
    device = "cuda"
    model = ReflectionSurfelModel()
    model._xyz = torch.nn.Parameter(torch.tensor([[0., 0., 1.]], device=device))
    model._rotation = torch.nn.Parameter(torch.tensor([[1., 0., 0., 0.]], device=device))
    model._scaling = torch.nn.Parameter(torch.log(torch.tensor([[0.5, 0.5]], device=device)))
    model._opacity = torch.nn.Parameter(torch.tensor([[0.]], device=device))
    model._color = torch.nn.Parameter(torch.zeros((1, 3), device=device))
    origins = torch.tensor([[0., 0., 0.], [0., 0., -1.]], device=device)
    directions = torch.tensor([[0., 0., 1.], [0., 0., 1.]], device=device)
    _, aux = trace_candidates(
        model, origins, directions,
        candidates=torch.tensor([0, 0], device=device),
        offsets=torch.tensor([0, 1, 2], device=device),
        cutoff_sigma=3.0, hit_threshold=1e-4, return_aux=True,
    )
    assert aux.contributing_indices.tolist() == [0, 0]
    assert aux.contributing_ray_indices.tolist() == [0, 1]
    assert torch.allclose(aux.contributing_depths, torch.tensor([1., 2.], device=device))
    assert torch.all(aux.contributing_weights > 0)


def test_transparent_off_gate_preserves_every_nontransparent_ray():
    values = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    selected = torch.tensor([False, True, False, True])
    result = apply_transparent_contribution_mode(values, None, selected, "off")
    assert torch.equal(result[~selected], values[~selected])
    assert torch.equal(result[selected], torch.zeros_like(result[selected]))


def test_support_safe_t_parameterization_survives_extreme_latents():
    cuboid = space().to(dtype=torch.float32)
    model = TransmittanceSurfelModel()
    model.create_random_support_safe_inside_cuboid(cuboid, count=128, seed=7)
    with torch.no_grad():
        model._xyz[0] = torch.tensor([1e6, -1e6, 1e6])
    model.assert_strictly_inside()
    classes = cuboid.classify_support(
        model.get_xyz, model.get_rotation, model.get_scaling, sigma=3.0,
    )
    assert torch.all(classes == SUPPORT_STRICT_INSIDE)
    assert model.capture()["position_parameterization"] == "cuboid_inside_support_sigmoid_v2"


def test_transferred_initialization_copies_geometry_color_and_caps_opacity():
    cuboid = space().to(dtype=torch.float32)
    diffuse = SimpleNamespace()
    diffuse._xyz = torch.tensor([[0., 0., 0.], [0.2, 0.1, 0.]])
    diffuse.get_xyz = diffuse._xyz
    diffuse._rotation = torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.]])
    diffuse._scaling = torch.log(torch.full((2, 2), 0.05))
    diffuse._base_color = torch.tensor([[0.1, -0.2, 0.3], [0.4, 0.5, -0.1]])
    diffuse._opacity = torch.tensor([[8.0], [-8.0]])
    model = TransmittanceSurfelModel()
    model.create_transferred_from_diffuse(
        diffuse, torch.tensor([0, 1]), cuboid, count=4, seed=9,
        selection_metadata={"eligible_count": 2},
    )
    assert torch.allclose(model.get_xyz[:2], diffuse._xyz, atol=1e-5)
    assert torch.equal(model._color[:2], diffuse._base_color)
    opacity = torch.sigmoid(model._opacity[:2])
    assert float(opacity.max()) <= 0.050001 and float(opacity.min()) >= 0.004999
    assert model.initialization["transferred_count"] == 2
    assert model.initialization["random_fill_count"] == 2


def test_v4_operator_arms_differ_only_by_t_initialization_and_cache_reuse():
    arm_a = training_command("random_strict_inside")
    arm_b = training_command("transferred_d_inside")
    assert common_training_contract(arm_a) == common_training_contract(arm_b)
    for command, arm in ((arm_a, "random_strict_inside"), (arm_b, "transferred_d_inside")):
        assert command[command.index("--transmittance_init_mode") + 1] == arm
        assert command[command.index("--iterations") + 1] == "15500"
        assert command[command.index("--transparent_path_mode") + 1] == "cuboid_front_v1"
        assert command[command.index("--transparent_direct_mode") + 1] == "off"
        assert command[command.index("--transparent_reflection_mode") + 1] == "off"
        assert command[command.index("--cout_ownership_mode") + 1] == "support_safe_outside"
        assert command[command.index("--stage_d_static_cache_path") + 1] == str(CACHE)
    assert "--stage_d_reuse_static_cache" not in arm_a
    assert "--stage_d_reuse_static_cache" in arm_b


def test_ownership_mode_requires_release_cuboid_space():
    assert _requires_cuboid_space(SimpleNamespace(
        stage_d_semantic_repair_pilot=False, stage_d_ownership_pilot=True,
    ))
    assert _requires_cuboid_space(SimpleNamespace(
        stage_d_semantic_repair_pilot=True, stage_d_ownership_pilot=False,
    ))
    assert not _requires_cuboid_space(SimpleNamespace(
        stage_d_semantic_repair_pilot=False, stage_d_ownership_pilot=False,
    ))


def test_operator_archives_only_the_known_zero_step_preflight_failure(tmp_path):
    output = tmp_path / "pilot"
    (output / "logs").mkdir(parents=True)
    (output / "logs/random_strict_inside.log").write_text(
        "ValueError: support-safe T initialization requires cuboid space\n",
        encoding="utf-8",
    )
    (output / "ownership_operator_record.json").write_text(
        json.dumps({
            "status": "CUBOID_PATH_OWNERSHIP_PILOT_BLOCKED",
            "git_commit": RETRYABLE_PREFLIGHT_COMMIT,
            "operator_error": "RuntimeError: ownership arm failed: random_strict_inside exit=1",
            "arms": {"random_strict_inside": {"exit_code": 1}},
            "source_sha256_before": "050500d607e1910ca088049ae73619949ad183e23c85354a8408bb29571fbe84",
            "source_sha256_after": "050500d607e1910ca088049ae73619949ad183e23c85354a8408bb29571fbe84",
            "release_aggregate_before": "4fedb22dc2f2e6415a3d3948ab26fba54df91ba06b66d951a09b3dc5f761188d",
            "release_aggregate_after": "4fedb22dc2f2e6415a3d3948ab26fba54df91ba06b66d951a09b3dc5f761188d",
        }),
        encoding="utf-8",
    )
    archive = archive_retryable_preflight_failure(output)
    assert not output.exists()
    assert archive.is_dir()
    with pytest.raises(FileExistsError, match="not a directory"):
        output.write_text("occupied", encoding="utf-8")
        archive_retryable_preflight_failure(output)


def test_renderer_contract_uses_incompatible_v4_cache_identity():
    dataset = SimpleNamespace(
        resolution=8, ray_chunk_size=2048, ray_cutoff_sigma=3.0,
        ray_hit_threshold=1e-4, ray_epsilon_scale=1e-4,
        material_alpha_threshold=1e-4, roughness_min=0.03,
        roughness_remap=False, ray_background="scene",
        transmittance_compose="alpha_over",
        transparent_path_mode="cuboid_front_v1",
        transparent_direct_mode="off", transparent_reflection_mode="off",
        cout_ownership_mode="support_safe_outside",
        _semantic_cuboid_space_metadata={"schema": "rtgs_cuboid_space_v1"},
    )
    contract = renderer_contract(dataset)
    assert contract["schema"] == "rtgs_stage_d_cuboid_front_renderer_contract_v4"
    assert contract["support_classification"] == "cuboid_local_finite_3sigma_v1"
    assert OWNERSHIP_CACHE_SCHEMA.endswith("_v4")


def test_legacy_semantic_cache_contract_remains_center_outside_only():
    dataset = SimpleNamespace(
        resolution=8, ray_chunk_size=2048, ray_cutoff_sigma=3.0,
        ray_hit_threshold=1e-4, ray_epsilon_scale=1e-4,
        material_alpha_threshold=1e-4, roughness_min=0.03,
        roughness_remap=False, ray_background="scene",
        transmittance_compose="alpha_over",
        transparent_path_mode="legacy_d_gbuffer",
        transparent_reflection_mode="legacy", cout_ownership_mode="legacy",
        _semantic_cuboid_space_metadata={"schema": "rtgs_cuboid_space_v1"},
    )
    contract = renderer_contract(dataset)
    assert contract["schema"] == "rtgs_stage_d_renderer_contract_v1"
    assert contract["r_transparent_spatial_filter"] == "outside_only"
    assert contract["cout_spatial_filter"] == "outside_only"
