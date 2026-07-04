from types import SimpleNamespace

import pytest
import torch

from geometry.cuboid_space import CuboidSpace, SUPPORT_STRICT_INSIDE
from raytracer.acceleration_structure import CudaLBVH
from raytracer.candidate_parameters import (
    decode_candidate_parameters, pack_reflection_parameters,
)
from raytracer.tracer import raytrace
from scene.transmittance_surfel_model import TransmittanceSurfelModel
from stage_d_training import (
    TSCALE_RECOVERY_LONG_ENDPOINT, TSCALE_RECOVERY_LONG_NODES,
    TSCALE_RECOVERY_PREFLIGHT_ENDPOINT, TSCALE_RECOVERY_PREFLIGHT_NODES,
    _tscale_recovery_guard_result,
)
from tools.run_stage_d_tscale_recovery import training_command
from utils.stage_d_static_cache import state_sha256


def space(device="cpu", dtype=torch.float32):
    return CuboidSpace(
        axes=torch.eye(3, device=device, dtype=dtype),
        lower=torch.tensor([-1.0, -1.0, -1.0], device=device, dtype=dtype),
        upper=torch.tensor([1.0, 1.0, 1.0], device=device, dtype=dtype),
        interface_margin=0.1, epsilon=1e-6,
    )


def args():
    return SimpleNamespace(
        transmittance_percent_dense=0.01,
        transmittance_position_lr_init=0.00016,
        transmittance_position_lr_final=0.0000016,
        transmittance_position_lr_delay_mult=0.01,
        transmittance_position_lr_max_steps=20000,
        transmittance_color_lr=0.0025,
        transmittance_opacity_lr=0.025,
        transmittance_scaling_lr=0.005,
        transmittance_rotation_lr=0.001,
    )


def initialized_model(device="cpu"):
    model = TransmittanceSurfelModel()
    model.create_random_support_safe_inside_cuboid(space(device), count=8, seed=17)
    model.training_setup(args())
    for group in model.optimizer.param_groups:
        group["params"][0].grad = torch.ones_like(group["params"][0]) * 1e-3
    model.optimizer.step(); model.optimizer.zero_grad(set_to_none=True)
    return model


def test_projection_preserves_active_geometry_and_only_clears_affected_scale_adam_rows():
    model = initialized_model()
    with torch.no_grad():
        model._scaling[0] = torch.log(torch.tensor([20.0, 10.0]))
    active_before = model.get_scaling.detach().clone()
    world_before = model.get_xyz.detach().clone()
    parameters_before = {
        name: group["params"][0].detach().clone()
        for group in model.optimizer.param_groups for name in [group["name"]]
        if name != "scaling"
    }
    state_before = {
        name: state_sha256(model.optimizer.state[group["params"][0]])
        for group in model.optimizer.param_groups for name in [group["name"]]
        if name != "scaling"
    }
    scaling_group = next(
        group for group in model.optimizer.param_groups if group["name"] == "scaling"
    )
    scale_state = model.optimizer.state[scaling_group["params"][0]]
    unaffected_exp_avg = scale_state["exp_avg"][1:].clone()
    step_before = scale_state["step"].clone()

    report = model.project_raw_scaling_to_active_(migrate_legacy=True)
    assert report["affected_count"] == 1
    assert report["schema"] == "cuboid_support_projected_cap_v2"
    assert torch.allclose(model.get_scaling, active_before, atol=2e-7, rtol=2e-7)
    assert torch.allclose(model.get_xyz, world_before, atol=2e-7, rtol=2e-7)
    assert torch.allclose(torch.exp(model._scaling), model.get_scaling, atol=2e-7, rtol=2e-7)
    assert torch.count_nonzero(scale_state["exp_avg"][0]) == 0
    assert torch.count_nonzero(scale_state["exp_avg_sq"][0]) == 0
    assert torch.equal(scale_state["exp_avg"][1:], unaffected_exp_avg)
    assert torch.equal(scale_state["step"], step_before)
    for group in model.optimizer.param_groups:
        name, parameter = group["name"], group["params"][0]
        if name == "scaling":
            continue
        assert torch.equal(parameter, parameters_before[name])
        assert state_sha256(model.optimizer.state[parameter]) == state_before[name]


def test_projected_checkpoint_roundtrip_keeps_raw_active_and_support_legal():
    model = initialized_model()
    with torch.no_grad():
        model._scaling[:2] = torch.log(torch.tensor([[20.0, 10.0], [8.0, 12.0]]))
    model.project_raw_scaling_to_active_(migrate_legacy=True)
    state = model.capture()
    restored = TransmittanceSurfelModel(); restored.restore(state, args())
    assert restored.scaling_parameterization == "cuboid_support_projected_cap_v2"
    assert torch.allclose(
        torch.exp(restored._scaling), restored.get_scaling, atol=2e-7, rtol=2e-7
    )
    classes = restored.cuboid_space.classify_support(
        restored.get_xyz, restored.get_rotation, restored.get_scaling, sigma=3.0,
    )
    assert torch.all(classes == SUPPORT_STRICT_INSIDE)
    second = restored.project_raw_scaling_to_active_()
    assert second["affected_count"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA raytrace parity is CUDA-only")
def test_cuda_raytrace_forward_is_invariant_across_scale_projection():
    device, dtype = "cuda", torch.float64
    model = TransmittanceSurfelModel()
    model.create_random_support_safe_inside_cuboid(
        space(device=device, dtype=dtype), count=1, seed=3,
    )
    model.training_setup(args())
    with torch.no_grad():
        model._xyz.zero_(); model._rotation[:] = torch.tensor([1., 0., 0., 0.], device=device)
        model._scaling[:] = torch.log(torch.tensor([[20., 10.]], device=device, dtype=dtype))
        model._opacity.zero_(); model._color[:] = torch.tensor([[0.2, -0.1, 0.3]], device=device)
    # An off-axis ray makes opacity depend on scale.  The old candidate pack
    # decoded exp(_scaling), so this case caught the raw/active split that a
    # center ray (radius=0) could not observe.
    origins = torch.tensor([[0.35, 0., -2.]], device=device, dtype=dtype)
    directions = torch.tensor([[0., 0., 1.]], device=device, dtype=dtype)
    packed_before = decode_candidate_parameters(pack_reflection_parameters(model))
    assert torch.equal(packed_before["scaling"], model.get_scaling)
    assert torch.equal(packed_before["xyz"], model.get_xyz)
    before = raytrace(
        model, origins, directions, acceleration=CudaLBVH(), chunk_size=1
    )[:3]
    model.project_raw_scaling_to_active_(migrate_legacy=True)
    packed_after = decode_candidate_parameters(pack_reflection_parameters(model))
    assert torch.equal(packed_after["scaling"], model.get_scaling)
    assert torch.equal(packed_before["scaling"], packed_after["scaling"])
    assert torch.equal(packed_before["xyz"], packed_after["xyz"])
    after = raytrace(
        model, origins, directions, acceleration=CudaLBVH(), chunk_size=1
    )[:3]
    for first, second in zip(before, after):
        assert torch.allclose(first, second, atol=1e-10, rtol=1e-10)
    loss_before = before[0].sum() + before[1].sum() + before[2].sum()
    loss_after = after[0].sum() + after[1].sum() + after[2].sum()
    assert torch.allclose(loss_before, loss_after, atol=1e-10, rtol=1e-10)


def test_recovery_commands_use_exact_sources_bounds_and_checkpoint_nodes():
    preflight = training_command("preflight")
    long = training_command("long")
    assert "--stage_d_tscale_recovery_preflight" in preflight
    assert preflight[preflight.index("--start_checkpoint") + 1].endswith(
        "ownership_tlong_g15500_g20000_v1/chkpnt16000.pth"
    )
    assert preflight[preflight.index("--iterations") + 1] == str(
        TSCALE_RECOVERY_PREFLIGHT_ENDPOINT
    )
    assert tuple(map(int, preflight[preflight.index("--checkpoint_iterations") + 1:])) \
        == TSCALE_RECOVERY_PREFLIGHT_NODES
    assert "--stage_d_tscale_recovery_long" in long
    assert long[long.index("--start_checkpoint") + 1].endswith("chkpnt16050.pth")
    assert long[long.index("--iterations") + 1] == str(TSCALE_RECOVERY_LONG_ENDPOINT)
    assert tuple(map(int, long[long.index("--checkpoint_iterations") + 1:])) \
        == TSCALE_RECOVERY_LONG_NODES
    assert all(b - a <= 250 for a, b in zip(TSCALE_RECOVERY_LONG_NODES, TSCALE_RECOVERY_LONG_NODES[1:]))


def test_recovery_guard_accepts_projected_state_and_blocks_escape_or_collapse():
    healthy = [{
        "saturation": 0.13, "black": 0.001, "capped_fraction": 0.0,
        "projection_affected_fraction": 3 / 4096,
        "minimum_forward_factor": 0.9999999,
        "minimum_pre_projection_factor": 0.995,
        "post_projection_raw_active_abs": 1e-8,
        "t_energy": 0.32, "cin_energy": 0.17,
    }] * 50
    summary, failures = _tscale_recovery_guard_result(healthy)
    assert failures == [] and summary["window"] == 50
    escaped = [dict(healthy[0], minimum_pre_projection_factor=0.5)]
    _, failures = _tscale_recovery_guard_result(escaped)
    assert "single-update raw scale escape" in failures
    collapsed = [dict(healthy[0], t_energy=0.01, cin_energy=0.01)]
    _, failures = _tscale_recovery_guard_result(collapsed)
    assert "T contribution collapse" in failures and "Cin collapse" in failures
