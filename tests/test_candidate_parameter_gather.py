from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import gaussian_renderer.reflection_renderer as reflection_renderer
import raytracer.differentiable_raytrace as differentiable_raytrace
from gaussian_renderer.reflection_renderer import StageBRenderState
from raytracer.candidate_parameters import (
    decode_candidate_parameters,
    gather_candidate_parameters,
    pack_reflection_parameters,
)
from raytracer.tracer import raytrace
from scene.reflection_surfel_model import ReflectionSurfelModel
from utils.loss_utils import (
    l1_loss,
    monocular_normal_loss,
    normal_depth_consistency_loss,
    ssim,
)
from utils.perceptual_loss import VGG16PerceptualLoss
from utils.surfel_utils import quaternion_to_rotation_matrix


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="candidate gather/reduce is CUDA-only"
)


def _model(count=7, dtype=torch.float64):
    generator = torch.Generator(device="cuda").manual_seed(17)
    model = ReflectionSurfelModel()
    model._xyz = torch.nn.Parameter(
        0.2 * torch.randn((count, 3), generator=generator, device="cuda", dtype=dtype)
    )
    model._xyz.data[:, 2] += torch.linspace(0.7, 1.5, count, device="cuda", dtype=dtype)
    model._rotation = torch.nn.Parameter(
        torch.randn((count, 4), generator=generator, device="cuda", dtype=dtype)
    )
    model._scaling = torch.nn.Parameter(
        torch.log(
            0.8
            + 0.2
            * torch.rand((count, 2), generator=generator, device="cuda", dtype=dtype)
        )
    )
    model._opacity = torch.nn.Parameter(
        torch.randn((count, 1), generator=generator, device="cuda", dtype=dtype)
    )
    model._color = torch.nn.Parameter(
        torch.randn((count, 3), generator=generator, device="cuda", dtype=dtype)
    )
    model.topology_version = 0
    return model


def _copy_model(source):
    copied = ReflectionSurfelModel()
    for name in ("_xyz", "_rotation", "_scaling", "_opacity", "_color"):
        setattr(copied, name, torch.nn.Parameter(getattr(source, name).detach().clone()))
    copied.topology_version = source.topology_version
    return copied


def _legacy_candidate_parameters(model, candidate_ids):
    return {
        "xyz": model.get_xyz[candidate_ids],
        "rotation": quaternion_to_rotation_matrix(model.get_rotation)[candidate_ids],
        "scaling": model.get_scaling[candidate_ids],
        "opacity": model.get_opacity[candidate_ids],
        "color": model.get_color[candidate_ids],
    }


def test_repeated_candidate_forward_and_all_raw_parameter_gradients_match_legacy():
    optimized = _model()
    legacy = _copy_model(optimized)
    fixed_ids = torch.tensor(
        [0, 0, 0, 1, 1, 2, 0, 2, 2, 2, 3, 1, 0, 3, 3, 3] * 256,
        device="cuda",
        dtype=torch.int64,
    ).reshape(64, 64)

    optimized_values = decode_candidate_parameters(
        gather_candidate_parameters(pack_reflection_parameters(optimized), fixed_ids)
    )
    legacy_values = _legacy_candidate_parameters(legacy, fixed_ids)
    generator = torch.Generator(device="cuda").manual_seed(23)
    optimized_loss = optimized._xyz.new_zeros(())
    legacy_loss = legacy._xyz.new_zeros(())
    for name in ("xyz", "rotation", "scaling", "opacity", "color"):
        assert torch.allclose(
            optimized_values[name], legacy_values[name], atol=1e-12, rtol=1e-12
        )
        weight = torch.randn(
            optimized_values[name].shape,
            generator=generator,
            device="cuda",
            dtype=optimized_values[name].dtype,
        )
        optimized_loss = optimized_loss + (optimized_values[name] * weight).sum()
        legacy_loss = legacy_loss + (legacy_values[name] * weight).sum()

    optimized_loss.backward()
    legacy_loss.backward()
    for name in ("_xyz", "_rotation", "_scaling", "_opacity", "_color"):
        assert torch.allclose(
            getattr(optimized, name).grad,
            getattr(legacy, name).grad,
            atol=2e-10,
            rtol=2e-10,
        ), name


def _raytrace_loss(model, origins, directions, reference_gather=False):
    original = differentiable_raytrace.gather_candidate_parameters
    if reference_gather:
        differentiable_raytrace.gather_candidate_parameters = (
            lambda table, ids: table[ids]
        )
    try:
        color, alpha, depth, hit = raytrace(
            model, origins, directions, chunk_size=16
        )
    finally:
        differentiable_raytrace.gather_candidate_parameters = original
    loss = (
        color @ color.new_tensor([[0.7], [1.1], [1.3]])
        + 0.4 * alpha
        + 0.2 * depth
    ).sum()
    return (color, alpha, depth, hit), loss


def test_full_raytrace_outputs_and_r_and_ray_gradients_match_legacy_gather():
    optimized = _model(dtype=torch.float64)
    legacy = _copy_model(optimized)
    grid = torch.linspace(-0.12, 0.12, 8, device="cuda", dtype=torch.float64)
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    origins = torch.stack((xx, yy, torch.zeros_like(xx)), dim=-1).reshape(-1, 3)
    directions = torch.stack(
        (0.03 * xx, -0.02 * yy, torch.ones_like(xx)), dim=-1
    ).reshape(-1, 3)
    optimized_origins = origins.detach().clone().requires_grad_(True)
    optimized_directions = directions.detach().clone().requires_grad_(True)
    legacy_origins = origins.detach().clone().requires_grad_(True)
    legacy_directions = directions.detach().clone().requires_grad_(True)

    optimized_outputs, optimized_loss = _raytrace_loss(
        optimized, optimized_origins, optimized_directions
    )
    legacy_outputs, legacy_loss = _raytrace_loss(
        legacy, legacy_origins, legacy_directions, reference_gather=True
    )
    for actual, expected in zip(optimized_outputs[:3], legacy_outputs[:3]):
        assert torch.allclose(actual, expected, atol=2e-10, rtol=2e-10)
    assert torch.equal(optimized_outputs[3], legacy_outputs[3])
    assert torch.allclose(optimized_loss, legacy_loss, atol=2e-10, rtol=2e-10)

    optimized_loss.backward()
    legacy_loss.backward()
    for name in ("_xyz", "_rotation", "_scaling", "_opacity", "_color"):
        assert torch.allclose(
            getattr(optimized, name).grad,
            getattr(legacy, name).grad,
            atol=2e-9,
            rtol=2e-9,
        ), name
    assert torch.allclose(
        optimized_origins.grad, legacy_origins.grad, atol=2e-9, rtol=2e-9
    )
    assert torch.allclose(
        optimized_directions.grad, legacy_directions.grad, atol=2e-9, rtol=2e-9
    )


def _diffuse_package(size=16):
    dtype, device = torch.float32, "cuda"
    coordinates = torch.linspace(-0.2, 0.2, size, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")

    def leaf(value):
        return value.detach().clone().requires_grad_(True)

    position = torch.stack((xx, yy, torch.ones_like(xx)), dim=-1)
    normal = torch.zeros_like(position)
    normal[..., 2] = -1.0
    alpha = torch.ones((size, size, 1), device=device, dtype=dtype)
    leaves = {
        "Cd": leaf(torch.full((size, size, 3), 0.3, device=device, dtype=dtype)),
        "alpha": leaf(alpha),
        "depth": leaf(torch.ones((size, size, 1), device=device, dtype=dtype)),
        "position": leaf(position),
        "normal": leaf(normal),
        "roughness": leaf(0.5 * alpha),
        "f0": leaf(torch.full((size, size, 3), 0.04, device=device, dtype=dtype)),
        "ks": leaf(0.2 * alpha),
    }
    leaves.update(
        {
            "render": leaves["Cd"].permute(2, 0, 1),
            "viewspace_points": torch.zeros(
                (1, 3), device=device, requires_grad=True
            ),
            "visibility_filter": torch.tensor([0], device=device),
            "radii": torch.ones(1, device=device),
        }
    )
    return leaves


def _single_reflector():
    model = ReflectionSurfelModel()
    model._xyz = torch.nn.Parameter(torch.tensor([[0.0, 0.0, 0.5]], device="cuda"))
    model._rotation = torch.nn.Parameter(
        torch.tensor([[1.0, 0.04, -0.02, 0.01]], device="cuda")
    )
    model._scaling = torch.nn.Parameter(
        torch.log(torch.tensor([[1.0, 1.0]], device="cuda"))
    )
    model._opacity = torch.nn.Parameter(torch.tensor([[0.7]], device="cuda"))
    model._color = torch.nn.Parameter(torch.tensor([[0.2, -0.1, 0.4]], device="cuda"))
    model.topology_version = 0
    return model


def _stage_b_loss(package, model, perceptual, reference_gather=False):
    state = StageBRenderState(
        diffuse=object(), reflection=model, scene_radius=1.0, ray_chunk_size=64
    )
    original_render = reflection_renderer.render_diffuse
    original_gather = differentiable_raytrace.gather_candidate_parameters
    reflection_renderer.render_diffuse = lambda *args, **kwargs: package
    if reference_gather:
        differentiable_raytrace.gather_candidate_parameters = (
            lambda table, ids: table[ids]
        )
    try:
        output = reflection_renderer.render(
            SimpleNamespace(camera_center=torch.zeros(3, device="cuda")),
            state,
            SimpleNamespace(),
            torch.zeros(3, device="cuda"),
        )
    finally:
        reflection_renderer.render_diffuse = original_render
        differentiable_raytrace.gather_candidate_parameters = original_gather
    image = output["render"]
    gt = torch.full_like(image, 0.35)
    prior = torch.zeros_like(output["normal"])
    prior[..., 2] = -1.0
    valid = torch.ones_like(output["alpha"], dtype=torch.bool)
    loss = 0.8 * l1_loss(image, gt) + 0.2 * (1.0 - ssim(image, gt))
    loss = loss + 0.04 * normal_depth_consistency_loss(
        output["normal"], output["position"], output["alpha"]
    )
    loss = loss + 0.01 * monocular_normal_loss(
        output["normal"], prior, output["alpha"], valid
    )
    loss = loss + 0.01 * perceptual(image, gt)
    return output, loss


def test_complete_stage_b_loss_and_d_r_gradients_match_legacy_gather():
    torch.manual_seed(31)
    perceptual = VGG16PerceptualLoss(pretrained=False).cuda().eval()
    optimized_package = _diffuse_package()
    legacy_package = {
        name: value.detach().clone().requires_grad_(value.requires_grad)
        if torch.is_tensor(value)
        else value
        for name, value in optimized_package.items()
    }
    optimized_model = _single_reflector()
    legacy_model = _copy_model(optimized_model)

    optimized_output, optimized_loss = _stage_b_loss(
        optimized_package, optimized_model, perceptual
    )
    legacy_output, legacy_loss = _stage_b_loss(
        legacy_package, legacy_model, perceptual, reference_gather=True
    )
    assert torch.allclose(
        optimized_output["render"], legacy_output["render"], atol=2e-6, rtol=2e-6
    )
    assert torch.allclose(optimized_loss, legacy_loss, atol=2e-6, rtol=2e-6)

    optimized_loss.backward()
    legacy_loss.backward()
    for name in ("_xyz", "_rotation", "_scaling", "_opacity", "_color"):
        assert torch.allclose(
            getattr(optimized_model, name).grad,
            getattr(legacy_model, name).grad,
            atol=3e-5,
            rtol=3e-4,
        ), name
    for name in ("Cd", "alpha", "position", "normal", "roughness", "f0", "ks"):
        assert torch.allclose(
            optimized_package[name].grad,
            legacy_package[name].grad,
            atol=3e-5,
            rtol=3e-4,
        ), name
