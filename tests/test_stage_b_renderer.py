from types import SimpleNamespace

import pytest
import torch

import gaussian_renderer.reflection_renderer as reflection_renderer
from gaussian_renderer.reflection_renderer import StageBRenderState
from scene.reflection_surfel_model import ReflectionSurfelModel


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Stage B renderer is CUDA-only")


def _reflection():
    model = ReflectionSurfelModel()
    model._xyz = torch.nn.Parameter(torch.tensor([[0.0, 0.0, 0.5]], device="cuda"))
    model._rotation = torch.nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda"))
    model._scaling = torch.nn.Parameter(torch.log(torch.tensor([[0.8, 0.8]], device="cuda")))
    model._opacity = torch.nn.Parameter(torch.tensor([[1.0]], device="cuda"))
    model._color = torch.nn.Parameter(torch.tensor([[0.2, -0.1, 0.4]], device="cuda"))
    model.topology_version = 0
    return model


def _diffuse_package():
    device = "cuda"
    leaves = {
        "Cd": torch.tensor([[[0.4, 0.3, 0.2], [0.0, 0.0, 0.0]]], device=device, requires_grad=True),
        "alpha": torch.tensor([[[1.0], [0.0]]], device=device, requires_grad=True),
        "depth": torch.tensor([[[1.0], [0.0]]], device=device, requires_grad=True),
        "position": torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]], device=device, requires_grad=True),
        "normal": torch.tensor([[[0.0, 0.0, -1.0], [0.0, 0.0, 0.0]]], device=device, requires_grad=True),
        "roughness": torch.tensor([[[0.5], [0.0]]], device=device, requires_grad=True),
        "f0": torch.tensor([[[0.04, 0.04, 0.04], [0.0, 0.0, 0.0]]], device=device, requires_grad=True),
        "ks": torch.tensor([[[0.2], [0.0]]], device=device, requires_grad=True),
    }
    leaves.update(
        {
            "render": leaves["Cd"].permute(2, 0, 1),
            "viewspace_points": torch.zeros((1, 3), device=device, requires_grad=True),
            "visibility_filter": torch.tensor([0], device=device),
            "radii": torch.ones(1, device=device),
        }
    )
    return leaves


def test_stage_b_renderer_composes_real_reflection_and_gradients(monkeypatch):
    package = _diffuse_package()
    monkeypatch.setattr(reflection_renderer, "render_diffuse", lambda *args, **kwargs: package)
    state = StageBRenderState(diffuse=object(), reflection=_reflection(), scene_radius=1.0, ray_chunk_size=8)
    camera = SimpleNamespace(camera_center=torch.zeros(3, device="cuda"))
    output = reflection_renderer.render(
        camera,
        state,
        SimpleNamespace(),
        torch.zeros(3, device="cuda"),
        return_ray_aux=True,
        return_ray_diagnostics=True,
    )
    required = {
        "final", "reflection_color", "reflection_alpha", "reflection_depth", "reflection_hit_mask",
        "microfacet_D", "microfacet_F", "microfacet_G", "microfacet_fr", "microfacet_wr",
        "diffuse_contribution", "reflection_contribution",
        "ray_candidate_count", "ray_exact_intersection_count", "ray_diagnostics",
    }
    assert required <= set(output)
    assert output["valid_ray_count"] == 1
    assert output["reflection_hit_mask"][0, 0, 0] == 1
    assert output["ray_candidate_count"][0, 0, 0] >= output["ray_exact_intersection_count"][0, 0, 0]
    assert output["ray_exact_intersection_count"][0, 0, 0] >= 1
    assert output["ray_diagnostics"].reflection_surfel_count == 1
    assert torch.allclose(
        output["final"],
        (output["diffuse_contribution"] + output["reflection_contribution"]).clamp(0, 1),
        atol=1e-6,
    )
    assert not any(
        token in key for key in output for token in ("transmittance", "inside", "outside", "two_hit", "near_depth", "far_depth", "depth_violation")
    )
    output["final"].sum().backward()
    for parameter in (
        state.reflection._xyz,
        state.reflection._rotation,
        state.reflection._scaling,
        state.reflection._opacity,
        state.reflection._color,
    ):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_no_valid_surface_skips_raytrace_and_returns_background(monkeypatch):
    package = _diffuse_package()
    package["alpha"] = torch.zeros_like(package["alpha"])
    package["depth"] = torch.zeros_like(package["depth"])
    monkeypatch.setattr(reflection_renderer, "render_diffuse", lambda *args, **kwargs: package)
    state = StageBRenderState(diffuse=object(), reflection=_reflection(), scene_radius=1.0)
    camera = SimpleNamespace(camera_center=torch.zeros(3, device="cuda"))
    background = torch.ones(3, device="cuda")
    output = reflection_renderer.render(camera, state, SimpleNamespace(), background)
    assert output["valid_ray_count"] == 0
    assert torch.equal(output["final"], torch.ones_like(output["final"]))
    assert not state.acceleration.rebuild_count
