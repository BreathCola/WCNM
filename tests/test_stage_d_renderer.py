from types import SimpleNamespace

import numpy as np
import pytest
import torch

import gaussian_renderer.transmittance_renderer as renderer
from gaussian_renderer.transmittance_renderer import StageDRenderState
from scene.diffuse_surfel_model import DiffuseSurfelModel
from scene.transmittance_surfel_model import TransmittanceSurfelModel


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Stage D renderer is CUDA-only")


class _Geometry:
    def load_view(self, stem):
        assert stem == "toy"
        return {
            "valid_two_hit": np.array([[True, False]]),
            "t_near": np.array([[1.0, 0.0]], np.float32),
            "t_far": np.array([[2.0, 0.0]], np.float32),
            "back_position": np.array([[[0, 0, 2.0], [0, 0, 0]]], np.float32),
        }


def _ray_model(model, z, color):
    model._xyz = torch.nn.Parameter(torch.tensor([[0.0, 0.0, z]], device="cuda"))
    model._rotation = torch.nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda"))
    model._scaling = torch.nn.Parameter(torch.log(torch.tensor([[0.8, 0.8]], device="cuda")))
    model._opacity = torch.nn.Parameter(torch.tensor([[2.0]], device="cuda"))
    if isinstance(model, DiffuseSurfelModel):
        model._base_color = torch.nn.Parameter(torch.tensor([color], device="cuda"))
    else:
        model._color = torch.nn.Parameter(torch.tensor([color], device="cuda"))
        model.topology_version = 0
    return model


def _stage_b_package():
    zeros3 = torch.zeros((1, 2, 3), device="cuda")
    zeros1 = torch.zeros((1, 2, 1), device="cuda")
    position = torch.tensor([[[0.0, 0.0, 1.0], [0, 0, 0]]], device="cuda", requires_grad=True)
    return {
        "alpha": torch.tensor([[[1.0], [0.0]]], device="cuda", requires_grad=True),
        "position": position,
        "surface_ks": torch.tensor([[[0.8], [0.0]]], device="cuda", requires_grad=True),
        "microfacet_F": torch.full((1, 2, 3), 0.04, device="cuda", requires_grad=True),
        "diffuse_contribution": zeros3.clone(),
        "reflection_contribution": zeros3.clone(),
        "ray_aux": None,
        "normal": torch.tensor([[[0.0, 0.0, -1.0], [0, 0, 0]]], device="cuda"),
        "depth": torch.tensor([[[1.0], [0.0]]], device="cuda"),
        "Cd": zeros3.clone(), "roughness": zeros1.clone(), "f0": zeros3.clone(),
        "ks": zeros1.clone(), "render": zeros3.permute(2, 0, 1),
    }


def test_stage_d_renderer_traces_both_bounces_and_gradients(monkeypatch):
    monkeypatch.setattr(renderer, "render_stage_b", lambda *args, **kwargs: _stage_b_package())
    diffuse = _ray_model(DiffuseSurfelModel(), 3.0, [0.3, -0.2, 0.1])
    transmittance = _ray_model(TransmittanceSurfelModel(), 1.5, [0.1, 0.2, -0.1])
    state = StageDRenderState(
        diffuse=diffuse, reflection=object(), transmittance=transmittance,
        geometry_release=_Geometry(), scene_radius=1.0, ray_chunk_size=8,
        ray_checkpoint_chunks=False,
    )
    camera = SimpleNamespace(camera_center=torch.zeros(3, device="cuda"), image_name="toy")
    output = renderer.render(camera, state, SimpleNamespace(), torch.zeros(3, device="cuda"))
    assert output["transmittance_valid_count"] == 1
    assert output["inside_alpha"][0, 0, 0] > 0
    assert output["outside_alpha"][0, 0, 0] > 0
    assert output["inside_depth"][0, 0, 0] < output["far_depth"][0, 0, 0]
    assert torch.isfinite(output["final"]).all()
    output["final"].sum().backward()
    assert transmittance._color.grad is not None
    assert diffuse._base_color.grad is not None
    assert torch.isfinite(transmittance._color.grad).all()
    assert torch.isfinite(diffuse._base_color.grad).all()
