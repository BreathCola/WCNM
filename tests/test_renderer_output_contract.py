from types import SimpleNamespace

import pytest
import torch

from utils.graphics_utils import getProjectionMatrix


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA rasterizer test")


class _Camera:
    image_height = 24
    image_width = 32
    FoVx = 0.8
    FoVy = 0.7
    image_name = "toy"

    def __init__(self):
        self.world_view_transform = torch.eye(4, device="cuda")
        self.projection_matrix = getProjectionMatrix(0.01, 100.0, self.FoVx, self.FoVy).transpose(0, 1).cuda()
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = torch.zeros(3, device="cuda")


def _model():
    from scene.diffuse_surfel_model import DiffuseSurfelModel

    model = DiffuseSurfelModel()
    model._xyz = torch.nn.Parameter(torch.tensor([[0.0, 0.0, 2.0]], device="cuda"))
    model._rotation = torch.nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda"))
    model._scaling = torch.nn.Parameter(torch.log(torch.tensor([[0.7, 0.6]], device="cuda")))
    model._opacity = torch.nn.Parameter(torch.tensor([[2.0]], device="cuda"))
    model._base_color = torch.nn.Parameter(torch.tensor([[0.2, -0.1, 0.4]], device="cuda"))
    model._roughness = torch.nn.Parameter(torch.tensor([[0.1]], device="cuda"))
    model._f0 = torch.nn.Parameter(torch.tensor([[-2.0, -2.5, -3.0]], device="cuda"))
    model._ks = torch.nn.Parameter(torch.tensor([[-0.5]], device="cuda"))
    return model


def test_renderer_contract_is_finite_and_differentiable():
    pytest.importorskip("diff_surfel_rasterization")
    from gaussian_renderer.surfel_renderer import OUTPUT_CONTRACT, render

    model = _model()
    camera = _Camera()
    pipe = SimpleNamespace(debug=False, compute_cov3D_python=False)
    output = render(camera, model, pipe, torch.zeros(3, device="cuda"))

    for name, channels in OUTPUT_CONTRACT.items():
        assert output[name].shape == (camera.image_height, camera.image_width, channels)
        assert torch.isfinite(output[name]).all()
    visible = output["alpha"][..., 0] > 1e-4
    assert visible.any()
    assert torch.allclose(output["normal"][visible].norm(dim=-1), torch.ones_like(output["alpha"][..., 0][visible]), atol=1e-4)
    wo = torch.nn.functional.normalize(camera.camera_center - output["position"][visible], dim=-1)
    assert ((output["normal"][visible] * wo).sum(dim=-1) >= -1e-5).all()
    assert torch.allclose(
        output["position"][..., 2:3][visible], output["depth"][visible], atol=1e-4, rtol=1e-4
    )

    alpha = output["alpha"]
    assert torch.allclose(output["roughness"], alpha * model.get_roughness[0], atol=1e-5, rtol=1e-4)
    assert torch.allclose(output["f0"], alpha * model.get_f0[0], atol=1e-5, rtol=1e-4)
    assert torch.allclose(output["ks"], alpha * model.get_ks[0], atol=1e-5, rtol=1e-4)

    loss = (
        output["Cd"].sum()
        + output["depth"].sum() * 1e-3
        + output["normal"][..., 0].sum()
        + output["roughness"].sum()
        + output["f0"].sum()
        + output["ks"].sum()
    )
    loss.backward()
    for parameter in (
        model._xyz,
        model._rotation,
        model._scaling,
        model._opacity,
        model._base_color,
        model._roughness,
        model._f0,
        model._ks,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
