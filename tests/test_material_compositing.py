import torch

from scene.diffuse_surfel_model import DiffuseSurfelModel
from utils.surfel_utils import alpha_composite


def test_material_maps_share_identical_alpha_weights():
    alpha = torch.tensor([[0.2], [0.5], [0.75]])
    roughness = torch.tensor([[0.2], [0.6], [0.9]])
    f0 = torch.tensor([[0.04, 0.05, 0.06], [0.1, 0.2, 0.3], [0.7, 0.8, 0.9]])
    ks = torch.tensor([[0.1], [0.4], [0.8]])

    expected_weights = torch.tensor([[0.2], [0.4], [0.3]])
    assert torch.allclose(alpha_composite(roughness, alpha), (roughness * expected_weights).sum(0))
    assert torch.allclose(alpha_composite(f0, alpha), (f0 * expected_weights).sum(0))
    assert torch.allclose(alpha_composite(ks, alpha), (ks * expected_weights).sum(0))


def test_material_activations_are_bounded_and_finite():
    model = DiffuseSurfelModel(roughness_min=0.03)
    model._roughness = torch.nn.Parameter(torch.tensor([[-100.0], [0.0], [100.0]]))
    model._f0 = torch.nn.Parameter(torch.tensor([[-100.0] * 3, [0.0] * 3, [100.0] * 3]))
    model._ks = torch.nn.Parameter(torch.tensor([[-100.0], [0.0], [100.0]]))

    for value in (model.get_roughness, model.get_f0, model.get_ks):
        assert torch.isfinite(value).all()
    assert (model.get_roughness >= 0.03).all()
    assert (model.get_roughness <= 1.0).all()
    assert (model.get_f0 >= 0.0).all() and (model.get_f0 <= 1.0).all()
    assert (model.get_ks >= 0.0).all() and (model.get_ks <= 1.0).all()
