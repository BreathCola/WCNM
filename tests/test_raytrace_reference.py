import torch

from raytracer.reference import raytrace_bruteforce
from scene.reflection_surfel_model import ReflectionSurfelModel


def _single_surfel():
    model = ReflectionSurfelModel()
    model._xyz = torch.nn.Parameter(torch.tensor([[0.0, 0.0, 1.0]]))
    model._rotation = torch.nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    model._scaling = torch.nn.Parameter(torch.log(torch.tensor([[1.0, 1.0]])))
    model._opacity = torch.nn.Parameter(torch.tensor([[0.0]]))
    model._color = torch.nn.Parameter(torch.tensor([[0.0, 0.0, 0.0]]))
    return model


def test_reference_contract_hit_miss_depth_and_premultiplied_color():
    model = _single_surfel()
    origins = torch.tensor([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    directions = torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 1.0]])
    color, alpha, depth, hit = raytrace_bruteforce(model, origins, directions, chunk_size=1)
    assert torch.allclose(alpha, torch.tensor([[0.5], [0.0]]), atol=1e-6)
    assert torch.allclose(color, torch.tensor([[0.25, 0.25, 0.25], [0.0, 0.0, 0.0]]), atol=1e-6)
    assert torch.allclose(depth, torch.tensor([[1.0], [0.0]]), atol=1e-6)
    assert hit.tolist() == [[True], [False]]


def test_reference_chunking_is_equivalent_and_differentiable():
    model = _single_surfel()
    origins = torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], requires_grad=True)
    directions = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], requires_grad=True)
    one = raytrace_bruteforce(model, origins, directions, chunk_size=1)
    two = raytrace_bruteforce(model, origins, directions, chunk_size=8)
    for lhs, rhs in zip(one[:3], two[:3]):
        assert torch.allclose(lhs, rhs)
    sum(value.sum() for value in one[:3]).backward()
    for value in (model._xyz, model._rotation, model._scaling, model._opacity, model._color, origins, directions):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
