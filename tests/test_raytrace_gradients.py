import pytest
import torch

from raytracer.acceleration_structure import CudaLBVH
from raytracer.tracer import raytrace
from scene.reflection_surfel_model import ReflectionSurfelModel


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Stage B gradient test is CUDA-only")


def _scene():
    device, dtype = "cuda", torch.float64
    model = ReflectionSurfelModel()
    model._xyz = torch.nn.Parameter(torch.tensor([[0.04, -0.03, 1.2]], device=device, dtype=dtype))
    model._rotation = torch.nn.Parameter(torch.tensor([[1.0, 0.12, -0.08, 0.04]], device=device, dtype=dtype))
    model._scaling = torch.nn.Parameter(torch.log(torch.tensor([[0.75, 0.55]], device=device, dtype=dtype)))
    model._opacity = torch.nn.Parameter(torch.tensor([[0.15]], device=device, dtype=dtype))
    model._color = torch.nn.Parameter(torch.tensor([[0.2, -0.25, 0.35]], device=device, dtype=dtype))
    model.topology_version = 0
    origin = torch.tensor([[0.11, 0.04, 0.0]], device=device, dtype=dtype, requires_grad=True)
    direction = torch.tensor([[0.035, -0.018, 1.0]], device=device, dtype=dtype, requires_grad=True)
    return model, origin, direction


def _loss(model, origin, direction, acceleration):
    color, alpha, depth, _ = raytrace(
        model, origin, direction, acceleration=acceleration, chunk_size=1
    )
    return color @ torch.tensor([[0.7], [1.1], [1.3]], device=color.device, dtype=color.dtype) + 0.4 * alpha + 0.2 * depth


def _finite_difference(model, origin, direction, acceleration, tensor, index, delta=1e-5):
    with torch.no_grad():
        original = tensor[index].item()
        tensor[index] = original + delta
    plus = _loss(model, origin, direction, acceleration).item()
    with torch.no_grad():
        tensor[index] = original - delta
    minus = _loss(model, origin, direction, acceleration).item()
    with torch.no_grad():
        tensor[index] = original
    return (plus - minus) / (2.0 * delta)


def test_cuda_raytrace_gradients_match_finite_difference_for_all_required_inputs():
    model, origin, direction = _scene()
    acceleration = CudaLBVH()
    loss = _loss(model, origin, direction, acceleration).sum()
    loss.backward()
    checks = [
        ("xyz", model._xyz, (0, 0)),
        ("rotation", model._rotation, (0, 1)),
        ("scaling", model._scaling, (0, 0)),
        ("opacity", model._opacity, (0, 0)),
        ("color", model._color, (0, 0)),
        ("ray_origin", origin, (0, 0)),
        ("ray_direction", direction, (0, 0)),
    ]
    for name, tensor, index in checks:
        analytic = tensor.grad[index].item()
        numeric = _finite_difference(model, origin, direction, acceleration, tensor, index)
        assert torch.isfinite(torch.tensor(analytic)), name
        assert abs(analytic) > 1e-7, (name, analytic)
        assert abs(analytic - numeric) <= 2e-3 + 2e-2 * abs(numeric), (name, analytic, numeric)
