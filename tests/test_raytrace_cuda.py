import pytest
import torch

from raytracer.acceleration_structure import CudaLBVH, load_cuda_extension
from raytracer.reference import raytrace_bruteforce
from raytracer.tracer import raytrace
from scene.reflection_surfel_model import ReflectionSurfelModel


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Stage B production tracer is CUDA-only")


def _model(dtype=torch.float32):
    device = "cuda"
    model = ReflectionSurfelModel()
    model._xyz = torch.nn.Parameter(torch.tensor([[0.0, 0.0, 1.0], [0.3, 0.0, 2.0]], device=device, dtype=dtype))
    model._rotation = torch.nn.Parameter(
        torch.tensor([[1.0, 0.05, 0.02, 0.0], [1.0, -0.03, 0.0, 0.02]], device=device, dtype=dtype)
    )
    model._scaling = torch.nn.Parameter(torch.log(torch.tensor([[0.8, 0.6], [0.7, 0.5]], device=device, dtype=dtype)))
    model._opacity = torch.nn.Parameter(torch.tensor([[0.2], [-0.1]], device=device, dtype=dtype))
    model._color = torch.nn.Parameter(torch.tensor([[0.1, -0.2, 0.3], [-0.3, 0.2, 0.1]], device=device, dtype=dtype))
    model.topology_version = 0
    return model


def test_cuda_extension_loads_and_cuda_matches_reference_with_chunking():
    assert load_cuda_extension() is not None
    model = _model()
    origins = torch.tensor(
        [[0.05, 0.02, 0.0], [0.4, 0.0, 0.0], [4.0, 4.0, 0.0]], device="cuda"
    )
    directions = torch.tensor(
        [[0.02, 0.01, 1.0], [-0.03, 0.0, 1.0], [0.0, 0.0, 1.0]], device="cuda"
    )
    acceleration = CudaLBVH()
    cuda = raytrace(model, origins, directions, acceleration=acceleration, chunk_size=1)
    oracle = raytrace_bruteforce(model, origins, directions, chunk_size=8)
    for actual, expected in zip(cuda[:3], oracle[:3]):
        assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-5)
    assert torch.equal(cuda[3], oracle[3])
    second = raytrace(model, origins, directions, acceleration=acceleration, chunk_size=8)
    for actual, expected in zip(cuda[:3], second[:3]):
        assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-6)


def test_lbvh_refits_parameter_updates_and_rebuilds_topology_updates():
    model = _model()
    acceleration = CudaLBVH()
    acceleration.ensure_current(model)
    assert acceleration.rebuild_count == 1 and acceleration.refit_count == 0
    with torch.no_grad():
        model._xyz[0, 0] += 0.01
    acceleration.ensure_current(model)
    assert acceleration.rebuild_count == 1 and acceleration.refit_count == 1
    model.topology_version += 1
    acceleration.ensure_current(model)
    assert acceleration.rebuild_count == 2


def test_production_tracer_is_fail_closed_on_cpu():
    model = ReflectionSurfelModel()
    model._xyz = torch.nn.Parameter(torch.zeros((1, 3)))
    model._rotation = torch.nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    model._scaling = torch.nn.Parameter(torch.zeros((1, 2)))
    model._opacity = torch.nn.Parameter(torch.zeros((1, 1)))
    model._color = torch.nn.Parameter(torch.zeros((1, 3)))
    with pytest.raises(RuntimeError, match="CUDA-only"):
        raytrace(model, torch.zeros((1, 3)), torch.ones((1, 3)))
