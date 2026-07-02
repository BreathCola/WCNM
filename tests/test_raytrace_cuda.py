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


def test_checkpointed_chunks_match_outputs_gradients_and_detach_aux():
    plain_model = _model()
    checkpoint_model = _model()
    for name in ("_xyz", "_rotation", "_scaling", "_opacity", "_color"):
        setattr(
            checkpoint_model,
            name,
            torch.nn.Parameter(getattr(plain_model, name).detach().clone()),
        )
    origins = torch.tensor(
        [[0.05, 0.02, 0.0], [0.4, 0.0, 0.0], [4.0, 4.0, 0.0]],
        device="cuda",
        requires_grad=True,
    )
    directions = torch.tensor(
        [[0.02, 0.01, 1.0], [-0.03, 0.0, 1.0], [0.0, 0.0, 1.0]],
        device="cuda",
        requires_grad=True,
    )
    retry_origins = origins.detach().clone().requires_grad_(True)
    retry_directions = directions.detach().clone().requires_grad_(True)
    plain, plain_aux = raytrace(
        plain_model, origins, directions, chunk_size=2, return_aux=True
    )
    bounded, bounded_aux = raytrace(
        checkpoint_model,
        retry_origins,
        retry_directions,
        chunk_size=2,
        return_aux=True,
        checkpoint_chunks=True,
    )
    for actual, expected in zip(bounded[:3], plain[:3]):
        assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-6)
    assert torch.equal(bounded[3], plain[3])
    assert plain_aux.contributing_weights.requires_grad is False
    assert bounded_aux.contributing_weights.requires_grad is False
    weights = (0.7, 0.4, 0.2)
    plain_loss = sum(weight * value.sum() for weight, value in zip(weights, plain[:3]))
    bounded_loss = sum(weight * value.sum() for weight, value in zip(weights, bounded[:3]))
    plain_loss.backward()
    bounded_loss.backward()
    for name in ("_xyz", "_rotation", "_scaling", "_opacity", "_color"):
        assert torch.allclose(
            getattr(checkpoint_model, name).grad,
            getattr(plain_model, name).grad,
            atol=1e-6,
            rtol=1e-5,
        ), name
    assert torch.allclose(retry_origins.grad, origins.grad, atol=1e-6, rtol=1e-5)
    assert torch.allclose(retry_directions.grad, directions.grad, atol=1e-6, rtol=1e-5)


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


def test_raytrace_diagnostics_report_candidates_exact_hits_timing_and_memory():
    model = _model()
    origins = torch.tensor(
        [[0.05, 0.02, 0.0], [0.4, 0.0, 0.0], [4.0, 4.0, 0.0]], device="cuda"
    )
    directions = torch.tensor(
        [[0.02, 0.01, 1.0], [-0.03, 0.0, 1.0], [0.0, 0.0, 1.0]], device="cuda"
    )
    outputs, diagnostics = raytrace(
        model, origins, directions, chunk_size=2, return_diagnostics=True
    )
    plain_outputs = raytrace(model, origins, directions, chunk_size=1)
    for diagnosed, plain in zip(outputs[:3], plain_outputs[:3]):
        assert torch.allclose(diagnosed, plain, atol=1e-7, rtol=1e-6)
    assert torch.equal(outputs[3], plain_outputs[3])

    _, single_chunk_diagnostics = raytrace(
        model, origins, directions, chunk_size=3, return_diagnostics=True
    )

    assert diagnostics.candidate_counts.shape == (3,)
    assert diagnostics.exact_intersection_counts.shape == (3,)
    assert torch.all(diagnostics.candidate_counts >= diagnostics.exact_intersection_counts)
    assert torch.all(diagnostics.exact_intersection_counts >= outputs[3][:, 0])
    assert torch.equal(diagnostics.candidate_counts, single_chunk_diagnostics.candidate_counts)
    assert torch.equal(
        diagnostics.exact_intersection_counts,
        single_chunk_diagnostics.exact_intersection_counts,
    )
    assert diagnostics.chunk_count == 2
    assert diagnostics.reflection_surfel_count == 2
    assert diagnostics.bvh_rebuild_delta == 1
    assert diagnostics.bvh_refit_delta == 0
    assert diagnostics.peak_memory_allocated_bytes > 0
    assert diagnostics.peak_memory_delta_bytes >= 0
    assert set(diagnostics.timing_ms) == {
        "bvh_sync_cuda", "traversal_cuda", "intersection_composite_cuda", "raytrace_wall"
    }
    assert all(torch.isfinite(torch.tensor(value)) and value >= 0 for value in diagnostics.timing_ms.values())


def test_production_tracer_is_fail_closed_on_cpu():
    model = ReflectionSurfelModel()
    model._xyz = torch.nn.Parameter(torch.zeros((1, 3)))
    model._rotation = torch.nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    model._scaling = torch.nn.Parameter(torch.zeros((1, 2)))
    model._opacity = torch.nn.Parameter(torch.zeros((1, 1)))
    model._color = torch.nn.Parameter(torch.zeros((1, 3)))
    with pytest.raises(RuntimeError, match="CUDA-only"):
        raytrace(model, torch.zeros((1, 3)), torch.ones((1, 3)))
