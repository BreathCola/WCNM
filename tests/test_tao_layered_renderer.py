import torch

from gaussian_renderer.tao_layered_renderer import (
    FORMAL_TRACE_NAMES, TraceSample, render_tao_layered_v1,
    run_tao_layered_review_diagnostics,
)
from utils.tao_layered_exports import layered_export_tensors, save_tao_layered_exports


def _render(*, cin=0.2, ain=0.5, f0=0.0):
    shape = (1, 1, 3)
    direction = torch.tensor([[[0.0, 0.0, 1.0]]])
    front_normal = torch.tensor([[[0.0, 0.0, -1.0]]])
    back_normal = torch.tensor([[[0.0, 0.0, 1.0]]])
    calls = []
    values = {
        "R_front": (0.3, 0.4), "T_direct": (cin, ain),
        "R_back_from_T": (0.4, 0.6), "Cout": (0.5, 0.7),
    }

    def trace(request):
        calls.append(request)
        rgb, alpha = values[request.name]
        return TraceSample(
            torch.full(shape, rgb, requires_grad=True),
            torch.full((1, 1, 1), alpha, requires_grad=True),
            request.radiance_namespace, request.ownership,
        )

    package = render_tao_layered_v1(
        camera_direction=direction, front_position=torch.zeros(shape),
        front_normal=front_normal, back_position=torch.tensor([[[0.0, 0.0, 2.0]]]),
        back_normal=back_normal, interface_alpha=torch.ones((1, 1, 1)),
        interface_ks=torch.ones((1, 1, 1)),
        interface_f0=torch.full((1, 1, 1), f0),
        d_direct=torch.full(shape, 0.1), uncovered_background=torch.full(shape, 0.05),
        trace_reflection=trace, trace_transmittance=trace, trace_diffuse=trace,
    )
    return package, calls


def test_formal_trace_count_ownership_and_back_direction():
    package, calls = _render()
    assert tuple(call.name for call in calls) == FORMAL_TRACE_NAMES
    assert [(call.radiance_namespace, call.ownership) for call in calls] == [
        ("R", "strict_outside"), ("T", "strict_inside"),
        ("T", "strict_inside"), ("D", "strict_outside"),
    ]
    assert package["formal_trace_count"] == 4
    assert calls[0].directions[0, 0, 2] < 0
    assert calls[2].directions[0, 0, 2] < 0
    assert calls[3].directions[0, 0, 2] > 0


def test_premultiplied_cin_is_not_multiplied_by_ain_and_linear_closure_holds():
    package, _ = _render(cin=0.2, ain=0.5, f0=0.0)
    assert torch.allclose(package["internal_only"], torch.full((1, 1, 3), 0.2))
    assert torch.allclose(package["internal_intrinsic_rgb"], torch.full((1, 1, 3), 0.4))
    assert torch.equal(package["internal_alpha"], package["Ain"])
    closure = package["final"] - package["no_reflection"] - package["reflection_only"]
    assert float(closure.abs().max()) < 1e-7
    assert torch.equal(package["linear_closure_error"], torch.zeros_like(closure))
    package["final"].sum().backward()


def test_review_diagnostics_are_detached_from_training_backward():
    parameter = torch.tensor(2.0, requires_grad=True)
    outputs = run_tao_layered_review_diagnostics({"unfiltered": lambda: parameter * 3})
    assert outputs["unfiltered"].requires_grad is False
    assert parameter.grad is None


def test_versioned_export_preserves_float_and_closure(tmp_path):
    package, _ = _render()
    ground_truth = torch.full((3, 1, 1), 1.25)
    exports = layered_export_tensors(package, ground_truth)
    assert float(exports["ground_truth"].max()) == 1.25
    target = tmp_path / "view"
    metadata = save_tao_layered_exports(package, ground_truth, target)
    saved = torch.load(target / "ground_truth.pt", weights_only=True)
    assert float(saved.max()) == 1.25
    assert metadata["linear_closure_max_abs"] < 1e-7
