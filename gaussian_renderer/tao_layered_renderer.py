"""Tao layered early-joint renderer contract, version 1.

This module is intentionally separate from the TiHuBird Stage-D renderer.  It
owns the four formal Gaussian traces and the unclamped linear composition, but
does not own a particular ray tracer implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import torch
import torch.nn.functional as F


LAYERED_RENDERER_SCHEMA = "rtgs_tao_layered_renderer_v1"
FORMAL_TRACE_NAMES = ("R_front", "T_direct", "R_back_from_T", "Cout")


@dataclass(frozen=True)
class TraceRequest:
    """One ownership-qualified black-background Gaussian trace."""

    name: str
    radiance_namespace: str
    ownership: str
    origins: torch.Tensor
    directions: torch.Tensor
    background: str = "black"


@dataclass(frozen=True)
class TraceSample:
    """Premultiplied result returned by a formal trace callback."""

    raw_rgb: torch.Tensor
    alpha: torch.Tensor
    radiance_namespace: str
    ownership: str
    background: str = "black"


TraceCallback = Callable[[TraceRequest], TraceSample]


def reflect(direction: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
    """Reflect travel direction about a unit surface normal."""
    direction = F.normalize(direction, dim=-1, eps=1e-8)
    normal = F.normalize(normal, dim=-1, eps=1e-8)
    return F.normalize(
        direction - 2.0 * (direction * normal).sum(dim=-1, keepdim=True) * normal,
        dim=-1,
        eps=1e-8,
    )


def schlick_fresnel(
    f0: torch.Tensor,
    travel_direction: torch.Tensor,
    surface_normal: torch.Tensor,
) -> torch.Tensor:
    """Schlick Fresnel for either oriented face of a closed interface."""
    travel_direction = F.normalize(travel_direction, dim=-1, eps=1e-8)
    surface_normal = F.normalize(surface_normal, dim=-1, eps=1e-8)
    cosine = (travel_direction * surface_normal).sum(dim=-1, keepdim=True).abs().clamp(0.0, 1.0)
    return f0 + (1.0 - f0) * (1.0 - cosine).pow(5)


def _assert_rgb(name: str, value: torch.Tensor, reference: torch.Tensor) -> None:
    if value.shape != reference.shape or value.shape[-1] != 3:
        raise ValueError(f"{name} must match RGB shape {tuple(reference.shape)}, got {tuple(value.shape)}")
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} contains NaN or Inf")


def _assert_scalar_or_rgb(name: str, value: torch.Tensor, reference: torch.Tensor) -> None:
    if value.shape not in (reference.shape[:-1] + (1,), reference.shape):
        raise ValueError(f"{name} must be one-channel or RGB alongside {tuple(reference.shape)}")
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} contains NaN or Inf")


def _run_trace(callback: TraceCallback, request: TraceRequest, reference: torch.Tensor) -> TraceSample:
    if request.background != "black":
        raise ValueError("formal layered traces require black background extraction")
    sample = callback(request)
    if not isinstance(sample, TraceSample):
        raise TypeError(f"{request.name} callback must return TraceSample")
    if sample.background != "black":
        raise ValueError(f"{request.name} did not return a black-background extraction")
    if sample.radiance_namespace != request.radiance_namespace:
        raise ValueError(f"{request.name} queried the wrong radiance namespace")
    if sample.ownership != request.ownership:
        raise ValueError(f"{request.name} violated support ownership")
    _assert_rgb(f"{request.name}.raw_rgb", sample.raw_rgb, reference)
    expected_alpha = reference.shape[:-1] + (1,)
    if sample.alpha.shape != expected_alpha or not torch.isfinite(sample.alpha).all():
        raise ValueError(f"{request.name}.alpha must have shape {expected_alpha}")
    return sample


def render_tao_layered_v1(
    *,
    camera_direction: torch.Tensor,
    front_position: torch.Tensor,
    front_normal: torch.Tensor,
    back_position: torch.Tensor,
    back_normal: torch.Tensor,
    interface_alpha: torch.Tensor,
    interface_ks: torch.Tensor,
    interface_f0: torch.Tensor,
    d_direct: torch.Tensor,
    uncovered_background: torch.Tensor,
    trace_reflection: TraceCallback,
    trace_transmittance: TraceCallback,
    trace_diffuse: TraceCallback,
    ray_epsilon: float = 1e-4,
) -> dict[str, torch.Tensor | tuple[TraceRequest, ...] | int | str]:
    """Run the four formal traces and compose the one-bounce layered result.

    ``raw_rgb`` from every callback is already alpha-premultiplied over black.
    In particular, ``Cin`` is never multiplied by ``Ain`` a second time.
    """
    reference = d_direct
    for name, value in (
        ("camera_direction", camera_direction), ("front_position", front_position),
        ("front_normal", front_normal), ("back_position", back_position),
        ("back_normal", back_normal), ("uncovered_background", uncovered_background),
    ):
        _assert_rgb(name, value, reference)
    for name, value in (
        ("interface_alpha", interface_alpha), ("interface_ks", interface_ks),
        ("interface_f0", interface_f0),
    ):
        _assert_scalar_or_rgb(name, value, reference)
    if ray_epsilon <= 0.0:
        raise ValueError("ray_epsilon must be positive")

    direction = F.normalize(camera_direction, dim=-1, eps=1e-8)
    front_reflected = reflect(direction, front_normal)
    back_reflected = reflect(direction, back_normal)
    requests = (
        TraceRequest(
            "R_front", "R", "strict_outside",
            front_position + float(ray_epsilon) * front_reflected, front_reflected,
        ),
        TraceRequest(
            "T_direct", "T", "strict_inside",
            front_position + float(ray_epsilon) * direction, direction,
        ),
        TraceRequest(
            "R_back_from_T", "T", "strict_inside",
            back_position + float(ray_epsilon) * back_reflected, back_reflected,
        ),
        TraceRequest(
            "Cout", "D", "strict_outside",
            back_position + float(ray_epsilon) * direction, direction,
        ),
    )
    r_front = _run_trace(trace_reflection, requests[0], reference)
    t_direct = _run_trace(trace_transmittance, requests[1], reference)
    r_back = _run_trace(trace_transmittance, requests[2], reference)
    cout = _run_trace(trace_diffuse, requests[3], reference)

    f_front = schlick_fresnel(interface_f0, direction, front_normal)
    f_back = schlick_fresnel(interface_f0, direction, back_normal)
    alpha_ks = interface_alpha * interface_ks
    cin = t_direct.raw_rgb
    ain = t_direct.alpha

    internal = alpha_ks * (1.0 - f_front) * cin
    front_only = alpha_ks * f_front * r_front.raw_rgb
    back_only = alpha_ks * (1.0 - f_front) * (1.0 - ain) * f_back * r_back.raw_rgb
    cout_only = alpha_ks * (1.0 - f_front) * (1.0 - ain) * (1.0 - f_back) * cout.raw_rgb
    reflection_only = front_only + back_only
    no_reflection = d_direct + internal + cout_only + uncovered_background
    final = no_reflection + reflection_only
    closure = final - (no_reflection + reflection_only)
    intrinsic = torch.where(ain > 1e-8, cin / ain.clamp_min(1e-8), torch.zeros_like(cin))

    result: dict[str, torch.Tensor | tuple[TraceRequest, ...] | int | str] = {
        "schema": LAYERED_RENDERER_SCHEMA,
        "final": final,
        "no_reflection": no_reflection,
        "reflection_only": reflection_only,
        "internal_only": internal,
        "front_reflection_only": front_only,
        "back_internal_reflection_only": back_only,
        "Cout_only": cout_only,
        "D_direct_only": d_direct,
        "Cin": cin,
        "Ain": ain,
        "internal_intrinsic_rgb": intrinsic,
        "internal_alpha": ain,
        "Ffront": f_front,
        "Fback": f_back,
        "Rfront_raw": r_front.raw_rgb,
        "Rback_raw": r_back.raw_rgb,
        "Cout_raw": cout.raw_rgb,
        "linear_closure_error": closure,
        "linear_closure_max_abs": closure.detach().abs().max(),
        "linear_closure_mean_abs": closure.detach().abs().mean(),
        "formal_trace_requests": requests,
        "formal_trace_count": len(requests),
    }
    for name, value in result.items():
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            raise FloatingPointError(f"layered output {name} contains NaN or Inf")
    return result


@torch.no_grad()
def run_tao_layered_review_diagnostics(
    diagnostics: Mapping[str, Callable[[], torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Run optional unfiltered/ownership diagnostics outside the training graph."""
    outputs: dict[str, torch.Tensor] = {}
    for name, callback in diagnostics.items():
        value = callback()
        if not torch.is_tensor(value):
            raise TypeError(f"diagnostic {name} must return a tensor")
        outputs[name] = value.detach()
    return outputs
