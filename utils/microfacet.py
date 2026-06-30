"""Full non-split-sum GGX reflection for Stage B."""

import math

import torch
import torch.nn.functional as F


def _require_finite(name, value):
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} contains NaN or Inf")


def pbrt_roughness_to_alpha(roughness: torch.Tensor) -> torch.Tensor:
    x = torch.log(roughness.clamp_min(1e-3))
    return 1.62142 + 0.819955 * x + 0.1734 * x.square() + 0.0171201 * x.pow(3) + 0.000640711 * x.pow(4)


def fresnel_schlick(f0: torch.Tensor, voh: torch.Tensor) -> torch.Tensor:
    return (f0 + (1.0 - f0) * (1.0 - voh).clamp(0.0, 1.0).pow(5)).clamp(0.0, 1.0)


def smith_ggx(no_v: torch.Tensor, no_l: torch.Tensor, alpha: torch.Tensor, eps: float = 1e-6):
    alpha2 = alpha.square()

    def g1(no_x):
        root = torch.sqrt((alpha2 + (1.0 - alpha2) * no_x.square()).clamp_min(eps))
        return (2.0 * no_x / (no_x + root).clamp_min(eps)).clamp(0.0, 1.0)

    return (g1(no_v) * g1(no_l)).clamp(0.0, 1.0)


def microfacet_reflection(
    normal: torch.Tensor,
    wo: torch.Tensor,
    wi: torch.Tensor,
    roughness: torch.Tensor,
    f0: torch.Tensor,
    roughness_min: float = 0.03,
    roughness_remap: bool = False,
    eps: float = 1e-6,
):
    for name, value in (("normal", normal), ("wo", wo), ("wi", wi), ("roughness", roughness), ("f0", f0)):
        _require_finite(name, value)
    n = F.normalize(normal, dim=-1, eps=1e-8)
    wo = F.normalize(wo, dim=-1, eps=1e-8)
    wi = F.normalize(wi, dim=-1, eps=1e-8)
    half = F.normalize(wi + wo, dim=-1, eps=1e-8)
    raw_no_v = (n * wo).sum(dim=-1, keepdim=True)
    raw_no_l = (n * wi).sum(dim=-1, keepdim=True)
    gate = (raw_no_v > 0.0) & (raw_no_l > 0.0)
    no_v = raw_no_v.clamp(eps, 1.0)
    no_l = raw_no_l.clamp(eps, 1.0)
    no_h = (n * half).sum(dim=-1, keepdim=True).clamp(eps, 1.0)
    vo_h = (wo * half).sum(dim=-1, keepdim=True).clamp(eps, 1.0)
    alpha = roughness.clamp(roughness_min, 1.0)
    if roughness_remap:
        alpha = pbrt_roughness_to_alpha(alpha)
    alpha = alpha.clamp(roughness_min, 1.0)
    alpha2 = alpha.square()
    distribution = alpha2 / (
        math.pi * ((no_h.square() * (alpha2 - 1.0) + 1.0).square()).clamp_min(eps)
    )
    distribution = distribution.clamp_min(0.0)
    fresnel = fresnel_schlick(f0.clamp(0.0, 1.0), vo_h)
    geometry = smith_ggx(no_v, no_l, alpha, eps)
    fr = (distribution * geometry * fresnel / (4.0 * no_v * no_l + eps)).clamp_min(0.0)
    wr = (fr * no_l).clamp_min(0.0)
    zero1 = torch.zeros_like(distribution)
    zero3 = torch.zeros_like(fr)
    distribution = torch.where(gate, distribution, zero1)
    fresnel = torch.where(gate.expand_as(fresnel), fresnel, zero3)
    geometry = torch.where(gate, geometry, zero1)
    fr = torch.where(gate.expand_as(fr), fr, zero3)
    wr = torch.where(gate.expand_as(wr), wr, zero3)
    result = {
        "D": distribution,
        "F": fresnel,
        "G": geometry,
        "fr": fr,
        "wr": wr,
        "NoV": torch.where(gate, no_v, torch.zeros_like(no_v)),
        "NoL": torch.where(gate, no_l, torch.zeros_like(no_l)),
        "alpha_ggx": alpha,
        "gate": gate,
    }
    for name, value in result.items():
        if torch.is_tensor(value):
            _require_finite(name, value)
    return result
