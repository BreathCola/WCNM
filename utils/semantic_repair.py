"""Engineering priors and metrics for the Stage D semantic-repair pilot."""

from __future__ import annotations

import torch
import torch.nn.functional as F


ANTI_VEIL_SCHEMA = "rtgs_stage_d_anti_veil_v1"


def luminance(rgb: torch.Tensor) -> torch.Tensor:
    if rgb.shape[-1] != 3:
        raise ValueError("RGB tensor must use a final channel dimension")
    weights = rgb.new_tensor((0.2126, 0.7152, 0.0722))
    return (rgb * weights).sum(dim=-1, keepdim=True)


def smooth_ramp(local_iteration: int, start: int, end: int) -> float:
    if end <= start:
        raise ValueError("anti-veil ramp end must exceed start")
    u = min(1.0, max(0.0, (int(local_iteration) - int(start)) / float(end - start)))
    return float(u * u * (3.0 - 2.0 * u))


@torch.no_grad()
def spatial_frequency_energy(rgb: torch.Tensor, mask: torch.Tensor) -> dict:
    """Transparent-domain first-difference and Laplacian energy for branch maps."""
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("spatial frequency metric expects HWC RGB")
    if mask.shape != rgb.shape[:2] + (1,):
        raise ValueError("spatial frequency mask shape mismatch")
    y = luminance(rgb.detach().float())
    valid = mask.detach().bool()
    dx_mask = valid[:, 1:] & valid[:, :-1]
    dy_mask = valid[1:] & valid[:-1]
    dx = torch.abs(y[:, 1:] - y[:, :-1])[dx_mask]
    dy = torch.abs(y[1:] - y[:-1])[dy_mask]
    edge_values = torch.cat((dx, dy)) if dx.numel() or dy.numel() else y.new_zeros((0,))
    if y.shape[0] >= 3 and y.shape[1] >= 3:
        laplacian = torch.abs(
            4.0 * y[1:-1, 1:-1] - y[:-2, 1:-1] - y[2:, 1:-1]
            - y[1:-1, :-2] - y[1:-1, 2:]
        )
        laplacian = laplacian[valid[1:-1, 1:-1]]
    else:
        laplacian = y.new_zeros((0,))
    return {
        "edge_l1": float(edge_values.mean()) if edge_values.numel() else 0.0,
        "laplacian_l1": float(laplacian.mean()) if laplacian.numel() else 0.0,
    }


def anti_veil_loss(
    inside_color: torch.Tensor,
    inside_alpha: torch.Tensor,
    ground_truth_hwc: torch.Tensor,
    transparent_mask_hwc: torch.Tensor,
    *,
    gt_luminance_threshold: float = 0.15,
    high_alpha_threshold: float = 0.80,
    black_luminance_threshold: float = 0.08,
    saturation_alpha_threshold: float = 0.95,
    target_saturation_coverage: float = 0.35,
    gate_temperature: float = 0.05,
    black_temperature: float = 0.02,
    coverage_temperature: float = 0.02,
    epsilon: float = 1e-6,
):
    if inside_color.shape[-1] != 3 or inside_alpha.shape[-1] != 1:
        raise ValueError("anti-veil expects HWC Cin and HW1 Ain")
    if ground_truth_hwc.shape != inside_color.shape:
        raise ValueError("anti-veil GT shape mismatch")
    if transparent_mask_hwc.shape != inside_alpha.shape:
        raise ValueError("anti-veil mask shape mismatch")
    if min(gate_temperature, black_temperature, coverage_temperature, epsilon) <= 0:
        raise ValueError("anti-veil temperatures and epsilon must be positive")
    conditional = inside_color / inside_alpha.clamp_min(float(epsilon))
    conditional_luma = luminance(conditional)
    gt_luma = luminance(ground_truth_hwc)
    mask = transparent_mask_hwc.clamp(0.0, 1.0)
    bright_gate = torch.sigmoid(
        (gt_luma - float(gt_luminance_threshold)) / float(gate_temperature)
    )
    high_gate = torch.sigmoid(
        (inside_alpha - float(high_alpha_threshold)) / float(gate_temperature)
    )
    dark_penalty = float(black_temperature) * F.softplus(
        (float(black_luminance_threshold) - conditional_luma)
        / float(black_temperature)
    )
    bright_support = mask * bright_gate
    denominator = bright_support.sum().clamp_min(float(epsilon))
    black = (bright_support * high_gate * dark_penalty).sum() / denominator

    saturation_gate = torch.sigmoid(
        (inside_alpha - float(saturation_alpha_threshold)) / float(gate_temperature)
    )
    saturation_coverage = (bright_support * saturation_gate).sum() / denominator
    saturation = float(coverage_temperature) * F.softplus(
        (saturation_coverage - float(target_saturation_coverage))
        / float(coverage_temperature)
    )
    return {
        "black": black,
        "saturation": saturation,
        "conditional_inside_color": conditional,
        "conditional_luminance": conditional_luma,
        "soft_saturation_coverage": saturation_coverage,
        "soft_high_alpha_coverage": (bright_support * high_gate).sum() / denominator,
    }


def anti_veil_config(opt) -> dict:
    return {
        "schema": ANTI_VEIL_SCHEMA,
        "lambda_black": float(opt.lambda_anti_veil_black),
        "lambda_saturation": float(opt.lambda_anti_veil_saturation),
        "gt_luminance_threshold": float(opt.anti_veil_gt_luminance_threshold),
        "high_alpha_threshold": float(opt.anti_veil_high_alpha_threshold),
        "black_luminance_threshold": float(opt.anti_veil_black_luminance_threshold),
        "saturation_alpha_threshold": float(opt.anti_veil_saturation_alpha_threshold),
        "target_saturation_coverage": float(opt.anti_veil_target_saturation_coverage),
        "gate_temperature": float(opt.anti_veil_gate_temperature),
        "black_temperature": float(opt.anti_veil_black_temperature),
        "coverage_temperature": float(opt.anti_veil_coverage_temperature),
        "epsilon": float(opt.anti_veil_epsilon),
        "ramp_start_local": int(opt.anti_veil_ramp_start),
        "ramp_end_local": int(opt.anti_veil_ramp_end),
        "engineering_prior": True,
        "paper_claim": False,
    }
