"""Stage D real D/R/T debug products and metadata."""

import json
import os

import torch
from torchvision.utils import save_image

from utils.reflection_debug import save_reflection_debug_maps
from utils.surfel_utils import visualize_depth


def _chw(output, name):
    return output[name].detach().permute(2, 0, 1)


def _stats(value, mask):
    selected = value.detach()[mask.expand_as(value)].float()
    finite = selected[torch.isfinite(selected)]
    if finite.numel() != selected.numel() or not finite.numel():
        return {"count": int(selected.numel()), "finite": False}
    return {
        "count": int(finite.numel()), "finite": True,
        "min": float(finite.min()), "mean": float(finite.mean()),
        "p50": float(torch.quantile(finite, 0.5)),
        "p95": float(torch.quantile(finite, 0.95)),
        "max": float(finite.max()),
    }


@torch.no_grad()
def save_transmittance_debug_maps(
    output, ground_truth, directory, specular_mask, mask_sha256,
    geometry_release_id, geometry_release_aggregate_sha256,
):
    os.makedirs(directory, exist_ok=True)
    save_reflection_debug_maps(
        output, ground_truth, directory,
        specular_mask=specular_mask, mask_sha256=mask_sha256,
    )
    for name in (
        "diffuse_contribution", "reflection_contribution",
        "transmittance_contribution", "inside_color", "outside_color",
        "transmittance_color",
    ):
        save_image(_chw(output, name).clamp(0, 1), os.path.join(directory, f"{name}.png"))
    for name in (
        "inside_alpha", "outside_alpha", "transmittance_alpha",
        "two_hit_valid", "transmittance_valid",
    ):
        save_image(
            _chw(output, name).repeat(3, 1, 1).clamp(0, 1),
            os.path.join(directory, f"{name}.png"),
        )
    for name, alpha_name in (
        ("inside_depth", "inside_alpha"),
        ("outside_depth", "outside_alpha"),
        ("near_depth", "two_hit_valid"),
        ("far_depth", "two_hit_valid"),
    ):
        save_image(
            visualize_depth(output[name], output[alpha_name]),
            os.path.join(directory, f"{name}.png"),
        )
    valid = output["transmittance_valid"] > 0.5
    violation = output["depth_violation"].detach().clamp_min(0)
    selected = violation[valid]
    scale = float(torch.quantile(selected, 0.99)) if selected.numel() else 0.0
    violation_vis = violation / (scale if scale > 0 else 1.0)
    save_image(
        _chw({"value": violation_vis}, "value").repeat(3, 1, 1).clamp(0, 1),
        os.path.join(directory, "depth_violation.png"),
    )
    depth_order = (
        output["inside_depth"][valid] <= output["far_depth"][valid]
    )
    metadata = {
        "geometry_release_id": geometry_release_id,
        "geometry_release_aggregate_sha256": geometry_release_aggregate_sha256,
        "transmittance_valid_count": int(valid.sum()),
        "valid_two_hit_fraction": float(output["two_hit_valid"].mean()),
        "din_le_t_far_fraction": float(depth_order.float().mean()) if depth_order.numel() else 0.0,
        "alpha_over_formula": "Ct = Cin + (1 - Ain) * Cout",
        "raw_stats": {
            name: _stats(output[name], valid)
            for name in (
                "inside_color", "inside_alpha", "inside_depth", "outside_color",
                "outside_alpha", "outside_depth", "transmittance_color",
                "transmittance_alpha", "transmittance_contribution", "depth_violation",
            )
        },
        "depth_violation_display_p99": scale,
    }
    with open(os.path.join(directory, "transmittance_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True, allow_nan=False)
