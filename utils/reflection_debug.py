"""Real Stage B Reflection debug-map writer; no later-stage placeholders."""

import json
import os

import torch
from torchvision.utils import save_image

from utils.surfel_debug import save_surfel_debug_maps
from utils.surfel_utils import visualize_depth


def _selected_finite_values(tensor, mask):
    values = tensor.detach().float()
    selected = mask.detach().bool()
    if selected.shape != values.shape:
        selected = selected.expand_as(values)
    values = values[selected]
    return values[torch.isfinite(values)]


def _raw_stats(tensor, mask):
    selected = tensor.detach().float()
    selected_mask = mask.detach().bool()
    if selected_mask.shape != selected.shape:
        selected_mask = selected_mask.expand_as(selected)
    selected = selected[selected_mask]
    finite_mask = torch.isfinite(selected)
    finite = selected[finite_mask]
    result = {
        "count": int(selected.numel()),
        "finite_count": int(finite.numel()),
        "nonfinite_count": int((~finite_mask).sum().item()),
    }
    if not finite.numel():
        result.update({key: None for key in ("min", "mean", "p50", "p95", "p99", "max")})
        result.update({"nonzero_count": 0, "nonzero_fraction": 0.0})
        return result
    quantiles = torch.quantile(
        finite, torch.tensor([0.5, 0.95, 0.99], device=finite.device)
    )
    nonzero_count = int((finite != 0.0).sum().item())
    result.update(
        {
            "min": float(finite.min().item()),
            "mean": float(finite.mean().item()),
            "p50": float(quantiles[0].item()),
            "p95": float(quantiles[1].item()),
            "p99": float(quantiles[2].item()),
            "max": float(finite.max().item()),
            "nonzero_count": nonzero_count,
            "nonzero_fraction": float(nonzero_count / finite.numel()),
        }
    )
    return result


def _display_map(tensor, mask, transform):
    values = tensor.detach().float().clamp_min(0.0)
    transformed = torch.log1p(values) if transform == "log1p" else values
    selected = _selected_finite_values(transformed, mask)
    scale = float(torch.quantile(selected, 0.99).item()) if selected.numel() else 0.0
    if scale <= 0.0 and selected.numel():
        scale = float(selected.max().item())
    denominator = scale if scale > 0.0 else 1.0
    return (transformed / denominator).clamp(0.0, 1.0), scale


@torch.no_grad()
def save_reflection_debug_maps(output, ground_truth, directory, specular_mask=None, mask_sha256=None):
    os.makedirs(directory, exist_ok=True)
    save_surfel_debug_maps(output, ground_truth, directory)

    def chw(name):
        return output[name].detach().permute(2, 0, 1)

    save_image(chw("final").clamp(0, 1), os.path.join(directory, "final.png"))
    save_image(chw("reflection_color").clamp(0, 1), os.path.join(directory, "reflection_color.png"))
    save_image(chw("reflection_alpha").repeat(3, 1, 1).clamp(0, 1), os.path.join(directory, "reflection_alpha.png"))
    save_image(
        visualize_depth(output["reflection_depth"], output["reflection_alpha"]),
        os.path.join(directory, "reflection_depth.png"),
    )
    save_image(chw("reflection_hit_mask").repeat(3, 1, 1).clamp(0, 1), os.path.join(directory, "reflection_hit_mask.png"))
    save_image(chw("microfacet_D").repeat(3, 1, 1).clamp(0, 1), os.path.join(directory, "microfacet_D.png"))
    save_image(chw("microfacet_F").clamp(0, 1), os.path.join(directory, "microfacet_F.png"))
    save_image(chw("microfacet_G").repeat(3, 1, 1).clamp(0, 1), os.path.join(directory, "microfacet_G.png"))
    save_image(chw("microfacet_fr").clamp(0, 1), os.path.join(directory, "microfacet_fr.png"))
    save_image(chw("microfacet_wr").clamp(0, 1), os.path.join(directory, "microfacet_wr.png"))
    save_image(chw("diffuse_contribution").clamp(0, 1), os.path.join(directory, "diffuse_contribution.png"))
    save_image(chw("reflection_contribution").clamp(0, 1), os.path.join(directory, "reflection_contribution.png"))
    save_image(chw("valid_surface_mask").repeat(3, 1, 1).clamp(0, 1), os.path.join(directory, "valid_surface_mask.png"))

    for name in ("ray_candidate_count", "ray_exact_intersection_count"):
        if name not in output:
            raise KeyError(f"Stage B debug output is missing required diagnostics: {name}")
    valid_mask = output["valid_surface_mask"] > 0.5
    hit_mask = output["reflection_hit_mask"] > 0.5
    contribution_vis, contribution_scale = _display_map(
        output["reflection_contribution"], valid_mask, "linear"
    )
    microfacet_d_vis, microfacet_d_scale = _display_map(
        output["microfacet_D"], valid_mask, "log1p"
    )
    candidate_vis, candidate_scale = _display_map(
        output["ray_candidate_count"], valid_mask, "linear"
    )
    exact_vis, exact_scale = _display_map(
        output["ray_exact_intersection_count"], valid_mask, "linear"
    )
    save_image(
        contribution_vis.permute(2, 0, 1),
        os.path.join(directory, "reflection_contribution_vis.png"),
    )
    save_image(
        microfacet_d_vis.permute(2, 0, 1).repeat(3, 1, 1),
        os.path.join(directory, "microfacet_D_log.png"),
    )
    save_image(
        candidate_vis.permute(2, 0, 1).repeat(3, 1, 1),
        os.path.join(directory, "ray_candidate_count.png"),
    )
    save_image(
        exact_vis.permute(2, 0, 1).repeat(3, 1, 1),
        os.path.join(directory, "ray_exact_intersection_count.png"),
    )

    raw_stats = {}
    for name in (
        "reflection_color", "reflection_raw_color", "reflection_alpha", "microfacet_D",
        "microfacet_F", "microfacet_G", "microfacet_fr", "microfacet_wr",
        "diffuse_contribution", "reflection_contribution", "ray_candidate_count",
        "ray_exact_intersection_count",
    ):
        raw_stats[name] = _raw_stats(output[name], valid_mask)
    raw_stats["reflection_depth"] = _raw_stats(output["reflection_depth"], hit_mask)

    diagnostics = output.get("ray_diagnostics")
    raytrace_metadata = None
    if diagnostics is not None:
        reflection_count = diagnostics.reflection_surfel_count
        candidate_fraction = output["ray_candidate_count"] / max(reflection_count, 1)
        exact_fraction = torch.where(
            output["ray_candidate_count"] > 0,
            output["ray_exact_intersection_count"]
            / output["ray_candidate_count"].clamp_min(1.0),
            torch.zeros_like(output["ray_candidate_count"]),
        )
        raytrace_metadata = {
            "timing_ms": {key: float(value) for key, value in diagnostics.timing_ms.items()},
            "chunk_count": int(diagnostics.chunk_count),
            "reflection_surfel_count": int(reflection_count),
            "peak_memory_allocated_bytes": int(diagnostics.peak_memory_allocated_bytes),
            "peak_memory_delta_bytes": int(diagnostics.peak_memory_delta_bytes),
            "bvh_rebuild_delta": int(diagnostics.bvh_rebuild_delta),
            "bvh_refit_delta": int(diagnostics.bvh_refit_delta),
            "candidate_fraction_of_field": _raw_stats(candidate_fraction, valid_mask),
            "exact_fraction_of_candidates": _raw_stats(exact_fraction, valid_mask),
        }

    metadata = {
        "valid_ray_count": int(output["valid_ray_count"]),
        "reflection_hit_count": int((output["reflection_hit_mask"] > 0.5).sum().item()),
        "reflection_hit_fraction_of_valid": float(
            (output["reflection_hit_mask"] > 0.5).sum().item()
            / max(int(output["valid_ray_count"]), 1)
        ),
        "roughness_min": float(output["surface_roughness"].min().item()),
        "roughness_max": float(output["surface_roughness"].max().item()),
        "specular_mask_sha256": mask_sha256,
        "raw_stats": raw_stats,
        "display_scales": {
            "reflection_contribution_vis": {
                "transform": "linear", "finite_valid_p99": contribution_scale
            },
            "microfacet_D_log": {
                "transform": "log1p", "finite_valid_p99": microfacet_d_scale
            },
            "ray_candidate_count": {
                "transform": "linear", "finite_valid_p99": candidate_scale
            },
            "ray_exact_intersection_count": {
                "transform": "linear", "finite_valid_p99": exact_scale
            },
        },
        "raytrace": raytrace_metadata,
    }
    if specular_mask is not None:
        mask = specular_mask.detach().clamp(0, 1)
        save_image(mask, os.path.join(directory, "transparent_mask.png"))
        save_image(mask.repeat(3, 1, 1), os.path.join(directory, "specular_mask.png"))
        overlay = 0.7 * ground_truth.detach().clamp(0, 1) + 0.3 * torch.cat(
            (mask, torch.zeros_like(mask), torch.zeros_like(mask)), dim=0
        )
        save_image(overlay.clamp(0, 1), os.path.join(directory, "overlay.png"))
    with open(os.path.join(directory, "reflection_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True, allow_nan=False)
