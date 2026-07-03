"""Stage D real D/R/T debug products and metadata."""

import json
import os

import torch
from PIL import Image, ImageDraw
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
    order_map = torch.zeros_like(output["final"])
    order_map[..., 1:2] = valid.float() * (output["depth_violation"] <= 0).float()
    order_map[..., 0:1] = valid.float() * (output["depth_violation"] > 0).float()
    save_image(
        order_map.detach().permute(2, 0, 1),
        os.path.join(directory, "din_vs_far_violation.png"),
    )
    gt_hwc = ground_truth.detach().permute(1, 2, 0)
    hard = specular_mask.detach() >= 0.5
    full_l1 = torch.abs(output["final"] - gt_hwc).mean()
    transparent_l1 = torch.abs(output["final"] - gt_hwc)[hard.expand_as(gt_hwc)].mean()
    outside_hard = ~hard
    metadata = {
        "geometry_release_id": geometry_release_id,
        "geometry_release_aggregate_sha256": geometry_release_aggregate_sha256,
        "transmittance_valid_count": int(valid.sum()),
        "valid_two_hit_fraction": float(output["two_hit_valid"].mean()),
        "din_le_t_far_fraction": float(depth_order.float().mean()) if depth_order.numel() else 0.0,
        "alpha_over_formula": "Ct = Cin + (1 - Ain) * Cout",
        "rgb_l1": {
            "full_image": float(full_l1),
            "transparent_hard": float(transparent_l1),
        },
        "energy": {
            "inside_color_valid_mean": float(output["inside_color"][valid.expand_as(output["inside_color"])].mean()),
            "outside_color_valid_mean": float(output["outside_color"][valid.expand_as(output["outside_color"])].mean()),
            "transmittance_contribution_transparent_mean": float(output["transmittance_contribution"][hard.expand_as(output["transmittance_contribution"])].mean()),
            "transmittance_contribution_outside_mean": float(output["transmittance_contribution"][outside_hard.expand_as(output["transmittance_contribution"])].mean()),
        },
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


def make_stage_d_contact_sheet(iteration_directory, stems):
    """Build the fixed nine-view review sheet from already saved debug PNGs."""
    columns = (
        ("ground_truth.png", "GT"), ("final.png", "final"),
        ("diffuse_contribution.png", "D contribution"),
        ("reflection_contribution.png", "R contribution"),
        ("transmittance_contribution.png", "T contribution"),
        ("inside_color.png", "Cin"), ("outside_color.png", "Cout"),
        ("din_vs_far_violation.png", "Din vs far"),
    )
    thumb = (320, 180)
    label_height = 28
    canvas = Image.new(
        "RGB", (thumb[0] * len(columns), (thumb[1] + label_height) * len(stems)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for row, stem in enumerate(stems):
        view = os.path.join(iteration_directory, stem)
        for col, (filename, label) in enumerate(columns):
            path = os.path.join(view, filename)
            if not os.path.isfile(path):
                raise FileNotFoundError(path)
            image = Image.open(path).convert("RGB")
            image.thumbnail(thumb, Image.Resampling.LANCZOS)
            x = col * thumb[0] + (thumb[0] - image.width) // 2
            y0 = row * (thumb[1] + label_height)
            y = y0 + label_height + (thumb[1] - image.height) // 2
            canvas.paste(image, (x, y))
            draw.text((col * thumb[0] + 4, y0 + 5), f"{stem} | {label}", fill="black")
    target = os.path.join(iteration_directory, "contact_sheet.png")
    canvas.save(target)
    return target
