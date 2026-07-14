"""Stage D real D/R/T debug products and metadata."""

import json
import os

import torch
from PIL import Image, ImageDraw
from torchvision.utils import save_image

from utils.reflection_debug import save_reflection_debug_maps
from utils.surfel_utils import visualize_depth
from utils.semantic_repair import spatial_frequency_energy


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


def _mask_hwc(mask):
    """Normalize a one-channel mask to the renderer's [H,W,1] map layout."""
    if mask.ndim == 2:
        return mask[..., None]
    if mask.ndim == 3 and mask.shape[-1] == 1:
        return mask
    if mask.ndim == 3 and mask.shape[0] == 1:
        return mask.permute(1, 2, 0)
    raise ValueError(f"expected a one-channel mask, found shape {tuple(mask.shape)}")


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
        "transmittance_color", "inside_contribution", "cout_contribution",
        "final_t_off", "final_cout_off", "final_d_direct_off", "final_r_off",
    ):
        if name in output:
            save_image(_chw(output, name).clamp(0, 1), os.path.join(directory, f"{name}.png"))
    gt_hwc = ground_truth.detach().permute(1, 2, 0)
    save_image(
        torch.abs(output["final"] - gt_hwc).permute(2, 0, 1).clamp(0, 1),
        os.path.join(directory, "final_diff.png"),
    )
    semantic_rgb = (
        "reflection_unfiltered", "reflection_inside", "reflection_interface",
        "reflection_outside", "reflection_final_filtered",
        "conditional_inside_color", "outside_unfiltered", "outside_inside",
        "outside_interface", "outside_outside", "outside_final_filtered",
        "t_spatial_class_map",
        "t_support_legal",
        "reflection_strict_inside_safe", "reflection_interface_margin",
        "reflection_strict_outside_safe", "reflection_crossing_or_ambiguous",
        "outside_strict_inside_safe", "outside_interface_margin",
        "outside_strict_outside_safe", "outside_crossing_or_ambiguous",
        "cout_ownership_class_map",
    )
    for name in semantic_rgb:
        if name in output:
            save_image(_chw(output, name).clamp(0, 1), os.path.join(directory, f"{name}.png"))
    for source, alias in (
        ("outside_unfiltered", "cout_unfiltered"),
        ("outside_inside", "cout_inside"),
        ("outside_interface", "cout_interface"),
        ("outside_outside", "cout_outside"),
        ("outside_final_filtered", "cout_final"),
        ("conditional_inside_color", "c_in_cond"),
    ):
        if source in output:
            save_image(_chw(output, source).clamp(0, 1), os.path.join(directory, f"{alias}.png"))
    for name in (
        "inside_alpha", "outside_alpha", "transmittance_alpha",
        "two_hit_valid", "transmittance_valid",
    ):
        save_image(
            _chw(output, name).repeat(3, 1, 1).clamp(0, 1),
            os.path.join(directory, f"{name}.png"),
        )
    for prefix in ("reflection_inside", "reflection_interface", "reflection_outside"):
        alpha_name, depth_name, hit_name = f"{prefix}_alpha", f"{prefix}_depth", f"{prefix}_hit"
        if alpha_name in output:
            save_image(_chw(output, alpha_name).repeat(3, 1, 1).clamp(0, 1), os.path.join(directory, f"{alpha_name}.png"))
            save_image(visualize_depth(output[depth_name], output[alpha_name]), os.path.join(directory, f"{depth_name}.png"))
            save_image(_chw(output, hit_name).repeat(3, 1, 1).clamp(0, 1), os.path.join(directory, f"{hit_name}.png"))
    for prefix in (
        "reflection_strict_inside_safe", "reflection_interface_margin",
        "reflection_strict_outside_safe", "reflection_crossing_or_ambiguous",
    ):
        alpha_name, depth_name, hit_name = f"{prefix}_alpha", f"{prefix}_depth", f"{prefix}_hit"
        if alpha_name in output:
            save_image(_chw(output, alpha_name).repeat(3, 1, 1).clamp(0, 1), os.path.join(directory, f"{alpha_name}.png"))
            save_image(visualize_depth(output[depth_name], output[alpha_name]), os.path.join(directory, f"{depth_name}.png"))
            save_image(_chw(output, hit_name).repeat(3, 1, 1).clamp(0, 1), os.path.join(directory, f"{hit_name}.png"))
    for prefix in ("outside_inside", "outside_interface", "outside_outside"):
        alpha_name, depth_name, hit_name = f"{prefix}_alpha", f"{prefix}_depth", f"{prefix}_hit"
        if alpha_name in output:
            save_image(_chw(output, alpha_name).repeat(3, 1, 1).clamp(0, 1), os.path.join(directory, f"{alpha_name}.png"))
            save_image(visualize_depth(output[depth_name], output[alpha_name]), os.path.join(directory, f"{depth_name}.png"))
            save_image(_chw(output, hit_name).repeat(3, 1, 1).clamp(0, 1), os.path.join(directory, f"{hit_name}.png"))
    for prefix in (
        "outside_strict_inside_safe", "outside_interface_margin",
        "outside_strict_outside_safe", "outside_crossing_or_ambiguous",
    ):
        alpha_name, depth_name, hit_name = f"{prefix}_alpha", f"{prefix}_depth", f"{prefix}_hit"
        if alpha_name in output:
            save_image(_chw(output, alpha_name).repeat(3, 1, 1).clamp(0, 1), os.path.join(directory, f"{alpha_name}.png"))
            save_image(visualize_depth(output[depth_name], output[alpha_name]), os.path.join(directory, f"{depth_name}.png"))
            save_image(_chw(output, hit_name).repeat(3, 1, 1).clamp(0, 1), os.path.join(directory, f"{hit_name}.png"))
    if "front_normal" in output:
        save_image(
            ((_chw(output, "front_normal") + 1.0) * 0.5).clamp(0, 1),
            os.path.join(directory, "front_normal.png"),
        )
        valid_front = output["transparent_path_valid"] > 0.5
        front = output["front_position"].detach()
        selected_front = front[valid_front.expand_as(front)]
        lo = float(selected_front.min()) if selected_front.numel() else 0.0
        hi = float(selected_front.max()) if selected_front.numel() else 1.0
        front_vis = (front - lo) / max(hi - lo, 1e-8)
        save_image(
            _chw({"front": front_vis}, "front").clamp(0, 1),
            os.path.join(directory, "front_position.png"),
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
    hard = _mask_hwc(specular_mask.detach()) >= 0.5
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
        "path_contract": {
            "transparent_path_mode": output.get("transparent_path_mode"),
            "transparent_direct_mode": output.get("transparent_direct_mode"),
            "transparent_reflection_mode": output.get("transparent_reflection_mode"),
            "front_origin_tnear_error": _stats(
                output.get("front_origin_tnear_error", torch.zeros_like(output["inside_alpha"])),
                valid,
            ),
            "front_normal_faceforward_dot": _stats(
                output.get("front_normal_faceforward_dot", torch.zeros_like(output["inside_alpha"])),
                valid,
            ),
            "front_plane_residual": _stats(
                output.get("front_plane_residual", torch.zeros_like(output["inside_alpha"])),
                valid,
            ),
            "back_tfar_residual": _stats(
                output.get(
                    "frozen_back_distance_residual",
                    torch.zeros_like(output["inside_alpha"]),
                ),
                valid,
            ),
        },
    }
    if "t_spatial_counts" in output:
        ain = output["inside_alpha"][hard]
        conditional_luma = (
            output["conditional_inside_color"]
            * output["conditional_inside_color"].new_tensor((0.2126, 0.7152, 0.0722))
        ).sum(dim=-1, keepdim=True)[hard]
        metadata["semantic_repair"] = {
            "t_spatial_counts": output["t_spatial_counts"],
            "t_support_scale_cap": output.get("t_support_scale_cap", {}),
            "r_spatial": output.get("semantic_r_stats", {}),
            "cout_spatial": output.get("semantic_cout_stats", {}),
            "r_filter": output.get("semantic_r_filter_stats", {}),
            "cout_filter": output.get("semantic_cout_filter_stats", {}),
            "ain": {
                "mean": float(ain.mean()), "p50": float(torch.quantile(ain, 0.50)),
                "p95": float(torch.quantile(ain, 0.95)),
                "p99": float(torch.quantile(ain, 0.99)),
                "saturation_fraction_ge_0_95": float((ain >= 0.95).float().mean()),
            } if ain.numel() else {},
            "conditional_inside_luminance": _stats(
                conditional_luma,
                torch.ones_like(conditional_luma, dtype=torch.bool),
            ),
            "high_ain_near_black_fraction": float(
                ((ain >= 0.80) & (conditional_luma < 0.08)).float().mean()
            ) if ain.numel() else 0.0,
            "spatial_frequency_energy": {
                branch: spatial_frequency_energy(output[name], hard)
                for branch, name in (
                    ("diffuse", "diffuse_contribution"),
                    ("reflection", "reflection_final_filtered"),
                    ("transmittance", "transmittance_contribution"),
                )
            },
            "bird_roi_available": False,
            "bird_level_quantification": "unavailable: no independent versioned bird ROI",
        }
    raw_names = (
        "final", "diffuse_contribution", "reflection_contribution",
        "transmittance_contribution", "inside_color", "inside_alpha",
        "inside_depth", "conditional_inside_color", "outside_color",
        "outside_alpha", "outside_depth", "transmittance_color",
        "transmittance_alpha", "inside_contribution", "cout_contribution",
        "front_position", "front_normal", "near_depth", "far_depth",
        "front_origin_tnear_error", "front_normal_faceforward_dot",
    )
    metadata["raw_float_statistics"] = {
        name: _stats(output[name], valid) for name in raw_names if name in output
    }
    metadata["raw_float_statistics_transparent_hard"] = {
        name: _stats(output[name], hard) for name in raw_names if name in output
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


def make_semantic_repair_contact_sheet(iteration_directory, stems):
    columns = (
        ("ground_truth.png", "GT"), ("final.png", "final"),
        ("diffuse_contribution.png", "D"),
        ("reflection_unfiltered.png", "R unfiltered"),
        ("reflection_inside.png", "R inside"),
        ("reflection_interface.png", "R interface"),
        ("reflection_outside.png", "R outside"),
        ("reflection_final_filtered.png", "R final"),
        ("transmittance_contribution.png", "T"),
        ("inside_color.png", "Cin"), ("inside_alpha.png", "Ain"),
        ("inside_depth.png", "Din"), ("c_in_cond.png", "Cin/Ain"),
        ("cout_unfiltered.png", "Cout unfiltered"),
        ("cout_inside.png", "Cout inside"),
        ("cout_interface.png", "Cout interface"),
        ("cout_outside.png", "Cout outside"),
        ("cout_final.png", "Cout final"),
        ("transmittance_color.png", "Ct"),
        ("transmittance_alpha.png", "At"),
        ("din_vs_far_violation.png", "Din vs far"),
        ("transparent_mask.png", "mask"), ("ks.png", "ks"),
        ("t_spatial_class_map.png", "T spatial"),
    )
    required = {
        "ground_truth.png", "final.png", "diffuse_contribution.png",
        "transmittance_contribution.png", "inside_color.png",
        "inside_alpha.png", "transparent_mask.png",
    }
    available_columns = []
    for filename, label in columns:
        exists_for_all = all(
            os.path.isfile(os.path.join(iteration_directory, stem, filename))
            for stem in stems
        )
        if exists_for_all:
            available_columns.append((filename, label))
        elif filename in required:
            raise FileNotFoundError(os.path.join(iteration_directory, stems[0], filename))
    columns = tuple(available_columns)
    thumb, label_height = (240, 135), 25
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
            draw.text((col * thumb[0] + 3, y0 + 4), f"{stem} | {label}", fill="black")
    target = os.path.join(iteration_directory, "semantic_repair_contact_sheet.png")
    canvas.save(target)
    canvas.save(os.path.join(iteration_directory, "contact_sheet.png"))
    return target
