"""D-015 zero-update semantic renderer ablation helpers."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from gaussian_renderer.reflection_renderer import StageBRenderState, render as render_stage_b
from gaussian_renderer.transmittance_renderer import (
    _scatter, _trace, render as render_stage_d, render_from_static_dr,
)
from geometry.cuboid_space import (
    SUPPORT_CROSSING, SUPPORT_INTERFACE, SUPPORT_STRICT_INSIDE,
    SUPPORT_STRICT_OUTSIDE,
)


SCHEMA = "rtgs_stage_d_semantic_renderer_ablation_v1"
ARM_NAMES = ("arm_0", "arm_1", "arm_2", "arm_3")
LUMA = (0.2126, 0.7152, 0.0722)
NEAR_BLACK_THRESHOLD = 0.10


def tensor_sha256(value: torch.Tensor) -> str:
    cpu = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tuple(cpu.shape)).encode("ascii"))
    digest.update(str(cpu.dtype).encode("ascii"))
    digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def nested_tensor_hash(value) -> str:
    rows = []

    def visit(prefix, child):
        if torch.is_tensor(child):
            rows.append((prefix, tensor_sha256(child)))
        elif isinstance(child, dict):
            for key in sorted(child):
                visit(f"{prefix}.{key}", child[key])
        elif isinstance(child, (list, tuple)):
            for index, item in enumerate(child):
                visit(f"{prefix}[{index}]", item)

    visit("root", value)
    payload = json.dumps(rows, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def luminance(rgb: torch.Tensor) -> torch.Tensor:
    weights = rgb.new_tensor(LUMA)
    return (rgb * weights).sum(dim=-1, keepdim=True)


def _mask_hwc(mask: torch.Tensor, *, reference=None) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask[..., None]
    elif mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask.permute(1, 2, 0)
    if mask.ndim != 3 or mask.shape[-1] != 1:
        raise ValueError(f"expected one-channel mask, got {tuple(mask.shape)}")
    if reference is not None:
        mask = mask.to(device=reference.device, dtype=reference.dtype)
    return mask


def _fraction(mask: torch.Tensor) -> float:
    if mask.numel() == 0:
        return 0.0
    return float(mask.float().mean())


def _stats(values: torch.Tensor) -> dict:
    values = values.detach().float()
    if values.numel() == 0:
        return {"count": 0, "finite": True}
    finite = values[torch.isfinite(values)]
    if finite.numel() != values.numel():
        return {"count": int(values.numel()), "finite": False}
    return {
        "count": int(values.numel()),
        "finite": True,
        "min": float(finite.min()),
        "mean": float(finite.mean()),
        "p50": float(torch.quantile(finite, 0.50)),
        "p95": float(torch.quantile(finite, 0.95)),
        "max": float(finite.max()),
    }


def boundary_distance(mask: torch.Tensor, max_radius: int = 8) -> torch.Tensor:
    """Approximate pixel distance to the hard-mask boundary up to max_radius."""
    hard = _mask_hwc(mask).permute(2, 0, 1).unsqueeze(0).float()
    padded_same = F.max_pool2d(hard, 3, stride=1, padding=1)
    padded_inv = F.max_pool2d(1.0 - hard, 3, stride=1, padding=1)
    boundary = ((padded_same > 0) & (padded_inv > 0)).float()
    distance = torch.full_like(hard, float(max_radius + 1))
    distance = torch.where(boundary > 0, torch.zeros_like(distance), distance)
    grown = boundary
    for radius in range(1, int(max_radius) + 1):
        grown = F.max_pool2d(grown, 3, stride=1, padding=1)
        newly = (grown > 0) & (distance > max_radius)
        distance = torch.where(newly, torch.full_like(distance, float(radius)), distance)
    return distance.squeeze(0).permute(1, 2, 0)


def mask_boundary_diagnostics(mask_hard, valid_two_hit, final_rgb) -> tuple[dict, dict]:
    hard = _mask_hwc(mask_hard, reference=final_rgb) >= 0.5
    valid = _mask_hwc(valid_two_hit, reference=final_rgb) >= 0.5
    near_black = luminance(final_rgb) < NEAR_BLACK_THRESHOLD
    distance = boundary_distance(hard.to(final_rgb), 8)
    rows = {}
    domain = hard
    invalid = hard & ~valid
    for radius in (1, 2, 4, 8):
        band = domain & (distance <= radius)
        outside_band = domain & (distance > radius)
        rows[f"{radius}px"] = {
            "pixel_count": int(band.sum()),
            "near_black_fraction": _fraction(near_black[band]),
            "invalid_two_hit_fraction_in_band": _fraction(invalid[band]),
            "invalid_two_hit_fraction_outside_band": _fraction(invalid[outside_band]),
        }
    maps = {
        "mask_boundary": (distance == 0).to(final_rgb),
        "validity_disagreement": (hard ^ valid).to(final_rgb),
        "boundary_distance": distance.to(final_rgb),
    }
    return rows, maps


def black_pixel_attribution(package: dict, mask_hard: torch.Tensor) -> tuple[dict, dict]:
    hard = _mask_hwc(mask_hard, reference=package["final"]) >= 0.5
    valid = package["two_hit_valid"] >= 0.5
    final_black = hard & (luminance(package["final"]) < NEAR_BLACK_THRESHOLD)
    cin_luma = luminance(package.get("conditional_inside_color", package["inside_color"]))
    ain = package["inside_alpha"]
    cout = package["outside_color"]
    cout_unfiltered = package.get("outside_unfiltered", cout)
    formal_r = package.get("reflection_contribution", torch.zeros_like(package["final"]))
    safe_r = package.get("reflection_strict_outside_safe", formal_r)
    labels = {
        "two_hit_invalid": final_black & ~valid,
        "T_no_hit": final_black & valid & (package["inside_hit_mask"] < 0.5),
        "Cin_low_energy": final_black & valid & (luminance(package["inside_color"]) < 0.03),
        "Ain_high_and_Cin_dark": final_black & valid & (ain >= 0.80) & (cin_luma < 0.08),
        "strict_Cout_low": final_black & valid & (luminance(cout) < 0.03),
        "unfiltered_Cout_high_but_strict_low": (
            final_black & valid & (luminance(cout_unfiltered) >= 0.10)
            & (luminance(cout) < 0.03)
        ),
        "safe_R_has_energy_but_formal_R_off": (
            final_black & (luminance(safe_r) > 0.01) & (luminance(formal_r) <= 1e-8)
        ),
    }
    union = torch.zeros_like(final_black)
    for value in labels.values():
        union = union | value
    labels["residual_unattributed"] = final_black & ~union
    stats = {
        "near_black_threshold": NEAR_BLACK_THRESHOLD,
        "hard_mask_pixel_count": int(hard.sum()),
        "near_black_count": int(final_black.sum()),
        "near_black_fraction": _fraction(final_black[hard]),
        "labels": {
            name: {"count": int(value.sum()), "fraction_of_near_black": _fraction(value[final_black])}
            for name, value in labels.items()
        },
    }
    maps = {name: value.to(package["final"]) for name, value in labels.items()}
    maps["near_black"] = final_black.to(package["final"])
    return stats, maps


def overbright_diagnostics(package: dict, arm0_package: dict | None = None) -> dict:
    final_linear = package.get("final_linear", package["final"])
    over = final_linear > 1.0
    branch_names = ("reflection_contribution", "inside_contribution", "cout_contribution")
    branch_energy = {
        name: float(package[name].detach().float().abs().mean())
        for name in branch_names if name in package
    }
    nonzero = {
        name: luminance(package[name]).abs() > 1e-8
        for name in branch_names if name in package
    }
    overlap = torch.zeros_like(over[..., :1])
    if len(nonzero) == 3:
        overlap = nonzero["reflection_contribution"] & nonzero["inside_contribution"] \
            & nonzero["cout_contribution"]
    dominant = {}
    if bool(over.any()):
        over_pixels = over.any(dim=-1, keepdim=True)
        energies = {
            name: luminance(package[name]).abs()[over_pixels]
            for name in branch_names if name in package
        }
        if energies:
            stacked = torch.cat([value.reshape(-1, 1) for value in energies.values()], dim=1)
            winner = stacked.argmax(dim=1)
            for index, name in enumerate(energies):
                dominant[name] = int((winner == index).sum())
    delta = None
    if arm0_package is not None:
        delta_map = final_linear - arm0_package.get("final_linear", arm0_package["final"])
        delta = {
            "mean_positive": float(delta_map.clamp_min(0).mean()),
            "max_positive": float(delta_map.clamp_min(0).max()),
        }
    return {
        "pre_clamp_over_1_pixel_fraction": _fraction(over.any(dim=-1, keepdim=True)),
        "pre_clamp_max_rgb": float(final_linear.detach().float().max()) if final_linear.numel() else 0.0,
        "branch_contribution_energy": branch_energy,
        "r_cin_cout_simultaneously_nonzero_fraction": _fraction(overlap),
        "positive_energy_delta_from_arm0": delta,
        "overbright_dominant_branch_counts": dominant,
    }


def assert_outside_mask_bitwise_parity(arm0: dict, candidate: dict, mask_hard: torch.Tensor):
    hard = _mask_hwc(mask_hard, reference=arm0["final"]) >= 0.5
    outside = ~hard
    checks = {}
    for name in ("final", "diffuse_contribution", "reflection_contribution"):
        if name in arm0 and name in candidate:
            equal = torch.equal(arm0[name][outside.expand_as(arm0[name])],
                                candidate[name][outside.expand_as(candidate[name])])
            checks[name] = bool(equal)
            if not equal:
                raise RuntimeError(f"outside-mask bitwise parity failed for {name}")
    return checks


def apply_legacy_fallback(arm0: dict, legacy: dict, mask_hard: torch.Tensor) -> dict:
    hard = _mask_hwc(mask_hard, reference=arm0["final"]) >= 0.5
    invalid = hard & ~(arm0["two_hit_valid"] >= 0.5)
    result = dict(arm0)
    for name in ("final", "diffuse_contribution", "reflection_contribution"):
        if name in legacy and name in result:
            updated = result[name].clone()
            updated[invalid.expand_as(updated)] = legacy[name][invalid.expand_as(legacy[name])]
            result[name] = updated
    result["render"] = result["final"].clamp(0.0, 1.0).permute(2, 0, 1)
    result["fallback_pixel_mask"] = invalid.to(arm0["final"])
    result["fallback_validation"] = {
        "pixel_count": int(invalid.sum()),
        "domain": "mask_hard & !valid_two_hit",
        "legacy_final_sha256": tensor_sha256(legacy["final"][invalid.expand_as(legacy["final"])]),
        "applied_final_sha256": tensor_sha256(result["final"][invalid.expand_as(result["final"])]),
    }
    return result


def exit_face_normals(cuboid, back_points: torch.Tensor):
    local = cuboid.world_to_local(back_points)
    lower = cuboid.lower.to(local)
    upper = cuboid.upper.to(local)
    distances = torch.cat(((local - lower).abs(), (upper - local).abs()), dim=-1)
    face = distances.argmin(dim=-1)
    normals_local = torch.zeros_like(local)
    axis = face % 3
    sign = torch.where(face < 3, -torch.ones_like(face, dtype=local.dtype), torch.ones_like(face, dtype=local.dtype))
    normals_local.scatter_(1, axis[:, None], sign[:, None])
    return cuboid.local_to_world(normals_local)


def make_back_face_candidate_filter(cuboid, candidate_classes, back_points, delta: float):
    normals = exit_face_normals(cuboid, back_points)
    delta = float(delta)
    cursor = {"start": 0}

    def _filter(candidate_ids, exact_valid, points, distance, origins, directions):
        start = cursor["start"]
        end = start + int(origins.shape[0])
        if end > back_points.shape[0]:
            raise RuntimeError("Arm 3 candidate filter received more rays than back-face references")
        chunk_back_points = back_points[start:end].to(points)
        chunk_normals = normals[start:end].to(points)
        cursor["start"] = end
        safe_ids = candidate_ids.clamp_min(0)
        classes = candidate_classes.to(candidate_ids.device)[safe_ids]
        strict_outside = classes == SUPPORT_STRICT_OUTSIDE
        handoff_class = (classes == SUPPORT_INTERFACE) | (classes == SUPPORT_CROSSING)
        clearance = cuboid.signed_clearance(points)
        exit_distance = ((points - chunk_back_points[:, None, :]) * chunk_normals[:, None, :]).sum(dim=-1)
        accepted_handoff = handoff_class & (exit_distance > delta) & (clearance < -delta)
        strict_inside = classes == SUPPORT_STRICT_INSIDE
        return (strict_outside | accepted_handoff) & ~strict_inside

    return _filter


def render_arm3_with_handoff(state, background, static_inputs, delta: float = 1e-5) -> dict:
    if state.cuboid_space is None:
        raise RuntimeError("Arm 3 requires cuboid_space")
    classes = state.cuboid_space.classify_support(
        state.diffuse.get_xyz.detach(), state.diffuse.get_rotation.detach(),
        state.diffuse.get_scaling.detach(), sigma=state.support_sigma,
    )
    eligible = (
        (classes == SUPPORT_STRICT_OUTSIDE)
        | (classes == SUPPORT_INTERFACE)
        | (classes == SUPPORT_CROSSING)
    )
    handoff_only = (classes == SUPPORT_INTERFACE) | (classes == SUPPORT_CROSSING)
    candidate_filter = make_back_face_candidate_filter(
        state.cuboid_space, classes, static_inputs["back_position_selected"], delta
    )
    outputs, _, accepted_diag = _trace(
        state.diffuse_raytrace, static_inputs["second_origin"], static_inputs["direction"],
        state.diffuse_acceleration, state, False, True,
        surfel_filter=eligible, candidate_hit_filter=candidate_filter,
    )
    _, _, all_handoff_diag = _trace(
        state.diffuse_raytrace, static_inputs["second_origin"], static_inputs["direction"],
        state.diffuse_acceleration, state, False, True,
        surfel_filter=handoff_only,
    )
    arm_inputs = dict(static_inputs)
    arm_inputs.update({
        "outside_raw": outputs[0],
        "outside_alpha": outputs[1],
        "outside_relative_depth": outputs[2],
        "outside_hit": outputs[3],
    })
    package = render_from_static_dr(state, background, arm_inputs, return_ray_diagnostics=True)
    height, width = int(static_inputs["height"]), int(static_inputs["width"])
    indices = static_inputs["indices"]
    accepted = accepted_diag.exact_intersection_counts
    rejected = (all_handoff_diag.exact_intersection_counts - accepted).clamp_min(0)
    ray_background = background.reshape(1, 3).to(outputs[0])
    strict_color = (
        static_inputs["outside_raw"]
        + (1.0 - static_inputs["outside_alpha"]) * ray_background
    )
    package["cout_strict"] = _scatter(strict_color, indices, height, width, 3)
    handoff_color = outputs[0] + (1.0 - outputs[1]) * ray_background
    package["cout_handoff"] = _scatter(handoff_color, indices, height, width, 3)
    package["cout_handoff_difference"] = torch.abs(package["cout_handoff"] - package["cout_strict"])
    package["handoff_accepted_hit_count"] = _scatter(
        accepted.to(outputs[0])[:, None], indices, height, width, 1
    )
    package["handoff_rejected_hit_count"] = _scatter(
        rejected.to(outputs[0])[:, None], indices, height, width, 1
    )
    package["handoff_metadata"] = {
        "delta": float(delta),
        "strict_inside_candidate_count": int((classes == SUPPORT_STRICT_INSIDE).sum()),
        "strict_inside_admitted": 0,
        "accepted_exact_hit_count": int(accepted.sum()),
        "rejected_exact_hit_count": int(rejected.sum()),
    }
    return package


def save_tensor_products(package: dict, directory: Path, *, maps: dict | None = None):
    directory.mkdir(parents=True, exist_ok=True)
    tensor_dir = directory / "tensors"
    tensor_dir.mkdir(exist_ok=True)
    names = (
        "final", "diffuse_contribution", "diffuse_unfiltered",
        "reflection_contribution", "reflection_unfiltered",
        "reflection_strict_outside_safe", "inside_color", "inside_alpha",
        "inside_depth", "conditional_inside_color", "outside_color",
        "outside_alpha", "outside_depth", "outside_unfiltered",
        "outside_strict_outside_safe", "transmittance_color",
        "transmittance_alpha", "transmittance_contribution",
        "inside_contribution", "cout_contribution", "two_hit_valid",
        "transmittance_valid", "fallback_pixel_mask", "cout_strict",
        "cout_handoff", "cout_handoff_difference",
        "handoff_accepted_hit_count", "handoff_rejected_hit_count",
    )
    for name in names:
        if name in package and torch.is_tensor(package[name]):
            torch.save(package[name].detach().cpu(), tensor_dir / f"{name}.pt")
            value = package[name].detach()
            if value.ndim == 3 and value.shape[-1] in (1, 3):
                image = value
                if value.shape[-1] == 1:
                    image = value.repeat(1, 1, 3)
                save_image(image.permute(2, 0, 1).clamp(0, 1), directory / f"{name}.png")
    for name, value in (maps or {}).items():
        torch.save(value.detach().cpu(), tensor_dir / f"{name}.pt")
        image = value
        if image.ndim == 3 and image.shape[-1] == 1:
            image = image.repeat(1, 1, 3)
        if image.ndim == 3 and image.shape[-1] == 3:
            save_image(image.permute(2, 0, 1).clamp(0, 1), directory / f"{name}.png")


def write_aggregate(rows: list[dict], directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "semantic_renderer_ablation_aggregate.json"
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    csv_path = directory / "semantic_renderer_ablation_aggregate.csv"
    flat_rows = []
    for row in rows:
        flat = {
            "group": row["group"], "stem": row["stem"], "arm": row["arm"],
            "near_black_fraction": row["black_attribution"]["near_black_fraction"],
            "overbright_fraction": row["overbright"]["pre_clamp_over_1_pixel_fraction"],
            "outside_parity": all(row["outside_mask_parity"].values()),
        }
        flat_rows.append(flat)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]) if flat_rows else [])
        if flat_rows:
            writer.writeheader()
            writer.writerows(flat_rows)
    report = directory / "semantic_renderer_ablation_report.md"
    report.write_text(
        "# D-015 semantic-renderer zero-update ablation\n\n"
        "Verdict: `AWAITING_USER_REVIEW`\n\n"
        "All formal metrics are computed from saved float tensors; PNG files are review-only. "
        "No optimizer update, checkpoint, PLY, or semantic-separation claim is produced.\n",
        encoding="utf-8",
    )
    return {"json": str(json_path), "csv": str(csv_path), "markdown": str(report)}
