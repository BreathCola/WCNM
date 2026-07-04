"""Stage D D/R/T rendering with frozen two-hit geometry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.autograd.profiler import record_function

from gaussian_renderer.reflection_renderer import StageBRenderState, render as render_stage_b
from geometry.cuboid_space import SUPPORT_CLASS_NAMES, SUPPORT_STRICT_INSIDE
from raytracer.acceleration_structure import CudaLBVH
from raytracer.tracer import raytrace


class DiffuseRaytraceAdapter:
    """Expose Diffuse base color through the common 13-channel raytrace contract."""

    def __init__(self, diffuse):
        self.diffuse = diffuse
        self.topology_version = 0

    @property
    def _xyz(self): return self.diffuse._xyz
    @property
    def _rotation(self): return self.diffuse._rotation
    @property
    def _scaling(self): return self.diffuse._scaling
    @property
    def _opacity(self): return self.diffuse._opacity
    @property
    def _color(self): return self.diffuse._base_color
    @property
    def get_xyz(self): return self.diffuse.get_xyz
    @property
    def get_rotation(self): return self.diffuse.get_rotation
    @property
    def get_scaling(self): return self.diffuse.get_scaling


@dataclass
class StageDRenderState:
    diffuse: object
    reflection: object
    transmittance: object
    geometry_release: object
    scene_radius: float
    ray_chunk_size: int = 512
    ray_cutoff_sigma: float = 3.0
    ray_hit_threshold: float = 1e-4
    ray_epsilon_scale: float = 1e-4
    material_alpha_threshold: float = 1e-4
    roughness_min: float = 0.03
    roughness_remap: bool = False
    ray_checkpoint_chunks: bool = True
    reflection_acceleration: object = None
    transmittance_acceleration: object = None
    diffuse_acceleration: object = None
    cuboid_space: object = None
    semantic_repair: bool = False
    transparent_path_mode: str = "legacy_d_gbuffer"
    transparent_direct_mode: str = "legacy"
    transparent_reflection_mode: str = "legacy"
    cout_ownership_mode: str = "legacy"
    support_sigma: float = 3.0
    transfer_depth_margin: float = 0.05

    def __post_init__(self):
        self.reflection_acceleration = self.reflection_acceleration or CudaLBVH(self.ray_cutoff_sigma)
        self.transmittance_acceleration = self.transmittance_acceleration or CudaLBVH(self.ray_cutoff_sigma)
        self.diffuse_acceleration = self.diffuse_acceleration or CudaLBVH(self.ray_cutoff_sigma)
        self.diffuse_raytrace = DiffuseRaytraceAdapter(self.diffuse)
        self.stage_b = StageBRenderState(
            diffuse=self.diffuse, reflection=self.reflection,
            scene_radius=self.scene_radius, ray_chunk_size=self.ray_chunk_size,
            ray_cutoff_sigma=self.ray_cutoff_sigma,
            ray_hit_threshold=self.ray_hit_threshold,
            ray_epsilon_scale=self.ray_epsilon_scale,
            material_alpha_threshold=self.material_alpha_threshold,
            roughness_min=self.roughness_min,
            roughness_remap=self.roughness_remap,
            ray_background="scene", ray_checkpoint_chunks=self.ray_checkpoint_chunks,
            acceleration=self.reflection_acceleration,
            cuboid_space=self.cuboid_space,
            semantic_repair=self.semantic_repair,
            geometry_release=self.geometry_release,
            transparent_path_mode=self.transparent_path_mode,
            transparent_direct_mode=self.transparent_direct_mode,
            transparent_reflection_mode=self.transparent_reflection_mode,
            support_sigma=self.support_sigma,
        )
        if self.cout_ownership_mode not in ("legacy", "support_safe_outside"):
            raise ValueError("unsupported Cout ownership mode")

    def mark_parameters_updated(self):
        self.reflection_acceleration.mark_parameters_updated()
        self.transmittance_acceleration.mark_parameters_updated()
        self.diffuse_acceleration.mark_parameters_updated()

    def mark_transmittance_updated(self):
        self.transmittance_acceleration.mark_parameters_updated()


def _scatter(values, indices, height, width, channels, fill=0.0):
    output = values.new_full((height * width, channels), float(fill))
    if indices.numel():
        output = output.index_copy(0, indices, values)
    return output.reshape(height, width, channels)


def _assert_finite_outputs(output, names):
    checks = torch.stack([torch.isfinite(output[name]).all() for name in names])
    if not bool(checks.all()):
        values = checks.detach().cpu().tolist()
        failed = [name for name, finite in zip(names, values) if not finite]
        raise FloatingPointError(f"Stage D outputs contain NaN or Inf: {failed}")


def alpha_over_transmittance(inside_color, inside_alpha, outside_color, outside_alpha):
    color = inside_color + (1.0 - inside_alpha) * outside_color
    alpha = inside_alpha + (1.0 - inside_alpha) * outside_alpha
    return color, alpha


def semantic_cout_outside_only(components):
    """Select the audited outside class as the only formal second-bounce Cout."""
    if set(components) != {"inside", "interface", "outside"}:
        raise ValueError("semantic Cout requires disjoint inside/interface/outside components")
    return components["outside"]


def support_safe_cout_outside_only(components):
    if set(components) != set(SUPPORT_CLASS_NAMES):
        raise ValueError("support-safe Cout requires the four disjoint support classes")
    return components["strict_outside_safe"]


def cuboid_front_transmission_inputs(front_position, direction, t_near, epsilon):
    """Return the T first origin and absolute Din baseline from frozen geometry."""
    if front_position.shape != direction.shape or front_position.shape[-1] != 3:
        raise ValueError("cuboid-front T position/direction shapes do not match")
    if t_near.shape != front_position.shape[:-1] + (1,):
        raise ValueError("cuboid-front T t_near shape does not match")
    if not torch.isfinite(front_position).all() or not torch.isfinite(direction).all() \
            or not torch.isfinite(t_near).all():
        raise FloatingPointError("cuboid-front T inputs are non-finite")
    return front_position + float(epsilon) * direction, t_near + float(epsilon)


def _trace(
    model, origins, directions, acceleration, state, return_aux, diagnostics,
    surfel_filter=None,
):
    traced = raytrace(
        model, origins, directions, acceleration=acceleration,
        chunk_size=state.ray_chunk_size, cutoff_sigma=state.ray_cutoff_sigma,
        hit_threshold=state.ray_hit_threshold, return_aux=return_aux,
        return_diagnostics=diagnostics,
        checkpoint_chunks=bool(state.ray_checkpoint_chunks and not diagnostics),
        surfel_filter=surfel_filter,
    )
    if return_aux and diagnostics:
        outputs, aux, diagnostic = traced
    elif return_aux:
        outputs, aux = traced
        diagnostic = None
    elif diagnostics:
        outputs, diagnostic = traced
        aux = None
    else:
        outputs, aux, diagnostic = traced, None, None
    return outputs, aux, diagnostic


def build_static_dr_inputs(
    camera, state, pipe, background,
    return_ray_aux=False, return_ray_diagnostics=False,
):
    """Compute the D/R and frozen-geometry values that do not depend on T."""
    state.stage_b.ray_chunk_size = state.ray_chunk_size
    state.stage_b.ray_checkpoint_chunks = bool(
        state.ray_checkpoint_chunks and not return_ray_diagnostics
    )
    with record_function("stage_d.stage_b_forward"):
        package = render_stage_b(
            camera, state.stage_b, pipe, background,
            return_ray_aux=return_ray_aux,
            return_ray_diagnostics=return_ray_diagnostics,
        )
    height, width = package["alpha"].shape[:2]
    cache = state.geometry_release.load_view(Path(str(camera.image_name)).stem)
    device, dtype = package["position"].device, package["position"].dtype
    valid_cache = torch.from_numpy(cache["valid_two_hit"]).to(device=device, dtype=torch.bool)
    t_near = torch.from_numpy(cache["t_near"]).to(device=device, dtype=dtype)
    t_far = torch.from_numpy(cache["t_far"]).to(device=device, dtype=dtype)
    back_position = torch.from_numpy(cache["back_position"]).to(device=device, dtype=dtype)
    if valid_cache.shape != (height, width):
        raise ValueError("frozen two-hit cache resolution does not match the training camera")
    position = package["position"]
    surface_valid = (
        (package["alpha"][..., 0] > state.material_alpha_threshold)
        & torch.isfinite(position).all(dim=-1)
    )
    if state.transparent_path_mode == "cuboid_front_v1":
        required = (
            "front_position", "front_normal", "frozen_camera_direction",
            "transparent_path_valid", "front_plane_residual",
        )
        missing = [name for name in required if name not in package]
        if missing:
            raise RuntimeError(f"cuboid-front Stage B package is missing {missing}")
        valid = package["transparent_path_valid"][..., 0] >= 0.5
        hard = camera.specular_mask
        if hard.ndim == 3 and hard.shape[0] == 1:
            hard = hard.permute(1, 2, 0)
        elif hard.ndim == 2:
            hard = hard[..., None]
        if hard.ndim != 3 or hard.shape[-1] != 1:
            raise ValueError("cuboid-front transparent mask must be one-channel")
        if not torch.equal(valid, valid_cache & (hard[..., 0] >= 0.5)):
            raise RuntimeError("cuboid-front valid domain drifted from mask_hard & valid_two_hit")
    else:
        valid = valid_cache & surface_valid
    indices = valid.reshape(-1).nonzero(as_tuple=False)[:, 0]
    flat_position = (
        package["front_position"] if state.transparent_path_mode == "cuboid_front_v1"
        else position
    ).reshape(-1, 3)[indices]
    flat_back = back_position.reshape(-1, 3)[indices]
    center = camera.camera_center.reshape(1, 3).to(flat_position)
    direction = (
        package["frozen_camera_direction"].reshape(-1, 3)[indices]
        if state.transparent_path_mode == "cuboid_front_v1"
        else F.normalize(flat_back - center, dim=-1, eps=1e-8)
    )
    epsilon = float(state.ray_epsilon_scale * state.scene_radius)
    if state.transparent_path_mode == "cuboid_front_v1":
        first_origin, first_distance = cuboid_front_transmission_inputs(
            flat_position, direction, t_near.reshape(-1, 1)[indices], epsilon,
        )
    else:
        first_origin = flat_position + epsilon * direction
        first_distance = torch.linalg.vector_norm(
            flat_position - center, dim=-1, keepdim=True,
        )
    second_origin = flat_back + epsilon * direction
    front_origin_tnear_error = (
        torch.linalg.vector_norm(first_origin - center, dim=-1, keepdim=True)
        - (t_near.reshape(-1, 1)[indices] + epsilon)
    )
    front_normal_selected = (
        package["front_normal"].reshape(-1, 3)[indices]
        if state.transparent_path_mode == "cuboid_front_v1" else torch.zeros_like(direction)
    )
    front_normal_faceforward_dot = (front_normal_selected * (-direction)).sum(
        dim=-1, keepdim=True
    )

    with record_function("stage_d.outside_d_forward"):
        outside_outputs, outside_aux, outside_diagnostics = _trace(
            state.diffuse_raytrace, second_origin, direction,
            state.diffuse_acceleration, state, return_ray_aux,
            return_ray_diagnostics,
        )
    outside_raw, outside_alpha, outside_relative_depth, outside_hit = outside_outputs
    semantic_cout_components = None
    semantic_cout_stats = None
    if state.semantic_repair or state.cout_ownership_mode == "support_safe_outside":
        if state.cuboid_space is None:
            raise RuntimeError("semantic Cout filtering requires cuboid_space")
        if state.cout_ownership_mode == "support_safe_outside":
            class_names = SUPPORT_CLASS_NAMES
            class_masks = state.cuboid_space.support_masks(
                state.diffuse.get_xyz.detach(), state.diffuse.get_rotation.detach(),
                state.diffuse.get_scaling.detach(), sigma=state.support_sigma,
            )
        else:
            class_names = ("inside", "interface", "outside")
            class_masks = state.cuboid_space.masks(state.diffuse.get_xyz.detach())
        semantic_cout_components = {}
        semantic_cout_stats = {}
        for class_name in class_names:
            class_outputs, _, class_diagnostics = _trace(
                state.diffuse_raytrace, second_origin, direction,
                state.diffuse_acceleration, state, False,
                return_ray_diagnostics, surfel_filter=class_masks[class_name],
            )
            class_raw, class_alpha, class_depth, class_hit = class_outputs
            semantic_cout_components[class_name] = {
                "raw": class_raw, "alpha": class_alpha, "depth": class_depth,
                "hit": class_hit,
                "surfel_count": int(class_masks[class_name].sum()),
                "candidate_count": (
                    int(class_diagnostics.eligible_candidate_counts.sum())
                    if class_diagnostics is not None else None
                ),
                "hit_count": int(class_hit.sum()),
            }
            semantic_cout_stats[class_name] = {
                "surfel_count": int(class_masks[class_name].sum()),
                "candidate_count": (
                    int(class_diagnostics.eligible_candidate_counts.sum())
                    if class_diagnostics is not None else None
                ),
                "hit_count": int(class_hit.sum()),
                "contribution_energy": float(class_raw.detach().mean())
                if class_raw.numel() else 0.0,
            }
        formal = (
            support_safe_cout_outside_only(semantic_cout_components)
            if state.cout_ownership_mode == "support_safe_outside"
            else semantic_cout_outside_only(semantic_cout_components)
        )
        outside_raw = formal["raw"]
        outside_alpha = formal["alpha"]
        outside_relative_depth = formal["depth"]
        outside_hit = formal["hit"]
    transfer_evidence = None
    if state.transparent_path_mode == "cuboid_front_v1":
        support_masks = state.cuboid_space.support_masks(
            state.diffuse.get_xyz.detach(), state.diffuse.get_rotation.detach(),
            state.diffuse.get_scaling.detach(), sigma=state.support_sigma,
        )
        _, transfer_aux, _ = _trace(
            state.diffuse_raytrace, first_origin, direction,
            state.diffuse_acceleration, state, True, False,
            surfel_filter=support_masks["strict_inside_safe"],
        )
        if transfer_aux.contributing_ray_indices is None \
                or transfer_aux.contributing_depths is None:
            raise RuntimeError("cuboid-front transfer trace lacks per-ray/depth attribution")
        ray_ids = transfer_aux.contributing_ray_indices.long()
        absolute_depth = (
            t_near.reshape(-1)[indices][ray_ids]
            + epsilon + transfer_aux.contributing_depths
        )
        safe = (
            absolute_depth > t_near.reshape(-1)[indices][ray_ids] + float(state.transfer_depth_margin)
        ) & (
            absolute_depth < t_far.reshape(-1)[indices][ray_ids] - float(state.transfer_depth_margin)
        ) & (transfer_aux.contributing_weights > 0)
        ids = transfer_aux.contributing_indices[safe]
        weights = transfer_aux.contributing_weights[safe]
        if ids.numel():
            unique, inverse = torch.unique(ids, sorted=True, return_inverse=True)
            weight_sum = weights.new_zeros((unique.numel(),))
            hit_count = torch.zeros((unique.numel(),), dtype=torch.int64, device=ids.device)
            weight_sum.scatter_add_(0, inverse, weights)
            hit_count.scatter_add_(0, inverse, torch.ones_like(inverse, dtype=torch.int64))
        else:
            unique = ids
            weight_sum = weights
            hit_count = torch.zeros((0,), dtype=torch.int64, device=ids.device)
        transfer_evidence = {
            "surfel_ids": unique,
            "weight_sum": weight_sum,
            "hit_count": hit_count,
            "safe_contribution_count": int(ids.numel()),
            "depth_margin": float(state.transfer_depth_margin),
            "support_sigma": float(state.support_sigma),
        }
    return {
        "package": package,
        "height": int(height), "width": int(width), "indices": indices,
        "first_origin": first_origin, "direction": direction,
        "first_distance": first_distance,
        "far": t_far.reshape(-1, 1)[indices],
        "outside_raw": outside_raw, "outside_alpha": outside_alpha,
        "outside_relative_depth": outside_relative_depth,
        "outside_hit": outside_hit,
        "outside_unfiltered_raw": outside_outputs[0],
        "outside_unfiltered_alpha": outside_outputs[1],
        "outside_unfiltered_relative_depth": outside_outputs[2],
        "outside_unfiltered_hit": outside_outputs[3],
        "semantic_cout_components": semantic_cout_components,
        "semantic_cout_stats": semantic_cout_stats,
        "near_depth": t_near[..., None], "far_depth": t_far[..., None],
        "two_hit_valid": valid_cache[..., None].to(dtype),
        "transmittance_valid": valid[..., None].to(dtype),
        "outside_ray_aux": outside_aux,
        "outside_ray_diagnostics": outside_diagnostics,
        "outside_unfiltered_candidate_count": (
            int(outside_diagnostics.candidate_counts.sum())
            if outside_diagnostics is not None else None
        ),
        "front_position": package.get("front_position"),
        "front_normal": package.get("front_normal"),
        "frozen_camera_direction": package.get("frozen_camera_direction"),
        "front_plane_residual": package.get("front_plane_residual"),
        "transparent_path_mode": state.transparent_path_mode,
        "cout_ownership_mode": state.cout_ownership_mode,
        "transfer_evidence": transfer_evidence,
        "front_origin_tnear_error": _scatter(
            front_origin_tnear_error, indices, height, width, 1
        ),
        "front_normal_faceforward_dot": _scatter(
            front_normal_faceforward_dot, indices, height, width, 1
        ),
    }


def render_from_static_dr(
    state, background, static_inputs,
    return_ray_aux=False, return_ray_diagnostics=False,
):
    """Trace T and apply the unchanged Stage D composition to static D/R inputs."""
    package = dict(static_inputs["package"])
    height, width = int(static_inputs["height"]), int(static_inputs["width"])
    indices = static_inputs["indices"]
    with record_function("stage_d.inside_t_forward"):
        inside_outputs, inside_aux, inside_diagnostics = _trace(
            state.transmittance, static_inputs["first_origin"], static_inputs["direction"],
            state.transmittance_acceleration, state, return_ray_aux,
            return_ray_diagnostics,
        )
    inside_raw, inside_alpha, inside_relative_depth, inside_hit = inside_outputs
    outside_raw = static_inputs["outside_raw"]
    outside_alpha = static_inputs["outside_alpha"]
    outside_relative_depth = static_inputs["outside_relative_depth"]
    outside_hit = static_inputs["outside_hit"]
    dtype = package["position"].dtype
    ray_background = background.reshape(1, 3).to(outside_raw)
    outside_color = outside_raw + (1.0 - outside_alpha) * ray_background
    outside_unfiltered_raw = static_inputs.get("outside_unfiltered_raw", outside_raw)
    outside_unfiltered_alpha = static_inputs.get("outside_unfiltered_alpha", outside_alpha)
    outside_unfiltered_color = (
        outside_unfiltered_raw + (1.0 - outside_unfiltered_alpha) * ray_background
    )
    transmittance_color, transmittance_alpha = alpha_over_transmittance(
        inside_raw, inside_alpha, outside_color, outside_alpha
    )

    inside_depth = static_inputs["first_distance"] + inside_relative_depth
    far = static_inputs["far"]
    outside_depth = far + outside_relative_depth
    violation = torch.relu(inside_depth - far)
    conditional_inside = inside_raw / inside_alpha.clamp_min(1e-6)

    alpha = package["alpha"].reshape(-1, 1)[indices]
    ks = package["surface_ks"].reshape(-1, 1)[indices]
    fresnel = package["microfacet_F"].reshape(-1, 3)[indices]
    wt = (1.0 - fresnel).clamp(0.0, 1.0)
    transmission_weight = alpha * ks * wt
    inside_contribution = transmission_weight * inside_raw
    cout_contribution = transmission_weight * (1.0 - inside_alpha) * outside_color
    transmittance_contribution = inside_contribution + cout_contribution
    contribution_map = _scatter(transmittance_contribution, indices, height, width, 3)
    inside_contribution_map = _scatter(inside_contribution, indices, height, width, 3)
    cout_contribution_map = _scatter(cout_contribution, indices, height, width, 3)
    final = (
        package["diffuse_contribution"] + package["reflection_contribution"]
        + contribution_map
        + (1.0 - package["alpha"]) * background.reshape(1, 1, 3)
    )

    package.update({
        "final": final.clamp(0.0, 1.0),
        "render": final.clamp(0.0, 1.0).permute(2, 0, 1),
        "transmittance_contribution": contribution_map,
        "inside_contribution": inside_contribution_map,
        "cout_contribution": cout_contribution_map,
        "inside_color": _scatter(inside_raw, indices, height, width, 3),
        "inside_alpha": _scatter(inside_alpha, indices, height, width, 1),
        "inside_depth": _scatter(inside_depth, indices, height, width, 1),
        "inside_hit_mask": _scatter(inside_hit.to(dtype), indices, height, width, 1),
        "outside_color": _scatter(outside_color, indices, height, width, 3),
        "outside_alpha": _scatter(outside_alpha, indices, height, width, 1),
        "outside_depth": _scatter(outside_depth, indices, height, width, 1),
        "outside_hit_mask": _scatter(outside_hit.to(dtype), indices, height, width, 1),
        "conditional_inside_color": _scatter(
            conditional_inside, indices, height, width, 3
        ),
        "outside_unfiltered": _scatter(
            outside_unfiltered_color, indices, height, width, 3
        ),
        "transmittance_color": _scatter(transmittance_color, indices, height, width, 3),
        "transmittance_alpha": _scatter(transmittance_alpha, indices, height, width, 1),
        "depth_violation": _scatter(violation, indices, height, width, 1),
        "near_depth": static_inputs["near_depth"],
        "far_depth": static_inputs["far_depth"],
        "two_hit_valid": static_inputs["two_hit_valid"],
        "transmittance_valid": static_inputs["transmittance_valid"],
        "transmittance_valid_count": int(indices.numel()),
        "inside_ray_aux": inside_aux,
        "outside_ray_aux": static_inputs.get("outside_ray_aux"),
        "inside_ray_diagnostics": inside_diagnostics,
        "outside_ray_diagnostics": static_inputs.get("outside_ray_diagnostics"),
        "front_origin_tnear_error": static_inputs.get(
            "front_origin_tnear_error", package["alpha"].new_zeros((height, width, 1))
        ),
        "front_normal_faceforward_dot": static_inputs.get(
            "front_normal_faceforward_dot", package["alpha"].new_zeros((height, width, 1))
        ),
    })
    package["final_t_off"] = (package["final"] - contribution_map).clamp(0.0, 1.0)
    package["final_cout_off"] = (package["final"] - cout_contribution_map).clamp(0.0, 1.0)
    package["final_d_direct_off"] = (
        package["final"] - package["diffuse_contribution"]
    ).clamp(0.0, 1.0)
    package["final_r_off"] = (
        package["final"] - package["reflection_contribution"]
    ).clamp(0.0, 1.0)
    semantic_components = static_inputs.get("semantic_cout_components")
    if state.semantic_repair or state.cout_ownership_mode == "support_safe_outside":
        stats = dict(static_inputs.get("semantic_cout_stats") or {})
        if semantic_components is not None:
            stats = {}
            class_alpha_maps = []
            for class_name, component in semantic_components.items():
                absolute_depth = far + component["depth"]
                package[f"outside_{class_name}"] = _scatter(
                    component["raw"], indices, height, width, 3
                )
                package[f"outside_{class_name}_alpha"] = _scatter(
                    component["alpha"], indices, height, width, 1
                )
                class_alpha_maps.append(package[f"outside_{class_name}_alpha"])
                package[f"outside_{class_name}_depth"] = _scatter(
                    absolute_depth, indices, height, width, 1
                )
                package[f"outside_{class_name}_hit"] = _scatter(
                    component["hit"].to(dtype), indices, height, width, 1
                )
                stats[class_name] = {
                    "surfel_count": component["surfel_count"],
                    "candidate_count": component["candidate_count"],
                    "hit_count": component["hit_count"],
                    "contribution_energy": float(component["raw"].detach().mean())
                    if component["raw"].numel() else 0.0,
                }
            if class_alpha_maps:
                stacked = torch.cat(class_alpha_maps, dim=-1)
                winner = stacked.argmax(dim=-1)
                palette = package["final"].new_tensor([
                    [0.1, 0.8, 0.1], [0.9, 0.8, 0.1],
                    [0.1, 0.4, 0.9], [0.9, 0.1, 0.1],
                ])[: stacked.shape[-1]]
                ownership = palette[winner] * (stacked.max(dim=-1).values > 0)[..., None]
                package["cout_ownership_class_map"] = ownership
        package["outside_final_filtered"] = package["outside_color"]
        package["semantic_cout_stats"] = stats
        cout_difference = torch.abs(outside_unfiltered_color - outside_color)
        package["semantic_cout_filter_stats"] = {
            "formal_mode": state.cout_ownership_mode,
            "support_sigma": float(state.support_sigma),
            "unfiltered_surfel_count": int(state.diffuse.get_xyz.shape[0]),
            "unfiltered_candidate_count": (
                static_inputs.get("outside_unfiltered_candidate_count")
            ),
            "unfiltered_hit_count": int(static_inputs["outside_unfiltered_hit"].sum()),
            "rgb_difference_mean_abs": float(cout_difference.detach().mean())
            if cout_difference.numel() else 0.0,
            "rgb_difference_max_abs": float(cout_difference.detach().max())
            if cout_difference.numel() else 0.0,
        }
        if state.transparent_path_mode == "cuboid_front_v1":
            t_classes = state.cuboid_space.classify_support(
                state.transmittance.get_xyz.detach(),
                state.transmittance.get_rotation.detach(),
                state.transmittance.get_scaling.detach(), sigma=state.support_sigma,
            )
            t_names = SUPPORT_CLASS_NAMES
            t_safe = bool((t_classes == SUPPORT_STRICT_INSIDE).all())
        else:
            t_classes = state.cuboid_space.classify(state.transmittance.get_xyz.detach())
            t_names = ("inside", "interface", "outside")
            t_safe = bool((t_classes == 0).all())
        t_map = package["final"].new_zeros((height, width, 3))
        if t_safe:
            t_map[..., 1:2] = package["transmittance_valid"]
        else:
            t_map[..., 0:1] = package["transmittance_valid"]
        package["t_spatial_class_map"] = t_map
        package["t_spatial_counts"] = {
            name: int((t_classes == index).sum())
            for index, name in enumerate(t_names)
        }
        package["t_support_legal"] = t_map
    if state.transparent_path_mode == "cuboid_front_v1":
        hard_domain = package["transparent_mask_hard"] > 0.5
        d_values = package["diffuse_contribution"][hard_domain.expand_as(package["diffuse_contribution"])]
        r_values = package["reflection_contribution"][hard_domain.expand_as(package["reflection_contribution"])]
        if state.transparent_direct_mode == "off" and bool((d_values != 0).any()):
            raise RuntimeError("transparent direct contribution is nonzero in ownership-off mode")
        if state.transparent_reflection_mode == "off" and bool((r_values != 0).any()):
            raise RuntimeError("transparent reflection contribution is nonzero in ownership-off mode")
    _assert_finite_outputs(package, (
        "final", "transmittance_contribution", "inside_color", "inside_alpha",
        "inside_depth", "outside_color", "outside_alpha", "outside_depth",
        "transmittance_color", "transmittance_alpha", "depth_violation",
        "near_depth", "far_depth",
    ))
    return package


def render(
    camera,
    state: StageDRenderState,
    pipe,
    background: torch.Tensor,
    return_ray_aux: bool = False,
    return_ray_diagnostics: bool = False,
):
    static_inputs = build_static_dr_inputs(
        camera, state, pipe, background,
        return_ray_aux=return_ray_aux,
        return_ray_diagnostics=return_ray_diagnostics,
    )
    return render_from_static_dr(
        state, background, static_inputs,
        return_ray_aux=return_ray_aux,
        return_ray_diagnostics=return_ray_diagnostics,
    )
