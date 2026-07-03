"""Stage D D/R/T rendering with frozen two-hit geometry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.autograd.profiler import record_function

from gaussian_renderer.reflection_renderer import StageBRenderState, render as render_stage_b
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
        )

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
    valid = valid_cache & surface_valid
    indices = valid.reshape(-1).nonzero(as_tuple=False)[:, 0]
    flat_position = position.reshape(-1, 3)[indices]
    flat_back = back_position.reshape(-1, 3)[indices]
    center = camera.camera_center.reshape(1, 3).to(flat_position)
    direction = F.normalize(flat_back - center, dim=-1, eps=1e-8)
    epsilon = float(state.ray_epsilon_scale * state.scene_radius)
    first_origin = flat_position + epsilon * direction
    second_origin = flat_back + epsilon * direction

    with record_function("stage_d.outside_d_forward"):
        outside_outputs, outside_aux, outside_diagnostics = _trace(
            state.diffuse_raytrace, second_origin, direction,
            state.diffuse_acceleration, state, return_ray_aux,
            return_ray_diagnostics,
        )
    outside_raw, outside_alpha, outside_relative_depth, outside_hit = outside_outputs
    semantic_cout_components = None
    semantic_cout_stats = None
    if state.semantic_repair:
        if state.cuboid_space is None:
            raise RuntimeError("semantic Cout filtering requires cuboid_space")
        class_masks = state.cuboid_space.masks(state.diffuse.get_xyz.detach())
        semantic_cout_components = {}
        semantic_cout_stats = {}
        for class_name in ("inside", "interface", "outside"):
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
        formal = semantic_cout_outside_only(semantic_cout_components)
        outside_raw = formal["raw"]
        outside_alpha = formal["alpha"]
        outside_relative_depth = formal["depth"]
        outside_hit = formal["hit"]
    return {
        "package": package,
        "height": int(height), "width": int(width), "indices": indices,
        "first_origin": first_origin, "direction": direction,
        "first_distance": torch.linalg.vector_norm(flat_position - center, dim=-1, keepdim=True),
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
    transmittance_contribution = alpha * ks * wt * transmittance_color
    contribution_map = _scatter(transmittance_contribution, indices, height, width, 3)
    final = (
        package["diffuse_contribution"] + package["reflection_contribution"]
        + contribution_map
        + (1.0 - package["alpha"]) * background.reshape(1, 1, 3)
    )

    package.update({
        "final": final.clamp(0.0, 1.0),
        "render": final.clamp(0.0, 1.0).permute(2, 0, 1),
        "transmittance_contribution": contribution_map,
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
    })
    semantic_components = static_inputs.get("semantic_cout_components")
    if state.semantic_repair:
        stats = dict(static_inputs.get("semantic_cout_stats") or {})
        if semantic_components is not None:
            stats = {}
            for class_name, component in semantic_components.items():
                absolute_depth = far + component["depth"]
                package[f"outside_{class_name}"] = _scatter(
                    component["raw"], indices, height, width, 3
                )
                package[f"outside_{class_name}_alpha"] = _scatter(
                    component["alpha"], indices, height, width, 1
                )
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
        package["outside_final_filtered"] = package["outside_color"]
        package["semantic_cout_stats"] = stats
        cout_difference = torch.abs(outside_unfiltered_color - outside_color)
        package["semantic_cout_filter_stats"] = {
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
        t_classes = state.cuboid_space.classify(state.transmittance.get_xyz.detach())
        t_map = package["final"].new_zeros((height, width, 3))
        if bool((t_classes == 0).all()):
            t_map[..., 1:2] = package["transmittance_valid"]
        else:
            t_map[..., 0:1] = package["transmittance_valid"]
        package["t_spatial_class_map"] = t_map
        package["t_spatial_counts"] = {
            "inside": int((t_classes == 0).sum()),
            "interface": int((t_classes == 1).sum()),
            "outside": int((t_classes == 2).sum()),
        }
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
