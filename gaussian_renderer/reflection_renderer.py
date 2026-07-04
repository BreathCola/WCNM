"""Stage B Diffuse rasterization plus Reflection ray tracing and full GGX."""

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from gaussian_renderer.surfel_renderer import render as render_diffuse
from geometry.cuboid_path import build_cuboid_front_path, scatter_front_path
from geometry.cuboid_space import SUPPORT_CLASS_NAMES
from raytracer.acceleration_structure import CudaLBVH
from raytracer.ray_utils import decode_stage_a_gbuffer, generate_reflection_rays
from raytracer.tracer import raytrace
from utils.microfacet import microfacet_reflection


@dataclass
class StageBRenderState:
    diffuse: object
    reflection: object
    scene_radius: float
    ray_chunk_size: int = 4096
    ray_cutoff_sigma: float = 3.0
    ray_hit_threshold: float = 1e-4
    ray_epsilon_scale: float = 1e-4
    material_alpha_threshold: float = 1e-4
    roughness_min: float = 0.03
    roughness_remap: bool = False
    ray_background: str = "scene"
    ray_checkpoint_chunks: bool = False
    acceleration: object = None
    model_type: str = "stage_b"
    cuboid_space: object = None
    semantic_repair: bool = False
    geometry_release: object = None
    transparent_path_mode: str = "legacy_d_gbuffer"
    transparent_direct_mode: str = "legacy"
    transparent_reflection_mode: str = "legacy"
    support_sigma: float = 3.0

    def __post_init__(self):
        if self.ray_background != "scene":
            raise ValueError("Stage B supports only deterministic --ray_background scene")
        if self.acceleration is None:
            self.acceleration = CudaLBVH(self.ray_cutoff_sigma)
        if self.transparent_path_mode not in ("legacy_d_gbuffer", "cuboid_front_v1"):
            raise ValueError("unsupported transparent path mode")
        if self.transparent_direct_mode not in ("legacy", "interface_only", "off"):
            raise ValueError("unsupported transparent direct mode")
        if self.transparent_reflection_mode not in (
            "legacy", "support_safe_outside", "off",
        ):
            raise ValueError("unsupported transparent reflection mode")
        if self.transparent_path_mode == "cuboid_front_v1" and (
            self.geometry_release is None or self.cuboid_space is None
        ):
            raise ValueError("cuboid-front path requires frozen geometry release and cuboid space")


def _scatter(values, indices, height, width, channels, fill=0.0):
    output = values.new_full((height * width, channels), float(fill))
    if indices.numel():
        output = output.index_copy(0, indices, values)
    return output.reshape(height, width, channels)


def _replace_selected(base, selected, replacement):
    result = base.clone()
    if bool(selected.any()):
        result[selected] = replacement
    return result


def semantic_transparent_outside_only(unfiltered, outside, transparent_rays):
    """Replace only transparent-mask rays; outside-mask R remains bit-identical."""
    return _replace_selected(unfiltered, transparent_rays, outside)


def apply_transparent_contribution_mode(unfiltered, selected, transparent_rays, mode):
    """Apply a contribution-only gate while preserving non-transparent pixels."""
    if mode == "legacy":
        return unfiltered
    if mode == "off":
        selected = torch.zeros_like(unfiltered[transparent_rays])
    return _replace_selected(unfiltered, transparent_rays, selected)


def support_safe_outside_mask(model, cuboid_space, sigma=3.0):
    """The only candidate mask admitted by support_safe_outside mode."""
    return cuboid_space.support_masks(
        model.get_xyz.detach(), model.get_rotation.detach(),
        model.get_scaling.detach(), sigma=float(sigma),
    )["strict_outside_safe"]


def apply_cuboid_front_reflection_path(
    rays, transparent_rays, front_position, front_normal, camera_direction, epsilon,
):
    """Replace only transparent R path fields with frozen cuboid-front geometry."""
    selected_count = int(transparent_rays.sum())
    expected = (selected_count, 3)
    for name, value in (
        ("front_position", front_position), ("front_normal", front_normal),
        ("camera_direction", camera_direction),
    ):
        if value.shape != expected or not torch.isfinite(value).all():
            raise RuntimeError(f"cuboid-front {name} is missing/non-finite")
    reflection_direction = F.normalize(
        camera_direction
        - 2.0 * (camera_direction * front_normal).sum(dim=-1, keepdim=True)
        * front_normal,
        dim=-1, eps=1e-8,
    )
    result = dict(rays)
    for key in ("origins", "directions", "d_cam", "wo", "normal"):
        result[key] = rays[key].clone()
    result["origins"][transparent_rays] = front_position + float(epsilon) * reflection_direction
    result["directions"][transparent_rays] = reflection_direction
    result["d_cam"][transparent_rays] = camera_direction
    result["wo"][transparent_rays] = -camera_direction
    result["normal"][transparent_rays] = front_normal
    return result


class _FilteredDiffuseView:
    """Differentiable parameter subset used by interface-only direct rasterization."""
    def __init__(self, diffuse, mask):
        self.source = diffuse; self.mask = mask

    @property
    def get_xyz(self): return self.source.get_xyz[self.mask]
    @property
    def get_rotation(self): return self.source.get_rotation[self.mask]
    @property
    def get_scaling(self): return self.source.get_scaling[self.mask]
    @property
    def get_opacity(self): return self.source.get_opacity[self.mask]
    @property
    def get_base_color(self): return self.source.get_base_color[self.mask]
    @property
    def get_roughness(self): return self.source.get_roughness[self.mask]
    @property
    def get_f0(self): return self.source.get_f0[self.mask]
    @property
    def get_ks(self): return self.source.get_ks[self.mask]


def _transparent_front_contract(viewpoint_camera, state, diffuse, legacy_rays):
    cache = state.geometry_release.load_view(Path(str(viewpoint_camera.image_name)).stem)
    device, dtype = diffuse["position"].device, diffuse["position"].dtype
    valid_cache = torch.from_numpy(cache["valid_two_hit"]).to(device=device, dtype=torch.bool)
    near = torch.from_numpy(cache["t_near"]).to(device=device, dtype=dtype)
    far = torch.from_numpy(cache["t_far"]).to(device=device, dtype=dtype)
    back = torch.from_numpy(cache["back_position"]).to(device=device, dtype=dtype)
    mask = viewpoint_camera.specular_mask
    if mask is None:
        raise RuntimeError("cuboid-front path requires hard transparent mask")
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask.permute(1, 2, 0)
    if mask.ndim == 2:
        mask = mask[..., None]
    if mask.ndim != 3 or mask.shape[-1] != 1:
        raise ValueError("cuboid-front transparent mask must be one-channel")
    hard = mask[..., 0] >= 0.5
    formal_valid = hard & valid_cache
    missing_surface = formal_valid & ~legacy_rays["valid_mask"]
    if bool(missing_surface.any()):
        raise RuntimeError("cuboid-front valid transparent ray lacks required D material surface")
    path = build_cuboid_front_path(
        viewpoint_camera.camera_center.to(dtype=dtype), back, near, far,
        formal_valid, state.cuboid_space,
    )
    maps = scatter_front_path(
        path, diffuse["position"].shape[0], diffuse["position"].shape[1],
        diffuse["position"],
    )
    maps.update({
        "near_depth": near[..., None], "far_depth": far[..., None],
        "two_hit_valid": valid_cache[..., None].to(dtype),
        "transparent_path_valid": formal_valid[..., None].to(dtype),
        "transparent_mask_hard": hard[..., None].to(dtype),
        "frozen_back_position": back,
    })
    return path, maps


def _zero_ray_outputs(reference, count):
    return (
        reference.new_zeros((count, 3)), reference.new_zeros((count, 1)),
        reference.new_zeros((count, 1)),
        torch.zeros((count, 1), dtype=torch.bool, device=reference.device),
    )


def render(
    viewpoint_camera,
    state: StageBRenderState,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier: float = 1.0,
    separate_sh: bool = False,
    override_color=None,
    use_trained_exp: bool = False,
    return_ray_aux: bool = False,
    return_ray_diagnostics: bool = False,
):
    diffuse = render_diffuse(
        viewpoint_camera,
        state.diffuse,
        pipe,
        bg_color,
        scaling_modifier=scaling_modifier,
        separate_sh=separate_sh,
        override_color=override_color,
        use_trained_exp=use_trained_exp,
    )
    height, width = diffuse["alpha"].shape[:2]
    decoded = decode_stage_a_gbuffer(
        diffuse,
        bg_color,
        roughness_min=state.roughness_min,
        alpha_threshold=state.material_alpha_threshold,
    )
    rays = generate_reflection_rays(
        diffuse,
        viewpoint_camera.camera_center,
        state.scene_radius,
        ray_epsilon_scale=state.ray_epsilon_scale,
        alpha_threshold=state.material_alpha_threshold,
    )
    indices = rays["flat_indices"]
    background = bg_color.reshape(1, 3).to(diffuse["Cd"])
    front_maps = {}
    transparent_rays = torch.zeros((indices.numel(),), dtype=torch.bool, device=indices.device)
    ownership_rays = transparent_rays
    if state.transparent_path_mode == "cuboid_front_v1":
        _, front_maps = _transparent_front_contract(
            viewpoint_camera, state, diffuse, rays,
        )
        transparent_rays = (
            front_maps["transparent_path_valid"].reshape(-1)[indices] >= 0.5
        )
        ownership_rays = front_maps["transparent_mask_hard"].reshape(-1)[indices] >= 0.5
        selected_indices = indices[transparent_rays]
        selected_direction = front_maps["frozen_camera_direction"].reshape(-1, 3)[selected_indices]
        selected_normal = front_maps["front_normal"].reshape(-1, 3)[selected_indices]
        selected_front = front_maps["front_position"].reshape(-1, 3)[selected_indices]
        epsilon = float(state.ray_epsilon_scale * state.scene_radius)
        rays = apply_cuboid_front_reflection_path(
            rays, transparent_rays, selected_front, selected_normal,
            selected_direction, epsilon,
        )

    if indices.numel():
        traced = raytrace(
            state.reflection,
            rays["origins"],
            rays["directions"],
            acceleration=state.acceleration,
            chunk_size=state.ray_chunk_size,
            cutoff_sigma=state.ray_cutoff_sigma,
            hit_threshold=state.ray_hit_threshold,
            return_aux=return_ray_aux,
            return_diagnostics=return_ray_diagnostics,
            checkpoint_chunks=state.ray_checkpoint_chunks,
        )
        if return_ray_aux and return_ray_diagnostics:
            (raw_color, reflection_alpha, reflection_depth, reflection_hit), ray_aux, ray_diagnostics = traced
        elif return_ray_aux:
            (raw_color, reflection_alpha, reflection_depth, reflection_hit), ray_aux = traced
            ray_diagnostics = None
        elif return_ray_diagnostics:
            (raw_color, reflection_alpha, reflection_depth, reflection_hit), ray_diagnostics = traced
            ray_aux = None
        else:
            raw_color, reflection_alpha, reflection_depth, reflection_hit = traced
            ray_aux = None
            ray_diagnostics = None
        reflection_color = raw_color + (1.0 - reflection_alpha) * background
        flat = lambda name: decoded[name].reshape(-1, decoded[name].shape[-1])[indices]
        material = microfacet_reflection(
            rays["normal"],
            rays["wo"],
            rays["directions"],
            flat("roughness"),
            flat("f0"),
            roughness_min=state.roughness_min,
            roughness_remap=state.roughness_remap,
        )
        alpha = flat("alpha")
        ks = flat("ks")
        cd_surface = flat("Cd")
        diffuse_contribution = alpha * (1.0 - ks) * cd_surface
        diffuse_unfiltered_contribution = diffuse_contribution
        if state.transparent_path_mode == "cuboid_front_v1":
            if state.transparent_direct_mode == "off":
                diffuse_contribution = apply_transparent_contribution_mode(
                    diffuse_contribution, None, ownership_rays, "off",
                )
            elif state.transparent_direct_mode == "interface_only":
                d_masks = state.cuboid_space.support_masks(
                    state.diffuse.get_xyz.detach(), state.diffuse.get_rotation.detach(),
                    state.diffuse.get_scaling.detach(), sigma=state.support_sigma,
                )
                selected_indices = indices[ownership_rays]
                if bool(d_masks["interface_margin"].any()):
                    interface_diffuse = render_diffuse(
                        viewpoint_camera,
                        _FilteredDiffuseView(state.diffuse, d_masks["interface_margin"]),
                        pipe, bg_color, scaling_modifier=scaling_modifier,
                        separate_sh=separate_sh, override_color=None,
                        use_trained_exp=False,
                    )
                    interface_decoded = decode_stage_a_gbuffer(
                        interface_diffuse, bg_color, roughness_min=state.roughness_min,
                        alpha_threshold=state.material_alpha_threshold,
                    )
                    ia = interface_decoded["alpha"].reshape(-1, 1)[selected_indices]
                    iks = interface_decoded["ks"].reshape(-1, 1)[selected_indices]
                    icd = interface_decoded["Cd"].reshape(-1, 3)[selected_indices]
                    interface_contribution = ia * (1.0 - iks) * icd
                else:
                    interface_contribution = diffuse_contribution.new_zeros(
                        (selected_indices.numel(), 3)
                    )
                diffuse_contribution = apply_transparent_contribution_mode(
                    diffuse_contribution, interface_contribution,
                    ownership_rays, "interface_only",
                )
        reflection_contribution = alpha * ks * material["wr"] * reflection_color
        reflection_unfiltered_contribution = reflection_contribution
        reflection_unfiltered_hit = reflection_hit
        semantic_r_components = None
        if state.semantic_repair or state.transparent_path_mode == "cuboid_front_v1":
            if state.cuboid_space is None:
                raise RuntimeError("semantic R filtering requires cuboid_space")
            mask = viewpoint_camera.specular_mask
            if mask is None:
                raise RuntimeError("semantic R filtering requires transparent mask")
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask.permute(1, 2, 0)
            if state.transparent_path_mode != "cuboid_front_v1":
                transparent_rays = mask.reshape(-1)[indices] >= 0.5
            selected_count = int(transparent_rays.sum())
            if state.transparent_path_mode == "cuboid_front_v1":
                class_names = SUPPORT_CLASS_NAMES
                class_masks = state.cuboid_space.support_masks(
                    state.reflection.get_xyz.detach(), state.reflection.get_rotation.detach(),
                    state.reflection.get_scaling.detach(), sigma=state.support_sigma,
                )
                class_masks["strict_outside_safe"] = support_safe_outside_mask(
                    state.reflection, state.cuboid_space, sigma=state.support_sigma,
                )
            else:
                class_names = ("inside", "interface", "outside")
                class_masks = state.cuboid_space.masks(state.reflection.get_xyz.detach())
            semantic_r_components = {}
            ray_origins = rays["origins"][transparent_rays]
            ray_directions = rays["directions"][transparent_rays]
            selected_alpha = alpha[transparent_rays]
            selected_ks = ks[transparent_rays]
            selected_wr = material["wr"][transparent_rays]
            for class_name in class_names:
                if selected_count:
                    class_trace = raytrace(
                        state.reflection, ray_origins, ray_directions,
                        acceleration=state.acceleration,
                        chunk_size=state.ray_chunk_size,
                        cutoff_sigma=state.ray_cutoff_sigma,
                        hit_threshold=state.ray_hit_threshold,
                        return_aux=False,
                        return_diagnostics=return_ray_diagnostics,
                        checkpoint_chunks=state.ray_checkpoint_chunks,
                        surfel_filter=class_masks[class_name],
                    )
                    if return_ray_diagnostics:
                        class_outputs, class_diagnostics = class_trace
                    else:
                        class_outputs, class_diagnostics = class_trace, None
                else:
                    class_outputs = _zero_ray_outputs(raw_color, 0)
                    class_diagnostics = None
                class_raw, class_alpha, class_depth, class_hit = class_outputs
                class_color = class_raw + (1.0 - class_alpha) * background
                class_contribution = selected_alpha * selected_ks * selected_wr * class_raw
                formal_contribution = selected_alpha * selected_ks * selected_wr * class_color
                semantic_r_components[class_name] = {
                    "raw": class_raw, "color": class_color, "alpha": class_alpha,
                    "depth": class_depth, "hit": class_hit,
                    "contribution": class_contribution,
                    "formal_contribution": formal_contribution,
                    "surfel_count": int(class_masks[class_name].sum()),
                    "candidate_count": (
                        int(class_diagnostics.eligible_candidate_counts.sum())
                        if class_diagnostics is not None else None
                    ),
                    "hit_count": int(class_hit.sum()),
                }
            outside_name = (
                "strict_outside_safe"
                if state.transparent_path_mode == "cuboid_front_v1" else "outside"
            )
            outside = semantic_r_components[outside_name]
            if state.transparent_reflection_mode == "support_safe_outside" \
                    or state.transparent_path_mode != "cuboid_front_v1":
                raw_color = semantic_transparent_outside_only(raw_color, outside["raw"], transparent_rays)
                reflection_color = semantic_transparent_outside_only(
                    reflection_color, outside["color"], transparent_rays
                )
                reflection_alpha = semantic_transparent_outside_only(
                    reflection_alpha, outside["alpha"], transparent_rays
                )
                reflection_depth = semantic_transparent_outside_only(
                    reflection_depth, outside["depth"], transparent_rays
                )
                reflection_hit = semantic_transparent_outside_only(
                    reflection_hit, outside["hit"], transparent_rays
                )
                reflection_contribution = semantic_transparent_outside_only(
                    reflection_contribution, outside["formal_contribution"], transparent_rays
                )
            elif state.transparent_reflection_mode == "off":
                reflection_contribution = apply_transparent_contribution_mode(
                    reflection_contribution, None, ownership_rays, "off",
                )
        final_valid = diffuse_contribution + reflection_contribution + (1.0 - alpha) * background
    else:
        raw_color = diffuse["Cd"].new_zeros((0, 3))
        reflection_alpha = diffuse["alpha"].new_zeros((0, 1))
        reflection_depth = diffuse["depth"].new_zeros((0, 1))
        reflection_hit = torch.zeros((0, 1), dtype=torch.bool, device=diffuse["Cd"].device)
        reflection_color = raw_color
        material = {
            "D": diffuse["alpha"].new_zeros((0, 1)),
            "F": diffuse["Cd"].new_zeros((0, 3)),
            "G": diffuse["alpha"].new_zeros((0, 1)),
            "fr": diffuse["Cd"].new_zeros((0, 3)),
            "wr": diffuse["Cd"].new_zeros((0, 3)),
        }
        diffuse_contribution = raw_color
        diffuse_unfiltered_contribution = raw_color
        reflection_contribution = raw_color
        reflection_unfiltered_contribution = raw_color
        reflection_unfiltered_hit = reflection_hit
        semantic_r_components = None
        final_valid = raw_color
        ray_aux = None
        ray_diagnostics = None

    final = bg_color.reshape(1, 1, 3).expand(height, width, 3).clone()
    if indices.numel():
        final = final.reshape(-1, 3).index_copy(0, indices, final_valid).reshape(height, width, 3)
    output = dict(diffuse)
    output.update(
        {
            "final": final.clamp(0.0, 1.0),
            "render": final.clamp(0.0, 1.0).permute(2, 0, 1),
            "reflection_color": _scatter(reflection_color, indices, height, width, 3, 0.0),
            "reflection_raw_color": _scatter(raw_color, indices, height, width, 3, 0.0),
            "reflection_alpha": _scatter(reflection_alpha, indices, height, width, 1, 0.0),
            "reflection_depth": _scatter(reflection_depth, indices, height, width, 1, 0.0),
            "reflection_hit_mask": _scatter(reflection_hit.to(diffuse["alpha"]), indices, height, width, 1, 0.0),
            "microfacet_D": _scatter(material["D"], indices, height, width, 1, 0.0),
            "microfacet_F": _scatter(material["F"], indices, height, width, 3, 0.0),
            "microfacet_G": _scatter(material["G"], indices, height, width, 1, 0.0),
            "microfacet_fr": _scatter(material["fr"], indices, height, width, 3, 0.0),
            "microfacet_wr": _scatter(material["wr"], indices, height, width, 3, 0.0),
            "diffuse_contribution": _scatter(diffuse_contribution, indices, height, width, 3, 0.0),
            "diffuse_unfiltered": _scatter(
                diffuse_unfiltered_contribution, indices, height, width, 3, 0.0
            ),
            "reflection_contribution": _scatter(reflection_contribution, indices, height, width, 3, 0.0),
            "reflection_unfiltered": _scatter(
                reflection_unfiltered_contribution, indices, height, width, 3, 0.0
            ),
            "surface_Cd": decoded["Cd"],
            "surface_roughness": decoded["roughness"],
            "surface_f0": decoded["f0"],
            "surface_ks": decoded["ks"],
            "valid_surface_mask": rays["valid_mask"][..., None].to(diffuse["alpha"]),
            "valid_ray_count": int(indices.numel()),
            "ray_aux": ray_aux,
        }
    )
    output.update(front_maps)
    output["transparent_path_mode"] = state.transparent_path_mode
    output["transparent_direct_mode"] = state.transparent_direct_mode
    output["transparent_reflection_mode"] = state.transparent_reflection_mode
    if semantic_r_components is not None:
        transparent_indices = indices[transparent_rays]
        stats = {}
        for class_name, component in semantic_r_components.items():
            contribution_map = _scatter(
                component["contribution"], transparent_indices, height, width, 3, 0.0
            )
            output[f"reflection_{class_name}"] = contribution_map
            output[f"reflection_{class_name}_alpha"] = _scatter(
                component["alpha"], transparent_indices, height, width, 1, 0.0
            )
            output[f"reflection_{class_name}_depth"] = _scatter(
                component["depth"], transparent_indices, height, width, 1, 0.0
            )
            output[f"reflection_{class_name}_hit"] = _scatter(
                component["hit"].to(diffuse["alpha"]),
                transparent_indices, height, width, 1, 0.0,
            )
            stats[class_name] = {
                "surfel_count": component["surfel_count"],
                "candidate_count": component["candidate_count"],
                "hit_count": component["hit_count"],
                "contribution_energy": float(component["contribution"].detach().mean())
                if component["contribution"].numel() else 0.0,
            }
        output["reflection_final_filtered"] = output["reflection_contribution"]
        output["semantic_r_stats"] = stats
        output["semantic_r_transparent_ray_count"] = selected_count
        comparison_name = (
            "strict_outside_safe"
            if state.transparent_path_mode == "cuboid_front_v1" else "outside"
        )
        comparison = (
            torch.zeros_like(reflection_unfiltered_contribution[transparent_rays])
            if state.transparent_reflection_mode == "off"
            else semantic_r_components[comparison_name]["formal_contribution"]
        )
        filter_difference = torch.abs(
            reflection_unfiltered_contribution[transparent_rays] - comparison
        )
        output["semantic_r_filter_stats"] = {
            "formal_mode": state.transparent_reflection_mode,
            "support_sigma": float(state.support_sigma),
            "unfiltered_surfel_count": int(state.reflection.get_xyz.shape[0]),
            "unfiltered_candidate_count": (
                int(ray_diagnostics.candidate_counts[transparent_rays].sum())
                if ray_diagnostics is not None else None
            ),
            "unfiltered_hit_count": int(reflection_unfiltered_hit[transparent_rays].sum()),
            "rgb_difference_mean_abs": float(filter_difference.detach().mean())
            if filter_difference.numel() else 0.0,
            "rgb_difference_max_abs": float(filter_difference.detach().max())
            if filter_difference.numel() else 0.0,
        }
    if return_ray_diagnostics:
        candidate_counts = (
            ray_diagnostics.candidate_counts.to(diffuse["alpha"])
            if ray_diagnostics is not None else diffuse["alpha"].new_zeros((indices.numel(),))
        )
        exact_counts = (
            ray_diagnostics.exact_intersection_counts.to(diffuse["alpha"])
            if ray_diagnostics is not None else diffuse["alpha"].new_zeros((indices.numel(),))
        )
        output.update(
            {
                "ray_candidate_count": _scatter(
                    candidate_counts[:, None], indices, height, width, 1, 0.0
                ),
                "ray_exact_intersection_count": _scatter(
                    exact_counts[:, None], indices, height, width, 1, 0.0
                ),
                "ray_diagnostics": ray_diagnostics,
            }
        )
    for name in (
        "final", "reflection_color", "reflection_alpha", "reflection_depth",
        "microfacet_D", "microfacet_F", "microfacet_G", "microfacet_fr", "microfacet_wr",
        "diffuse_contribution", "reflection_contribution",
    ):
        if not torch.isfinite(output[name]).all():
            raise FloatingPointError(f"{name} contains NaN or Inf")
    return output
