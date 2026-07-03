"""Stage B Diffuse rasterization plus Reflection ray tracing and full GGX."""

from dataclasses import dataclass

import torch

from gaussian_renderer.surfel_renderer import render as render_diffuse
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

    def __post_init__(self):
        if self.ray_background != "scene":
            raise ValueError("Stage B supports only deterministic --ray_background scene")
        if self.acceleration is None:
            self.acceleration = CudaLBVH(self.ray_cutoff_sigma)


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
        reflection_contribution = alpha * ks * material["wr"] * reflection_color
        reflection_unfiltered_contribution = reflection_contribution
        reflection_unfiltered_hit = reflection_hit
        semantic_r_components = None
        if state.semantic_repair:
            if state.cuboid_space is None:
                raise RuntimeError("semantic R filtering requires cuboid_space")
            mask = viewpoint_camera.specular_mask
            if mask is None:
                raise RuntimeError("semantic R filtering requires transparent mask")
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask.permute(1, 2, 0)
            transparent_rays = mask.reshape(-1)[indices] >= 0.5
            selected_count = int(transparent_rays.sum())
            class_masks = state.cuboid_space.masks(state.reflection.get_xyz.detach())
            semantic_r_components = {}
            ray_origins = rays["origins"][transparent_rays]
            ray_directions = rays["directions"][transparent_rays]
            selected_alpha = alpha[transparent_rays]
            selected_ks = ks[transparent_rays]
            selected_wr = material["wr"][transparent_rays]
            for class_name in ("inside", "interface", "outside"):
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
            outside = semantic_r_components["outside"]
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
        filter_difference = torch.abs(
            reflection_unfiltered_contribution[transparent_rays]
            - semantic_r_components["outside"]["formal_contribution"]
        )
        output["semantic_r_filter_stats"] = {
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
