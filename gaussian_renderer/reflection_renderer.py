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
    acceleration: object = None
    model_type: str = "stage_b"

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
        )
        if return_ray_aux:
            (raw_color, reflection_alpha, reflection_depth, reflection_hit), ray_aux = traced
        else:
            raw_color, reflection_alpha, reflection_depth, reflection_hit = traced
            ray_aux = None
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
        final_valid = raw_color
        ray_aux = None

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
            "surface_Cd": decoded["Cd"],
            "surface_roughness": decoded["roughness"],
            "surface_f0": decoded["f0"],
            "surface_ks": decoded["ks"],
            "valid_surface_mask": rays["valid_mask"][..., None].to(diffuse["alpha"]),
            "valid_ray_count": int(indices.numel()),
            "ray_aux": ray_aux,
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
