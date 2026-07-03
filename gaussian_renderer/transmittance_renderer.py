"""Stage D D/R/T rendering with frozen two-hit geometry."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

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
        )

    def mark_parameters_updated(self):
        self.reflection_acceleration.mark_parameters_updated()
        self.transmittance_acceleration.mark_parameters_updated()
        self.diffuse_acceleration.mark_parameters_updated()


def _scatter(values, indices, height, width, channels, fill=0.0):
    output = values.new_full((height * width, channels), float(fill))
    if indices.numel():
        output = output.index_copy(0, indices, values)
    return output.reshape(height, width, channels)


def alpha_over_transmittance(inside_color, inside_alpha, outside_color, outside_alpha):
    color = inside_color + (1.0 - inside_alpha) * outside_color
    alpha = inside_alpha + (1.0 - inside_alpha) * outside_alpha
    return color, alpha


def _trace(model, origins, directions, acceleration, state, return_aux, diagnostics):
    traced = raytrace(
        model, origins, directions, acceleration=acceleration,
        chunk_size=state.ray_chunk_size, cutoff_sigma=state.ray_cutoff_sigma,
        hit_threshold=state.ray_hit_threshold, return_aux=return_aux,
        return_diagnostics=diagnostics,
        checkpoint_chunks=bool(state.ray_checkpoint_chunks and not diagnostics),
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


def render(
    camera,
    state: StageDRenderState,
    pipe,
    background: torch.Tensor,
    return_ray_aux: bool = False,
    return_ray_diagnostics: bool = False,
):
    state.stage_b.ray_chunk_size = state.ray_chunk_size
    state.stage_b.ray_checkpoint_chunks = bool(
        state.ray_checkpoint_chunks and not return_ray_diagnostics
    )
    package = render_stage_b(
        camera, state.stage_b, pipe, background,
        return_ray_aux=return_ray_aux,
        return_ray_diagnostics=return_ray_diagnostics,
    )
    height, width = package["alpha"].shape[:2]
    cache = state.geometry_release.load_view(str(camera.image_name))
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

    inside_outputs, inside_aux, inside_diagnostics = _trace(
        state.transmittance, first_origin, direction,
        state.transmittance_acceleration, state, return_ray_aux,
        return_ray_diagnostics,
    )
    outside_outputs, outside_aux, outside_diagnostics = _trace(
        state.diffuse_raytrace, second_origin, direction,
        state.diffuse_acceleration, state, return_ray_aux,
        return_ray_diagnostics,
    )
    inside_raw, inside_alpha, inside_relative_depth, inside_hit = inside_outputs
    outside_raw, outside_alpha, outside_relative_depth, outside_hit = outside_outputs
    ray_background = background.reshape(1, 3).to(outside_raw)
    outside_color = outside_raw + (1.0 - outside_alpha) * ray_background
    transmittance_color, transmittance_alpha = alpha_over_transmittance(
        inside_raw, inside_alpha, outside_color, outside_alpha
    )

    first_distance = torch.linalg.vector_norm(flat_position - center, dim=-1, keepdim=True)
    inside_depth = first_distance + inside_relative_depth
    far = t_far.reshape(-1, 1)[indices]
    outside_depth = far + outside_relative_depth
    violation = torch.relu(inside_depth - far)

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
        "transmittance_color": _scatter(transmittance_color, indices, height, width, 3),
        "transmittance_alpha": _scatter(transmittance_alpha, indices, height, width, 1),
        "depth_violation": _scatter(violation, indices, height, width, 1),
        "near_depth": t_near[..., None], "far_depth": t_far[..., None],
        "two_hit_valid": valid_cache[..., None].to(dtype),
        "transmittance_valid": valid[..., None].to(dtype),
        "transmittance_valid_count": int(indices.numel()),
        "inside_ray_aux": inside_aux, "outside_ray_aux": outside_aux,
        "inside_ray_diagnostics": inside_diagnostics,
        "outside_ray_diagnostics": outside_diagnostics,
    })
    for name in (
        "final", "transmittance_contribution", "inside_color", "inside_alpha",
        "inside_depth", "outside_color", "outside_alpha", "outside_depth",
        "transmittance_color", "transmittance_alpha", "depth_violation",
        "near_depth", "far_depth",
    ):
        if not torch.isfinite(package[name]).all():
            raise FloatingPointError(f"Stage D {name} contains NaN or Inf")
    return package
