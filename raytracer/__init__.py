"""Stage B differentiable Gaussian ray tracing package."""

from .ray_utils import decode_stage_a_gbuffer, generate_reflection_rays, valid_diffuse_surface_mask

__all__ = ["decode_stage_a_gbuffer", "generate_reflection_rays", "valid_diffuse_surface_mask"]
