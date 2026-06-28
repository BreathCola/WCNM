"""Differentiable Stage A renderer for diffuse 2D Gaussian surfels."""

import math

import torch
import torch.nn.functional as F
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from scene.diffuse_surfel_model import DiffuseSurfelModel
from utils.surfel_utils import face_forward, unproject_depth


OUTPUT_CONTRACT = {
    "Cd": 3,
    "alpha": 1,
    "depth": 1,
    "position": 3,
    "normal": 3,
    "roughness": 1,
    "f0": 3,
    "ks": 1,
}


def _settings(camera, background, pipe, scaling_modifier):
    return GaussianRasterizationSettings(
        image_height=int(camera.image_height),
        image_width=int(camera.image_width),
        tanfovx=math.tan(camera.FoVx * 0.5),
        tanfovy=math.tan(camera.FoVy * 0.5),
        bg=background,
        scale_modifier=scaling_modifier,
        viewmatrix=camera.world_view_transform,
        projmatrix=camera.full_proj_transform,
        sh_degree=0,
        campos=camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )


def _rasterize(rasterizer, pc, screenspace_points, colors):
    return rasterizer(
        means3D=pc.get_xyz,
        means2D=screenspace_points,
        shs=None,
        colors_precomp=colors.contiguous(),
        opacities=pc.get_opacity,
        scales=pc.get_scaling,
        rotations=pc.get_rotation,
        cov3D_precomp=None,
    )


def render(
    viewpoint_camera,
    pc: DiffuseSurfelModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier: float = 1.0,
    separate_sh: bool = False,
    override_color=None,
    use_trained_exp: bool = False,
):
    """Render the complete diffuse surfel G-buffer.

    Contract fields use HWC layout as required by ``RTGS_MASTER_PLAN.md``.
    Compatibility aliases such as ``render`` remain CHW for the baseline
    training/reporting code.
    """
    del separate_sh
    if pipe.compute_cov3D_python:
        raise ValueError("2D surfel mode requires CUDA scale/rotation projection")
    if bg_color.shape != (3,):
        raise ValueError("bg_color must contain three RGB values")

    screenspace_points = torch.zeros_like(
        pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device=pc.get_xyz.device
    )
    screenspace_points.retain_grad()

    color_settings = _settings(viewpoint_camera, bg_color, pipe, scaling_modifier)
    color_rasterizer = GaussianRasterizer(raster_settings=color_settings)
    base_color = pc.get_base_color if override_color is None else override_color
    rendered_color, radii, allmap = _rasterize(color_rasterizer, pc, screenspace_points, base_color)

    material_settings = _settings(viewpoint_camera, torch.zeros_like(bg_color), pipe, scaling_modifier)
    material_rasterizer = GaussianRasterizer(raster_settings=material_settings)
    roughness_ks = torch.cat(
        (pc.get_roughness, pc.get_ks, torch.zeros_like(pc.get_ks)), dim=-1
    )
    rendered_roughness_ks, _, _ = _rasterize(
        material_rasterizer, pc, screenspace_points, roughness_ks
    )
    rendered_f0, _, _ = _rasterize(material_rasterizer, pc, screenspace_points, pc.get_f0)

    alpha_chw = allmap[1:2].clamp(0, 1)
    expected_depth = torch.nan_to_num(
        allmap[0:1] / alpha_chw.clamp_min(1e-8), nan=0.0, posinf=0.0, neginf=0.0
    )
    position = unproject_depth(viewpoint_camera, expected_depth)

    normal_view = allmap[2:5].permute(1, 2, 0)
    view_to_world = viewpoint_camera.world_view_transform[:3, :3].transpose(0, 1)
    normal_world = normal_view @ view_to_world
    normal_world = F.normalize(normal_world, dim=-1, eps=1e-12)
    normal_world = face_forward(normal_world, position, viewpoint_camera.camera_center)
    normal_world = torch.where(
        alpha_chw.permute(1, 2, 0) > 1e-4, normal_world, torch.zeros_like(normal_world)
    )

    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_color = (
            torch.matmul(rendered_color.permute(1, 2, 0), exposure[:3, :3])
            + exposure[:3, 3]
        ).permute(2, 0, 1)

    rendered_color = rendered_color.clamp(0, 1)
    output = {
        "Cd": rendered_color.permute(1, 2, 0),
        "alpha": alpha_chw.permute(1, 2, 0),
        "depth": expected_depth.permute(1, 2, 0),
        "position": position,
        "normal": normal_world,
        "roughness": rendered_roughness_ks[0:1].permute(1, 2, 0),
        "f0": rendered_f0.permute(1, 2, 0),
        "ks": rendered_roughness_ks[1:2].permute(1, 2, 0),
        "render": rendered_color,
        "viewspace_points": screenspace_points,
        "visibility_filter": (radii > 0).nonzero(),
        "radii": radii,
    }
    _validate_output_contract(output, viewpoint_camera.image_height, viewpoint_camera.image_width)
    return output


def _validate_output_contract(output, height, width):
    for name, channels in OUTPUT_CONTRACT.items():
        expected = (int(height), int(width), channels)
        if output[name].shape != expected:
            raise RuntimeError(f"{name} has shape {tuple(output[name].shape)}, expected {expected}")
        if not torch.isfinite(output[name]).all():
            raise RuntimeError(f"{name} contains NaN or Inf")
