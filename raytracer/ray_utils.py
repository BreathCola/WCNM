"""Reflection-ray generation and Stage A G-buffer decoding."""

import torch
import torch.nn.functional as F


def valid_diffuse_surface_mask(output, camera_center, alpha_threshold: float = 1e-4):
    alpha = output["alpha"][..., 0]
    depth = output["depth"][..., 0]
    position = output["position"]
    normal = output["normal"]
    wo = F.normalize(camera_center.reshape(1, 1, 3) - position, dim=-1, eps=1e-8)
    valid = alpha > alpha_threshold
    valid &= torch.isfinite(alpha) & torch.isfinite(depth) & (depth > 0.0)
    valid &= torch.isfinite(position).all(dim=-1)
    valid &= torch.isfinite(normal).all(dim=-1) & (torch.linalg.vector_norm(normal, dim=-1) > 1e-6)
    for name in ("roughness", "f0", "ks"):
        valid &= torch.isfinite(output[name]).all(dim=-1)
    valid &= (normal * wo).sum(dim=-1) > 0.0
    return valid


def generate_reflection_rays(
    output,
    camera_center: torch.Tensor,
    scene_radius: float,
    ray_epsilon_scale: float = 1e-4,
    alpha_threshold: float = 1e-4,
):
    if scene_radius <= 0 or ray_epsilon_scale <= 0:
        raise ValueError("scene_radius and ray_epsilon_scale must be positive")
    valid = valid_diffuse_surface_mask(output, camera_center, alpha_threshold)
    flat_indices = valid.reshape(-1).nonzero(as_tuple=False)[:, 0]
    position = output["position"].reshape(-1, 3)[flat_indices]
    normal = output["normal"].reshape(-1, 3)[flat_indices]
    d_cam = F.normalize(position - camera_center.reshape(1, 3), dim=-1, eps=1e-8)
    direction = F.normalize(
        d_cam - 2.0 * (d_cam * normal).sum(dim=-1, keepdim=True) * normal,
        dim=-1,
        eps=1e-8,
    )
    origin = position + float(ray_epsilon_scale * scene_radius) * direction
    return {
        "valid_mask": valid,
        "flat_indices": flat_indices,
        "origins": origin,
        "directions": direction,
        "d_cam": d_cam,
        "wo": -d_cam,
        "normal": normal,
    }


def decode_stage_a_gbuffer(
    output,
    background: torch.Tensor,
    roughness_min: float,
    alpha_threshold: float = 1e-4,
):
    alpha = output["alpha"]
    if background.shape != (3,):
        raise ValueError("background must have shape [3]")
    denominator = alpha.clamp_min(alpha_threshold)
    bg = background.reshape(1, 1, 3).to(output["Cd"])
    cd_premultiplied = output["Cd"] - (1.0 - alpha) * bg
    return {
        "Cd": (cd_premultiplied / denominator).clamp(0.0, 1.0),
        "roughness": (output["roughness"] / denominator).clamp(roughness_min, 1.0),
        "f0": (output["f0"] / denominator).clamp(0.0, 1.0),
        "ks": (output["ks"] / denominator).clamp(0.0, 1.0),
        "normal": output["normal"],
        "alpha": alpha,
    }
