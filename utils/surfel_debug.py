"""Stage A debug image writer."""

import os

import torch
from torchvision.utils import save_image

from utils.surfel_utils import visualize_depth


@torch.no_grad()
def save_surfel_debug_maps(output, ground_truth, directory):
    os.makedirs(directory, exist_ok=True)

    def chw(name):
        return output[name].detach().permute(2, 0, 1).clamp(0, 1)

    save_image(chw("Cd"), os.path.join(directory, "diffuse_color.png"))
    depth_vis = visualize_depth(output["depth"], output["alpha"])
    save_image(depth_vis, os.path.join(directory, "depth.png"))
    save_image(depth_vis, os.path.join(directory, "diffuse_depth.png"))
    save_image((output["normal"].detach().permute(2, 0, 1) * 0.5 + 0.5).clamp(0, 1), os.path.join(directory, "normal.png"))
    save_image(chw("roughness").repeat(3, 1, 1), os.path.join(directory, "roughness.png"))
    save_image(chw("f0"), os.path.join(directory, "f0.png"))
    save_image(chw("ks").repeat(3, 1, 1), os.path.join(directory, "ks.png"))
    save_image(chw("alpha").repeat(3, 1, 1), os.path.join(directory, "alpha.png"))
    save_image(ground_truth.detach().clamp(0, 1), os.path.join(directory, "ground_truth.png"))
