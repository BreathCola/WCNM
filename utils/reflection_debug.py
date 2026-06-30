"""Real Stage B Reflection debug-map writer; no later-stage placeholders."""

import json
import os

import torch
from torchvision.utils import save_image

from utils.surfel_debug import save_surfel_debug_maps
from utils.surfel_utils import visualize_depth


@torch.no_grad()
def save_reflection_debug_maps(output, ground_truth, directory, specular_mask=None, mask_sha256=None):
    os.makedirs(directory, exist_ok=True)
    save_surfel_debug_maps(output, ground_truth, directory)

    def chw(name):
        return output[name].detach().permute(2, 0, 1)

    save_image(chw("final").clamp(0, 1), os.path.join(directory, "final.png"))
    save_image(chw("reflection_color").clamp(0, 1), os.path.join(directory, "reflection_color.png"))
    save_image(chw("reflection_alpha").repeat(3, 1, 1).clamp(0, 1), os.path.join(directory, "reflection_alpha.png"))
    save_image(
        visualize_depth(output["reflection_depth"], output["reflection_alpha"]),
        os.path.join(directory, "reflection_depth.png"),
    )
    save_image(chw("reflection_hit_mask").repeat(3, 1, 1).clamp(0, 1), os.path.join(directory, "reflection_hit_mask.png"))
    save_image(chw("microfacet_D").repeat(3, 1, 1).clamp(0, 1), os.path.join(directory, "microfacet_D.png"))
    save_image(chw("microfacet_F").clamp(0, 1), os.path.join(directory, "microfacet_F.png"))
    save_image(chw("microfacet_G").repeat(3, 1, 1).clamp(0, 1), os.path.join(directory, "microfacet_G.png"))
    save_image(chw("microfacet_fr").clamp(0, 1), os.path.join(directory, "microfacet_fr.png"))
    save_image(chw("microfacet_wr").clamp(0, 1), os.path.join(directory, "microfacet_wr.png"))
    save_image(chw("diffuse_contribution").clamp(0, 1), os.path.join(directory, "diffuse_contribution.png"))
    save_image(chw("reflection_contribution").clamp(0, 1), os.path.join(directory, "reflection_contribution.png"))
    save_image(chw("valid_surface_mask").repeat(3, 1, 1).clamp(0, 1), os.path.join(directory, "valid_surface_mask.png"))

    metadata = {
        "valid_ray_count": int(output["valid_ray_count"]),
        "reflection_hit_count": int((output["reflection_hit_mask"] > 0.5).sum().item()),
        "roughness_min": float(output["surface_roughness"].min().item()),
        "roughness_max": float(output["surface_roughness"].max().item()),
        "specular_mask_sha256": mask_sha256,
    }
    if specular_mask is not None:
        mask = specular_mask.detach().clamp(0, 1)
        save_image(mask.repeat(3, 1, 1), os.path.join(directory, "specular_mask.png"))
        overlay = 0.7 * ground_truth.detach().clamp(0, 1) + 0.3 * torch.cat(
            (mask, torch.zeros_like(mask), torch.zeros_like(mask)), dim=0
        )
        save_image(overlay.clamp(0, 1), os.path.join(directory, "overlay.png"))
    with open(os.path.join(directory, "reflection_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
