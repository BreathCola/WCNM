from pathlib import Path

import torch

from utils.reflection_debug import save_reflection_debug_maps


def _output():
    height, width = 4, 5
    scalar = torch.full((height, width, 1), 0.5)
    rgb = torch.full((height, width, 3), 0.25)
    normal = torch.zeros_like(rgb)
    normal[..., 2] = 1.0
    return {
        "Cd": rgb, "alpha": scalar, "depth": scalar, "position": rgb, "normal": normal,
        "roughness": scalar, "f0": rgb, "ks": scalar, "final": rgb,
        "reflection_color": rgb, "reflection_alpha": scalar, "reflection_depth": scalar,
        "reflection_hit_mask": scalar, "microfacet_D": scalar, "microfacet_F": rgb,
        "microfacet_G": scalar, "microfacet_fr": rgb, "microfacet_wr": rgb,
        "diffuse_contribution": rgb, "reflection_contribution": rgb,
        "valid_surface_mask": scalar, "surface_roughness": scalar,
        "valid_ray_count": height * width,
    }


def test_debug_writer_emits_only_real_stage_b_maps_and_no_mask_when_disabled(tmp_path):
    output = _output()
    save_reflection_debug_maps(output, torch.zeros((3, 4, 5)), str(tmp_path))
    names = {path.name for path in Path(tmp_path).iterdir()}
    assert {"final.png", "reflection_color.png", "reflection_alpha.png", "reflection_depth.png",
            "reflection_hit_mask.png", "microfacet_D.png", "microfacet_F.png", "microfacet_G.png",
            "microfacet_fr.png", "microfacet_wr.png", "diffuse_contribution.png",
            "reflection_contribution.png", "reflection_metadata.json"} <= names
    assert "specular_mask.png" not in names and "overlay.png" not in names
    assert not any(
        token in name for name in names
        for token in ("transmittance", "inside", "outside", "two_hit", "near_depth", "far_depth", "depth_violation")
    )


def test_debug_writer_emits_real_specular_mask_only_when_provided(tmp_path):
    mask = torch.ones((1, 4, 5))
    save_reflection_debug_maps(
        _output(), torch.zeros((3, 4, 5)), str(tmp_path), specular_mask=mask, mask_sha256="abc"
    )
    assert (tmp_path / "specular_mask.png").is_file()
    assert (tmp_path / "overlay.png").is_file()
