import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from utils.reflection_debug import save_reflection_debug_maps


def _output():
    height, width = 4, 5
    scalar = torch.full((height, width, 1), 0.5)
    rgb = torch.full((height, width, 3), 0.25)
    candidate_count = torch.arange(1, height * width + 1, dtype=torch.float32).reshape(height, width, 1)
    exact_count = torch.arange(0, height * width, dtype=torch.float32).reshape(height, width, 1)
    normal = torch.zeros_like(rgb)
    normal[..., 2] = 1.0
    return {
        "Cd": rgb, "alpha": scalar, "depth": scalar, "position": rgb, "normal": normal,
        "roughness": scalar, "f0": rgb, "ks": scalar, "final": rgb,
        "reflection_color": rgb, "reflection_raw_color": rgb,
        "reflection_alpha": scalar, "reflection_depth": scalar,
        "reflection_hit_mask": torch.ones_like(scalar), "microfacet_D": scalar, "microfacet_F": rgb,
        "microfacet_G": scalar, "microfacet_fr": rgb, "microfacet_wr": rgb,
        "diffuse_contribution": rgb, "reflection_contribution": rgb,
        "valid_surface_mask": torch.ones_like(scalar), "surface_roughness": scalar,
        "valid_ray_count": height * width,
        "ray_candidate_count": candidate_count,
        "ray_exact_intersection_count": exact_count,
        "ray_diagnostics": SimpleNamespace(
            timing_ms={
                "bvh_sync_cuda": 1.0,
                "traversal_cuda": 2.0,
                "intersection_composite_cuda": 3.0,
                "raytrace_wall": 7.0,
            },
            chunk_count=2,
            reflection_surfel_count=32,
            peak_memory_allocated_bytes=4096,
            peak_memory_delta_bytes=1024,
            bvh_rebuild_delta=1,
            bvh_refit_delta=0,
        ),
    }


def test_debug_writer_emits_only_real_stage_b_maps_and_no_mask_when_disabled(tmp_path):
    output = _output()
    save_reflection_debug_maps(output, torch.zeros((3, 4, 5)), str(tmp_path))
    names = {path.name for path in Path(tmp_path).iterdir()}
    assert {"final.png", "reflection_color.png", "reflection_alpha.png", "reflection_depth.png",
            "reflection_hit_mask.png", "microfacet_D.png", "microfacet_F.png", "microfacet_G.png",
            "microfacet_fr.png", "microfacet_wr.png", "diffuse_contribution.png",
            "reflection_contribution.png", "reflection_contribution_vis.png",
            "microfacet_D_log.png", "ray_candidate_count.png",
            "ray_exact_intersection_count.png", "reflection_metadata.json"} <= names
    assert "specular_mask.png" not in names and "overlay.png" not in names
    assert not any(
        token in name for name in names
        for token in ("transmittance", "inside", "outside", "two_hit", "near_depth", "far_depth", "depth_violation")
    )


def test_debug_writer_preserves_physical_maps_and_records_display_scales_and_raw_stats(tmp_path):
    output = _output()
    height, width = output["reflection_contribution"].shape[:2]
    tiny = torch.linspace(1e-6, 3e-4, height * width * 3).reshape(height, width, 3)
    output["reflection_contribution"] = tiny
    output["microfacet_D"] = torch.linspace(1.0, 16.0, height * width).reshape(height, width, 1)

    save_reflection_debug_maps(output, torch.zeros((3, height, width)), str(tmp_path))

    physical_contribution = np.asarray(Image.open(tmp_path / "reflection_contribution.png"))
    visible_contribution = np.asarray(Image.open(tmp_path / "reflection_contribution_vis.png"))
    physical_d = np.asarray(Image.open(tmp_path / "microfacet_D.png"))
    visible_d = np.asarray(Image.open(tmp_path / "microfacet_D_log.png"))
    assert not physical_contribution.any()
    assert visible_contribution.max() == 255 and visible_contribution.min() < 255
    assert physical_d.min() == 255
    assert visible_d.max() == 255 and visible_d.min() < 255

    metadata = json.loads((tmp_path / "reflection_metadata.json").read_text())
    assert metadata["raw_stats"]["reflection_contribution"]["p99"] > 0
    assert metadata["raw_stats"]["ray_candidate_count"]["max"] == height * width
    assert metadata["display_scales"]["reflection_contribution_vis"]["transform"] == "linear"
    assert metadata["display_scales"]["microfacet_D_log"]["transform"] == "log1p"
    assert metadata["raytrace"]["chunk_count"] == 2
    assert metadata["raytrace"]["reflection_surfel_count"] == 32


def test_debug_writer_emits_real_specular_mask_only_when_provided(tmp_path):
    mask = torch.ones((1, 4, 5))
    save_reflection_debug_maps(
        _output(), torch.zeros((3, 4, 5)), str(tmp_path), specular_mask=mask, mask_sha256="abc"
    )
    assert (tmp_path / "specular_mask.png").is_file()
    assert (tmp_path / "overlay.png").is_file()
