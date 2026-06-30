from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from stage_b_training import _config, _validate_stage_b_args
from utils.loss_utils import specular_constraint_loss
from utils.specular_mask import validate_specular_mask_set


def _scene(tmp_path, count=2):
    images = tmp_path / "images"
    masks = tmp_path / "specular_masks"
    images.mkdir()
    masks.mkdir()
    for index in range(count):
        Image.new("RGB", (6, 4), (20, 30, 40)).save(images / f"{index:06d}.jpg")
        Image.fromarray(np.full((4, 6), 128, dtype=np.uint8)).save(masks / f"{index:06d}.png")
    return images, masks


def test_complete_soft_mask_set_is_strictly_validated_and_hashed(tmp_path):
    _scene(tmp_path)
    result = validate_specular_mask_set(tmp_path, "images", "specular_masks")
    assert result["count"] == 2
    assert len(result["aggregate_sha256"]) == 64
    assert all(len(entry["sha256"]) == 64 for entry in result["entries"])


def test_missing_extra_rgb_or_wrong_size_masks_fail(tmp_path):
    _, masks = _scene(tmp_path)
    (masks / "000001.png").unlink()
    with pytest.raises(ValueError, match="match images exactly"):
        validate_specular_mask_set(tmp_path, "images", "specular_masks")
    Image.new("RGB", (6, 4)).save(masks / "000001.png")
    with pytest.raises(ValueError, match="single-channel"):
        validate_specular_mask_set(tmp_path, "images", "specular_masks")
    Image.new("L", (5, 4)).save(masks / "000001.png")
    with pytest.raises(ValueError, match="size mismatch"):
        validate_specular_mask_set(tmp_path, "images", "specular_masks")


def test_specular_constraint_and_lambda_zero_mask_boundary():
    ks = torch.tensor([[[0.2], [0.95]]])
    mask = torch.tensor([[[1.0], [1.0]]])
    assert torch.allclose(specular_constraint_loss(ks, mask, 0.9), torch.tensor(0.35))
    dataset = SimpleNamespace(
        model_type="surfel", stage="stage_b", ray_background="scene", specular_masks="masks",
        reflection_init_mode="random_bbox", reflection_init_count=4,
    )
    opt = SimpleNamespace(random_background=False, lambda_spec=0.0)
    with pytest.raises(ValueError, match="empty --specular_masks"):
        _validate_stage_b_args(dataset, opt, None, "diffuse.pth")
    dataset.specular_masks = ""
    opt.lambda_spec = 0.2
    with pytest.raises(ValueError, match="requires a complete"):
        _validate_stage_b_args(dataset, opt, None, "diffuse.pth")


def test_stage_b_checkpoint_config_records_both_schedules():
    dataset = SimpleNamespace(
        roughness_remap=False, material_alpha_threshold=1e-4, ray_background="scene",
        ray_chunk_size=16, ray_cutoff_sigma=3.0, ray_hit_threshold=1e-4,
        ray_epsilon_scale=1e-4,
    )
    opt = SimpleNamespace(
        lambda_norm=0.04, lambda_mono=0.01, lambda_perc=0.01, lambda_spec=0.0, specular_k0=0.9,
        position_lr_init=1e-4, position_lr_final=1e-6, position_lr_delay_mult=0.01,
        position_lr_max_steps=100, densify_from_iter=10, densify_until_iter=50,
        densification_interval=5, densify_grad_threshold=2e-4,
        reflection_position_lr_init=2e-4, reflection_position_lr_final=2e-6,
        reflection_position_lr_delay_mult=0.02, reflection_position_lr_max_steps=80,
        reflection_color_lr=2e-3, reflection_opacity_lr=2e-2, reflection_scaling_lr=3e-3,
        reflection_rotation_lr=4e-3, reflection_percent_dense=0.02,
        reflection_densify_from_iter=4, reflection_densify_until_iter=60,
        reflection_densification_interval=6, reflection_densify_grad_threshold=3e-4,
        reflection_min_opacity=0.01, reflection_prune_unhit_after=20,
    )
    diffuse = SimpleNamespace(roughness_min=0.03)
    config = _config(dataset, opt, diffuse, None)
    assert config["diffuse_schedule"]["position_lr_max_steps"] == 100
    assert config["reflection_schedule"]["position_lr_init"] == 2e-4
    assert config["reflection_schedule"]["prune_unhit_after"] == 20
