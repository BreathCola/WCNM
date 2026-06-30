from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from stage_b_training import _validate_stage_b_args
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
