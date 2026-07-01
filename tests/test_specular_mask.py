import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from stage_b_training import _config, _validate_stage_b_args, specular_gradient_diagnostics
from tools.create_formal_specular_masks import REPAIR_STEMS, create_archive
from utils.loss_utils import l1_loss, specular_constraint_loss
from utils.specular_mask import (
    MASK_INTERPOLATION,
    load_resized_formal_mask,
    validate_specular_mask_set,
)


def _mask(index, size=(6, 4)):
    values = np.zeros((size[1], size[0]), dtype=np.uint8)
    values[:, 1:5] = 255
    values[0, 1:5] = 128 + index % 2
    return values


def _formal_scene(tmp_path):
    scene = tmp_path / "data" / "TiHuBird"
    images = scene / "images"
    proposal = tmp_path / "output" / "proposals"
    repair = tmp_path / "output" / "repairs"
    images.mkdir(parents=True)
    (repair / "repair_candidates").mkdir(parents=True)
    for index in range(111):
        stem = f"{index:06d}"
        Image.new("RGB", (6, 4), (20 + index % 20, 30, 40)).save(images / f"{stem}.jpg")
        values = _mask(index)
        directory = proposal / "proposal_soft" / stem
        directory.mkdir(parents=True)
        Image.fromarray(values).save(directory / "proposal_soft.png")
        if stem in REPAIR_STEMS:
            repaired = values.copy()
            repaired[0, 4] = 0
            Image.fromarray(repaired).save(repair / "repair_candidates" / f"{stem}.png")
    destination = scene / "specular_masks_reviewed_v1"
    create_archive(scene, proposal, repair, destination, read_only=False)
    return scene, proposal, repair, destination


def test_formal_archive_is_complete_hashed_and_uses_exact_accepted_sources(tmp_path):
    scene, proposal, repair, destination = _formal_scene(tmp_path)
    result = validate_specular_mask_set(scene, "images", destination / "manifest.json")
    manifest = json.loads((destination / "manifest.json").read_text())
    assert result["count"] == 111
    assert result["mask_interpolation"] == MASK_INTERPOLATION
    assert len(result["aggregate_sha256"]) == 64
    assert manifest["padding_exclusion_proof"]["mixed_count"] == 0
    for entry in manifest["entries"]:
        stem = entry["stem"]
        source = (
            repair / "repair_candidates" / f"{stem}.png"
            if stem in REPAIR_STEMS
            else proposal / "proposal_soft" / stem / "proposal_soft.png"
        )
        assert (destination / f"{stem}.png").read_bytes() == source.read_bytes()
        assert entry["source"] == ("repair_candidate_v1" if stem in REPAIR_STEMS else "proposal_v1")
        assert entry["human_status"] == "accepted"


def test_raw_proposal_or_repair_directory_is_never_a_training_manifest(tmp_path):
    scene, proposal, repair, _ = _formal_scene(tmp_path)
    with pytest.raises(FileNotFoundError, match="formal specular mask manifest"):
        validate_specular_mask_set(scene, "images", proposal / "proposal_soft")
    with pytest.raises(FileNotFoundError, match="formal specular mask manifest"):
        validate_specular_mask_set(scene, "images", repair / "repair_candidates")


@pytest.mark.parametrize("damage", ["missing", "extra", "corrupt_hash", "bad_json"])
def test_formal_manifest_and_files_fail_closed(tmp_path, damage):
    scene, _, _, destination = _formal_scene(tmp_path)
    manifest_path = destination / "manifest.json"
    if damage == "missing":
        (destination / "000010.png").unlink()
    elif damage == "extra":
        Image.fromarray(_mask(0)).save(destination / "000111.png")
    elif damage == "corrupt_hash":
        Image.fromarray(np.flipud(_mask(0))).save(destination / "000010.png")
    else:
        manifest_path.write_text("{not-json")
    with pytest.raises((ValueError, FileNotFoundError)):
        validate_specular_mask_set(scene, "images", manifest_path)


def test_size_and_non_l_mode_fail_closed_during_archive_creation(tmp_path):
    scene, proposal, repair, destination = _formal_scene(tmp_path)
    # Build a fresh destination after damaging one source.
    for path in destination.iterdir():
        path.unlink()
    destination.rmdir()
    Image.new("RGB", (6, 4)).save(proposal / "proposal_soft" / "000010" / "proposal_soft.png")
    with pytest.raises(ValueError, match="mode L uint8"):
        create_archive(scene, proposal, repair, destination, read_only=False)


def test_formal_mask_resize_is_continuous_and_hash_checked(tmp_path):
    scene, _, _, destination = _formal_scene(tmp_path)
    result = validate_specular_mask_set(scene, "images", destination / "manifest.json")
    values, digest, _ = load_resized_formal_mask(result, "000000", (6, 4), (12, 8))
    assert values.shape == (8, 12)
    assert values.dtype == np.float32 and 0 <= values.min() <= values.max() <= 1
    assert np.any((values > 0) & (values < 1))
    assert digest == result["entries"]["000000"]["sha256"]


def test_specular_constraint_gradient_is_inside_only_and_rgb_loss_remains_full_image():
    ks = torch.tensor([[[0.2], [0.95]], [[0.3], [0.4]]], requires_grad=True)
    mask = torch.tensor([[[1.0, 0.0], [0.5, 0.0]]])
    valid = torch.ones_like(ks)
    value = specular_constraint_loss(ks, mask, 0.9)
    gradient = torch.autograd.grad(0.2 * value, ks)[0]
    audit = specular_gradient_diagnostics(ks, mask, valid, gradient)
    assert value > 0
    assert audit["l_spec_gradient"]["inside_nonzero_count"] > 0
    assert audit["l_spec_gradient"]["outside_max_abs"] == 0
    assert audit["l_spec_gradient"]["gradient_descent_ks_direction"] == "increase"

    prediction = torch.zeros((3, 2, 2))
    target = torch.zeros_like(prediction)
    target[:, 0, 1] = 1.0  # Error deliberately lies outside mask support.
    assert l1_loss(prediction, target) > 0


def test_lambda_and_diagnostic_boundaries():
    dataset = SimpleNamespace(
        model_type="surfel", stage="stage_b", ray_background="scene", specular_masks="manifest.json",
        reflection_init_mode="random_bbox", reflection_init_count=4,
    )
    opt = SimpleNamespace(random_background=False, lambda_spec=0.0, specular_smoke_diagnostics=False)
    with pytest.raises(ValueError, match="empty --specular_masks"):
        _validate_stage_b_args(dataset, opt, None, "diffuse.pth")
    dataset.specular_masks = ""
    opt.lambda_spec = 0.2
    with pytest.raises(ValueError, match="formal"):
        _validate_stage_b_args(dataset, opt, None, "diffuse.pth")
    opt.specular_smoke_diagnostics = True
    opt.lambda_spec = 0.0
    with pytest.raises(ValueError, match="diagnostics"):
        _validate_stage_b_args(dataset, opt, None, "diffuse.pth")


def test_stage_b_checkpoint_config_records_schedules_and_formal_manifest():
    dataset = SimpleNamespace(
        roughness_remap=False, material_alpha_threshold=1e-4, ray_background="scene",
        ray_chunk_size=16, ray_cutoff_sigma=3.0, ray_hit_threshold=1e-4,
        ray_epsilon_scale=1e-4,
    )
    opt = SimpleNamespace(
        lambda_norm=0.04, lambda_mono=0.01, lambda_perc=0.01, lambda_spec=0.2, specular_k0=0.9,
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
    manifest = {
        "role": "stage_b_formal_reviewed_specular_soft_masks", "count": 111,
        "aggregate_sha256": "a" * 64, "manifest_payload_sha256": "b" * 64,
        "manifest_file_sha256": "c" * 64, "mask_interpolation": MASK_INTERPOLATION,
    }
    config = _config(dataset, opt, SimpleNamespace(roughness_min=0.03), manifest)
    assert config["diffuse_schedule"]["position_lr_max_steps"] == 100
    assert config["reflection_schedule"]["position_lr_init"] == 2e-4
    assert config["specular_mask"]["count"] == 111
    assert config["specular_mask"]["mask_interpolation"] == MASK_INTERPOLATION
