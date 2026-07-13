import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from utils.internal_object_mask import (
    EXPECTED_STEMS,
    MASK_INTERPOLATION,
    PROPOSAL_ROLE,
    REVIEWED_ROLE,
    SCHEMA_VERSION,
    canonical_payload_sha256,
    clip_mask_to_glass,
    load_resized_internal_object_masks,
    object_domain_metrics,
    object_occupancy_domains,
    object_occupancy_loss,
    proposal_view_metadata,
    sha256_file,
    validate_internal_object_mask_set,
)


def _write_rgb(path, color=(60, 70, 80)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((6, 8, 3), color, dtype=np.uint8)).save(path)


def _write_mask(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.asarray(values, dtype=np.uint8) * 255)).save(path)


def _formal_manifest(scene, root):
    images = scene / "images"
    root.mkdir(parents=True)
    for role in ("bird", "base", "union"):
        (root / role).mkdir()
    entries = []
    import hashlib
    aggregate = hashlib.sha256()
    for stem in EXPECTED_STEMS:
        rgb = images / f"{stem}.jpg"
        _write_rgb(rgb)
        bird = np.zeros((6, 8), dtype=np.uint8)
        base = np.zeros((6, 8), dtype=np.uint8)
        bird[1:3, 2:4] = 1
        base[3:5, 3:6] = 1
        union = np.maximum(bird, base)
        masks = {"bird": bird, "base": base, "union": union}
        entry = {
            "stem": stem,
            "human_status": "accepted",
            "rgb_path": f"images/{stem}.jpg",
            "rgb_sha256": sha256_file(rgb),
        }
        for role, mask in masks.items():
            path = root / role / f"{stem}.png"
            _write_mask(path, mask)
            digest = sha256_file(path)
            entry[f"{role}_mask_path"] = f"{role}/{stem}.png"
            entry[f"{role}_mask_sha256"] = digest
            aggregate.update(f"{stem} {role} {digest}\n".encode("utf-8"))
        entries.append(entry)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "role": REVIEWED_ROLE,
        "human_status": "accepted",
        "count": 111,
        "ordered_stems": EXPECTED_STEMS,
        "mask_interpolation": MASK_INTERPOLATION,
        "aggregate_mask_sha256": aggregate.hexdigest(),
        "provenance": {"method": "unit-test"},
        "entries": entries,
    }
    payload["manifest_payload_sha256"] = canonical_payload_sha256(payload)
    (root / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    return root / "manifest.json"


def test_reviewed_manifest_validates_and_rejects_proposal(tmp_path):
    scene = tmp_path / "scene"
    manifest = _formal_manifest(scene, tmp_path / "reviewed")
    validated = validate_internal_object_mask_set(scene, "images", manifest)
    assert validated["role"] == REVIEWED_ROLE
    assert validated["count"] == 111

    proposal = tmp_path / "proposal.json"
    proposal.write_text(
        json.dumps({
            "schema_version": SCHEMA_VERSION,
            "role": PROPOSAL_ROLE,
            "count": 111,
            "ordered_stems": EXPECTED_STEMS,
            "entries": [],
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="proposal"):
        validate_internal_object_mask_set(scene, "images", proposal)


def test_reviewed_manifest_rgb_hash_mismatch_fails(tmp_path):
    scene = tmp_path / "scene"
    manifest = _formal_manifest(scene, tmp_path / "reviewed")
    _write_rgb(scene / "images" / "000000.jpg", color=(1, 2, 3))
    with pytest.raises(ValueError, match="RGB hash mismatch"):
        validate_internal_object_mask_set(scene, "images", manifest)


def test_resized_loader_checks_union_semantics(tmp_path):
    scene = tmp_path / "scene"
    manifest = _formal_manifest(scene, tmp_path / "reviewed")
    validated = validate_internal_object_mask_set(scene, "images", manifest)
    loaded = load_resized_internal_object_masks(validated, "000000", (8, 6), (4, 3))
    assert loaded["bird"].shape == (1, 3, 4)
    assert loaded["base"].shape == (1, 3, 4)
    assert loaded["union"].shape == (1, 3, 4)


def test_raw_clipped_accounting_and_review_flags(tmp_path):
    rgb = tmp_path / "000000.jpg"
    _write_rgb(rgb)
    raw_bird = np.zeros((6, 8), dtype=np.uint8)
    raw_bird[:, :4] = 1
    raw_base = np.zeros((6, 8), dtype=np.uint8)
    raw_base[2:4, 4:6] = 1
    glass = np.zeros((6, 8), dtype=np.uint8)
    glass[:, 2:6] = 1
    _write_mask(tmp_path / "bird.png", raw_bird)
    _write_mask(tmp_path / "base.png", raw_base)
    _write_mask(tmp_path / "glass.png", glass)
    clipped, stats = clip_mask_to_glass(raw_bird, glass)
    assert stats["raw_area"] == 24
    assert stats["clipped_area"] == 12
    assert stats["removed_outside_glass_area"] == 12
    metadata = proposal_view_metadata(
        "000000", rgb, tmp_path / "bird.png", tmp_path / "base.png", tmp_path / "glass.png",
    )
    assert metadata["clipping"]["union"]["removed_fraction"] > 0.25
    assert "large_prediction_outside_glass" in metadata["review_flags"]
    assert clipped.sum() == 12


def test_object_domains_do_not_change_transmittance_ray_domain():
    obj = torch.zeros((5, 5, 1))
    obj[2, 2, 0] = 1
    glass = torch.zeros((5, 5, 1))
    glass[1:4, 1:4, 0] = 1
    valid = torch.ones((5, 5, 1))
    domains = object_occupancy_domains(obj, glass, valid, erode_px=0, dilate_px=1)
    assert torch.equal(domains["domain"], glass * valid)
    assert int(domains["Mpos"].sum()) == 1
    assert int(domains["Mneg"].sum()) == 0
    assert int(domains["Mignore"].sum()) == 8


def test_object_loss_is_float_and_only_requires_ain_grad():
    obj = torch.zeros((4, 4, 1))
    obj[:2, :2, 0] = 1
    glass = torch.ones((4, 4, 1))
    valid = torch.ones((4, 4, 1))
    domains = object_occupancy_domains(obj, glass, valid, erode_px=0, dilate_px=0)
    ain = torch.full((4, 4, 1), 0.2, requires_grad=True)
    frozen = torch.full((4, 4, 1), 0.5, requires_grad=True)
    loss = object_occupancy_loss(
        ain + frozen.detach() * 0.0,
        domains,
        alpha_floor=0.4,
        lambda_positive=1.0,
        lambda_negative=1.0,
    )
    loss["total"].backward()
    assert torch.isfinite(loss["total"])
    assert ain.grad is not None and ain.grad.abs().sum() > 0
    assert frozen.grad is None


def test_object_metrics_keep_cout_in_negative_region():
    package = {
        "inside_alpha": torch.zeros((2, 2, 1)),
        "inside_color": torch.zeros((2, 2, 3)),
        "outside_color": torch.ones((2, 2, 3)),
    }
    domains = {
        "Mpos": torch.tensor([[[1.0], [0.0]], [[0.0], [0.0]]]),
        "Mneg": torch.tensor([[[0.0], [1.0]], [[1.0], [1.0]]]),
    }
    metrics = object_domain_metrics(package, domains)
    assert metrics["cout_energy_mneg"] == 1.0
