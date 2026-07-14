import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from utils.internal_object_mask import (
    DEFAULT_CANDIDATE_GUARDS,
    EXPECTED_STEMS,
    INTERNAL_OBJECT_SEMANTICS_VERSION,
    MASK_INTERPOLATION,
    PROPOSAL_ROLE,
    REVIEWED_ROLE,
    SCHEMA_VERSION,
    candidate_metrics,
    canonical_payload_sha256,
    clip_mask_to_glass,
    load_resized_internal_object_masks,
    object_domain_metrics,
    object_occupancy_domains,
    object_occupancy_loss,
    proposal_view_metadata,
    score_candidate,
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
    glass_root = scene / "specular_masks_reviewed_v1"
    root.mkdir(parents=True)
    for role in ("bird", "internal_base", "internal_object_union", "glass_hard"):
        (root / role).mkdir()
    (root / "internal_ignore").mkdir()
    entries = []
    import hashlib
    aggregate = hashlib.sha256()
    rgb_hashes = {}
    glass_hashes = {}
    glass_hard_hashes = {}
    bird_hashes = {}
    internal_base_hashes = {}
    union_hashes = {}
    for stem in EXPECTED_STEMS:
        rgb = images / f"{stem}.jpg"
        _write_rgb(rgb)
        glass = np.ones((6, 8), dtype=np.uint8)
        glass_path = glass_root / f"{stem}.png"
        _write_mask(glass_path, glass)
        glass_hard_path = root / "glass_hard" / f"{stem}.png"
        _write_mask(glass_hard_path, glass)
        bird = np.zeros((6, 8), dtype=np.uint8)
        internal_base = np.zeros((6, 8), dtype=np.uint8)
        bird[1:3, 2:4] = 1
        internal_base[3:5, 3:6] = 1
        union = np.maximum(bird, internal_base)
        masks = {"bird": bird, "internal_base": internal_base, "internal_object_union": union}
        entry = {
            "stem": stem,
            "human_status": "accepted",
            "rgb_path": f"images/{stem}.jpg",
            "rgb_sha256": sha256_file(rgb),
            "glass_mask_path": f"specular_masks_reviewed_v1/{stem}.png",
            "glass_mask_sha256": sha256_file(glass_path),
            "glass_hard_mask_path": f"glass_hard/{stem}.png",
            "glass_hard_mask_sha256": sha256_file(glass_hard_path),
        }
        rgb_hashes[stem] = entry["rgb_sha256"]
        glass_hashes[stem] = entry["glass_mask_sha256"]
        glass_hard_hashes[stem] = entry["glass_hard_mask_sha256"]
        for role, mask in masks.items():
            path = root / role / f"{stem}.png"
            _write_mask(path, mask)
            digest = sha256_file(path)
            entry[f"{role}_mask_path"] = f"{role}/{stem}.png"
            entry[f"{role}_mask_sha256"] = digest
            if role == "bird":
                bird_hashes[stem] = digest
            elif role == "internal_base":
                internal_base_hashes[stem] = digest
            else:
                union_hashes[stem] = digest
            aggregate.update(f"{stem} {role} {digest}\n".encode("utf-8"))
        entries.append(entry)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_role": REVIEWED_ROLE,
        "role": REVIEWED_ROLE,
        "internal_object_semantics_version": INTERNAL_OBJECT_SEMANTICS_VERSION,
        "legacy_aliases": {
            "base": "legacy_not_auto_promoted_to_internal_base",
            "bird_support": "legacy_not_auto_promoted_to_internal_base",
        },
        "human_status": "accepted",
        "accepted_count": 111,
        "count": 111,
        "ordered_stems": EXPECTED_STEMS,
        "accepted_stems": EXPECTED_STEMS,
        "accepted_with_warning": [],
        "mask_interpolation": MASK_INTERPOLATION,
        "aggregate_mask_sha256": aggregate.hexdigest(),
        "source_proposal_payload_hash": "0" * 64,
        "rgb_hashes": rgb_hashes,
        "glass_hashes": glass_hashes,
        "glass_hard_hashes": glass_hard_hashes,
        "bird_hashes": bird_hashes,
        "internal_base_hashes": internal_base_hashes,
        "internal_object_union_hashes": union_hashes,
        "provenance": {"method": "unit-test"},
        "entries": entries,
    }
    payload["canonical_payload_sha256"] = canonical_payload_sha256(payload)
    payload["manifest_payload_sha256"] = payload["canonical_payload_sha256"]
    (root / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    return root / "manifest.json"


def test_reviewed_manifest_validates_and_rejects_proposal(tmp_path):
    scene = tmp_path / "scene"
    manifest = _formal_manifest(scene, tmp_path / "reviewed")
    validated = validate_internal_object_mask_set(scene, "images", manifest)
    assert validated["role"] == REVIEWED_ROLE
    assert validated["count"] == 111
    assert validated["accepted_count"] == 111

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


def test_reviewed_manifest_glass_hash_mismatch_fails(tmp_path):
    scene = tmp_path / "scene"
    manifest = _formal_manifest(scene, tmp_path / "reviewed")
    _write_mask(scene / "specular_masks_reviewed_v1" / "000000.png", np.zeros((6, 8), dtype=np.uint8))
    with pytest.raises(ValueError, match="glass hash mismatch"):
        validate_internal_object_mask_set(scene, "images", manifest)


def test_reviewed_manifest_missing_and_extra_stem_fail(tmp_path):
    scene = tmp_path / "scene"
    root = tmp_path / "reviewed"
    manifest = _formal_manifest(scene, root)
    (root / "bird" / "000110.png").unlink()
    with pytest.raises(ValueError, match="missing or extra stems"):
        validate_internal_object_mask_set(scene, "images", manifest)

    manifest = _formal_manifest(scene, tmp_path / "reviewed_extra")
    _write_mask(tmp_path / "reviewed_extra" / "bird" / "999999.png", np.zeros((6, 8), dtype=np.uint8))
    with pytest.raises(ValueError, match="missing or extra stems"):
        validate_internal_object_mask_set(scene, "images", manifest)


def test_reviewed_manifest_mask_hash_mismatch_fails(tmp_path):
    scene = tmp_path / "scene"
    root = tmp_path / "reviewed"
    manifest = _formal_manifest(scene, root)
    mask = np.zeros((6, 8), dtype=np.uint8)
    mask[0, 0] = 1
    _write_mask(root / "bird" / "000000.png", mask)
    with pytest.raises(ValueError, match="bird hash mismatch"):
        validate_internal_object_mask_set(scene, "images", manifest)


def test_reviewed_manifest_union_outside_glass_fails(tmp_path):
    scene = tmp_path / "scene"
    root = tmp_path / "reviewed"
    manifest = _formal_manifest(scene, root)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    glass_hard = np.ones((6, 8), dtype=np.uint8)
    glass_hard[1, 2] = 0
    glass_hard_path = root / "glass_hard" / "000000.png"
    _write_mask(glass_hard_path, glass_hard)
    payload["entries"][0]["glass_hard_mask_sha256"] = sha256_file(glass_hard_path)
    payload["glass_hard_hashes"]["000000"] = sha256_file(glass_hard_path)
    payload["canonical_payload_sha256"] = canonical_payload_sha256(payload)
    payload["manifest_payload_sha256"] = payload["canonical_payload_sha256"]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="union is not subset of glass_hard"):
        validate_internal_object_mask_set(scene, "images", manifest)


def test_legacy_v2_bird_support_manifest_cannot_claim_v3(tmp_path):
    scene = tmp_path / "scene"
    root = tmp_path / "reviewed"
    manifest = _formal_manifest(scene, root)
    (root / "internal_base").rename(root / "bird_support")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    for entry in payload["entries"]:
        entry["bird_support_mask_path"] = entry.pop("internal_base_mask_path").replace("internal_base/", "bird_support/")
        entry["bird_support_mask_sha256"] = entry.pop("internal_base_mask_sha256")
    payload["internal_base_hashes"] = {}
    payload["canonical_payload_sha256"] = canonical_payload_sha256(payload)
    payload["manifest_payload_sha256"] = payload["canonical_payload_sha256"]
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="missing role directory|hash map"):
        validate_internal_object_mask_set(scene, "images", manifest)


def test_legacy_alias_cannot_auto_promote_bird_support_to_internal_base(tmp_path):
    scene = tmp_path / "scene"
    root = tmp_path / "reviewed"
    manifest = _formal_manifest(scene, root)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["legacy_aliases"] = {"bird_support": "internal_base"}
    payload["canonical_payload_sha256"] = canonical_payload_sha256(payload)
    payload["manifest_payload_sha256"] = payload["canonical_payload_sha256"]
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="may not be auto-promoted"):
        validate_internal_object_mask_set(scene, "images", manifest)


def test_resized_loader_checks_union_semantics(tmp_path):
    scene = tmp_path / "scene"
    manifest = _formal_manifest(scene, tmp_path / "reviewed")
    validated = validate_internal_object_mask_set(scene, "images", manifest)
    loaded = load_resized_internal_object_masks(validated, "000000", (8, 6), (4, 3))
    assert loaded["bird"].shape == (1, 3, 4)
    assert loaded["internal_base"].shape == (1, 3, 4)
    assert loaded["internal_object_union"].shape == (1, 3, 4)


def test_accepted_with_warning_does_not_block_loading(tmp_path):
    scene = tmp_path / "scene"
    root = tmp_path / "reviewed"
    manifest = _formal_manifest(scene, root)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["accepted_with_warning"] = ["000012"]
    payload["entries"][12]["human_status"] = "accepted_with_warning"
    payload["canonical_payload_sha256"] = canonical_payload_sha256(payload)
    payload["manifest_payload_sha256"] = payload["canonical_payload_sha256"]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    validated = validate_internal_object_mask_set(scene, "images", manifest)
    assert validated["accepted_with_warning"] == ["000012"]


def test_wrong_role_status_or_version_fails(tmp_path):
    scene = tmp_path / "scene"
    for field, value, pattern in (
        ("artifact_role", "stage_d_internal_object_mask_proposal_111_v3", "formal reviewed"),
        ("human_status", "proposal_requires_review", "accepted"),
        ("internal_object_semantics_version", "v2", "semantic version"),
    ):
        root = tmp_path / f"reviewed_{field}"
        manifest = _formal_manifest(scene, root)
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload[field] = value
        payload["canonical_payload_sha256"] = canonical_payload_sha256(payload)
        payload["manifest_payload_sha256"] = payload["canonical_payload_sha256"]
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match=pattern):
            validate_internal_object_mask_set(scene, "images", manifest)


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
    _write_mask(tmp_path / "internal_base.png", raw_base)
    _write_mask(tmp_path / "glass.png", glass)
    clipped, stats = clip_mask_to_glass(raw_bird, glass)
    assert stats["raw_area"] == 24
    assert stats["clipped_area"] == 12
    assert stats["removed_outside_glass_area"] == 12
    metadata = proposal_view_metadata(
        "000000", rgb, tmp_path / "bird.png", tmp_path / "internal_base.png", tmp_path / "glass.png",
    )
    assert metadata["clipping"]["internal_object_union"]["removed_fraction"] > 0.25
    assert "large_prediction_outside_glass" in metadata["review_flags"]
    assert clipped.sum() == 12


def test_candidate_scoring_rejects_glass_fill_leaks_and_fragmented_bird():
    glass = np.zeros((10, 10), dtype=bool)
    glass[1:9, 1:9] = True
    whole_glass = glass.copy()
    metrics = candidate_metrics(whole_glass, glass)
    scored = score_candidate("yellow_base_board", 0.9, metrics, DEFAULT_CANDIDATE_GUARDS)
    assert "candidate_too_similar_to_glass_hard" in scored["reject_reasons"]
    assert "mask_glass_ratio_too_large" in scored["reject_reasons"]

    leaking = np.zeros((10, 10), dtype=bool)
    leaking[1:4, 1:4] = True
    leaking[0:3, 0:3] = True
    metrics = candidate_metrics(leaking, glass)
    scored = score_candidate("white_platform", 0.8, metrics, DEFAULT_CANDIDATE_GUARDS)
    assert "raw_outside_glass_ratio_too_high" in scored["reject_reasons"]

    fragments = np.zeros((10, 10), dtype=bool)
    fragments[2:4, 2:4] = True
    fragments[6:8, 6:8] = True
    metrics = candidate_metrics(fragments, glass)
    scored = score_candidate("bird", 0.8, metrics, DEFAULT_CANDIDATE_GUARDS)
    assert "multiple_distant_bird_components" in scored["reject_reasons"]


def test_candidate_scoring_rejects_isolated_side_pole_fixture():
    glass = np.ones((32, 32), dtype=bool)
    bird = np.zeros((32, 32), dtype=bool)
    bird[6:14, 10:18] = True
    yellow = np.zeros((32, 32), dtype=bool)
    yellow[24:29, 6:26] = True
    white = np.zeros((32, 32), dtype=bool)
    white[20:23, 12:18] = True
    pole = np.zeros((32, 32), dtype=bool)
    pole[8:26, 27:29] = True
    pole[25:29, 25:31] = True
    metrics = candidate_metrics(
        pole, glass, bird_mask=bird,
        yellow_base_board_mask=yellow, white_platform_mask=white,
    )
    scored = score_candidate("connected_fixture", 0.9, metrics, DEFAULT_CANDIDATE_GUARDS)
    assert "fixture_not_connected_bird_to_base" in scored["reject_reasons"]
    assert "isolated_vertical_pole_or_rail" in scored["reject_reasons"]


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


def test_object_domains_put_explicit_ignore_outside_positive_and_negative():
    obj = torch.zeros((5, 5, 1))
    obj[2, 2, 0] = 1
    glass = torch.ones((5, 5, 1))
    valid = torch.ones((5, 5, 1))
    ignore = torch.zeros((5, 5, 1))
    ignore[0, 0, 0] = 1
    domains = object_occupancy_domains(
        obj, glass, valid, internal_ignore=ignore, erode_px=0, dilate_px=0,
    )
    assert int(domains["Mpos"][2, 2, 0]) == 1
    assert int(domains["Mignore"][0, 0, 0]) == 1
    assert int(domains["Mneg"][0, 0, 0]) == 0
    assert int((domains["Mpos"] * domains["Mignore"]).sum()) == 0
    assert int((domains["Mignore"] * domains["Mneg"]).sum()) == 0


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
