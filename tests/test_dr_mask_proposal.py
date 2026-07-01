import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from utils.dr_mask_proposal import (
    PROPOSAL_FILES,
    ProposalAuditError,
    audit_dr_artifacts,
    generate_proposals,
)
from utils.specular_mask import validate_specular_mask_set
from utils.dr_mask_review import (
    BOUNDARY_CROP_NAMES,
    build_review_package,
    generate_all_real_proposals,
    validate_proposal_frames,
)
from utils.dr_mask_high_risk_review import (
    REVIEW_FIELDS,
    generate_high_risk_review_pack,
    proposal_tree_digest,
)


def _save_rgb(path: Path, values: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(values.astype(np.uint8)).save(path)


def _fixture(tmp_path: Path, count=2):
    scene = tmp_path / "scene"
    image_root = scene / "images"
    raw_root = tmp_path / "raw"
    artifact_root = raw_root / "raw" / "TiHuBird"
    image_root.mkdir(parents=True)
    artifact_root.mkdir(parents=True)
    source_size = (64, 40)
    native_size = (32, 20)
    records = []
    for index in range(count):
        stem = f"{index:06d}"
        yy, xx = np.mgrid[:source_size[1], :source_size[0]]
        source = np.stack(
            [
                (xx * 3 + index * 7) % 255,
                (yy * 5 + index * 11) % 255,
                ((xx + yy) * 2 + 40) % 255,
            ],
            axis=-1,
        ).astype(np.uint8)
        source_path = image_root / f"{stem}.jpg"
        Image.fromarray(source).save(source_path, quality=95)
        source_rgb = np.asarray(Image.open(source_path).convert("RGB"), dtype=np.uint8)
        resized = np.asarray(
            Image.fromarray(source_rgb).resize(native_size, Image.Resampling.BILINEAR),
            dtype=np.uint8,
        )
        prefix = f"0000.{index:04d}"
        normal = np.full((native_size[1], native_size[0], 3), (127, 127, 255), dtype=np.uint8)
        depth = np.full((native_size[1], native_size[0], 3), 210, dtype=np.uint8)
        depth[4:18, 8:25] = 20
        basecolor = np.full_like(depth, 35)
        basecolor[4:18, 8:25] = 235
        albedo = np.full_like(depth, 10)
        for kind, values in (
            ("rgb", resized),
            ("normal", normal),
            ("depth", depth),
            ("basecolor", basecolor),
            ("diffuse_albedo", albedo),
        ):
            _save_rgb(artifact_root / f"{prefix}.{kind}.png", values)
        import hashlib
        digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        mapping = {
            "source_resolution": {"width": source_size[0], "height": source_size[1]},
            "native_prior_resolution": {"width": native_size[0], "height": native_size[1]},
            "resize_filter": "PIL.Image.BILINEAR",
            "crop": None,
            "pad": None,
        }
        records.append(
            {
                "raw_slot": index,
                "chunk_index": 0,
                "frame_index": index,
                "frame_kind": "real",
                "is_padding": False,
                "rtgs_real_frame_index": index,
                "rtgs_input_file": f"data/TiHuBird/images/{stem}.jpg",
                "input_sha256": digest,
                "diffusion_renderer_rgb_file": f"raw/TiHuBird/{prefix}.rgb.png",
                "raw_prior_files": {
                    kind: f"raw/TiHuBird/{prefix}.{kind}.png"
                    for kind in ("normal", "depth", "basecolor", "diffuse_albedo")
                },
                "source_to_prior_mapping": mapping,
            }
        )
    padding_index = count
    padding_prefix = f"0000.{padding_index:04d}"
    for kind in ("rgb", "normal", "depth", "basecolor", "diffuse_albedo"):
        source = artifact_root / f"0000.{count - 1:04d}.{kind}.png"
        target = artifact_root / f"{padding_prefix}.{kind}.png"
        target.write_bytes(source.read_bytes())
    records.append(
        {
            "raw_slot": padding_index,
            "chunk_index": 0,
            "frame_index": padding_index,
            "frame_kind": "padding",
            "is_padding": True,
            "rtgs_real_frame_index": None,
            "rtgs_input_file": None,
            "diffusion_renderer_rgb_file": f"raw/TiHuBird/{padding_prefix}.rgb.png",
            "raw_prior_files": {
                kind: f"raw/TiHuBird/{padding_prefix}.{kind}.png"
                for kind in ("normal", "depth", "basecolor", "diffuse_albedo")
            },
        }
    )
    manifest = {
        "real_frame_count": count,
        "padding_frame_count": 1,
        "source_to_prior_mapping": records[0]["source_to_prior_mapping"],
        "frames_and_padding": records,
    }
    (raw_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (raw_root / "validation_summary.json").write_text(
        json.dumps(
            {
                "validation": "PASS",
                "real_frames": count,
                "pixel_mapping_global_max_abs_difference": 0,
            }
        ),
        encoding="utf-8",
    )
    return scene, raw_root, records


def test_strict_audit_maps_stems_and_excludes_padding(tmp_path):
    scene, raw_root, _ = _fixture(tmp_path)
    audit = audit_dr_artifacts(scene, raw_root, expected_count=2)
    assert audit["status"] == "PASS"
    assert [entry["stem"] for entry in audit["entries"]] == ["000000", "000001"]
    assert all(not entry["is_padding"] for entry in audit["entries"])
    assert len(audit["padding_slots_excluded"]) == 1
    assert audit["padding_slots_excluded"][0]["is_padding"] is True
    assert audit["rgb_mapping_global_max_abs_difference"] == 0


def test_audit_fails_closed_on_artifact_size_mismatch(tmp_path):
    scene, raw_root, records = _fixture(tmp_path)
    path = raw_root / records[0]["raw_prior_files"]["depth"]
    Image.new("RGB", (31, 20)).save(path)
    with pytest.raises(ProposalAuditError, match="size mismatch"):
        audit_dr_artifacts(scene, raw_root, expected_count=2)


@pytest.mark.parametrize("kind", ["normal", "depth"])
def test_audit_fails_closed_on_missing_required_artifact(tmp_path, kind):
    scene, raw_root, records = _fixture(tmp_path)
    (raw_root / records[0]["raw_prior_files"][kind]).unlink()
    with pytest.raises(ProposalAuditError, match=f"missing {kind}"):
        audit_dr_artifacts(scene, raw_root, expected_count=2)


def test_audit_rejects_invalid_normal_encoding(tmp_path):
    scene, raw_root, records = _fixture(tmp_path)
    path = raw_root / records[0]["raw_prior_files"]["normal"]
    Image.new("L", (32, 20), 128).save(path)
    with pytest.raises(ProposalAuditError, match="expected RGB"):
        audit_dr_artifacts(scene, raw_root, expected_count=2)


def test_audit_rejects_near_zero_decoded_normal(tmp_path):
    scene, raw_root, records = _fixture(tmp_path)
    path = raw_root / records[0]["raw_prior_files"]["normal"]
    _save_rgb(path, np.full((20, 32, 3), 127, dtype=np.uint8))
    with pytest.raises(ProposalAuditError, match="near-zero pixels"):
        audit_dr_artifacts(scene, raw_root, expected_count=2)


def test_proposal_outputs_have_source_size_range_and_complete_files(tmp_path):
    scene, raw_root, _ = _fixture(tmp_path)
    audit = audit_dr_artifacts(scene, raw_root, expected_count=2)
    output = tmp_path / "proposal_output"
    metadata = generate_proposals(audit, output, ["000000", "000001"])
    assert len(metadata) == 2
    for stem in ("000000", "000001"):
        directory = output / "proposal_soft" / stem
        assert {path.name for path in directory.iterdir()} == set(PROPOSAL_FILES)
        for name in PROPOSAL_FILES[:-1]:
            with Image.open(directory / name) as image:
                assert image.size == (64, 40)
        soft = np.asarray(Image.open(directory / "proposal_soft.png"), dtype=np.uint8)
        assert soft.min() >= 0 and soft.max() <= 255 and soft.max() > soft.min()
        payload = json.loads((directory / "proposal_metadata.json").read_text())
        assert payload["training_role"] is None
        assert set(payload["risks"]) == {
            "bird_may_be_included", "background_may_be_included",
            "glass_edge_may_be_missing", "reflection_may_be_misclassified",
            "low_confidence_region",
        }
    assert (output / "contact_sheet_9views.png").is_file()
    assert not (output / "reviewed_soft").exists()


def test_proposal_directory_is_not_a_valid_training_mask_set(tmp_path):
    scene, raw_root, _ = _fixture(tmp_path)
    audit = audit_dr_artifacts(scene, raw_root, expected_count=2)
    output = tmp_path / "proposal_output"
    generate_proposals(audit, output, ["000000", "000001"])
    with pytest.raises(ValueError, match="must match images exactly"):
        validate_specular_mask_set(scene, "images", str(output / "proposal_soft"))


def test_review_package_is_complete_sorted_and_has_boundary_crops(tmp_path):
    scene, raw_root, _ = _fixture(tmp_path)
    audit = audit_dr_artifacts(scene, raw_root, expected_count=2)
    output = tmp_path / "review_output"
    generate_all_real_proposals(audit, output, expected_count=2)
    manifest = build_review_package(audit, output, expected_count=2)
    assert manifest["status"] == "PASS"
    assert manifest["completeness"]["validated_real_frames"] == 2
    assert manifest["review_queue_count"] == 2
    assert manifest["padding_exclusion_proof"]["padding_proposal_count"] == 0
    assert len(manifest["chronological_contact_sheets"]) == 1
    assert len(manifest["priority_contact_sheets"]) == 1
    assert (output / "distribution_plot.png").is_file()
    assert (output / "anomalies.json").is_file()
    queue = json.loads((output / "review_queue" / "review_queue.json").read_text())
    assert {item["stem"] for item in queue["items"]} == {"000000", "000001"}
    for stem in ("000000", "000001"):
        crop_root = output / "boundary_crops" / stem
        assert {path.name for path in crop_root.glob("*.png")} == set(BOUNDARY_CROP_NAMES)
    assert not (output / "reviewed_soft").exists()


def test_review_validation_fails_closed_on_missing_proposal(tmp_path):
    scene, raw_root, _ = _fixture(tmp_path)
    audit = audit_dr_artifacts(scene, raw_root, expected_count=2)
    output = tmp_path / "review_output"
    generate_all_real_proposals(audit, output, expected_count=2)
    (output / "proposal_soft" / "000001" / "proposal_soft.png").unlink()
    with pytest.raises(ProposalAuditError, match="incomplete proposal files"):
        validate_proposal_frames(audit, output, expected_count=2)


def test_review_validation_fails_closed_on_degenerate_proposal(tmp_path):
    scene, raw_root, _ = _fixture(tmp_path)
    audit = audit_dr_artifacts(scene, raw_root, expected_count=2)
    output = tmp_path / "review_output"
    generate_all_real_proposals(audit, output, expected_count=2)
    Image.new("L", (64, 40), 0).save(
        output / "proposal_soft" / "000000" / "proposal_soft.png"
    )
    with pytest.raises(ProposalAuditError, match="both zero exterior and 255"):
        validate_proposal_frames(audit, output, expected_count=2)


def test_high_risk_review_pack_is_read_only_and_human_fillable(tmp_path):
    scene, raw_root, _ = _fixture(tmp_path)
    audit = audit_dr_artifacts(scene, raw_root, expected_count=2)
    proposal_output = tmp_path / "proposal_review"
    generate_all_real_proposals(audit, proposal_output, expected_count=2)
    build_review_package(audit, proposal_output, expected_count=2)
    before = proposal_tree_digest(proposal_output / "proposal_soft")
    review_output = tmp_path / "high_risk_review"
    groups = {
        "background_risk": ("000000",),
        "reflection_risk": (),
        "highest_uncertainty": ("000001",),
    }
    manifest = generate_high_risk_review_pack(proposal_output, review_output, groups)
    after = proposal_tree_digest(proposal_output / "proposal_soft")
    assert before == after
    assert manifest["status"] == "PASS"
    assert manifest["view_count"] == 2
    assert manifest["proposal_files_modified"] is False
    assert (review_output / "high_risk_contact_sheet.png").is_file()
    checklist = json.loads((review_output / "review_checklist.json").read_text())
    assert len(checklist["items"]) == 2
    assert all(item[field] is None for item in checklist["items"] for field in REVIEW_FIELDS)
    for stem in ("000000", "000001"):
        metadata = json.loads(
            (review_output / "views" / stem / "review_pack_metadata.json").read_text()
        )
        assert (review_output / "views" / stem / "review_pack.png").is_file()
        for crop in metadata["crops"].values():
            box = crop["box_xyxy_source"]
            assert crop["resampling"] is None
            assert crop["source_pixel_size"] == [box[2] - box[0], box[3] - box[1]]
    assert not (review_output / "reviewed_soft").exists()


def test_high_risk_review_pack_rejects_duplicate_stems(tmp_path):
    scene, raw_root, _ = _fixture(tmp_path)
    audit = audit_dr_artifacts(scene, raw_root, expected_count=2)
    proposal_output = tmp_path / "proposal_review"
    generate_all_real_proposals(audit, proposal_output, expected_count=2)
    build_review_package(audit, proposal_output, expected_count=2)
    groups = {
        "background_risk": ("000000",),
        "reflection_risk": ("000000",),
        "highest_uncertainty": (),
    }
    with pytest.raises(ProposalAuditError, match="duplicate stems"):
        generate_high_risk_review_pack(
            proposal_output, tmp_path / "high_risk_review", groups
        )
