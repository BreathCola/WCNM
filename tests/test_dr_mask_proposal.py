import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from utils.dr_mask_proposal import (
    PROPOSAL_FILES,
    ProposalAuditError,
    audit_dr_artifacts,
    generate_proposals,
    sha256_file,
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
from utils.dr_mask_repair import (
    Line,
    apply_subtractive_top_repair,
    detect_repair_top_line,
    reference_top_line,
)
from utils.dr_mask_full_review_v2 import generate_full_review_v2_proposal


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
    with pytest.raises(FileNotFoundError, match="formal specular mask manifest"):
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


def test_local_top_repair_uses_visible_edge_and_is_strictly_subtractive():
    height, width = 40, 64
    rgb = np.full((height, width, 3), 24, dtype=np.uint8)
    cv2.line(rgb, (10, 8), (54, 10), (20, 235, 40), 3, cv2.LINE_AA)
    original = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(
        original,
        np.asarray([[10, 0], [54, 0], [52, 35], [12, 35]], dtype=np.int32),
        255,
    )
    normal = np.full((20, 32, 3), (127, 127, 255), dtype=np.uint8)
    depth = np.full((20, 32, 3), 150, dtype=np.uint8)
    references = (Line(10, 7, 54, 9), Line(10, 9, 54, 11))
    top_line, evidence = detect_repair_top_line(
        rgb, normal, depth, original, references
    )
    repaired = apply_subtractive_top_repair(original, top_line, feather_pixels=2)
    assert evidence["rgb_green_excess"] > 0
    assert 5 <= top_line.y_at(32) <= 13
    assert np.all(repaired <= original)
    assert np.count_nonzero(repaired > original) == 0
    assert np.count_nonzero(repaired < original) > 0
    assert repaired[0].max() == 0
    assert repaired[20].max() == 255


def test_reference_top_line_selects_long_upper_polygon_edge():
    mask = np.zeros((50, 80), dtype=np.uint8)
    polygon = np.asarray([[12, 9], [67, 12], [63, 44], [16, 46]], dtype=np.int32)
    cv2.fillConvexPoly(mask, polygon, 255)
    line = reference_top_line(mask)
    assert line.length > 50
    assert abs(line.slope) < 0.1
    assert line.y_at(40) < 15


def _write_reviewed_v1_fixture(
    scene: Path,
    proposal_root: Path,
    repair_root: Path,
    count: int,
    repair_stems=("000001",),
):
    reviewed = scene / "specular_masks_reviewed_v1"
    reviewed.mkdir()
    repair_candidate_root = repair_root / "repair_candidates"
    repair_candidate_root.mkdir(parents=True)
    entries = []
    for index in range(count):
        stem = f"{index:06d}"
        source = proposal_root / "proposal_soft" / stem / "proposal_soft.png"
        if stem == "000000":
            mask = np.zeros((40, 64), dtype=np.uint8)
            mask[8:32, 11:53] = 255
            mask[15:21, 26:33] = 0
            mask[2:5, 2:5] = 255
            Image.fromarray(mask).save(source)
        source_kind = "proposal_v1"
        if stem in repair_stems:
            repair_source = repair_candidate_root / f"{stem}.png"
            repair_source.write_bytes(source.read_bytes())
            source = repair_source
            source_kind = "repair_candidate_v1"
        target = reviewed / f"{stem}.png"
        target.write_bytes(source.read_bytes())
        with Image.open(target) as image:
            values = np.asarray(image, dtype=np.uint8)
        hard = values >= 128
        ys, xs = np.nonzero(hard)
        entries.append(
            {
                "stem": stem,
                "rgb_path": f"images/{stem}.jpg",
                "rgb_sha256": sha256_file(scene / "images" / f"{stem}.jpg"),
                "mask_path": f"{stem}.png",
                "mask_sha256": sha256_file(target),
                "source": source_kind,
                "source_reference": str(source),
                "source_sha256": sha256_file(source),
                "size": [64, 40],
                "mode": "L",
                "dtype": "uint8",
                "area_ratio": float(hard.mean()),
                "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                "human_status": "accepted",
            }
        )
    manifest = {
        "schema_version": 1,
        "role": "stage_b_formal_reviewed_specular_soft_masks",
        "version": "reviewed_v1",
        "human_status": "accepted",
        "count": count,
        "ordered_stems": [f"{index:06d}" for index in range(count)],
        "mask_interpolation": "opencv.INTER_LINEAR",
        "aggregate_mask_sha256": "mini-fixture",
        "padding_exclusion_proof": {"excluded_stems": [], "mixed_count": 0},
        "entries": entries,
    }
    (reviewed / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    repair_report = {
        "schema_version": 1,
        "artifact": "three localized automatic glass-mask repair candidates",
        "status": "PASS",
        "repair_stems": list(repair_stems),
        "repairs": {
            stem: {"repair_operation": "subtractive top-boundary clipping only"}
            for stem in repair_stems
        },
    }
    (repair_root / "repair_report.json").write_text(json.dumps(repair_report), encoding="utf-8")
    return reviewed / "manifest.json"


def test_full_v2_proposal_audits_v1_scope_and_allows_add_remove(tmp_path):
    scene, raw_root, _ = _fixture(tmp_path, count=4)
    proposal_root = tmp_path / "proposal_v1"
    audit = audit_dr_artifacts(scene, raw_root, expected_count=4)
    generate_all_real_proposals(audit, proposal_root, expected_count=4)
    build_review_package(audit, proposal_root, expected_count=4)
    high_risk_root = tmp_path / "high_risk_v1"
    high_risk_root.mkdir()
    (high_risk_root / "review_pack_manifest.json").write_text(
        json.dumps({"status": "PASS", "view_count": 1}), encoding="utf-8"
    )
    repair_root = tmp_path / "repair_v1"
    reviewed_manifest = _write_reviewed_v1_fixture(
        scene, proposal_root, repair_root, count=4, repair_stems=("000001",)
    )

    output = tmp_path / "full_v2"
    manifest = generate_full_review_v2_proposal(
        scene=scene,
        raw_root=raw_root,
        reviewed_v1_manifest=reviewed_manifest,
        proposal_v1_root=proposal_root,
        high_risk_root=high_risk_root,
        repair_v1_root=repair_root,
        output_root=output,
        expected_count=4,
        source_size=(64, 40),
        require_formal_loader=False,
        repair_stems=("000001",),
    )

    assert manifest["status"] == "PASS"
    assert manifest["source_audit"]["source_counts"] == {
        "proposal_v1": 3,
        "repair_candidate_v1": 1,
    }
    assert manifest["source_audit"]["old_repair_scope"][
        "old_repair_was_subtractive_only"
    ]
    assert len(list((output / "candidate_masks").glob("*.png"))) == 4
    assert len(list((output / "review_pages").glob("*_review.png"))) == 4
    assert (output / "review_queue.csv").is_file()
    assert (output / "review_template.json").is_file()
    assert (output / "proposal_summary.json").is_file()
    assert (output / "per_frame_metrics.csv").is_file()
    assert (output / "source_tree_hashes.json").is_file()
    assert not (scene / "specular_masks_reviewed_v2").exists()

    rows = {row["stem"]: row for row in manifest["frames"]}
    assert rows["000000"]["added_hard_pixel_count"] > 0
    assert rows["000000"]["removed_hard_pixel_count"] > 0
    with Image.open(output / "candidate_masks" / "000000.png") as candidate:
        assert candidate.mode == "L"
        assert candidate.size == (64, 40)
        values = np.asarray(candidate, dtype=np.uint8)
    assert values.min() == 0
    assert values.max() == 255
    assert manifest["outputs"]["source_tree_hashes_unchanged"] is True
