import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from tools.promote_internal_object_masks import (
    ACCEPTED_WITH_WARNING,
    DEFAULT_PROPOSAL,
    FIXED_NINE,
    audit_proposal,
    promote_reviewed_release,
)
from utils.internal_object_mask import (
    EXPECTED_STEMS,
    INTERNAL_OBJECT_SEMANTICS_VERSION,
    MASK_ROLES,
    PROPOSAL_ROLE,
    REVIEWED_ROLE,
    SCHEMA_VERSION,
    canonical_payload_sha256,
    sha256_file,
    validate_internal_object_mask_set,
)
from utils.specular_mask import (
    canonical_payload_sha256 as canonical_specular_payload_sha256,
    inspect_mask as inspect_specular_mask,
)


def _write_specular_manifest(scene: Path) -> None:
    import hashlib

    aggregate = hashlib.sha256()
    entries = []
    for stem in EXPECTED_STEMS:
        rgb = scene / "images" / f"{stem}.jpg"
        mask = scene / "specular_masks_reviewed_v1" / f"{stem}.png"
        facts = inspect_specular_mask(mask, (8, 6))
        mask_sha = sha256_file(mask)
        aggregate.update(f"{stem} {mask_sha}\n".encode("utf-8"))
        entries.append({
            "stem": stem,
            "human_status": "accepted",
            "source": "proposal_v1",
            "source_reference": "unit-test",
            "source_sha256": mask_sha,
            "rgb_path": f"images/{stem}.jpg",
            "rgb_sha256": sha256_file(rgb),
            "mask_path": f"{stem}.png",
            "mask_sha256": mask_sha,
            **facts,
        })
    payload = {
        "schema_version": 1,
        "role": "stage_b_formal_reviewed_specular_soft_masks",
        "human_status": "accepted",
        "count": 111,
        "ordered_stems": EXPECTED_STEMS,
        "mask_interpolation": "opencv.INTER_LINEAR",
        "padding_exclusion_proof": {
            "excluded_stems": [f"{index:06d}" for index in range(111, 120)],
            "mixed_count": 0,
        },
        "aggregate_mask_sha256": aggregate.hexdigest(),
        "entries": entries,
    }
    payload["manifest_payload_sha256"] = canonical_specular_payload_sha256(payload)
    (scene / "specular_masks_reviewed_v1" / "manifest.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _write_rgb(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((6, 8, 3), 128, dtype=np.uint8), mode="RGB").save(path)


def _write_mask(path: Path, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.asarray(values, dtype=np.uint8) * 255), mode="L").save(path)


def _make_scene_and_proposal(tmp_path: Path) -> tuple[Path, Path]:
    scene = tmp_path / "scene"
    proposal = tmp_path / "proposal"
    entries = []
    rows = []
    for stem in EXPECTED_STEMS:
        rgb = scene / "images" / f"{stem}.jpg"
        _write_rgb(rgb)
        glass = np.zeros((6, 8), dtype=np.uint8)
        glass[1:5, 1:7] = 1
        glass_path = scene / "specular_masks_reviewed_v1" / f"{stem}.png"
        _write_mask(glass_path, glass)
        _write_mask(proposal / "processed" / "glass_hard" / f"{stem}.png", glass)

        bird = np.zeros((6, 8), dtype=np.uint8)
        base = np.zeros((6, 8), dtype=np.uint8)
        bird[1:3, 2:4] = 1
        base[3:5, 3:6] = 1
        union = np.maximum(bird, base)
        masks = {"bird": bird, "internal_base": base, "internal_object_union": union}
        entry = {
            "stem": stem,
            "frame_status": "anchor_accepted" if stem in FIXED_NINE else "auto_candidate_ready",
            "rgb_sha256": sha256_file(rgb),
            "glass_sha256": sha256_file(glass_path),
            "review_flags": "",
        }
        for role, mask in masks.items():
            path = proposal / "processed" / role / f"{stem}.png"
            _write_mask(path, mask)
            entry[f"{role}_sha256"] = sha256_file(path)
            entry[f"{role}_area"] = str(int(mask.sum()))
        rows.append({
            **entry,
            "union_glass_ratio": "0.1",
            "raw_outside_glass_ratio": "0.0",
            "neighbor_iou_previous": "1.0",
            "area_ratio_previous": "1.0",
            "bbox_displacement_fraction": "0.0",
            "centroid_jump_fraction": "0.0",
            "raw_component_count": "1",
            "significant_component_count": "1",
            "largest_component_ratio": "1.0",
            "secondary_component_total_ratio": "0.0",
            "distant_significant_component_count": "0",
            "distant_significant_component_area_ratio": "0.0",
            "bird_forward_backward_iou": "1.0",
            "yellow_forward_backward_iou": "1.0",
            "white_forward_backward_iou": "1.0",
        })
        entries.append(entry)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "role": PROPOSAL_ROLE,
        "proposal_kind": "stage_d_internal_object_mask_proposal_111_v3",
        "human_status": "proposal_requires_review",
        "internal_object_semantics_version": INTERNAL_OBJECT_SEMANTICS_VERSION,
        "count": 111,
        "ordered_stems": EXPECTED_STEMS,
        "entries": entries,
    }
    manifest["manifest_payload_sha256"] = canonical_payload_sha256(manifest)
    (proposal / "proposal_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (proposal / "proposal_summary.json").write_text(
        json.dumps({
            "artifact_role": "stage_d_internal_object_mask_proposal_111_v3",
            "human_status": "proposal_requires_review",
            "internal_object_semantics_version": INTERNAL_OBJECT_SEMANTICS_VERSION,
            "count": 111,
            "review_queue_count": 0,
            "status_counts": {
                "anchor_accepted": 9,
                "auto_candidate_ready": 102,
                "manual_edit_required": 0,
                "review_required": 0,
            },
        }),
        encoding="utf-8",
    )
    (proposal / "generation_code_identity.json").write_text(
        json.dumps({
            "proposal_manifest_sha256": sha256_file(proposal / "proposal_manifest.json"),
            "final_commit_sha": "unit-test",
        }),
        encoding="utf-8",
    )
    fixed_entries = {}
    for stem in FIXED_NINE:
        fixed_entries[stem] = {
            "status": "accepted",
            "mask_sha256": {
                role: sha256_file(proposal / "processed" / role / f"{stem}.png")
                for role in MASK_ROLES
            },
        }
    (proposal / "fixed_nine_human_review_v3.json").write_text(
        json.dumps({
            "accepted_stems": list(FIXED_NINE),
            "review_authorization": {"type": "explicit_user_approval", "source": "current_codex_task_prompt"},
            "entries": fixed_entries,
        }),
        encoding="utf-8",
    )
    with (proposal / "per_frame_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (proposal / "review_queue.csv").open("w", newline="", encoding="utf-8") as handle:
        handle.write("stem,status,reasons\n")
    _write_specular_manifest(scene)
    return scene, proposal


def test_promote_reviewed_release_byte_copies_and_readonly_loads(tmp_path):
    scene, proposal = _make_scene_and_proposal(tmp_path)
    output = tmp_path / "reviewed"
    source_before = audit_proposal(scene=scene, proposal=proposal)["proposal_processed_hash"]

    result = promote_reviewed_release(scene=scene, proposal=proposal, output=output)

    assert result["aggregate_mask_sha256"] == result["validated"]["aggregate_sha256"]
    assert result["source_proposal_processed_hash_before"] == source_before
    assert result["source_proposal_processed_hash_after"] == source_before
    assert result["validated"]["role"] == REVIEWED_ROLE
    assert result["validated"]["accepted_with_warning"] == list(ACCEPTED_WITH_WARNING)
    for stem in EXPECTED_STEMS:
        for role in MASK_ROLES:
            assert (output / role / f"{stem}.png").read_bytes() == (
                proposal / "processed" / role / f"{stem}.png"
            ).read_bytes()
    assert not (output.stat().st_mode & 0o222)
    validate_internal_object_mask_set(scene, "images", output / "manifest.json")


def test_promotion_refuses_existing_output(tmp_path):
    scene, proposal = _make_scene_and_proposal(tmp_path)
    output = tmp_path / "reviewed"
    output.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        promote_reviewed_release(scene=scene, proposal=proposal, output=output)


def test_audit_rejects_union_equation_and_glass_escape(tmp_path):
    scene, proposal = _make_scene_and_proposal(tmp_path)
    bad_union = np.zeros((6, 8), dtype=np.uint8)
    _write_mask(proposal / "processed" / "internal_object_union" / "000000.png", bad_union)
    with pytest.raises(Exception, match="hash mismatch|union equation"):
        audit_proposal(scene=scene, proposal=proposal)

    scene, proposal = _make_scene_and_proposal(tmp_path / "glass")
    glass = np.zeros((6, 8), dtype=np.uint8)
    glass[1:5, 1:7] = 1
    glass[1, 2] = 0
    _write_mask(scene / "specular_masks_reviewed_v1" / "000000.png", glass)
    _write_mask(proposal / "processed" / "glass_hard" / "000000.png", glass)
    _write_specular_manifest(scene)
    manifest = json.loads((proposal / "proposal_manifest.json").read_text(encoding="utf-8"))
    row_path = proposal / "per_frame_metrics.csv"
    rows = list(csv.DictReader(row_path.open("r", newline="", encoding="utf-8")))
    rows[0]["glass_sha256"] = sha256_file(scene / "specular_masks_reviewed_v1" / "000000.png")
    with row_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest["entries"][0]["glass_sha256"] = rows[0]["glass_sha256"]
    manifest["manifest_payload_sha256"] = canonical_payload_sha256(manifest)
    (proposal / "proposal_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    gen = json.loads((proposal / "generation_code_identity.json").read_text(encoding="utf-8"))
    gen["proposal_manifest_sha256"] = sha256_file(proposal / "proposal_manifest.json")
    (proposal / "generation_code_identity.json").write_text(json.dumps(gen), encoding="utf-8")
    with pytest.raises(Exception, match="union subset glass_hard"):
        audit_proposal(scene=scene, proposal=proposal)


def test_loader_rejects_real_proposal_path_by_contract():
    proposal_manifest = DEFAULT_PROPOSAL / "proposal_manifest.json"
    if not proposal_manifest.is_file():
        pytest.skip("real proposal artifact is not present")
    with pytest.raises(ValueError, match="proposal"):
        validate_internal_object_mask_set(Path("data/TiHuBird"), "images", proposal_manifest)
