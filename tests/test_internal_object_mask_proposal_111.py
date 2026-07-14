import json
from pathlib import Path

import numpy as np
from PIL import Image

from tools.generate_internal_object_mask_proposal_111 import (
    FIXED_NINE,
    OBJECTS,
    choose_bidirectional_candidate,
    ensure_new_output_dir,
    make_fixed_nine_human_review,
    mask_iou,
    significant_component_stats,
)
from utils.internal_object_mask import (
    INTERNAL_OBJECT_SEMANTICS_VERSION,
    PROPOSAL_ROLE,
    SCHEMA_VERSION,
    canonical_payload_sha256,
    sha256_file,
)


def _write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)


def _fixed_root(tmp_path: Path) -> Path:
    root = tmp_path / "fixed_nine"
    for directory in (
        "processed/bird", "processed/internal_base", "processed/internal_object_union",
        "raw/yellow_base_board", "raw/white_platform",
    ):
        (root / directory).mkdir(parents=True)
    selection = {}
    for stem in FIXED_NINE:
        bird = np.zeros((8, 8), dtype=bool)
        bird[1:3, 2:4] = True
        yellow = np.zeros((8, 8), dtype=bool)
        yellow[5:7, 1:7] = True
        white = np.zeros((8, 8), dtype=bool)
        white[4:5, 3:5] = True
        base = yellow | white
        union = bird | base
        _write_mask(root / "processed/bird" / f"{stem}.png", bird)
        _write_mask(root / "raw/yellow_base_board" / f"{stem}.png", yellow)
        _write_mask(root / "raw/white_platform" / f"{stem}.png", white)
        _write_mask(root / "processed/internal_base" / f"{stem}.png", base)
        _write_mask(root / "processed/internal_object_union" / f"{stem}.png", union)
        selection[stem] = {
            "auto_suggestion": {
                "bird_candidate_id": f"{stem}_bird_000",
                "yellow_base_board_candidate_id": f"{stem}_yellow_base_board_000",
                "white_platform_candidate_id": f"{stem}_white_platform_000",
                "connected_fixture_candidate_id": None,
            },
            "quality_guards": ["old_component_warning"],
        }
    (root / "candidate_selection.json").write_text(json.dumps(selection), encoding="utf-8")
    (root / "proposal_summary.json").write_text(json.dumps({"rows": []}), encoding="utf-8")
    (root / "review_selection_template.json").write_text(json.dumps({stem: {} for stem in FIXED_NINE}), encoding="utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "role": PROPOSAL_ROLE,
        "internal_object_semantics_version": INTERNAL_OBJECT_SEMANTICS_VERSION,
        "count": len(FIXED_NINE),
        "ordered_stems": list(FIXED_NINE),
        "human_status": "proposal_requires_review",
    }
    manifest["manifest_payload_sha256"] = canonical_payload_sha256(manifest)
    (root / "proposal_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_fixed_nine_human_review_records_exact_anchor_hashes_without_reviewer_identity(tmp_path):
    fixed = _fixed_root(tmp_path)
    output = tmp_path / "out"
    record = make_fixed_nine_human_review(fixed, output, "abc123")

    assert record["accepted_stems"] == list(FIXED_NINE)
    assert record["review_authorization"]["type"] == "explicit_user_approval"
    assert "reviewer" not in json.dumps(record).lower()
    clean = dict(record)
    clean.pop("payload_sha256")
    assert record["payload_sha256"] == canonical_payload_sha256(clean)
    stem = FIXED_NINE[0]
    entry = record["entries"][stem]
    assert entry["status"] == "accepted"
    assert entry["connected_fixture_candidate_id"] is None
    assert entry["isolated_pole_included"] is False
    assert entry["bird_candidate_id"] == f"{stem}_bird_000"
    assert entry["mask_sha256"]["bird"] == sha256_file(fixed / "processed/bird" / f"{stem}.png")
    assert (output / "fixed_nine_human_review_v3.json").is_file()


def test_bidirectional_selection_records_disagreement_without_unioning():
    forward = np.zeros((8, 8), dtype=bool)
    forward[1:3, 1:3] = True
    backward = np.zeros((8, 8), dtype=bool)
    backward[5:7, 5:7] = True

    selected, meta = choose_bidirectional_candidate(
        forward, backward, anchor_distance_left=1, anchor_distance_right=4,
        guards={"forward_backward_iou_review": 0.75},
    )

    assert np.array_equal(selected, forward)
    assert not np.array_equal(selected, forward | backward)
    assert meta["forward_backward_iou"] == mask_iou(forward, backward)
    assert "low_forward_backward_iou" in meta["review_flags"]


def test_bidirectional_selection_uses_nearer_anchor_without_unioning():
    forward = np.zeros((8, 8), dtype=bool)
    forward[1:5, 1:5] = True
    forward[0, 0] = True
    backward = np.zeros((8, 8), dtype=bool)
    backward[1:5, 1:5] = True
    backward[6, 6] = True

    selected, meta = choose_bidirectional_candidate(
        forward,
        backward,
        anchor_distance_left=5,
        anchor_distance_right=2,
        guards={"forward_backward_iou_review": 0.10},
    )

    assert np.array_equal(selected, backward)
    assert not np.array_equal(selected, forward | backward)
    assert meta["selection_reason"] == "backward_nearer"
    assert meta["review_flags"] == []


def test_object_ids_and_existing_output_refusal_are_stable(tmp_path):
    assert OBJECTS == {"bird": 1, "yellow_base_board": 2, "white_platform": 3}
    existing = tmp_path / "existing"
    existing.mkdir()

    try:
        ensure_new_output_dir(existing)
    except FileExistsError as exc:
        assert "BLOCKED_BY_EXISTING_OUTPUT" in str(exc)
    else:
        raise AssertionError("existing output directory was not refused")


def test_significant_component_stats_do_not_block_tiny_nearby_fragments():
    mask = np.zeros((64, 64), dtype=bool)
    mask[20:44, 20:44] = True
    mask[18, 20] = True
    mask[45, 43] = True
    ref = np.zeros_like(mask)
    ref[20:44, 20:44] = True

    stats = significant_component_stats(mask, ref)

    assert stats["raw_component_count"] == 3
    assert stats["significant_component_count"] == 1
    assert stats["distant_significant_component_count"] == 0


def test_significant_component_stats_flags_large_distant_component():
    mask = np.zeros((128, 128), dtype=bool)
    mask[10:40, 10:40] = True
    mask[96:120, 96:120] = True
    ref = np.zeros_like(mask)
    ref[10:40, 10:40] = True

    stats = significant_component_stats(mask, ref, {"distant_component_min_distance_px": 32})

    assert stats["significant_component_count"] == 2
    assert stats["distant_significant_component_count"] == 1
    assert stats["distant_significant_component_area_ratio"] > 0.0
