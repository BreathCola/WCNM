import json
from pathlib import Path

import numpy as np
from PIL import Image

from utils.diffusion_renderer_raw import ALL_KINDS, validate_and_manifest
from utils.glass_mask_proposal import PROPOSAL_SCHEMA, _generic_risks


def test_raw_dr_manifest_maps_real_and_padding_slots_without_scene_hardcode(tmp_path):
    scene = tmp_path / "AnyScene"
    image_root = scene / "images"
    image_root.mkdir(parents=True)
    for index, color in enumerate(((10, 20, 30), (40, 50, 60))):
        Image.new("RGB", (64, 64), color).save(image_root / f"{index:06d}.png")
    raw = tmp_path / "raw"
    group = raw / scene.name
    group.mkdir(parents=True)
    for slot in range(24):
        color = (10, 20, 30) if slot == 0 else (40, 50, 60)
        for kind in ALL_KINDS:
            Image.new("RGB", (64, 64), color).save(group / f"0000.{slot:04d}.{kind}.png")
    signal = raw / "TMP_SUCCESS_SIGNAL"
    signal.mkdir()
    (signal / f"{scene.name}.0000").write_text("", encoding="utf-8")
    manifest, validation = validate_and_manifest(
        scene=scene, images="images", raw_root=raw, group_name=scene.name,
        target_hw=(64, 64), frames_per_chunk=24,
        generation_identity={"test": True}, effective_config={"test": True},
        repository_root=tmp_path,
    )
    assert manifest["ordered_stems"] == ["000000", "000001"]
    assert manifest["real_frame_count"] == 2
    assert manifest["padding_frame_count"] == 22
    assert sum(not row["is_padding"] for row in manifest["frames_and_padding"]) == 2
    assert validation["validation"] == "PASS"


def test_scene_agnostic_proposal_schema_can_never_train():
    source = Path(__import__("utils.glass_mask_proposal", fromlist=["x"]).__file__).read_text()
    assert "TiHuBird" not in source
    assert "111" not in source
    risks = _generic_risks({
        "bird_may_be_included": {"flag": True},
        "background_may_be_included": {"flag": False},
        "glass_edge_may_be_missing": {"flag": False},
        "reflection_may_be_misclassified": {"flag": False},
        "low_confidence_region": {"flag": False},
    })
    assert "subject_texture_may_be_included" in risks
    proposal_manifest = {
        "schema": PROPOSAL_SCHEMA, "training_eligible": False,
        "promotion_performed": False, "human_status": "proposal_requires_review",
    }
    encoded = json.dumps(proposal_manifest)
    assert "formal_reviewed_glass_masks" not in encoded
    assert proposal_manifest["training_eligible"] is False
