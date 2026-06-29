import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from tools.adapt_diffusion_renderer_normals import (
    adapt_raw_rgb,
    adapt_scene,
    decode_raw_normal,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, count: int = 2):
    scene = tmp_path / "TiHuBird"
    raw_root = tmp_path / "raw"
    (scene / "images").mkdir(parents=True)
    (raw_root / "raw" / "TiHuBird").mkdir(parents=True)
    records = []
    for index in range(count):
        source = scene / "images" / f"{index:06d}.jpg"
        Image.fromarray(np.full((5, 7, 3), 40 + index, np.uint8)).save(source)
        raw_relative = f"raw/TiHuBird/0000.{index:04d}.normal.png"
        raw = raw_root / raw_relative
        values = np.empty((3, 4, 3), np.uint8)
        values[..., 0] = 80 + index
        values[..., 1] = np.arange(4, dtype=np.uint8)[None] * 20 + 100
        values[..., 2] = 220 - index
        Image.fromarray(values).save(raw)
        records.append({
            "is_padding": False,
            "rtgs_real_frame_index": index,
            "rtgs_input_file": f"data/TiHuBird/images/{index:06d}.jpg",
            "input_sha256": _sha(source),
            "raw_prior_files": {"normal": raw_relative},
            "raw_slot": index,
            "chunk_index": 0,
            "frame_index": index,
            "source_to_prior_mapping": {"operation": "resize", "source_wh": [7, 5], "prior_wh": [4, 3]},
        })
    records.append({"is_padding": True, "rtgs_real_frame_index": None, "rtgs_input_file": None})
    (raw_root / "manifest.json").write_text(json.dumps({
        "real_frame_count": count,
        "padding_frame_count": 1,
        "frames_and_padding": records,
    }))
    return scene, raw_root


def test_decode_formula_and_black_has_no_invalid_special_case():
    raw = np.array([[[0, 127, 255]]], dtype=np.uint8)
    decoded = decode_raw_normal(raw)
    expected = raw.astype(np.float32) / np.float32(127.5) - np.float32(1.0)
    np.testing.assert_allclose(decoded, expected)
    assert decoded.dtype == np.float32
    assert np.isfinite(decode_raw_normal(np.zeros((1, 1, 3), np.uint8))).all()


def test_resize_is_float_bilinear_then_unit_normalized():
    raw = np.array([[[255, 127, 127], [127, 255, 127]], [[127, 127, 255], [255, 255, 127]]], dtype=np.uint8)
    adapted = adapt_raw_rgb(raw, (7, 9))
    assert adapted.shape == (7, 9, 3)
    assert adapted.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(adapted, axis=-1), 1.0, atol=2e-6)


def test_scene_adapter_excludes_padding_and_keeps_axis_unselected(tmp_path):
    scene, raw_root = _fixture(tmp_path)
    output = tmp_path / "candidate"
    manifest = adapt_scene(scene, raw_root, output, expected_count=2)
    assert len(manifest["files"]) == 2
    assert manifest["axis_mapping"] is None
    assert manifest["axis_selection_status"] == "awaiting_human_audit"
    assert sorted(path.name for path in (output / "raw_identity" / "normal").glob("*.npy")) == ["000000.npy", "000001.npy"]
