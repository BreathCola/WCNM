import json

import numpy as np
import pytest

from tests.test_diffusion_renderer_normal_adapter import _fixture
from tools.adapt_diffusion_renderer_normals import adapt_scene
from tools.validate_diffusion_renderer_normals import CandidateValidationError, validate_candidates


def test_strict_validation_accepts_traceable_hwc_float32_candidates(tmp_path):
    scene, raw_root = _fixture(tmp_path)
    output = tmp_path / "candidate"
    adapt_scene(scene, raw_root, output, expected_count=2)
    report = validate_candidates(scene, raw_root, output, expected_count=2)
    assert report["validation"] == "PASS"
    assert report["priors"] == 2
    assert report["raw_padding_records_excluded"] == 1
    assert report["axis_selection_status"] == "awaiting_human_audit"


def test_validation_rejects_selected_axis_or_missing_frame(tmp_path):
    scene, raw_root = _fixture(tmp_path)
    output = tmp_path / "candidate"
    adapt_scene(scene, raw_root, output, expected_count=2)
    manifest_path = output / "manifests" / "adapter_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["axis_mapping"] = "identity"
    manifest_path.write_text(json.dumps(manifest))
    (output / "raw_identity" / "normal" / "000001.npy").unlink()
    with pytest.raises(CandidateValidationError) as raised:
        validate_candidates(scene, raw_root, output, expected_count=2)
    errors = "\n".join(raised.value.report["errors"])
    assert "must not select an axis mapping" in errors
    assert "candidate prior set" in errors


def test_validation_rejects_non_float32(tmp_path):
    scene, raw_root = _fixture(tmp_path)
    output = tmp_path / "candidate"
    manifest = adapt_scene(scene, raw_root, output, expected_count=2)
    path = output / "raw_identity" / "normal" / "000000.npy"
    np.save(path, np.load(path).astype(np.float64))
    # Update only the hash so dtype validation is what remains authoritative.
    import hashlib
    manifest["files"][0]["output_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "manifests" / "adapter_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(CandidateValidationError) as raised:
        validate_candidates(scene, raw_root, output, expected_count=2)
    assert any("dtype=float64" in error for error in raised.value.report["errors"])
