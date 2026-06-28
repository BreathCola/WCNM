import json

import numpy as np
import pytest
from PIL import Image

from tools.validate_normal_priors import PriorValidationError, validate_scene


def _make_scene(tmp_path):
    image_root = tmp_path / "images"
    prior_root = tmp_path / "normal_priors"
    image_root.mkdir()
    prior_root.mkdir()
    entries = []
    for index in range(2):
        stem = f"{index:06d}"
        Image.new("RGB", (8, 6), color=(index, 64, 128)).save(image_root / f"{stem}.jpg")
        prior = np.zeros((6, 8, 3), dtype=np.float32)
        prior[..., 2] = -1.0
        np.save(prior_root / f"{stem}.npy", prior)
        entries.append(
            {
                "source_image": f"images/{stem}.jpg",
                "output_file": f"{stem}.npy",
                "shape": [6, 8, 3],
                "dtype": "float32",
                "layout": "HWC",
                "space": "camera",
                "generation_parameters": {
                    "model": "StableNormal",
                    "data_type": "indoor",
                    "resolution": 768,
                    "steps": 10,
                    "yoso_version": "yoso-normal-v0-3",
                    "diffusion_version": "stable-normal-v0-1",
                },
            }
        )
    (prior_root / "manifest.json").write_text(json.dumps({"files": entries}))
    return prior_root


def test_strict_validator_accepts_complete_valid_set(tmp_path):
    _make_scene(tmp_path)
    report = validate_scene(tmp_path)
    assert report["images"] == report["priors"] == report["manifest_entries"] == 2
    assert report["errors"] == []


def test_strict_validator_reports_missing_prior(tmp_path):
    prior_root = _make_scene(tmp_path)
    (prior_root / "000001.npy").unlink()
    with pytest.raises(PriorValidationError, match="missing prior: 000001.npy"):
        validate_scene(tmp_path)
