from pathlib import Path

import numpy as np
import pytest

from tools.generate_normal_priors import (
    atomic_save_npy,
    decode_rgb_normal,
    prior_path_for_image,
    validate_prior_file,
)


def test_prior_filename_matches_loader_stem_rule(tmp_path):
    image_path = Path("nested/frame.001.JPG")
    assert prior_path_for_image(image_path, tmp_path) == tmp_path / "frame.001.npy"


def test_saved_prior_is_finite_float32_hwc_unit_normal(tmp_path):
    rgb = np.array(
        [
            [[255, 128, 128], [128, 255, 128]],
            [[128, 128, 255], [0, 128, 128]],
        ],
        dtype=np.uint8,
    )
    normals = decode_rgb_normal(rgb)
    prior_path = tmp_path / "normal_priors" / "00000.npy"
    atomic_save_npy(prior_path, normals)
    loaded = validate_prior_file(prior_path, expected_hw=(2, 2))

    assert loaded.dtype == np.float32
    assert loaded.shape == (2, 2, 3)
    assert np.isfinite(loaded).all()
    assert np.allclose(np.linalg.norm(loaded, axis=-1), 1.0, atol=5e-4)


def test_rgb_decoder_rejects_non_finite_and_near_zero_vectors():
    with pytest.raises(ValueError, match="NaN or Inf"):
        decode_rgb_normal(np.array([[[np.nan, 0.0, 0.0]]], dtype=np.float32))
    with pytest.raises(ValueError, match="near-zero"):
        decode_rgb_normal(np.array([[[127.5, 127.5, 127.5]]], dtype=np.float32))
