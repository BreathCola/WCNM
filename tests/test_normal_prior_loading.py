from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

import scene  # Initialize the package before camera_utils imports scene.cameras.
from utils.camera_utils import loadCam


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Camera construction uses CUDA")


def test_npy_normal_prior_is_loaded_resized_and_normalized(tmp_path):
    image_path = tmp_path / "images" / "00000.png"
    normal_path = tmp_path / "normal_priors" / "00000.npy"
    image_path.parent.mkdir()
    normal_path.parent.mkdir()
    Image.new("RGB", (8, 6), color=(128, 64, 32)).save(image_path)
    prior = np.zeros((3, 4, 3), dtype=np.float32)
    prior[..., 2] = -2.0
    np.save(normal_path, prior)

    args = SimpleNamespace(
        source_path=str(tmp_path),
        normal_priors="normal_priors",
        normal_prior_space="camera",
        resolution=1,
        data_device="cuda",
        train_test_exp=False,
    )
    camera_info = SimpleNamespace(
        image_path=str(image_path),
        depth_path="",
        uid=0,
        R=np.eye(3, dtype=np.float32),
        T=np.zeros(3, dtype=np.float32),
        FovX=0.8,
        FovY=0.7,
        depth_params=None,
        image_name="00000.png",
        is_test=False,
    )

    camera = loadCam(args, 0, camera_info, 1.0, False, False)

    assert camera.normal_prior.shape == (3, 6, 8)
    assert camera.normal_prior_valid.all()
    assert torch.allclose(camera.normal_prior.norm(dim=0), torch.ones((6, 8), device="cuda"))
    assert torch.allclose(camera.normal_prior[2], -torch.ones((6, 8), device="cuda"))


def test_missing_normal_prior_keeps_view_unsupervised(tmp_path):
    image_path = tmp_path / "images" / "00001.png"
    image_path.parent.mkdir()
    Image.new("RGB", (8, 6), color=(128, 64, 32)).save(image_path)

    args = SimpleNamespace(
        source_path=str(tmp_path),
        normal_priors="normal_priors",
        normal_prior_space="camera",
        resolution=1,
        data_device="cuda",
        train_test_exp=False,
    )
    camera_info = SimpleNamespace(
        image_path=str(image_path),
        depth_path="",
        uid=1,
        R=np.eye(3, dtype=np.float32),
        T=np.zeros(3, dtype=np.float32),
        FovX=0.8,
        FovY=0.7,
        depth_params=None,
        image_name="00001.png",
        is_test=False,
    )

    camera = loadCam(args, 0, camera_info, 1.0, False, False)

    assert camera.normal_prior is None
    assert camera.normal_prior_valid is None
