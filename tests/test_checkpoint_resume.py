from io import BytesIO
from types import SimpleNamespace

import pytest
import torch

from scene.diffuse_surfel_model import DiffuseSurfelModel


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="surfel checkpoints are CUDA training checkpoints")


def _training_args():
    return SimpleNamespace(
        percent_dense=0.01,
        position_lr_init=1e-4,
        position_lr_final=1e-6,
        position_lr_delay_mult=0.01,
        position_lr_max_steps=100,
        feature_lr=1e-3,
        material_lr=2e-3,
        opacity_lr=1e-2,
        scaling_lr=1e-3,
        rotation_lr=1e-3,
        exposure_lr_init=1e-2,
        exposure_lr_final=1e-3,
        exposure_lr_delay_steps=0,
        exposure_lr_delay_mult=0.0,
        iterations=100,
    )


def _initialized_model():
    model = DiffuseSurfelModel()
    model.spatial_lr_scale = 1.0
    for name, shape in (
        ("_xyz", (2, 3)),
        ("_rotation", (2, 4)),
        ("_scaling", (2, 2)),
        ("_opacity", (2, 1)),
        ("_base_color", (2, 3)),
        ("_roughness", (2, 1)),
        ("_f0", (2, 3)),
        ("_ks", (2, 1)),
    ):
        value = torch.randn(shape, device="cuda")
        if name == "_rotation":
            value[:, 0] += 1.0
        setattr(model, name, torch.nn.Parameter(value))
    model._exposure = torch.nn.Parameter(torch.eye(3, 4, device="cuda")[None])
    model.exposure_mapping = {"toy": 0}
    model.max_radii2D = torch.zeros(2, device="cuda")
    model.training_setup(_training_args())
    return model


def test_checkpoint_restores_materials_and_optimizer_state():
    source = _initialized_model()
    sum(parameter.sum() for group in source.optimizer.param_groups for parameter in group["params"]).backward()
    source.optimizer.step()
    source.optimizer.zero_grad(set_to_none=True)

    buffer = BytesIO()
    torch.save(source.capture(), buffer)
    buffer.seek(0)
    state = torch.load(buffer)

    restored = DiffuseSurfelModel()
    restored.restore(state, _training_args())

    for name in ("_xyz", "_rotation", "_scaling", "_opacity", "_base_color", "_roughness", "_f0", "_ks"):
        assert torch.equal(getattr(source, name), getattr(restored, name))
    assert restored.optimizer.state_dict()["state"].keys() == source.optimizer.state_dict()["state"].keys()
