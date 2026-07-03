from types import SimpleNamespace

import pytest
import torch

from gaussian_renderer.transmittance_renderer import (
    DiffuseRaytraceAdapter, alpha_over_transmittance,
)
from scene.diffuse_surfel_model import DiffuseSurfelModel
from scene.reflection_surfel_model import ReflectionSurfelModel
from scene.stage_d_scene import StageDScene
from scene.stage_d_state import make_stage_d_checkpoint, restore_stage_d_checkpoint
from scene.transmittance_surfel_model import TransmittanceSurfelModel
from tests.test_stage_b_checkpoint_resume import diffuse_args, initialized_diffuse
from tests.test_reflection_surfel_model import reflection_args


def transmittance_args():
    source = reflection_args()
    values = {
        key.replace("reflection_", "transmittance_"): value
        for key, value in vars(source).items()
    }
    return SimpleNamespace(**values)


def initialized_fields():
    diffuse = initialized_diffuse()
    reflection = ReflectionSurfelModel()
    reflection.create_random_bbox(torch.zeros(3), torch.ones(3), 4, 3)
    reflection.training_setup(reflection_args())
    transmittance = TransmittanceSurfelModel()
    transmittance.create_random_bbox(torch.zeros(3), torch.ones(3), 5, 7)
    transmittance.training_setup(transmittance_args())
    return diffuse, reflection, transmittance


def test_transmittance_model_checkpoint_namespaces_are_independent():
    diffuse, reflection, transmittance = initialized_fields()
    checkpoint = make_stage_d_checkpoint(
        diffuse, reflection, transmittance, 15001, 12001, 1,
        {"sha256": "source"}, {"stage": "stage_d"},
        "stage_c_geometry_release_v1", "a" * 64,
        {"remaining_camera_indices": [0], "camera_count": 1},
    )
    assert checkpoint["format"] == "rtgs_stage_d"
    assert checkpoint["transmittance"]["model_type"] == "transmittance_surfel"
    restored = (DiffuseSurfelModel(), ReflectionSurfelModel(), TransmittanceSurfelModel())
    values = restore_stage_d_checkpoint(
        checkpoint, *restored, diffuse_args(), reflection_args(), transmittance_args(),
        "stage_c_geometry_release_v1", "a" * 64, restore_rng=False,
    )
    assert values[:3] == (15001, 12001, 1)
    assert len({model.get_xyz.data_ptr() for model in restored}) == 3
    assert len({id(model.optimizer) for model in restored}) == 3
    with pytest.raises(ValueError, match="aggregate mismatch"):
        restore_stage_d_checkpoint(
            checkpoint, *restored, diffuse_args(), reflection_args(),
            transmittance_args(), "stage_c_geometry_release_v1", "b" * 64,
            restore_rng=False,
        )


def test_alpha_over_and_diffuse_raytrace_adapter():
    cin = torch.tensor([[0.2, 0.1, 0.0]])
    ain = torch.tensor([[0.25]])
    cout = torch.tensor([[0.4, 0.8, 0.2]])
    aout = torch.tensor([[0.5]])
    color, alpha = alpha_over_transmittance(cin, ain, cout, aout)
    assert torch.allclose(color, cin + 0.75 * cout)
    assert torch.allclose(alpha, torch.tensor([[0.625]]))
    diffuse = initialized_diffuse()
    adapter = DiffuseRaytraceAdapter(diffuse)
    assert adapter._color is diffuse._base_color
    assert adapter._xyz is diffuse._xyz


def test_stage_d_scene_exports_three_separate_plys(tmp_path):
    diffuse, reflection, transmittance = initialized_fields()
    scene = StageDScene.__new__(StageDScene)
    scene.model_path = str(tmp_path)
    scene.diffuse, scene.reflection, scene.transmittance = diffuse, reflection, transmittance
    scene.save(9)
    for branch in ("diffuse", "reflection", "transmittance"):
        assert (
            tmp_path / "point_cloud" / branch / "iteration_9" / "point_cloud.ply"
        ).is_file()
