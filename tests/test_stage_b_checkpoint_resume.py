from types import SimpleNamespace

import torch

from scene.diffuse_surfel_model import DiffuseSurfelModel
from scene.reflection_surfel_model import ReflectionSurfelModel
from scene.stage_b_state import (
    initialize_stage_b_from_diffuse,
    make_stage_b_checkpoint,
    restore_stage_b_checkpoint,
    sha256_file,
)
from tests.test_reflection_surfel_model import reflection_args


def diffuse_args():
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


def initialized_diffuse():
    model = DiffuseSurfelModel()
    model.spatial_lr_scale = 1.0
    xyz = torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [0.5, 1.0, 2.0]])
    values = {
        "_xyz": xyz,
        "_rotation": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1),
        "_scaling": torch.zeros((3, 2)),
        "_opacity": torch.zeros((3, 1)),
        "_base_color": torch.zeros((3, 3)),
        "_roughness": torch.zeros((3, 1)),
        "_f0": torch.zeros((3, 3)),
        "_ks": torch.zeros((3, 1)),
    }
    for name, value in values.items():
        setattr(model, name, torch.nn.Parameter(value.clone()))
    model._exposure = torch.nn.Parameter(torch.eye(3, 4)[None])
    model.exposure_mapping = {"toy": 0}
    model.max_radii2D = torch.zeros(3)
    model.training_setup(diffuse_args())
    return model


def test_stage_a_initialization_and_stage_b_round_trip_keep_namespaces_separate(tmp_path):
    source = initialized_diffuse()
    stage_a = tmp_path / "stage_a.pth"
    torch.save({"format": "rtgs_stage_a", "iteration": 15000, "model_state": source.capture(), "config": {}}, stage_a)
    before = sha256_file(stage_a)

    diffuse = DiffuseSurfelModel()
    reflection = ReflectionSurfelModel()
    global_step, reflection_step, provenance = initialize_stage_b_from_diffuse(
        stage_a,
        diffuse,
        reflection,
        diffuse_args(),
        reflection_args(),
        reflection_count=8,
        reflection_seed=0,
        map_location="cpu",
    )
    assert (global_step, reflection_step) == (15000, 0)
    assert provenance["sha256"] == before == sha256_file(stage_a)
    assert diffuse.get_xyz.data_ptr() != reflection.get_xyz.data_ptr()
    assert diffuse.optimizer is not reflection.optimizer

    checkpoint = make_stage_b_checkpoint(
        diffuse,
        reflection,
        global_iteration=15002,
        reflection_iteration=2,
        provenance=provenance,
        config={"ray_background": "scene"},
    )
    assert checkpoint["format"] == "rtgs_stage_b"
    assert set(checkpoint) >= {"diffuse", "reflection", "global_iteration", "reflection_iteration"}
    restored_d = DiffuseSurfelModel()
    restored_r = ReflectionSurfelModel()
    values = restore_stage_b_checkpoint(
        checkpoint,
        restored_d,
        restored_r,
        diffuse_args(),
        reflection_args(),
        restore_rng=False,
    )
    assert values[:2] == (15002, 2)
    assert torch.equal(restored_d.get_xyz, diffuse.get_xyz)
    assert torch.equal(restored_r.get_xyz, reflection.get_xyz)
    assert restored_d.optimizer is not restored_r.optimizer


def test_checkpoint_format_rejections(tmp_path):
    diffuse = DiffuseSurfelModel()
    reflection = ReflectionSurfelModel()
    bad = tmp_path / "bad.pth"
    torch.save({"format": "rtgs_stage_b"}, bad)
    try:
        initialize_stage_b_from_diffuse(
            bad, diffuse, reflection, diffuse_args(), reflection_args(), 4, 0, map_location="cpu"
        )
    except ValueError as error:
        assert "rtgs_stage_a" in str(error)
    else:
        raise AssertionError("wrong checkpoint format was accepted")
