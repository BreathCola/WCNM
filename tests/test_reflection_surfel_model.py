from types import SimpleNamespace

import torch

from scene.reflection_surfel_model import ReflectionSurfelModel


def reflection_args():
    return SimpleNamespace(
        reflection_percent_dense=0.01,
        reflection_position_lr_init=1e-3,
        reflection_position_lr_final=1e-5,
        reflection_position_lr_delay_mult=0.01,
        reflection_position_lr_max_steps=100,
        reflection_color_lr=2e-3,
        reflection_opacity_lr=1e-2,
        reflection_scaling_lr=1e-3,
        reflection_rotation_lr=1e-3,
    )


def test_random_bbox_is_seeded_bounded_and_has_independent_optimizer_groups():
    bbox_min = torch.tensor([-2.0, -1.0, 0.5])
    bbox_max = torch.tensor([3.0, 4.0, 2.5])
    first = ReflectionSurfelModel()
    second = ReflectionSurfelModel()
    first.create_random_bbox(bbox_min, bbox_max, count=32, seed=7)
    second.create_random_bbox(bbox_min, bbox_max, count=32, seed=7)
    assert torch.equal(first.get_xyz, second.get_xyz)
    assert (first.get_xyz >= bbox_min).all() and (first.get_xyz <= bbox_max).all()
    assert torch.allclose(first.get_rotation.norm(dim=-1), torch.ones(32), atol=1e-6)
    assert torch.allclose(first.get_opacity, torch.full((32, 1), 0.01), atol=1e-6)
    assert torch.allclose(first.get_color, torch.full((32, 3), 0.5), atol=1e-6)
    first.training_setup(reflection_args())
    assert [group["name"] for group in first.optimizer.param_groups] == [
        "xyz", "color", "opacity", "scaling", "rotation"
    ]
    assert first.update_learning_rate(0) > first.update_learning_rate(100)


def test_reflection_ply_round_trip_and_prune(tmp_path):
    model = ReflectionSurfelModel()
    model.create_random_bbox(torch.zeros(3), torch.ones(3), count=6, seed=3)
    path = tmp_path / "reflection.ply"
    model.save_ply(str(path))
    loaded = ReflectionSurfelModel()
    loaded.load_ply(str(path))
    assert torch.allclose(loaded.get_xyz.cpu(), model.get_xyz.detach().cpu())
    assert torch.allclose(loaded.color_raw.cpu(), model.color_raw.detach().cpu())
    loaded.training_setup(reflection_args())
    with torch.no_grad():
        loaded._opacity[0] = -100.0
    before = loaded.get_xyz.shape[0]
    loaded.densify_and_prune(max_grad=1e9, min_opacity=1e-4, extent=1.0)
    assert loaded.get_xyz.shape[0] == before - 1
    assert loaded.topology_version == 1


def test_reflection_stats_are_separate_and_accumulate_hits():
    model = ReflectionSurfelModel()
    model.create_random_bbox(torch.zeros(3), torch.ones(3), count=4, seed=1)
    model.training_setup(reflection_args())
    model.get_xyz.sum().backward()
    model.add_densification_stats(torch.tensor([0, 2, 2]), torch.tensor([0.5, 0.25, 0.75]))
    assert (model.xyz_gradient_accum > 0).all()
    assert model.hit_count[:, 0].tolist() == [1.0, 0.0, 2.0, 0.0]
    assert torch.allclose(model.hit_weight_accum[:, 0], torch.tensor([0.5, 0.0, 1.0, 0.0]))


def test_reflection_densification_clones_high_gradient_small_surfel():
    model = ReflectionSurfelModel()
    model.create_random_bbox(torch.zeros(3), torch.ones(3), count=4, seed=1)
    model.training_setup(reflection_args())
    with torch.no_grad():
        model._scaling.fill_(torch.log(torch.tensor(1e-3)))
        model.xyz_gradient_accum[0] = 1.0
        model.denom[0] = 1.0
    model.densify_and_prune(max_grad=0.5, min_opacity=1e-5, extent=1.0)
    assert model.get_xyz.shape[0] == 5
    assert model.topology_version >= 1
