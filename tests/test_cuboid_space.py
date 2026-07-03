import math

import pytest
import torch

from geometry.cuboid_space import CuboidSpace, INSIDE, INTERFACE, OUTSIDE


def space(axes=None, margin=0.1, epsilon=1e-6):
    return CuboidSpace(
        torch.eye(3) if axes is None else axes,
        torch.tensor([-1.0, -2.0, -3.0]),
        torch.tensor([1.0, 2.0, 3.0]),
        interface_margin=margin, epsilon=epsilon,
    )


def test_cuboid_classifies_inside_interface_outside_and_epsilon():
    cuboid = space()
    points = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.9, 0.0, 0.0],
        [0.95, 0.0, 0.0],
        [1.05, 0.0, 0.0],
        [1.2, 0.0, 0.0],
        [1.0 + 0.1 + 2e-6, 0.0, 0.0],
    ])
    assert cuboid.classify(points).tolist() == [
        INSIDE, INTERFACE, INTERFACE, INTERFACE, OUTSIDE, OUTSIDE,
    ]


def test_cuboid_rotation_and_batch_shapes_are_coordinate_consistent():
    angle = math.pi / 3
    axes = torch.tensor([
        [math.cos(angle), -math.sin(angle), 0.0],
        [math.sin(angle), math.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    cuboid = space(axes=axes)
    local = torch.tensor([[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]])
    world = cuboid.local_to_world(local)
    assert torch.allclose(cuboid.world_to_local(world), local, atol=1e-6)
    assert cuboid.classify(world).shape == (1, 2)
    assert cuboid.classify(world).tolist() == [[INSIDE, OUTSIDE]]


def test_inside_latent_is_differentiable_and_strictly_inside_safe():
    cuboid = space()
    latent = torch.tensor([
        [-20.0, 0.0, 20.0], [0.3, -0.5, 1.0],
    ], requires_grad=True)
    world = cuboid.decode_inside_latent(latent)
    assert torch.all(cuboid.classify(world) == INSIDE)
    world.square().sum().backward()
    assert latent.grad is not None and torch.isfinite(latent.grad).all()
    encoded = cuboid.encode_inside_world(world.detach())
    assert torch.allclose(cuboid.decode_inside_latent(encoded), world.detach(), atol=2e-5)


def test_cuboid_rejects_nonorthogonal_axes_and_excessive_margin():
    with pytest.raises(ValueError, match="orthonormal"):
        space(axes=torch.ones((3, 3)))
    with pytest.raises(ValueError, match="no strict interior"):
        space(margin=1.0)
