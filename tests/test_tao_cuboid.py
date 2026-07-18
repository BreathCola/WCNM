import numpy as np
import torch

from geometry.tao_cuboid import intersect_cuboid_numpy, intersect_cuboid_torch, reflect


def test_runtime_cuboid_matches_audit_cache_fields_and_order(tmp_path):
    axes = np.eye(3, dtype=np.float64)
    lower = np.array([-1.0, -1.0, -1.0])
    upper = np.array([1.0, 1.0, 1.0])
    origins = np.array([[0, 0, -3], [3, 0, 0], [3, 3, 0]], np.float64)
    directions = np.array([[0, 0, 1], [-1, 0, 0], [-1, 0, 0]], np.float64)
    audited = intersect_cuboid_numpy(axes, lower, upper, origins, directions)
    np.savez_compressed(tmp_path / "audit.npz", **audited)
    runtime = intersect_cuboid_torch(
        torch.tensor(axes, dtype=torch.float64),
        torch.tensor(lower, dtype=torch.float64),
        torch.tensor(upper, dtype=torch.float64),
        torch.tensor(origins, dtype=torch.float64),
        torch.tensor(directions, dtype=torch.float64),
    )
    assert np.array_equal(runtime["valid_two_hit"].numpy(), audited["valid_two_hit"])
    for name in ("t_near", "t_far", "front_position", "back_position", "front_normal", "back_normal"):
        assert np.allclose(runtime[name].numpy(), audited[name], atol=1e-6)
    assert audited["crossing_count"].tolist() == [2, 2, 0]
    valid = audited["valid_two_hit"]
    assert np.all(audited["t_far"][valid] > audited["t_near"][valid])


def test_front_and_back_normals_produce_outside_and_inside_reflections():
    hit = intersect_cuboid_torch(
        torch.eye(3), torch.tensor([-1.0, -1.0, -1.0]),
        torch.tensor([1.0, 1.0, 1.0]), torch.tensor([[0.0, 0.0, -3.0]]),
        torch.tensor([[0.0, 0.0, 1.0]]),
    )
    direction = torch.tensor([[0.0, 0.0, 1.0]])
    front = reflect(direction, hit["front_normal"])
    back = reflect(direction, hit["back_normal"])
    assert torch.allclose(hit["front_normal"], torch.tensor([[0.0, 0.0, -1.0]]))
    assert torch.allclose(hit["back_normal"], torch.tensor([[0.0, 0.0, 1.0]]))
    assert front[0, 2] < 0 and back[0, 2] < 0
