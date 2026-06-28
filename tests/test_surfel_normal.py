import torch

from utils.surfel_utils import face_forward, surfel_frame


def test_surfel_normal_is_rotation_derived_and_normalized():
    rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]], requires_grad=True)
    scaling = torch.tensor([[2.0, 0.5]], requires_grad=True)
    tangent_u, tangent_v, normal = surfel_frame(rotation, scaling)

    assert torch.allclose(tangent_u, torch.tensor([[2.0, 0.0, 0.0]]))
    assert torch.allclose(tangent_v, torch.tensor([[0.0, 0.5, 0.0]]))
    assert torch.allclose(normal.norm(dim=-1), torch.ones(1), atol=1e-6)
    assert torch.allclose(normal, torch.tensor([[0.0, 0.0, 1.0]]), atol=1e-6)

    normal[:, 0].sum().backward()
    assert rotation.grad is not None


def test_face_forward_orients_normal_toward_camera():
    normal = torch.tensor([[[0.0, 0.0, 1.0]]])
    position = torch.tensor([[[0.0, 0.0, 2.0]]])
    camera = torch.tensor([0.0, 0.0, 0.0])

    oriented = face_forward(normal, position, camera)

    assert torch.allclose(oriented, torch.tensor([[[0.0, 0.0, -1.0]]]))
    wo = torch.nn.functional.normalize(camera - position, dim=-1)
    assert ((oriented * wo).sum(dim=-1) >= 0).all()
