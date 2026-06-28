import torch

from utils.loss_utils import monocular_normal_loss, normal_depth_consistency_loss


def _plane(height=7, width=9):
    x, y = torch.meshgrid(torch.arange(width), torch.arange(height), indexing="xy")
    position = torch.stack((x.float(), y.float(), torch.ones_like(x).float()), dim=-1)
    alpha = torch.ones(height, width, 1)
    return position, alpha


def test_normal_depth_loss_is_zero_for_consistent_plane():
    position, alpha = _plane()
    normal = torch.zeros_like(position)
    normal[..., 2] = -1.0

    loss = normal_depth_consistency_loss(normal, position, alpha)

    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_normal_depth_loss_penalizes_opposite_normal_and_masks_background():
    position, alpha = _plane()
    opposite = torch.zeros_like(position)
    opposite[..., 2] = 1.0
    assert torch.allclose(normal_depth_consistency_loss(opposite, position, alpha), torch.tensor(2.0))

    alpha.zero_()
    assert torch.allclose(normal_depth_consistency_loss(opposite, position, alpha), torch.tensor(0.0))


def test_monocular_normal_loss_supports_validity_mask():
    position, alpha = _plane()
    normal = torch.zeros_like(position)
    normal[..., 2] = -1.0
    valid = torch.ones_like(alpha, dtype=torch.bool)
    assert torch.allclose(monocular_normal_loss(normal, normal.clone(), alpha, valid), torch.tensor(0.0))

    prior = -normal
    valid[:, :4] = False
    assert torch.allclose(monocular_normal_loss(normal, prior, alpha, valid), torch.tensor(2.0))
