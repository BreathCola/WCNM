import torch

from raytracer.ray_utils import decode_stage_a_gbuffer, generate_reflection_rays, valid_diffuse_surface_mask


def _output():
    return {
        "Cd": torch.tensor([[[0.10, 0.20, 0.30], [0.0, 0.0, 0.0]]]),
        "alpha": torch.tensor([[[0.5], [0.0]]]),
        "depth": torch.tensor([[[1.0], [0.0]]]),
        "position": torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]]),
        "normal": torch.tensor([[[0.0, 0.0, -1.0], [0.0, 0.0, 0.0]]]),
        "roughness": torch.tensor([[[0.25], [0.0]]]),
        "f0": torch.tensor([[[0.02, 0.03, 0.04], [0.0, 0.0, 0.0]]]),
        "ks": torch.tensor([[[0.10], [0.0]]]),
    }


def test_reflection_direction_epsilon_and_valid_compaction():
    output = _output()
    camera = torch.zeros(3)
    assert valid_diffuse_surface_mask(output, camera).tolist() == [[True, False]]
    rays = generate_reflection_rays(output, camera, scene_radius=2.0, ray_epsilon_scale=1e-4)
    assert rays["flat_indices"].tolist() == [0]
    assert torch.allclose(rays["directions"], torch.tensor([[0.0, 0.0, -1.0]]))
    assert torch.allclose(rays["origins"], torch.tensor([[0.0, 0.0, 0.9998]]), atol=1e-7)


def test_stage_a_premultiplied_material_decode_does_not_modify_normal():
    output = _output()
    decoded = decode_stage_a_gbuffer(output, torch.zeros(3), roughness_min=0.03)
    assert torch.allclose(decoded["Cd"][0, 0], torch.tensor([0.2, 0.4, 0.6]))
    assert torch.allclose(decoded["roughness"][0, 0], torch.tensor([0.5]))
    assert torch.allclose(decoded["f0"][0, 0], torch.tensor([0.04, 0.06, 0.08]))
    assert torch.allclose(decoded["ks"][0, 0], torch.tensor([0.2]))
    assert decoded["normal"].data_ptr() == output["normal"].data_ptr()


def test_white_background_is_removed_before_diffuse_decode():
    output = _output()
    output["Cd"][0, 0] = torch.tensor([0.6, 0.7, 0.8])
    decoded = decode_stage_a_gbuffer(output, torch.ones(3), roughness_min=0.03)
    assert torch.allclose(decoded["Cd"][0, 0], torch.tensor([0.2, 0.4, 0.6]), atol=1e-6)
