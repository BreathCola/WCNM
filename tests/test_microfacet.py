import torch

from utils.microfacet import fresnel_schlick, microfacet_reflection, pbrt_roughness_to_alpha


def test_full_microfacet_is_finite_and_normal_incidence_fresnel_is_f0():
    normal = torch.tensor([[0.0, 0.0, 1.0]])
    result = microfacet_reflection(
        normal,
        wo=normal.clone(),
        wi=normal.clone(),
        roughness=torch.tensor([[0.5]]),
        f0=torch.tensor([[0.04, 0.05, 0.06]]),
    )
    assert torch.allclose(result["F"], torch.tensor([[0.04, 0.05, 0.06]]), atol=1e-6)
    for value in result.values():
        if torch.is_tensor(value):
            assert torch.isfinite(value).all()
    assert (result["D"] > 0).all() and (result["G"] > 0).all() and (result["wr"] > 0).all()


def test_fresnel_increases_toward_grazing_and_backfaces_are_gated():
    f0 = torch.full((1, 3), 0.04)
    normal = fresnel_schlick(f0, torch.ones((1, 1)))
    grazing = fresnel_schlick(f0, torch.full((1, 1), 0.05))
    assert (grazing > normal).all()
    result = microfacet_reflection(
        torch.tensor([[0.0, 0.0, 1.0]]),
        wo=torch.tensor([[0.0, 0.0, 1.0]]),
        wi=torch.tensor([[0.0, 0.0, -1.0]]),
        roughness=torch.tensor([[0.5]]),
        f0=f0,
    )
    assert torch.equal(result["wr"], torch.zeros_like(result["wr"]))
    assert not result["gate"].item()


def test_pbrt_remap_is_supported_and_clamped_by_microfacet():
    remapped = pbrt_roughness_to_alpha(torch.tensor([[0.2], [0.8]]))
    assert torch.isfinite(remapped).all()
    result = microfacet_reflection(
        torch.tensor([[0.0, 0.0, 1.0]]),
        wo=torch.tensor([[0.0, 0.0, 1.0]]),
        wi=torch.tensor([[0.0, 0.0, 1.0]]),
        roughness=torch.tensor([[0.03]]),
        f0=torch.full((1, 3), 0.04),
        roughness_remap=True,
    )
    assert (result["alpha_ggx"] >= 0.03).all() and (result["alpha_ggx"] <= 1.0).all()
