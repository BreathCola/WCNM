import torch

from utils.stage_d_paper_exports import paper_export_tensors


def test_paper_export_modes_are_read_only_decompositions():
    h, w = 2, 3
    final = torch.full((h, w, 3), 0.8)
    reflection = torch.full((h, w, 3), 0.2)
    package = {
        "final": final,
        "inside_color": torch.full((h, w, 3), 0.4),
        "inside_alpha": torch.full((h, w, 1), 0.5),
        "reflection_contribution": reflection,
        "diffuse_contribution": torch.full((h, w, 3), 0.1),
        "transmittance_contribution": torch.full((h, w, 3), 0.3),
        "normal": torch.zeros((h, w, 3)),
        "front_normal": torch.ones((h, w, 3)),
    }
    gt = torch.zeros((3, h, w))
    glass = torch.tensor([[[1.0], [0.0], [1.0]], [[0.0], [1.0], [0.0]]])
    exports = paper_export_tensors(package, gt, glass)
    assert torch.allclose(exports["ground_truth"], gt.permute(1, 2, 0))
    assert torch.allclose(exports["final"], final)
    assert torch.allclose(exports["no_reflection_full_scene"], final - reflection)
    assert exports["Cin_RGBA"].shape == (h, w, 4)
    assert exports["glass_internal_decomposition_no_R_no_Cout"][0, 1].sum() == 0
    assert exports["transmittance_surfel_normal"] == "unavailable_without_ray_normal_output"
