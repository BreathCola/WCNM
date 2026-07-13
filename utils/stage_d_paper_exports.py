"""Read-only Stage D paper/decomposition export tensors."""

from __future__ import annotations

from pathlib import Path
import json

import torch
from torchvision.utils import save_image


EXPORT_SCHEMA = "rtgs_stage_d_paper_export_modes_v1"


def _hwc1(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 2:
        return mask[..., None]
    if mask.ndim == 3 and mask.shape[0] == 1:
        return mask.permute(1, 2, 0)
    if mask.ndim == 3 and mask.shape[-1] == 1:
        return mask
    raise ValueError(f"expected one-channel mask, got {tuple(mask.shape)}")


def paper_export_tensors(package: dict[str, torch.Tensor], ground_truth: torch.Tensor, glass_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor | str]:
    """Create export-mode tensors without changing training/rendering semantics."""
    gt = ground_truth.detach()
    if gt.ndim == 3 and gt.shape[0] == 3:
        gt = gt.permute(1, 2, 0)
    if gt.ndim != 3 or gt.shape[-1] != 3:
        raise ValueError("ground_truth must be CHW or HWC RGB")
    final = package["final"].detach()
    cin = package["inside_color"].detach()
    ain = package["inside_alpha"].detach()
    if glass_mask is None:
        glass = torch.ones_like(ain)
    else:
        glass = (_hwc1(glass_mask.detach()).to(device=ain.device, dtype=ain.dtype) >= 0.5).to(ain)
    reflection = package.get("reflection_contribution", torch.zeros_like(final)).detach()
    diffuse = package.get("diffuse_contribution", torch.zeros_like(final)).detach()
    trans = package.get("transmittance_contribution", torch.zeros_like(final)).detach()
    no_reflection = (final - reflection).clamp(0, 1)
    inside_rgb = cin.clamp(0, 1)
    inside_rgba = torch.cat([inside_rgb, ain.clamp(0, 1)], dim=-1)
    decomposition = torch.where(glass.expand_as(inside_rgb) > 0.5, inside_rgb * ain.clamp(0, 1), torch.zeros_like(inside_rgb))
    exports: dict[str, torch.Tensor | str] = {
        "ground_truth": gt,
        "final": final,
        "inside_only": inside_rgb,
        "Cin": cin,
        "Ain": ain,
        "Cin_RGBA": inside_rgba,
        "no_reflection_full_scene": no_reflection,
        "glass_internal_decomposition_no_R_no_Cout": decomposition,
        "diffuse_normal": package.get("normal", torch.zeros_like(final)).detach(),
        "glass_cuboid_normal": package.get("front_normal", torch.zeros_like(final)).detach(),
        "transmittance_surfel_normal": "unavailable_without_ray_normal_output",
        "diffuse_contribution": diffuse,
        "transmittance_contribution": trans,
    }
    return exports


def save_paper_export_modes(package, ground_truth, glass_mask, directory: str | Path) -> dict:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    exports = paper_export_tensors(package, ground_truth, glass_mask)
    metadata = {
        "schema": EXPORT_SCHEMA,
        "float_tensor_files": {},
        "display_png_files": {},
        "normalization": "RGB/A tensors clamped to [0,1] for display PNG only; .pt tensors preserve float values",
    }
    for name, value in exports.items():
        if not torch.is_tensor(value):
            metadata["float_tensor_files"][name] = None
            metadata["display_png_files"][name] = str(value)
            continue
        tensor_path = directory / f"{name}.pt"
        torch.save(value.detach().cpu(), tensor_path)
        metadata["float_tensor_files"][name] = tensor_path.name
        display = value.detach()
        if display.ndim == 3 and display.shape[-1] in (1, 3, 4):
            display = display[..., :3].permute(2, 0, 1)
        elif display.ndim == 2:
            display = display[None]
        else:
            continue
        png_path = directory / f"{name}.png"
        save_image(display.float().clamp(0, 1), png_path)
        metadata["display_png_files"][name] = png_path.name
    (directory / "paper_export_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata
