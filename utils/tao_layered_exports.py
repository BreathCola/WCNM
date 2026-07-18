"""Versioned float/display exports for the Tao layered renderer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import torch
from torchvision.utils import save_image


EXPORT_SCHEMA = "rtgs_tao_layered_export_v1"
REQUIRED_LAYER_NAMES = (
    "final", "ground_truth", "no_reflection", "reflection_only", "internal_only",
    "front_reflection_only", "back_internal_reflection_only", "Cout_only",
    "D_direct_only", "Cin", "Ain", "internal_intrinsic_rgb", "internal_alpha",
)


def _hwc(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.ndim == 2:
        return value[..., None]
    if value.ndim != 3:
        raise ValueError(f"{name} must be a 2D/HWC/CHW image tensor")
    if value.shape[-1] in (1, 3):
        return value
    if value.shape[0] in (1, 3):
        return value.permute(1, 2, 0)
    raise ValueError(f"{name} has no recognizable channel dimension: {tuple(value.shape)}")


def layered_export_tensors(
    package: Mapping[str, object], ground_truth: torch.Tensor,
) -> dict[str, torch.Tensor]:
    gt = ground_truth.detach()
    if gt.ndim == 3 and gt.shape[0] == 3:
        gt = gt.permute(1, 2, 0)
    else:
        gt = _hwc(gt, "ground_truth")
    if gt.shape[-1] != 3:
        raise ValueError("ground_truth must be RGB")
    exports: dict[str, torch.Tensor] = {"ground_truth": gt}
    for name in REQUIRED_LAYER_NAMES:
        if name == "ground_truth":
            continue
        value = package.get(name)
        if not torch.is_tensor(value):
            raise KeyError(f"layered package is missing tensor {name}")
        exports[name] = _hwc(value.detach(), name)
    shape = exports["final"].shape[:2]
    if any(value.shape[:2] != shape for value in exports.values()):
        raise ValueError("layered export tensors do not share one image resolution")
    closure = exports["final"] - exports["no_reflection"] - exports["reflection_only"]
    if not torch.isfinite(closure).all():
        raise FloatingPointError("layered closure contains NaN or Inf")
    exports["linear_closure_error"] = closure
    return exports


def save_tao_layered_exports(
    package: Mapping[str, object], ground_truth: torch.Tensor, directory: str | Path,
) -> dict[str, object]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    exports = layered_export_tensors(package, ground_truth)
    closure = exports["linear_closure_error"]
    metadata: dict[str, object] = {
        "schema": EXPORT_SCHEMA,
        "float_tensor_files": {},
        "display_png_files": {},
        "linear_closure_max_abs": float(closure.abs().max().cpu()),
        "linear_closure_mean_abs": float(closure.abs().mean().cpu()),
        "linear_space": "unclamped float tensors; clamp is display-only",
        "Cin_contract": "premultiplied black-background RGB; Ain is not applied twice",
    }
    for name, value in exports.items():
        tensor_path = directory / f"{name}.pt"
        torch.save(value.cpu(), tensor_path)
        metadata["float_tensor_files"][name] = tensor_path.name
        display = value[..., :3].permute(2, 0, 1).float().clamp(0.0, 1.0)
        png_path = directory / f"{name}.png"
        save_image(display, png_path)
        metadata["display_png_files"][name] = png_path.name
    (directory / "layered_export_manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata
