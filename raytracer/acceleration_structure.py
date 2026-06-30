"""CUDA Morton-ordered linear BVH for finite Reflection surfel support."""

import importlib
import math
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from utils.surfel_utils import quaternion_to_rotation_matrix


_EXTENSION = None
_EXTENSION_ERROR = None


def load_cuda_extension():
    """Load or JIT-build the CUDA backend; never return a Python fallback."""
    global _EXTENSION, _EXTENSION_ERROR
    if _EXTENSION is not None:
        return _EXTENSION
    if _EXTENSION_ERROR is not None:
        raise RuntimeError("RT-GS CUDA LBVH extension is unavailable") from _EXTENSION_ERROR
    try:
        try:
            _EXTENSION = importlib.import_module("rtgs_bvh_cuda")
        except ImportError:
            root = Path(__file__).resolve().parent
            _EXTENSION = load(
                name="rtgs_bvh_cuda_v1",
                sources=[str(root / "csrc" / "bvh_bindings.cpp"), str(root / "csrc" / "bvh_cuda.cu")],
                extra_cflags=["-O2"],
                extra_cuda_cflags=["-O2"],
                verbose=False,
            )
    except Exception as error:
        _EXTENSION_ERROR = error
        raise RuntimeError(
            "RT-GS CUDA LBVH extension failed to compile/load; production ray tracing is fail-closed"
        ) from error
    return _EXTENSION


def _expand_bits(value):
    value = value & 0x000003FF
    value = (value | (value << 16)) & 0x030000FF
    value = (value | (value << 8)) & 0x0300F00F
    value = (value | (value << 4)) & 0x030C30C3
    value = (value | (value << 2)) & 0x09249249
    return value


def morton_codes(points: torch.Tensor) -> torch.Tensor:
    minimum = points.amin(dim=0)
    extent = (points.amax(dim=0) - minimum).clamp_min(1e-8)
    quantized = (((points - minimum) / extent).clamp(0.0, 1.0) * 1023.0).long()
    return _expand_bits(quantized[:, 0]) | (_expand_bits(quantized[:, 1]) << 1) | (_expand_bits(quantized[:, 2]) << 2)


def surfel_aabbs(model, cutoff_sigma: float = 3.0):
    rotation = quaternion_to_rotation_matrix(model.get_rotation.detach())
    tangent_u = rotation[:, :, 0]
    tangent_v = rotation[:, :, 1]
    scaling = model.get_scaling.detach()
    extent = float(cutoff_sigma) * (
        tangent_u.abs() * scaling[:, 0:1] + tangent_v.abs() * scaling[:, 1:2]
    )
    center = model.get_xyz.detach()
    return center - extent, center + extent


class CudaLBVH:
    """Complete balanced hierarchy over Morton-ordered Reflection leaves."""

    def __init__(self, cutoff_sigma: float = 3.0):
        if cutoff_sigma <= 0:
            raise ValueError("cutoff_sigma must be positive")
        self.cutoff_sigma = float(cutoff_sigma)
        self.leaf_base = 0
        self.leaf_indices = None
        self.node_min = None
        self.node_max = None
        self.topology_version = -1
        self._parameter_signature = None
        self._dirty = True
        self.rebuild_count = 0
        self.refit_count = 0

    @staticmethod
    def _signature(model):
        return tuple((id(value), value._version) for value in (model._xyz, model._rotation, model._scaling))

    def mark_parameters_updated(self):
        self._dirty = True

    def _validate_model(self, model):
        if not model.get_xyz.is_cuda:
            raise RuntimeError("production LBVH requires CUDA Reflection tensors")
        if model.get_xyz.shape[0] <= 0:
            raise RuntimeError("cannot build an LBVH for an empty Reflection field")

    def rebuild(self, model):
        self._validate_model(model)
        load_cuda_extension()
        count = model.get_xyz.shape[0]
        self.leaf_base = 1 << int(math.ceil(math.log2(max(count, 1))))
        order = torch.argsort(morton_codes(model.get_xyz.detach().float()))
        self.leaf_indices = torch.full(
            (self.leaf_base,), -1, dtype=torch.int64, device=model.get_xyz.device
        )
        self.leaf_indices[:count] = order
        self.topology_version = int(model.topology_version)
        self._refit_nodes(model)
        self._parameter_signature = self._signature(model)
        self._dirty = False
        self.rebuild_count += 1

    def _refit_nodes(self, model):
        minimum, maximum = surfel_aabbs(model, self.cutoff_sigma)
        minimum, maximum = minimum.float(), maximum.float()
        device = minimum.device
        self.node_min = torch.full((2 * self.leaf_base, 3), torch.inf, device=device)
        self.node_max = torch.full((2 * self.leaf_base, 3), -torch.inf, device=device)
        valid = self.leaf_indices >= 0
        indices = self.leaf_indices[valid]
        leaves = torch.arange(self.leaf_base, 2 * self.leaf_base, device=device)[valid]
        self.node_min[leaves] = minimum[indices]
        self.node_max[leaves] = maximum[indices]
        start = self.leaf_base // 2
        end = self.leaf_base
        while start >= 1:
            nodes = torch.arange(start, end, device=device)
            self.node_min[nodes] = torch.minimum(self.node_min[nodes * 2], self.node_min[nodes * 2 + 1])
            self.node_max[nodes] = torch.maximum(self.node_max[nodes * 2], self.node_max[nodes * 2 + 1])
            end = start
            start //= 2

    def refit(self, model):
        self._validate_model(model)
        if self.leaf_indices is None or int(model.topology_version) != self.topology_version:
            return self.rebuild(model)
        self._refit_nodes(model)
        self._parameter_signature = self._signature(model)
        self._dirty = False
        self.refit_count += 1

    def ensure_current(self, model):
        if self.leaf_indices is None or int(model.topology_version) != self.topology_version:
            self.rebuild(model)
        elif self._dirty or self._parameter_signature != self._signature(model):
            self.refit(model)

    def candidates(self, origins: torch.Tensor, directions: torch.Tensor):
        if self.leaf_indices is None:
            raise RuntimeError("LBVH has not been built")
        extension = load_cuda_extension()
        origins32 = origins.detach().float().contiguous()
        directions32 = directions.detach().float().contiguous()
        counts32 = extension.count_candidates(
            origins32,
            directions32,
            self.node_min,
            self.node_max,
            self.leaf_indices,
            self.leaf_base,
        )
        if (counts32 < 0).any().item():
            raise RuntimeError("CUDA LBVH traversal stack overflow")
        counts = counts32.to(torch.int64)
        offsets = torch.cat((torch.zeros(1, dtype=torch.int64, device=counts.device), counts.cumsum(0)))
        candidates = extension.fill_candidates(
            origins32,
            directions32,
            self.node_min,
            self.node_max,
            self.leaf_indices,
            offsets,
            self.leaf_base,
        )
        if candidates.numel() != int(offsets[-1].item()):
            raise RuntimeError("CUDA LBVH candidate count/fill mismatch")
        return candidates, offsets
