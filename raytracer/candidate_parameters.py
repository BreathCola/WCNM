"""Fused candidate parameter gather with one shared-ID gradient reduction."""

import torch
import torch.nn.functional as F

from raytracer.acceleration_structure import load_cuda_extension
from utils.surfel_utils import quaternion_to_rotation_matrix


PARAMETER_CHANNELS = 13


def pack_reflection_parameters(model) -> torch.Tensor:
    """Pack decoder-form parameters using each field's forward-scale contract."""
    return torch.cat(
        (
            model.get_xyz,
            model._rotation,
            model.raytrace_scaling_raw,
            model._opacity,
            model._color,
        ),
        dim=-1,
    )


class _CandidateParameterGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, parameter_table: torch.Tensor, candidate_ids: torch.Tensor):
        if not parameter_table.is_cuda or not candidate_ids.is_cuda:
            raise RuntimeError("production candidate parameter gather is CUDA-only")
        if parameter_table.ndim != 2 or parameter_table.shape[1] != PARAMETER_CHANNELS:
            raise ValueError("parameter_table must have shape [N,13]")
        ids = candidate_ids.reshape(-1).contiguous()
        extension = load_cuda_extension()
        gathered = extension.gather_candidate_parameters(
            parameter_table.contiguous(), ids
        )
        ctx.save_for_backward(ids)
        ctx.surfel_count = parameter_table.shape[0]
        ctx.candidate_shape = tuple(candidate_ids.shape)
        return gathered.reshape(*ctx.candidate_shape, PARAMETER_CHANNELS)

    @staticmethod
    def backward(ctx, candidate_gradients: torch.Tensor):
        (candidate_ids,) = ctx.saved_tensors
        extension = load_cuda_extension()
        table_gradients = extension.reduce_candidate_gradients(
            candidate_ids,
            candidate_gradients.reshape(-1, PARAMETER_CHANNELS).contiguous(),
            ctx.surfel_count,
        )
        return table_gradients, None


def gather_candidate_parameters(
    parameter_table: torch.Tensor, candidate_ids: torch.Tensor
) -> torch.Tensor:
    return _CandidateParameterGather.apply(parameter_table, candidate_ids)


def decode_candidate_parameters(raw: torch.Tensor):
    """Apply the original point-wise R activations after candidate gather."""
    if raw.shape[-1] != PARAMETER_CHANNELS:
        raise ValueError("raw candidate parameters must have 13 channels")
    quaternion = F.normalize(raw[..., 3:7], dim=-1, eps=1e-12)
    return {
        "xyz": raw[..., 0:3],
        # quaternion_to_rotation_matrix intentionally performs the second
        # normalization already present in the original model.get_rotation path.
        "rotation": quaternion_to_rotation_matrix(quaternion),
        "scaling": torch.exp(raw[..., 7:9]),
        "opacity": torch.sigmoid(raw[..., 9:10]),
        "color": torch.sigmoid(raw[..., 10:13]),
    }
