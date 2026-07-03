"""Independent Stage B Reflection 2D Gaussian surfel field."""

import os
from typing import Dict, Optional

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from torch import nn

from utils.general_utils import get_expon_lr_func, inverse_sigmoid
from utils.surfel_utils import quaternion_to_rotation_matrix, surfel_frame
from utils.system_utils import mkdir_p


def _raw_from_unit(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return inverse_sigmoid(value.clamp(eps, 1.0 - eps))


class ReflectionSurfelModel:
    """Reflection field with storage, optimization, and topology independent of D."""

    checkpoint_version = 1
    model_type = "reflection_surfel"

    def __init__(self):
        self._xyz = torch.empty(0)
        self._rotation = torch.empty(0)
        self._scaling = torch.empty(0)
        self._opacity = torch.empty(0)
        self._color = torch.empty(0)
        self.optimizer = None
        self.xyz_scheduler_args = None
        self.percent_dense = 0.01
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.hit_weight_accum = torch.empty(0)
        self.hit_count = torch.empty(0)
        self.topology_version = 0
        self.initialization = {}

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def xyz(self):
        return self.get_xyz

    @property
    def get_rotation(self):
        return torch.nn.functional.normalize(self._rotation, dim=-1, eps=1e-12)

    @property
    def rotation(self):
        return self.get_rotation

    @property
    def get_scaling(self):
        return torch.exp(self._scaling)

    @property
    def scaling_2d(self):
        return self.get_scaling

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity)

    @property
    def opacity(self):
        return self.get_opacity

    @property
    def opacity_raw(self):
        return self._opacity

    @property
    def get_color(self):
        return torch.sigmoid(self._color)

    @property
    def color(self):
        return self.get_color

    @property
    def color_raw(self):
        return self._color

    @property
    def get_normal(self):
        return surfel_frame(self.get_rotation, self.get_scaling)[2]

    @property
    def tangent_u(self):
        return surfel_frame(self.get_rotation, self.get_scaling)[0]

    @property
    def tangent_v(self):
        return surfel_frame(self.get_rotation, self.get_scaling)[1]

    def create_random_bbox(
        self,
        bbox_min: torch.Tensor,
        bbox_max: torch.Tensor,
        count: int,
        seed: int,
    ) -> None:
        if count <= 0:
            raise ValueError("reflection_init_count must be positive")
        if bbox_min.shape != (3,) or bbox_max.shape != (3,):
            raise ValueError("reflection bbox bounds must have shape [3]")
        if not torch.isfinite(bbox_min).all() or not torch.isfinite(bbox_max).all():
            raise ValueError("reflection bbox must be finite")
        if not torch.all(bbox_max > bbox_min):
            raise ValueError("reflection bbox must have positive extent on every axis")

        device, dtype = bbox_min.device, bbox_min.dtype
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        extent = bbox_max - bbox_min
        xyz = bbox_min + torch.rand((count, 3), generator=generator, device=device, dtype=dtype) * extent
        rotation = torch.randn((count, 4), generator=generator, device=device, dtype=dtype)
        rotation = torch.nn.functional.normalize(rotation, dim=-1, eps=1e-12)
        spacing = torch.pow(extent.prod().clamp_min(torch.finfo(dtype).eps) / float(count), 1.0 / 3.0)
        scale = (0.5 * spacing).clamp_min(torch.finfo(dtype).eps)
        scaling = torch.log(torch.full((count, 2), scale, device=device, dtype=dtype))
        opacity = _raw_from_unit(torch.full((count, 1), 0.01, device=device, dtype=dtype))
        color = _raw_from_unit(torch.full((count, 3), 0.5, device=device, dtype=dtype))

        self._xyz = nn.Parameter(xyz.requires_grad_(True))
        self._rotation = nn.Parameter(rotation.requires_grad_(True))
        self._scaling = nn.Parameter(scaling.requires_grad_(True))
        self._opacity = nn.Parameter(opacity.requires_grad_(True))
        self._color = nn.Parameter(color.requires_grad_(True))
        self._reset_stats(count, device)
        self.topology_version = 0
        self.initialization = {
            "mode": "random_bbox",
            "count": int(count),
            "seed": int(seed),
            "bbox_min": bbox_min.detach().cpu().tolist(),
            "bbox_max": bbox_max.detach().cpu().tolist(),
            "initial_opacity": 0.01,
            "initial_color": 0.5,
            "initial_scale": float(scale.detach().cpu()),
        }

    def _reset_stats(self, count: int, device: torch.device) -> None:
        self.xyz_gradient_accum = torch.zeros((count, 1), device=device)
        self.denom = torch.zeros((count, 1), device=device)
        self.hit_weight_accum = torch.zeros((count, 1), device=device)
        self.hit_count = torch.zeros((count, 1), device=device)

    def training_setup(self, args) -> None:
        if self.get_xyz.numel() == 0:
            raise RuntimeError("initialize ReflectionSurfelModel before training_setup")
        self.percent_dense = float(args.reflection_percent_dense)
        groups = [
            {"params": [self._xyz], "lr": args.reflection_position_lr_init, "name": "xyz"},
            {"params": [self._color], "lr": args.reflection_color_lr, "name": "color"},
            {"params": [self._opacity], "lr": args.reflection_opacity_lr, "name": "opacity"},
            {"params": [self._scaling], "lr": args.reflection_scaling_lr, "name": "scaling"},
            {"params": [self._rotation], "lr": args.reflection_rotation_lr, "name": "rotation"},
        ]
        self.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=args.reflection_position_lr_init,
            lr_final=args.reflection_position_lr_final,
            lr_delay_mult=args.reflection_position_lr_delay_mult,
            max_steps=args.reflection_position_lr_max_steps,
        )
        if self.xyz_gradient_accum.shape[0] != self.get_xyz.shape[0]:
            self._reset_stats(self.get_xyz.shape[0], self.get_xyz.device)

    def update_learning_rate(self, reflection_step: int):
        if self.optimizer is None or self.xyz_scheduler_args is None:
            raise RuntimeError("Reflection optimizer is not initialized")
        for group in self.optimizer.param_groups:
            if group["name"] == "xyz":
                group["lr"] = self.xyz_scheduler_args(reflection_step)
                return group["lr"]
        raise RuntimeError("Reflection optimizer has no xyz group")

    def capture(self) -> Dict:
        return {
            "checkpoint_version": self.checkpoint_version,
            "model_type": self.model_type,
            "xyz": self._xyz,
            "rotation": self._rotation,
            "scaling_2d": self._scaling,
            "opacity_raw": self._opacity,
            "color_raw": self._color,
            "xyz_gradient_accum": self.xyz_gradient_accum,
            "denom": self.denom,
            "hit_weight_accum": self.hit_weight_accum,
            "hit_count": self.hit_count,
            "topology_version": int(self.topology_version),
            "percent_dense": float(self.percent_dense),
            "initialization": dict(self.initialization),
            "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
        }

    def restore(self, state: Dict, args) -> None:
        if state.get("model_type") != self.model_type:
            raise ValueError("checkpoint is not a Reflection surfel checkpoint")
        if state.get("checkpoint_version") != self.checkpoint_version:
            raise ValueError(f"unsupported Reflection checkpoint version: {state.get('checkpoint_version')}")
        self._xyz = nn.Parameter(state["xyz"].detach().requires_grad_(True))
        self._rotation = nn.Parameter(state["rotation"].detach().requires_grad_(True))
        self._scaling = nn.Parameter(state["scaling_2d"].detach().requires_grad_(True))
        self._opacity = nn.Parameter(state["opacity_raw"].detach().requires_grad_(True))
        self._color = nn.Parameter(state["color_raw"].detach().requires_grad_(True))
        self.xyz_gradient_accum = state["xyz_gradient_accum"]
        self.denom = state["denom"]
        self.hit_weight_accum = state["hit_weight_accum"]
        self.hit_count = state["hit_count"]
        self.topology_version = int(state["topology_version"])
        self.percent_dense = float(state["percent_dense"])
        self.initialization = dict(state["initialization"])
        self.training_setup(args)
        if state["optimizer"] is not None:
            self.optimizer.load_state_dict(state["optimizer"])

    def construct_list_of_attributes(self):
        attrs = ["x", "y", "z", "nx", "ny", "nz"]
        attrs += [f"color_raw_{index}" for index in range(3)]
        attrs += ["opacity_raw"]
        attrs += [f"scale_2d_raw_{index}" for index in range(2)]
        attrs += [f"rot_{index}" for index in range(4)]
        return attrs

    def save_ply(self, path: str) -> None:
        mkdir_p(os.path.dirname(path))
        arrays = [
            self.get_xyz.detach().cpu().numpy(),
            self.get_normal.detach().cpu().numpy(),
            self._color.detach().cpu().numpy(),
            self._opacity.detach().cpu().numpy(),
            self._scaling.detach().cpu().numpy(),
            self._rotation.detach().cpu().numpy(),
        ]
        dtype = [(name, "f4") for name in self.construct_list_of_attributes()]
        elements = np.empty(self.get_xyz.shape[0], dtype=dtype)
        elements[:] = list(map(tuple, np.concatenate(arrays, axis=1)))
        PlyData([PlyElement.describe(elements, "vertex")]).write(path)

    def load_ply(self, path: str, use_train_test_exp: bool = False) -> None:
        del use_train_test_exp
        ply = PlyData.read(path)
        vertex = ply.elements[0]

        def stack(names):
            return np.stack([np.asarray(vertex[name]) for name in names], axis=1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._xyz = nn.Parameter(torch.tensor(stack(["x", "y", "z"]), dtype=torch.float32, device=device))
        self._color = nn.Parameter(torch.tensor(stack([f"color_raw_{i}" for i in range(3)]), dtype=torch.float32, device=device))
        self._opacity = nn.Parameter(torch.tensor(stack(["opacity_raw"]), dtype=torch.float32, device=device))
        self._scaling = nn.Parameter(torch.tensor(stack([f"scale_2d_raw_{i}" for i in range(2)]), dtype=torch.float32, device=device))
        self._rotation = nn.Parameter(torch.tensor(stack([f"rot_{i}" for i in range(4)]), dtype=torch.float32, device=device))
        self._reset_stats(self.get_xyz.shape[0], device)
        self.topology_version = 0
        self.initialization = {"mode": "ply", "path": os.path.abspath(path), "count": self.get_xyz.shape[0]}

    @torch.no_grad()
    def add_densification_stats(self, contributing_indices=None, contributing_weights=None) -> None:
        if self._xyz.grad is not None:
            gradient = torch.linalg.vector_norm(self._xyz.grad, dim=-1, keepdim=True)
            finite = torch.isfinite(gradient)
            self.xyz_gradient_accum += torch.where(finite, gradient, torch.zeros_like(gradient))
            self.denom += finite.to(self.denom.dtype)
        if contributing_indices is not None:
            indices = contributing_indices.reshape(-1).long()
            if contributing_weights is None:
                weights = torch.ones((indices.numel(), 1), device=self.get_xyz.device)
            else:
                weights = contributing_weights.reshape(-1, 1).to(self.get_xyz)
            valid = (indices >= 0) & (indices < self.get_xyz.shape[0])
            indices, weights = indices[valid], weights[valid]
            if indices.numel():
                self.hit_weight_accum.index_add_(0, indices, weights)
                self.hit_count.index_add_(0, indices, torch.ones_like(weights))

    def _assign(self, tensors: Dict[str, nn.Parameter]) -> None:
        self._xyz = tensors["xyz"]
        self._color = tensors["color"]
        self._opacity = tensors["opacity"]
        self._scaling = tensors["scaling"]
        self._rotation = tensors["rotation"]

    def _prune_optimizer(self, keep: torch.Tensor) -> Dict[str, nn.Parameter]:
        result = {}
        for group in self.optimizer.param_groups:
            old = group["params"][0]
            state = self.optimizer.state.get(old)
            if state is not None:
                state["exp_avg"] = state["exp_avg"][keep]
                state["exp_avg_sq"] = state["exp_avg_sq"][keep]
                del self.optimizer.state[old]
            parameter = nn.Parameter(old[keep].detach().requires_grad_(True))
            group["params"][0] = parameter
            if state is not None:
                self.optimizer.state[parameter] = state
            result[group["name"]] = parameter
        return result

    def prune_points(self, prune: torch.Tensor) -> None:
        if not prune.any():
            return
        keep = ~prune
        self._assign(self._prune_optimizer(keep))
        self.xyz_gradient_accum = self.xyz_gradient_accum[keep]
        self.denom = self.denom[keep]
        self.hit_weight_accum = self.hit_weight_accum[keep]
        self.hit_count = self.hit_count[keep]
        self.topology_version += 1

    def _cat_optimizer(self, extensions: Dict[str, torch.Tensor]) -> Dict[str, nn.Parameter]:
        result = {}
        for group in self.optimizer.param_groups:
            old = group["params"][0]
            extension = extensions[group["name"]]
            state = self.optimizer.state.get(old)
            if state is not None:
                state["exp_avg"] = torch.cat((state["exp_avg"], torch.zeros_like(extension)), dim=0)
                state["exp_avg_sq"] = torch.cat((state["exp_avg_sq"], torch.zeros_like(extension)), dim=0)
                del self.optimizer.state[old]
            parameter = nn.Parameter(torch.cat((old, extension), dim=0).detach().requires_grad_(True))
            group["params"][0] = parameter
            if state is not None:
                self.optimizer.state[parameter] = state
            result[group["name"]] = parameter
        return result

    def _append(self, extensions: Dict[str, torch.Tensor]) -> None:
        count = extensions["xyz"].shape[0]
        if count == 0:
            return
        self._assign(self._cat_optimizer(extensions))
        device = self.get_xyz.device
        zeros = torch.zeros((count, 1), device=device)
        self.xyz_gradient_accum = torch.cat((self.xyz_gradient_accum, zeros), dim=0)
        self.denom = torch.cat((self.denom, zeros), dim=0)
        self.hit_weight_accum = torch.cat((self.hit_weight_accum, zeros), dim=0)
        self.hit_count = torch.cat((self.hit_count, zeros), dim=0)
        self.topology_version += 1

    def _extensions(self, selected: torch.Tensor, repeats: int = 1) -> Dict[str, torch.Tensor]:
        repeat = (repeats, 1)
        return {
            "xyz": self._xyz[selected].repeat(*repeat),
            "color": self._color[selected].repeat(*repeat),
            "opacity": self._opacity[selected].repeat(*repeat),
            "scaling": self._scaling[selected].repeat(*repeat),
            "rotation": self._rotation[selected].repeat(*repeat),
        }

    @torch.no_grad()
    def densify_and_prune(
        self,
        max_grad: float,
        min_opacity: float,
        extent: float,
        allow_unhit_prune: bool = False,
        min_hit_count: float = 1.0,
    ) -> None:
        if self.optimizer is None:
            raise RuntimeError("Reflection optimizer is not initialized")
        mean_grad = torch.nan_to_num(self.xyz_gradient_accum / self.denom.clamp_min(1.0), nan=0.0)
        scale_max = self.get_scaling.max(dim=1).values
        high_grad = mean_grad[:, 0] >= max_grad
        clone = high_grad & (scale_max <= self.percent_dense * extent)
        split = high_grad & (scale_max > self.percent_dense * extent)

        clone_ext = self._extensions(clone)
        split_count = 2
        split_ext = self._extensions(split, split_count)
        split_scales = self.get_scaling[split].repeat(split_count, 1)
        split_matrices = quaternion_to_rotation_matrix(self.get_rotation[split]).repeat(split_count, 1, 1)
        split_centers = self.get_xyz[split].repeat(split_count, 1)

        self._append(clone_ext)
        if split.any():
            local = torch.cat(
                (torch.randn_like(split_scales) * split_scales, torch.zeros_like(split_scales[:, :1])),
                dim=-1,
            )
            split_ext["xyz"] = torch.bmm(split_matrices, local.unsqueeze(-1)).squeeze(-1) + split_centers
            split_ext["scaling"] = torch.log(split_scales / (0.8 * split_count))
            self._append(split_ext)
            original_split = torch.cat(
                (split, torch.zeros(self.get_xyz.shape[0] - split.shape[0], dtype=torch.bool, device=split.device))
            )
            self.prune_points(original_split)

        prune = (self.get_opacity < min_opacity).squeeze(-1)
        if allow_unhit_prune:
            prune |= self.hit_count[:, 0] < min_hit_count
        self.prune_points(prune)
        self._reset_stats(self.get_xyz.shape[0], self.get_xyz.device)
