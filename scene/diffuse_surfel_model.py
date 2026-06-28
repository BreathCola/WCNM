"""Stage A diffuse 2D Gaussian surfel model.

The original :mod:`scene.gaussian_model` remains untouched and is still used by
the default ``3dgs`` mode.
"""

import json
import os

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from torch import nn

from utils.general_utils import get_expon_lr_func, inverse_sigmoid
from utils.graphics_utils import BasicPointCloud
from utils.surfel_utils import surfel_frame
from utils.system_utils import mkdir_p

try:
    from simple_knn._C import distCUDA2
except ImportError:
    distCUDA2 = None


def _raw_from_unit(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return inverse_sigmoid(value.clamp(eps, 1.0 - eps))


def _quaternion_from_normals(normals: torch.Tensor) -> torch.Tensor:
    """Construct scalar-first quaternions rotating local +Z onto normals."""
    normals = torch.nn.functional.normalize(normals, dim=-1, eps=1e-12)
    q = torch.stack(
        (1.0 + normals[:, 2], -normals[:, 1], normals[:, 0], torch.zeros_like(normals[:, 0])),
        dim=-1,
    )
    opposite = normals[:, 2] < -0.9999
    q[opposite] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=normals.device, dtype=normals.dtype)
    return torch.nn.functional.normalize(q, dim=-1, eps=1e-12)


class DiffuseSurfelModel:
    checkpoint_version = 1
    model_type = "surfel"

    def __init__(self, roughness_min: float = 0.03, optimizer_type: str = "default"):
        if not 0.0 <= roughness_min < 1.0:
            raise ValueError("roughness_min must be in [0, 1)")
        if optimizer_type != "default":
            raise ValueError("DiffuseSurfelModel currently supports optimizer_type='default' only")
        self.roughness_min = float(roughness_min)
        self.optimizer_type = optimizer_type
        self.active_sh_degree = 0
        self.max_sh_degree = 0
        self._xyz = torch.empty(0)
        self._rotation = torch.empty(0)
        self._scaling = torch.empty(0)
        self._opacity = torch.empty(0)
        self._base_color = torch.empty(0)
        self._roughness = torch.empty(0)
        self._f0 = torch.empty(0)
        self._ks = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.exposure_optimizer = None
        self.percent_dense = 0.0
        self.spatial_lr_scale = 0.0
        self.exposure_mapping = {}
        self.pretrained_exposures = None

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
    def get_base_color(self):
        return torch.sigmoid(self._base_color)

    @property
    def base_color(self):
        return self.get_base_color

    @property
    def base_color_raw(self):
        return self._base_color

    @property
    def get_roughness(self):
        return self.roughness_min + (1.0 - self.roughness_min) * torch.sigmoid(self._roughness)

    @property
    def roughness(self):
        return self.get_roughness

    @property
    def roughness_raw(self):
        return self._roughness

    @property
    def get_f0(self):
        return torch.sigmoid(self._f0)

    @property
    def f0(self):
        return self.get_f0

    @property
    def f0_raw(self):
        return self._f0

    @property
    def get_ks(self):
        return torch.sigmoid(self._ks)

    @property
    def ks(self):
        return self.get_ks

    @property
    def ks_raw(self):
        return self._ks

    @property
    def get_normal(self):
        return surfel_frame(self.get_rotation, self.get_scaling)[2]

    @property
    def normal(self):
        return self.get_normal

    @property
    def tangent_u(self):
        return surfel_frame(self.get_rotation, self.get_scaling)[0]

    @property
    def tangent_v(self):
        return surfel_frame(self.get_rotation, self.get_scaling)[1]

    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is not None:
            return self.pretrained_exposures[image_name]
        return self._exposure[self.exposure_mapping[image_name]]

    def oneupSHdegree(self):
        return None

    def create_from_pcd(self, pcd: BasicPointCloud, cam_infos, spatial_lr_scale: float):
        if distCUDA2 is None:
            raise RuntimeError("simple-knn must be installed to initialize DiffuseSurfelModel")
        self.spatial_lr_scale = spatial_lr_scale
        points = torch.as_tensor(np.asarray(pcd.points), dtype=torch.float32, device="cuda")
        colors = torch.as_tensor(np.asarray(pcd.colors), dtype=torch.float32, device="cuda").clamp(0, 1)
        print("Number of diffuse surfels at initialisation:", points.shape[0])

        dist2 = torch.clamp_min(distCUDA2(points), 1e-7)
        scaling = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 2)

        pcd_normals = torch.as_tensor(np.asarray(pcd.normals), dtype=torch.float32, device="cuda")
        valid_normals = torch.linalg.vector_norm(pcd_normals, dim=-1) > 1e-6
        rotations = torch.randn((points.shape[0], 4), device="cuda")
        rotations = torch.nn.functional.normalize(rotations, dim=-1)
        if valid_normals.any():
            rotations[valid_normals] = _quaternion_from_normals(pcd_normals[valid_normals])

        opacity = _raw_from_unit(torch.full((points.shape[0], 1), 0.1, device="cuda"))
        rough_unit = (0.5 - self.roughness_min) / (1.0 - self.roughness_min)
        roughness = _raw_from_unit(torch.full((points.shape[0], 1), rough_unit, device="cuda"))
        f0 = _raw_from_unit(torch.full((points.shape[0], 3), 0.04, device="cuda"))
        ks = _raw_from_unit(torch.full((points.shape[0], 1), 0.1, device="cuda"))

        self._xyz = nn.Parameter(points.requires_grad_(True))
        self._rotation = nn.Parameter(rotations.requires_grad_(True))
        self._scaling = nn.Parameter(scaling.requires_grad_(True))
        self._opacity = nn.Parameter(opacity.requires_grad_(True))
        self._base_color = nn.Parameter(_raw_from_unit(colors).requires_grad_(True))
        self._roughness = nn.Parameter(roughness.requires_grad_(True))
        self._f0 = nn.Parameter(f0.requires_grad_(True))
        self._ks = nn.Parameter(ks.requires_grad_(True))
        self.max_radii2D = torch.zeros(points.shape[0], device="cuda")

        self.exposure_mapping = {cam.image_name: idx for idx, cam in enumerate(cam_infos)}
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))
        self.pretrained_exposures = None

    def training_setup(self, training_args):
        device = self.get_xyz.device
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        material_lr = getattr(training_args, "material_lr", training_args.feature_lr)
        groups = [
            {"params": [self._xyz], "lr": training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {"params": [self._base_color], "lr": training_args.feature_lr, "name": "base_color"},
            {"params": [self._opacity], "lr": training_args.opacity_lr, "name": "opacity"},
            {"params": [self._scaling], "lr": training_args.scaling_lr, "name": "scaling"},
            {"params": [self._rotation], "lr": training_args.rotation_lr, "name": "rotation"},
            {"params": [self._roughness], "lr": material_lr, "name": "roughness"},
            {"params": [self._f0], "lr": material_lr, "name": "f0"},
            {"params": [self._ks], "lr": material_lr, "name": "ks"},
        ]
        self.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)
        self.exposure_optimizer = torch.optim.Adam([self._exposure])
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )
        self.exposure_scheduler_args = get_expon_lr_func(
            training_args.exposure_lr_init,
            training_args.exposure_lr_final,
            lr_delay_steps=training_args.exposure_lr_delay_steps,
            lr_delay_mult=training_args.exposure_lr_delay_mult,
            max_steps=training_args.iterations,
        )

    def update_learning_rate(self, iteration):
        if self.pretrained_exposures is None:
            for group in self.exposure_optimizer.param_groups:
                group["lr"] = self.exposure_scheduler_args(iteration)
        for group in self.optimizer.param_groups:
            if group["name"] == "xyz":
                group["lr"] = self.xyz_scheduler_args(iteration)
                return group["lr"]
        return None

    def capture(self):
        return {
            "checkpoint_version": self.checkpoint_version,
            "model_type": self.model_type,
            "roughness_min": self.roughness_min,
            "xyz": self._xyz,
            "rotation": self._rotation,
            "scaling_2d": self._scaling,
            "opacity_raw": self._opacity,
            "base_color_raw": self._base_color,
            "roughness_raw": self._roughness,
            "f0_raw": self._f0,
            "ks_raw": self._ks,
            "exposure": self._exposure,
            "exposure_mapping": self.exposure_mapping,
            "max_radii2D": self.max_radii2D,
            "xyz_gradient_accum": self.xyz_gradient_accum,
            "denom": self.denom,
            "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
            "exposure_optimizer": self.exposure_optimizer.state_dict() if self.exposure_optimizer is not None else None,
            "spatial_lr_scale": self.spatial_lr_scale,
        }

    def restore(self, state, training_args):
        if state.get("model_type") != self.model_type:
            raise ValueError("checkpoint is not a diffuse surfel checkpoint")
        if state.get("checkpoint_version") != self.checkpoint_version:
            raise ValueError(f"unsupported surfel checkpoint version: {state.get('checkpoint_version')}")
        self.roughness_min = float(state["roughness_min"])
        self._xyz = state["xyz"]
        self._rotation = state["rotation"]
        self._scaling = state["scaling_2d"]
        self._opacity = state["opacity_raw"]
        self._base_color = state["base_color_raw"]
        self._roughness = state["roughness_raw"]
        self._f0 = state["f0_raw"]
        self._ks = state["ks_raw"]
        self._exposure = state["exposure"]
        self.exposure_mapping = state["exposure_mapping"]
        self.max_radii2D = state["max_radii2D"]
        self.spatial_lr_scale = state["spatial_lr_scale"]
        self.training_setup(training_args)
        self.xyz_gradient_accum = state["xyz_gradient_accum"]
        self.denom = state["denom"]
        if state["optimizer"] is not None:
            self.optimizer.load_state_dict(state["optimizer"])
        if state["exposure_optimizer"] is not None:
            self.exposure_optimizer.load_state_dict(state["exposure_optimizer"])

    def construct_list_of_attributes(self):
        attrs = ["x", "y", "z", "nx", "ny", "nz"]
        attrs += [f"base_color_raw_{idx}" for idx in range(3)]
        attrs += ["opacity_raw"]
        attrs += [f"scale_2d_raw_{idx}" for idx in range(2)]
        attrs += [f"rot_{idx}" for idx in range(4)]
        attrs += ["roughness_raw"]
        attrs += [f"f0_raw_{idx}" for idx in range(3)]
        attrs += ["ks_raw"]
        return attrs

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        arrays = [
            self._xyz.detach().cpu().numpy(),
            self.get_normal.detach().cpu().numpy(),
            self._base_color.detach().cpu().numpy(),
            self._opacity.detach().cpu().numpy(),
            self._scaling.detach().cpu().numpy(),
            self._rotation.detach().cpu().numpy(),
            self._roughness.detach().cpu().numpy(),
            self._f0.detach().cpu().numpy(),
            self._ks.detach().cpu().numpy(),
        ]
        dtype = [(name, "f4") for name in self.construct_list_of_attributes()]
        elements = np.empty(self.get_xyz.shape[0], dtype=dtype)
        elements[:] = list(map(tuple, np.concatenate(arrays, axis=1)))
        PlyData([PlyElement.describe(elements, "vertex")]).write(path)

    def load_ply(self, path, use_train_test_exp=False):
        ply = PlyData.read(path)
        vertex = ply.elements[0]

        def stack(names):
            return np.stack([np.asarray(vertex[name]) for name in names], axis=1)

        device = "cuda"
        self._xyz = nn.Parameter(torch.tensor(stack(["x", "y", "z"]), dtype=torch.float32, device=device))
        self._base_color = nn.Parameter(torch.tensor(stack([f"base_color_raw_{i}" for i in range(3)]), dtype=torch.float32, device=device))
        self._opacity = nn.Parameter(torch.tensor(stack(["opacity_raw"]), dtype=torch.float32, device=device))
        self._scaling = nn.Parameter(torch.tensor(stack([f"scale_2d_raw_{i}" for i in range(2)]), dtype=torch.float32, device=device))
        self._rotation = nn.Parameter(torch.tensor(stack([f"rot_{i}" for i in range(4)]), dtype=torch.float32, device=device))
        self._roughness = nn.Parameter(torch.tensor(stack(["roughness_raw"]), dtype=torch.float32, device=device))
        self._f0 = nn.Parameter(torch.tensor(stack([f"f0_raw_{i}" for i in range(3)]), dtype=torch.float32, device=device))
        self._ks = nn.Parameter(torch.tensor(stack(["ks_raw"]), dtype=torch.float32, device=device))
        self.max_radii2D = torch.zeros(self.get_xyz.shape[0], device=device)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r", encoding="utf-8") as handle:
                    values = json.load(handle)
                self.pretrained_exposures = {
                    name: torch.tensor(value, dtype=torch.float32, device=device) for name, value in values.items()
                }

    def reset_opacity(self):
        raw = _raw_from_unit(torch.minimum(self.get_opacity, torch.full_like(self.get_opacity, 0.01)))
        tensors = self.replace_tensor_to_optimizer(raw, "opacity")
        self._opacity = tensors["opacity"]

    def replace_tensor_to_optimizer(self, tensor, name):
        result = {}
        for group in self.optimizer.param_groups:
            if group["name"] != name:
                continue
            old = group["params"][0]
            state = self.optimizer.state.get(old)
            if state is not None:
                state["exp_avg"] = torch.zeros_like(tensor)
                state["exp_avg_sq"] = torch.zeros_like(tensor)
                del self.optimizer.state[old]
            group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
            if state is not None:
                self.optimizer.state[group["params"][0]] = state
            result[name] = group["params"][0]
        return result

    def _prune_optimizer(self, keep):
        result = {}
        for group in self.optimizer.param_groups:
            old = group["params"][0]
            state = self.optimizer.state.get(old)
            if state is not None:
                state["exp_avg"] = state["exp_avg"][keep]
                state["exp_avg_sq"] = state["exp_avg_sq"][keep]
                del self.optimizer.state[old]
            group["params"][0] = nn.Parameter(old[keep].requires_grad_(True))
            if state is not None:
                self.optimizer.state[group["params"][0]] = state
            result[group["name"]] = group["params"][0]
        return result

    def prune_points(self, prune):
        keep = ~prune
        tensors = self._prune_optimizer(keep)
        self._assign_optimizable(tensors)
        self.xyz_gradient_accum = self.xyz_gradient_accum[keep]
        self.denom = self.denom[keep]
        self.max_radii2D = self.max_radii2D[keep]
        if hasattr(self, "tmp_radii"):
            self.tmp_radii = self.tmp_radii[keep]

    def cat_tensors_to_optimizer(self, extensions):
        result = {}
        for group in self.optimizer.param_groups:
            old = group["params"][0]
            extension = extensions[group["name"]]
            state = self.optimizer.state.get(old)
            if state is not None:
                state["exp_avg"] = torch.cat((state["exp_avg"], torch.zeros_like(extension)), dim=0)
                state["exp_avg_sq"] = torch.cat((state["exp_avg_sq"], torch.zeros_like(extension)), dim=0)
                del self.optimizer.state[old]
            group["params"][0] = nn.Parameter(torch.cat((old, extension), dim=0).requires_grad_(True))
            if state is not None:
                self.optimizer.state[group["params"][0]] = state
            result[group["name"]] = group["params"][0]
        return result

    def _assign_optimizable(self, tensors):
        self._xyz = tensors["xyz"]
        self._base_color = tensors["base_color"]
        self._opacity = tensors["opacity"]
        self._scaling = tensors["scaling"]
        self._rotation = tensors["rotation"]
        self._roughness = tensors["roughness"]
        self._f0 = tensors["f0"]
        self._ks = tensors["ks"]

    def densification_postfix(self, extensions, new_radii):
        self._assign_optimizable(self.cat_tensors_to_optimizer(extensions))
        self.tmp_radii = torch.cat((self.tmp_radii, new_radii))
        device = self.get_xyz.device
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        self.max_radii2D = torch.zeros(self.get_xyz.shape[0], device=device)

    def _selected_extensions(self, selected, repeats=1):
        repeat2 = (repeats, 1)
        return {
            "base_color": self._base_color[selected].repeat(*repeat2),
            "opacity": self._opacity[selected].repeat(*repeat2),
            "scaling": self._scaling[selected].repeat(*repeat2),
            "rotation": self._rotation[selected].repeat(*repeat2),
            "roughness": self._roughness[selected].repeat(*repeat2),
            "f0": self._f0[selected].repeat(*repeat2),
            "ks": self._ks[selected].repeat(*repeat2),
        }

    def densify_and_split(self, grads, threshold, extent, count=2):
        initial_count = self.get_xyz.shape[0]
        padded = torch.zeros(initial_count, device=self.get_xyz.device)
        padded[: grads.shape[0]] = grads.squeeze()
        selected = (padded >= threshold) & (self.get_scaling.max(dim=1).values > self.percent_dense * extent)
        scales = self.get_scaling[selected].repeat(count, 1)
        tangent_samples = torch.normal(torch.zeros_like(scales), scales)
        local_samples = torch.cat((tangent_samples, torch.zeros_like(tangent_samples[:, :1])), dim=-1)
        matrices = self._rotation_matrices(self.get_rotation[selected]).repeat(count, 1, 1)
        xyz = torch.bmm(matrices, local_samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected].repeat(count, 1)
        extensions = self._selected_extensions(selected, count)
        extensions["xyz"] = xyz
        extensions["scaling"] = torch.log(scales / (0.8 * count))
        new_radii = self.tmp_radii[selected].repeat(count)
        self.densification_postfix(extensions, new_radii)
        new_count = count * int(selected.sum().item())
        prune = torch.cat((selected, torch.zeros(new_count, device=selected.device, dtype=torch.bool)))
        self.prune_points(prune)

    @staticmethod
    def _rotation_matrices(rotation):
        from utils.surfel_utils import quaternion_to_rotation_matrix

        return quaternion_to_rotation_matrix(rotation)

    def densify_and_clone(self, grads, threshold, extent):
        selected = (torch.linalg.vector_norm(grads, dim=-1) >= threshold) & (
            self.get_scaling.max(dim=1).values <= self.percent_dense * extent
        )
        extensions = self._selected_extensions(selected)
        extensions["xyz"] = self._xyz[selected]
        self.densification_postfix(extensions, self.tmp_radii[selected])

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = torch.nan_to_num(self.xyz_gradient_accum / self.denom, nan=0.0)
        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)
        prune = (self.get_opacity < min_opacity).squeeze(-1)
        if max_screen_size:
            prune |= self.max_radii2D > max_screen_size
            prune |= self.get_scaling.max(dim=1).values > 0.1 * extent
        self.prune_points(prune)
        self.tmp_radii = None
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_points, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.linalg.vector_norm(
            viewspace_points.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1
