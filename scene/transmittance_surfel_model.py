"""Independent Stage D Transmittance 2D Gaussian surfel field."""

import torch
from torch import nn

from geometry.cuboid_space import CuboidSpace, INSIDE
from scene.reflection_surfel_model import ReflectionSurfelModel, _raw_from_unit
from utils.general_utils import get_expon_lr_func


class TransmittanceSurfelModel(ReflectionSurfelModel):
    """T field with parameters, optimizer, schedule, and topology separate from D/R."""

    model_type = "transmittance_surfel"

    def __init__(self):
        super().__init__()
        self.position_parameterization = "world"
        self.cuboid_space = None

    @property
    def get_xyz(self):
        if self.position_parameterization == "cuboid_inside_sigmoid_v1":
            if self.cuboid_space is None:
                raise RuntimeError("cuboid T parameterization is missing its space contract")
            return self.cuboid_space.decode_inside_latent(self._xyz)
        return self._xyz

    def create_random_inside_cuboid(
        self, cuboid_space: CuboidSpace, count: int, seed: int,
    ) -> None:
        if count <= 0:
            raise ValueError("transmittance_init_count must be positive")
        self.cuboid_space = cuboid_space.to(
            device=cuboid_space.axes.device, dtype=cuboid_space.axes.dtype
        )
        device, dtype = self.cuboid_space.axes.device, self.cuboid_space.axes.dtype
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        unit = torch.rand((count, 3), generator=generator, device=device, dtype=dtype)
        unit = unit.clamp(1e-6, 1.0 - 1e-6)
        latent = torch.logit(unit)
        rotation = torch.randn((count, 4), generator=generator, device=device, dtype=dtype)
        rotation = torch.nn.functional.normalize(rotation, dim=-1, eps=1e-12)
        padding = float(self.cuboid_space.interface_margin + 4 * self.cuboid_space.epsilon)
        safe_extent = self.cuboid_space.upper - self.cuboid_space.lower - 2 * padding
        spacing = torch.pow(
            safe_extent.prod().clamp_min(torch.finfo(dtype).eps) / float(count), 1.0 / 3.0
        )
        scale = (0.5 * spacing).clamp_min(torch.finfo(dtype).eps)
        self._xyz = nn.Parameter(latent.requires_grad_(True))
        self._rotation = nn.Parameter(rotation.requires_grad_(True))
        self._scaling = nn.Parameter(torch.log(torch.full((count, 2), scale, device=device, dtype=dtype)))
        self._opacity = nn.Parameter(_raw_from_unit(torch.full((count, 1), 0.01, device=device, dtype=dtype)))
        self._color = nn.Parameter(_raw_from_unit(torch.full((count, 3), 0.5, device=device, dtype=dtype)))
        self.position_parameterization = "cuboid_inside_sigmoid_v1"
        self._reset_stats(count, device)
        self.topology_version = 0
        self.initialization = {
            "mode": "random_cuboid_inside_safe",
            "count": int(count), "seed": int(seed),
            "position_parameterization": self.position_parameterization,
            "cuboid_space": self.cuboid_space.metadata(),
            "initial_opacity": 0.01, "initial_color": 0.5,
            "initial_scale": float(scale.detach().cpu()),
        }
        self.assert_strictly_inside()

    def assert_strictly_inside(self) -> None:
        if self.position_parameterization != "cuboid_inside_sigmoid_v1":
            return
        classes = self.cuboid_space.classify(self.get_xyz.detach())
        if not bool((classes == INSIDE).all()):
            raise RuntimeError("T left the cuboid strict inside-safe region")

    def capture(self):
        state = super().capture()
        state["position_parameterization"] = self.position_parameterization
        state["cuboid_space"] = (
            self.cuboid_space.metadata() if self.cuboid_space is not None else None
        )
        return state

    def restore(self, state, args) -> None:
        mode = state.get("position_parameterization", "world")
        self.position_parameterization = mode
        metadata = state.get("cuboid_space")
        if mode == "cuboid_inside_sigmoid_v1":
            if not isinstance(metadata, dict):
                raise ValueError("cuboid T checkpoint is missing spatial metadata")
            self.cuboid_space = CuboidSpace(
                axes=torch.tensor(metadata["axes_columns"], dtype=state["xyz"].dtype, device=state["xyz"].device),
                lower=torch.tensor(metadata["lower"], dtype=state["xyz"].dtype, device=state["xyz"].device),
                upper=torch.tensor(metadata["upper"], dtype=state["xyz"].dtype, device=state["xyz"].device),
                interface_margin=float(metadata["interface_margin"]),
                epsilon=float(metadata["epsilon"]),
                interface_margin_mode=metadata["transparent_interface_margin_mode"],
            )
        elif mode == "world":
            self.cuboid_space = None
        else:
            raise ValueError(f"unsupported T position parameterization: {mode}")
        super().restore(state, args)
        self.assert_strictly_inside()

    def training_setup(self, args) -> None:
        if self.get_xyz.numel() == 0:
            raise RuntimeError("initialize TransmittanceSurfelModel before training_setup")
        self.percent_dense = float(args.transmittance_percent_dense)
        groups = [
            {"params": [self._xyz], "lr": args.transmittance_position_lr_init, "name": "xyz"},
            {"params": [self._color], "lr": args.transmittance_color_lr, "name": "color"},
            {"params": [self._opacity], "lr": args.transmittance_opacity_lr, "name": "opacity"},
            {"params": [self._scaling], "lr": args.transmittance_scaling_lr, "name": "scaling"},
            {"params": [self._rotation], "lr": args.transmittance_rotation_lr, "name": "rotation"},
        ]
        self.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=args.transmittance_position_lr_init,
            lr_final=args.transmittance_position_lr_final,
            lr_delay_mult=args.transmittance_position_lr_delay_mult,
            max_steps=args.transmittance_position_lr_max_steps,
        )
        if self.xyz_gradient_accum.shape[0] != self.get_xyz.shape[0]:
            self._reset_stats(self.get_xyz.shape[0], self.get_xyz.device)
