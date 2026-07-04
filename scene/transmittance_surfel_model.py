"""Independent Stage D Transmittance 2D Gaussian surfel field."""

from __future__ import annotations

import torch
from torch import nn

from geometry.cuboid_space import CuboidSpace, INSIDE, SUPPORT_STRICT_INSIDE
from scene.reflection_surfel_model import ReflectionSurfelModel, _raw_from_unit
from utils.general_utils import get_expon_lr_func


class TransmittanceSurfelModel(ReflectionSurfelModel):
    """T field with parameters, optimizer, schedule, and topology separate from D/R."""

    model_type = "transmittance_surfel"

    def __init__(self):
        super().__init__()
        self.position_parameterization = "world"
        self.scaling_parameterization = "exp_v1"
        self._projected_active_scaling = torch.empty(0)
        self.cuboid_space = None

    @property
    def get_xyz(self):
        if self.position_parameterization == "cuboid_inside_support_sigmoid_v2":
            if self.cuboid_space is None:
                raise RuntimeError("support-safe cuboid T parameterization is missing its space")
            return self.cuboid_space.decode_inside_support_latent(
                self._xyz, self.get_rotation, self.get_scaling, sigma=3.0,
            )
        if self.position_parameterization == "cuboid_inside_sigmoid_v1":
            if self.cuboid_space is None:
                raise RuntimeError("cuboid T parameterization is missing its space contract")
            return self.cuboid_space.decode_inside_latent(self._xyz)
        return self._xyz

    @property
    def get_scaling(self):
        scaling = super().get_scaling
        if self.position_parameterization == "cuboid_inside_support_sigmoid_v2":
            if self.cuboid_space is None:
                raise RuntimeError("support-safe cuboid T scaling is missing its space")
            bounded = self.cuboid_space.constrain_support_scaling(
                self.get_rotation, scaling, sigma=3.0,
            )
            if self.scaling_parameterization == "cuboid_support_projected_cap_v2":
                if self._projected_active_scaling.shape != bounded.shape:
                    raise RuntimeError("projected T active-scale state is missing or mismatched")
                # Preserve the exact checkpointed forward value while retaining
                # the bounded raw/rotation derivative through a straight-through
                # correction.  The buffer is refreshed after every optimizer step.
                active = self._projected_active_scaling.to(bounded)
                return bounded + (active - bounded).detach()
            return bounded
        return scaling

    @property
    def raytrace_scaling_raw(self):
        """Encode the actual T forward scale for the shared raytrace decoder."""
        if self.position_parameterization == "cuboid_inside_support_sigmoid_v2":
            return torch.log(
                self.get_scaling.clamp_min(torch.finfo(self._scaling.dtype).tiny)
            )
        return super().raytrace_scaling_raw

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
        self.scaling_parameterization = "exp_v1"
        self._projected_active_scaling = torch.empty(0, device=device, dtype=dtype)
        self._reset_stats(count, device)
        self.topology_version = 0
        self.initialization = {
            "mode": "random_cuboid_inside_safe",
            "count": int(count), "seed": int(seed),
            "position_parameterization": self.position_parameterization,
            "scaling_parameterization": "exp_v1",
            "cuboid_space": self.cuboid_space.metadata(),
            "initial_opacity": 0.01, "initial_color": 0.5,
            "initial_scale": float(scale.detach().cpu()),
        }
        self.assert_strictly_inside()

    def create_random_support_safe_inside_cuboid(
        self, cuboid_space: CuboidSpace, count: int, seed: int,
    ) -> None:
        """Initialize T so its complete ray-tracer 3-sigma support is inside-safe."""
        if count <= 0:
            raise ValueError("transmittance_init_count must be positive")
        self.cuboid_space = cuboid_space.to(
            device=cuboid_space.axes.device, dtype=cuboid_space.axes.dtype
        )
        device, dtype = self.cuboid_space.axes.device, self.cuboid_space.axes.dtype
        generator = torch.Generator(device=device); generator.manual_seed(int(seed))
        rotation = torch.randn((count, 4), generator=generator, device=device, dtype=dtype)
        rotation = torch.nn.functional.normalize(rotation, dim=-1, eps=1e-12)
        safe_extent = self.cuboid_space.upper - self.cuboid_space.lower \
            - 2 * float(self.cuboid_space.interface_margin + 4 * self.cuboid_space.epsilon)
        spacing = torch.pow(
            safe_extent.prod().clamp_min(torch.finfo(dtype).eps) / float(count), 1.0 / 3.0
        )
        # The initial support must fit for every orientation; keep the same
        # conservative density scale while accounting for the 3-sigma cutoff.
        scale = (0.12 * spacing).clamp_min(torch.finfo(dtype).eps)
        scaling = torch.log(torch.full((count, 2), scale, device=device, dtype=dtype))
        unit = torch.rand((count, 3), generator=generator, device=device, dtype=dtype)
        latent = torch.logit(unit.clamp(1e-6, 1.0 - 1e-6))
        self._xyz = nn.Parameter(latent.requires_grad_(True))
        self._rotation = nn.Parameter(rotation.requires_grad_(True))
        self._scaling = nn.Parameter(scaling.requires_grad_(True))
        self._opacity = nn.Parameter(
            _raw_from_unit(torch.full((count, 1), 0.01, device=device, dtype=dtype))
        )
        self._color = nn.Parameter(
            _raw_from_unit(torch.full((count, 3), 0.5, device=device, dtype=dtype))
        )
        self.position_parameterization = "cuboid_inside_support_sigmoid_v2"
        self.scaling_parameterization = "cuboid_support_uniform_cap_v1"
        self._projected_active_scaling = torch.empty(0, device=device, dtype=dtype)
        self._reset_stats(count, device); self.topology_version = 0
        self.initialization = {
            "mode": "random_strict_inside", "count": int(count), "seed": int(seed),
            "position_parameterization": self.position_parameterization,
            "scaling_parameterization": "cuboid_support_uniform_cap_v1",
            "support_sigma": 3.0, "cuboid_space": self.cuboid_space.metadata(),
            "initial_opacity": 0.01, "initial_color": 0.5,
            "initial_scale": float(scale.detach().cpu()),
        }
        self.assert_strictly_inside()

    def create_transferred_from_diffuse(
        self,
        diffuse,
        selected_indices: torch.Tensor,
        cuboid_space: CuboidSpace,
        count: int,
        seed: int,
        *,
        opacity_scale: float = 0.25,
        opacity_min: float = 0.005,
        opacity_max: float = 0.05,
        selection_metadata: dict | None = None,
    ) -> None:
        """Copy audited D geometry/color into fresh T and deterministically fill."""
        self.create_random_support_safe_inside_cuboid(cuboid_space, count, seed)
        selected = torch.as_tensor(selected_indices, device=diffuse.get_xyz.device, dtype=torch.long)
        selected = selected[:count]
        copied = int(selected.numel())
        if copied:
            with torch.no_grad():
                rotation = diffuse._rotation[selected].detach()
                scaling = diffuse._scaling[selected].detach()
                world = diffuse.get_xyz[selected].detach()
                active_scale = torch.exp(scaling)
                latent = self.cuboid_space.encode_inside_support_world(
                    world, rotation, active_scale, sigma=3.0,
                )
                self._rotation[:copied].copy_(rotation)
                self._scaling[:copied].copy_(scaling)
                self._xyz[:copied].copy_(latent)
                self._color[:copied].copy_(diffuse._base_color[selected].detach())
                source_opacity = torch.sigmoid(diffuse._opacity[selected].detach())
                calibrated = (source_opacity * float(opacity_scale)).clamp(
                    float(opacity_min), float(opacity_max)
                )
                self._opacity[:copied].copy_(_raw_from_unit(calibrated))
        self.initialization.update({
            "mode": "transferred_d_inside",
            "transferred_count": copied,
            "random_fill_count": int(count - copied),
            "fill_strategy": "random_strict_inside_same_seed_v1",
            "opacity_recalibration": {
                "schema": "source_quarter_capped_v1",
                "scale": float(opacity_scale), "minimum": float(opacity_min),
                "maximum": float(opacity_max),
            },
            "selection": dict(selection_metadata or {}),
        })
        self._reset_stats(count, self._xyz.device)
        self.assert_strictly_inside()

    def assert_strictly_inside(self) -> None:
        if self.position_parameterization not in (
            "cuboid_inside_sigmoid_v1", "cuboid_inside_support_sigmoid_v2",
        ):
            return
        if self.position_parameterization == "cuboid_inside_support_sigmoid_v2":
            classes = self.cuboid_space.classify_support(
                self.get_xyz.detach(), self.get_rotation.detach(),
                self.get_scaling.detach(), sigma=3.0,
            )
            if not bool((classes == SUPPORT_STRICT_INSIDE).all()):
                raise RuntimeError("T 3-sigma support left the cuboid strict inside-safe region")
        else:
            classes = self.cuboid_space.classify(self.get_xyz.detach())
            if not bool((classes == INSIDE).all()):
                raise RuntimeError("T left the cuboid strict inside-safe region")

    def capture(self):
        state = super().capture()
        state["position_parameterization"] = self.position_parameterization
        state["scaling_parameterization"] = self.scaling_parameterization
        if self.scaling_parameterization == "cuboid_support_projected_cap_v2":
            state["projected_active_scaling"] = self._projected_active_scaling
        state["cuboid_space"] = (
            self.cuboid_space.metadata() if self.cuboid_space is not None else None
        )
        return state

    def restore(self, state, args) -> None:
        mode = state.get("position_parameterization", "world")
        scaling_mode = state.get("scaling_parameterization", "exp_v1")
        if mode == "cuboid_inside_support_sigmoid_v2" \
                and scaling_mode not in (
                    "cuboid_support_uniform_cap_v1",
                    "cuboid_support_projected_cap_v2",
                ):
            raise ValueError(
                "support-safe T checkpoint has incompatible scale parameterization"
            )
        self.position_parameterization = mode
        self.scaling_parameterization = scaling_mode
        projected = state.get("projected_active_scaling")
        if scaling_mode == "cuboid_support_projected_cap_v2":
            if not torch.is_tensor(projected) or projected.shape != state["scaling_2d"].shape:
                raise ValueError("projected T checkpoint is missing exact active scale")
            self._projected_active_scaling = projected.detach().clone()
        else:
            self._projected_active_scaling = torch.empty(
                0, device=state["scaling_2d"].device, dtype=state["scaling_2d"].dtype,
            )
        metadata = state.get("cuboid_space")
        if mode in ("cuboid_inside_sigmoid_v1", "cuboid_inside_support_sigmoid_v2"):
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

    @torch.no_grad()
    def project_raw_scaling_to_active_(self, *, migrate_legacy: bool = False) -> dict:
        """Canonicalize support-safe raw scale and only its affected Adam rows.

        The cuboid cap remains a fail-safe in the forward path, but projected-v2
        checkpoints require the stored raw scale to equal that active scale after
        every optimizer update.  Multiplying both tangent axes by the same cap
        factor preserves anisotropy and the complete forward geometry.
        """
        if self.position_parameterization != "cuboid_inside_support_sigmoid_v2" \
                or self.cuboid_space is None:
            raise RuntimeError("T scale projection requires support-safe cuboid mode")
        allowed = {
            "cuboid_support_uniform_cap_v1", "cuboid_support_projected_cap_v2",
        }
        if self.scaling_parameterization not in allowed:
            raise RuntimeError("T scale projection received an incompatible parameterization")
        if self.optimizer is None:
            raise RuntimeError("T scale projection requires initialized optimizer state")

        raw_before = torch.exp(self._scaling)
        active_before = self.cuboid_space.constrain_support_scaling(
            self.get_rotation, raw_before, sigma=3.0,
        )
        ratio_before = active_before / raw_before.clamp_min(torch.finfo(raw_before.dtype).tiny)
        factor_before = ratio_before.amin(dim=-1)
        cap_tolerance = (
            2e-6 if self.scaling_parameterization == "cuboid_support_projected_cap_v2"
            else 1e-7
        )
        affected = factor_before.lt(1.0 - cap_tolerance)
        affected_count = int(affected.sum())

        state = self.optimizer.state.get(self._scaling, {})
        state_rows_zeroed = []
        if affected_count:
            self._scaling[affected] = torch.log(active_before[affected])
            for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                value = state.get(name)
                if torch.is_tensor(value):
                    if value.shape != self._scaling.shape:
                        raise RuntimeError(f"T scaling Adam {name} shape mismatch")
                    value[affected] = 0
                    state_rows_zeroed.append(name)

        raw_after = torch.exp(self._scaling)
        self.scaling_parameterization = "cuboid_support_projected_cap_v2"
        self._projected_active_scaling = active_before.detach().clone()
        active_after = self.get_scaling.detach()
        ratio_after = active_after / raw_after.clamp_min(torch.finfo(raw_after.dtype).tiny)
        factor_after = ratio_after.amin(dim=-1)
        difference = (active_after - raw_after).abs()
        tolerance = 8.0 * torch.finfo(raw_after.dtype).eps
        if not torch.isfinite(raw_after).all() or not torch.isfinite(active_after).all():
            raise FloatingPointError("T scale projection produced non-finite scale")
        if float(difference.max()) > tolerance * max(1.0, float(active_after.abs().max())):
            raise RuntimeError("T scale projection failed raw/active consistency")

        self.assert_strictly_inside()
        step = state.get("step")
        step_preserved = (
            True if step is None else (
                bool(torch.isfinite(step).all()) if torch.is_tensor(step) else True
            )
        )
        return {
            "schema": "cuboid_support_projected_cap_v2",
            "legacy_migration": bool(migrate_legacy),
            "affected_count": affected_count,
            "affected_fraction": affected_count / max(int(self._scaling.shape[0]), 1),
            "minimum_factor_before": float(factor_before.min()),
            "minimum_factor_after": float(factor_after.min()),
            "raw_scale_max_before": float(raw_before.max()),
            "active_scale_max_before": float(active_before.max()),
            "raw_scale_max_after": float(raw_after.max()),
            "active_scale_max_after": float(active_after.max()),
            "raw_active_max_abs_after": float(difference.max()),
            "adam_state_rows_zeroed": state_rows_zeroed,
            "adam_step_preserved": step_preserved,
        }

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
