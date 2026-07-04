"""Versioned cuboid-local spatial classification shared by Stage D branches."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch


SPACE_SCHEMA = "rtgs_cuboid_space_v1"
INSIDE = 0
INTERFACE = 1
OUTSIDE = 2
CLASS_NAMES = ("inside", "interface", "outside")
SUPPORT_STRICT_INSIDE = 0
SUPPORT_INTERFACE = 1
SUPPORT_STRICT_OUTSIDE = 2
SUPPORT_CROSSING = 3
SUPPORT_CLASS_NAMES = (
    "strict_inside_safe",
    "interface_margin",
    "strict_outside_safe",
    "crossing_or_ambiguous",
)


@dataclass(frozen=True)
class CuboidSpace:
    axes: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    interface_margin: float = 0.05
    epsilon: float = 1e-6
    interface_margin_mode: str = "exclude"

    def __post_init__(self):
        axes = torch.as_tensor(self.axes)
        lower = torch.as_tensor(self.lower)
        upper = torch.as_tensor(self.upper)
        if axes.shape != (3, 3) or lower.shape != (3,) or upper.shape != (3,):
            raise ValueError("cuboid axes/bounds have invalid shape")
        if not torch.isfinite(axes).all() or not torch.isfinite(lower).all() \
                or not torch.isfinite(upper).all():
            raise ValueError("cuboid axes/bounds must be finite")
        if not torch.all(upper > lower):
            raise ValueError("cuboid upper bounds must exceed lower bounds")
        gram = axes.T @ axes
        if not torch.allclose(gram, torch.eye(3, dtype=gram.dtype, device=gram.device), atol=1e-5, rtol=0):
            raise ValueError("cuboid axes must be orthonormal columns")
        if self.interface_margin <= 0 or self.epsilon <= 0:
            raise ValueError("cuboid margin and epsilon must be positive")
        if 2.0 * (self.interface_margin + 4.0 * self.epsilon) >= float((upper - lower).min()):
            raise ValueError("cuboid interface margin leaves no strict interior")
        if self.interface_margin_mode != "exclude":
            raise ValueError("only transparent_interface_margin_mode=exclude is supported")

    @classmethod
    def from_metadata(
        cls, path: Path, interface_margin: float = 0.05,
        epsilon: float = 1e-6, device=None, dtype=torch.float32,
    ):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            axes=torch.tensor(payload["axes_columns"], dtype=dtype, device=device),
            lower=torch.tensor(payload["fitted_lower"], dtype=dtype, device=device),
            upper=torch.tensor(payload["fitted_upper"], dtype=dtype, device=device),
            interface_margin=float(interface_margin), epsilon=float(epsilon),
        )

    def to(self, device=None, dtype=None):
        dtype = dtype or self.axes.dtype
        return CuboidSpace(
            self.axes.to(device=device, dtype=dtype),
            self.lower.to(device=device, dtype=dtype),
            self.upper.to(device=device, dtype=dtype),
            self.interface_margin, self.epsilon, self.interface_margin_mode,
        )

    def world_to_local(self, points: torch.Tensor) -> torch.Tensor:
        points = torch.as_tensor(points)
        if points.shape[-1] != 3:
            raise ValueError("cuboid points must end in three coordinates")
        return points @ self.axes.to(points)

    def local_to_world(self, points: torch.Tensor) -> torch.Tensor:
        points = torch.as_tensor(points)
        if points.shape[-1] != 3:
            raise ValueError("cuboid points must end in three coordinates")
        return points @ self.axes.to(points).T

    def signed_clearance(self, points: torch.Tensor) -> torch.Tensor:
        """Positive inside, negative outside, zero on a cuboid plane."""
        local = self.world_to_local(points)
        lower = self.lower.to(local)
        upper = self.upper.to(local)
        return torch.minimum(local - lower, upper - local).amin(dim=-1)

    def classify(self, points: torch.Tensor) -> torch.Tensor:
        clearance = self.signed_clearance(points)
        threshold = float(self.interface_margin + self.epsilon)
        result = torch.full_like(clearance, INTERFACE, dtype=torch.int64)
        result = torch.where(clearance > threshold, INSIDE, result)
        result = torch.where(clearance < -threshold, OUTSIDE, result)
        return result

    def masks(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        classes = self.classify(points)
        return {
            "inside": classes == INSIDE,
            "interface": classes == INTERFACE,
            "outside": classes == OUTSIDE,
        }

    def support_bounds(
        self,
        points: torch.Tensor,
        rotation: torch.Tensor,
        scaling_2d: torch.Tensor,
        sigma: float = 3.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cuboid-local bounds of each finite sigma-cutoff surfel support."""
        from utils.surfel_utils import quaternion_to_rotation_matrix

        points = torch.as_tensor(points)
        rotation = torch.as_tensor(rotation, device=points.device, dtype=points.dtype)
        scaling_2d = torch.as_tensor(scaling_2d, device=points.device, dtype=points.dtype)
        if points.ndim != 2 or points.shape[-1] != 3:
            raise ValueError("support points must have shape [N,3]")
        if rotation.shape != (points.shape[0], 4):
            raise ValueError("support rotations must have shape [N,4]")
        if scaling_2d.shape != (points.shape[0], 2):
            raise ValueError("support scales must have shape [N,2]")
        if sigma <= 0 or not torch.isfinite(points).all() \
                or not torch.isfinite(rotation).all() \
                or not torch.isfinite(scaling_2d).all() \
                or not torch.all(scaling_2d > 0):
            raise ValueError("support geometry must be finite with positive sigma/scales")
        matrix = quaternion_to_rotation_matrix(rotation)
        tangent_u = matrix[..., :, 0]
        tangent_v = matrix[..., :, 1]
        axes = self.axes.to(points)
        local = points @ axes
        local_u = tangent_u @ axes
        local_v = tangent_v @ axes
        radius = float(sigma) * (
            local_u.abs() * scaling_2d[:, 0:1]
            + local_v.abs() * scaling_2d[:, 1:2]
        )
        return local - radius, local + radius

    def classify_support(
        self,
        points: torch.Tensor,
        rotation: torch.Tensor,
        scaling_2d: torch.Tensor,
        sigma: float = 3.0,
    ) -> torch.Tensor:
        """Partition finite surfel supports into four ownership-safe classes."""
        support_lower, support_upper = self.support_bounds(
            points, rotation, scaling_2d, sigma=sigma,
        )
        lower = self.lower.to(support_lower)
        upper = self.upper.to(support_upper)
        threshold = float(self.interface_margin + self.epsilon)
        strict_inside = (
            (support_lower > lower + threshold)
            & (support_upper < upper - threshold)
        ).all(dim=-1)
        strict_outside = (
            (support_upper < lower - threshold)
            | (support_lower > upper + threshold)
        ).any(dim=-1)
        within_interface_envelope = (
            (support_lower >= lower - threshold)
            & (support_upper <= upper + threshold)
        ).all(dim=-1)
        interface = within_interface_envelope & ~strict_inside
        result = torch.full(
            (points.shape[0],), SUPPORT_CROSSING,
            dtype=torch.int64, device=points.device,
        )
        result = torch.where(strict_inside, SUPPORT_STRICT_INSIDE, result)
        result = torch.where(interface, SUPPORT_INTERFACE, result)
        result = torch.where(strict_outside, SUPPORT_STRICT_OUTSIDE, result)
        return result

    def support_masks(
        self,
        points: torch.Tensor,
        rotation: torch.Tensor,
        scaling_2d: torch.Tensor,
        sigma: float = 3.0,
    ) -> dict[str, torch.Tensor]:
        classes = self.classify_support(points, rotation, scaling_2d, sigma=sigma)
        return {
            name: classes == index for index, name in enumerate(SUPPORT_CLASS_NAMES)
        }

    def support_safe_local_bounds(
        self,
        rotation: torch.Tensor,
        scaling_2d: torch.Tensor,
        sigma: float = 3.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Center bounds that keep the complete finite support strictly inside."""
        from utils.surfel_utils import quaternion_to_rotation_matrix

        rotation = torch.as_tensor(rotation)
        scaling_2d = torch.as_tensor(
            scaling_2d, device=rotation.device, dtype=rotation.dtype,
        )
        matrix = quaternion_to_rotation_matrix(rotation)
        axes = self.axes.to(rotation)
        local_u = matrix[..., :, 0] @ axes
        local_v = matrix[..., :, 1] @ axes
        radius = float(sigma) * (
            local_u.abs() * scaling_2d[..., 0:1]
            + local_v.abs() * scaling_2d[..., 1:2]
        )
        padding = float(self.interface_margin + 4.0 * self.epsilon)
        lower = self.lower.to(rotation) + padding + radius
        upper = self.upper.to(rotation) - padding - radius
        if not torch.all(upper > lower):
            raise ValueError("surfel support is too large for the strict cuboid interior")
        return lower, upper

    def decode_inside_support_latent(
        self,
        latent: torch.Tensor,
        rotation: torch.Tensor,
        scaling_2d: torch.Tensor,
        sigma: float = 3.0,
    ) -> torch.Tensor:
        lower, upper = self.support_safe_local_bounds(rotation, scaling_2d, sigma=sigma)
        local = lower + torch.sigmoid(latent) * (upper - lower)
        return self.local_to_world(local)

    def encode_inside_support_world(
        self,
        points: torch.Tensor,
        rotation: torch.Tensor,
        scaling_2d: torch.Tensor,
        sigma: float = 3.0,
    ) -> torch.Tensor:
        local = self.world_to_local(points)
        lower, upper = self.support_safe_local_bounds(rotation, scaling_2d, sigma=sigma)
        unit = ((local - lower) / (upper - lower)).clamp(1e-6, 1.0 - 1e-6)
        return torch.logit(unit)

    def decode_inside_latent(self, latent: torch.Tensor) -> torch.Tensor:
        latent = torch.as_tensor(latent)
        if latent.shape[-1] != 3:
            raise ValueError("T position latent must end in three coordinates")
        padding = float(self.interface_margin + 4.0 * self.epsilon)
        lower = self.lower.to(latent) + padding
        upper = self.upper.to(latent) - padding
        local = lower + torch.sigmoid(latent) * (upper - lower)
        return self.local_to_world(local)

    def encode_inside_world(self, points: torch.Tensor) -> torch.Tensor:
        local = self.world_to_local(points)
        padding = float(self.interface_margin + 4.0 * self.epsilon)
        lower = self.lower.to(local) + padding
        upper = self.upper.to(local) - padding
        unit = ((local - lower) / (upper - lower)).clamp(1e-6, 1.0 - 1e-6)
        return torch.logit(unit)

    def metadata(self) -> dict:
        return {
            "schema": SPACE_SCHEMA,
            "axes_columns": self.axes.detach().cpu().double().tolist(),
            "lower": self.lower.detach().cpu().double().tolist(),
            "upper": self.upper.detach().cpu().double().tolist(),
            "interface_margin": float(self.interface_margin),
            "epsilon": float(self.epsilon),
            "transparent_interface_margin_mode": self.interface_margin_mode,
            "signed_clearance": "min(local-lower, upper-local); positive inside",
        }


def classify_numpy(points, axes, lower, upper, interface_margin=0.05, epsilon=1e-6):
    space = CuboidSpace(
        torch.as_tensor(np.asarray(axes), dtype=torch.float64),
        torch.as_tensor(np.asarray(lower), dtype=torch.float64),
        torch.as_tensor(np.asarray(upper), dtype=torch.float64),
        interface_margin=float(interface_margin), epsilon=float(epsilon),
    )
    return space.classify(torch.as_tensor(np.asarray(points), dtype=torch.float64)).numpy()
