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
