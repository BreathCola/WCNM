"""Independent Stage D Transmittance 2D Gaussian surfel field."""

from scene.reflection_surfel_model import ReflectionSurfelModel
from utils.general_utils import get_expon_lr_func


class TransmittanceSurfelModel(ReflectionSurfelModel):
    """T field with parameters, optimizer, schedule, and topology separate from D/R."""

    model_type = "transmittance_surfel"

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
        import torch
        self.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=args.transmittance_position_lr_init,
            lr_final=args.transmittance_position_lr_final,
            lr_delay_mult=args.transmittance_position_lr_delay_mult,
            max_steps=args.transmittance_position_lr_max_steps,
        )
        if self.xyz_gradient_accum.shape[0] != self.get_xyz.shape[0]:
            self._reset_stats(self.get_xyz.shape[0], self.get_xyz.device)
