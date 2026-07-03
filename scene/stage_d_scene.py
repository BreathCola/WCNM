"""Scene wrapper with independent Diffuse, Reflection, and Transmittance exports."""

import json
import os

from scene import Scene


class StageDScene:
    def __init__(self, args, diffuse, reflection, transmittance, shuffle=True):
        self.model_path = args.model_path
        self.diffuse = diffuse
        self.reflection = reflection
        self.transmittance = transmittance
        self.base = Scene(
            args, diffuse, shuffle=shuffle, initialize_model=False, write_metadata=True
        )
        self.cameras_extent = self.base.cameras_extent

    def _ply_path(self, branch, iteration):
        return os.path.join(
            self.model_path, "point_cloud", branch, f"iteration_{int(iteration)}",
            "point_cloud.ply",
        )

    def save(self, iteration):
        self.diffuse.save_ply(self._ply_path("diffuse", iteration))
        self.reflection.save_ply(self._ply_path("reflection", iteration))
        self.transmittance.save_ply(self._ply_path("transmittance", iteration))
        exposure = {
            name: self.diffuse.get_exposure_from_name(name).detach().cpu().numpy().tolist()
            for name in self.diffuse.exposure_mapping
        }
        with open(os.path.join(self.model_path, "exposure.json"), "w", encoding="utf-8") as handle:
            json.dump(exposure, handle, indent=2)

    def getTrainCameras(self, scale=1.0):
        return self.base.getTrainCameras(scale)

    def getTestCameras(self, scale=1.0):
        return self.base.getTestCameras(scale)
