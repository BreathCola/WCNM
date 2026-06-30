"""Scene wrapper with separate Diffuse and Reflection exports for Stage B."""

import json
import os

from scene import Scene
from utils.system_utils import searchForMaxIteration


class StageBScene:
    def __init__(
        self,
        args,
        diffuse,
        reflection,
        load_iteration=None,
        shuffle=True,
        write_metadata=True,
    ):
        self.model_path = args.model_path
        self.diffuse = diffuse
        self.reflection = reflection
        self.base = Scene(
            args,
            diffuse,
            shuffle=shuffle,
            initialize_model=False,
            write_metadata=write_metadata,
        )
        self.cameras_extent = self.base.cameras_extent
        self.loaded_iter = None
        if load_iteration is not None:
            self.loaded_iter = (
                searchForMaxIteration(os.path.join(self.model_path, "point_cloud", "diffuse"))
                if load_iteration == -1
                else int(load_iteration)
            )
            if self.loaded_iter is None or self.loaded_iter < 0:
                raise FileNotFoundError("no Stage B Diffuse iteration found")
            diffuse.load_ply(self._ply_path("diffuse", self.loaded_iter), args.train_test_exp)
            reflection.load_ply(self._ply_path("reflection", self.loaded_iter))

    def _ply_path(self, branch, iteration):
        return os.path.join(
            self.model_path,
            "point_cloud",
            branch,
            f"iteration_{int(iteration)}",
            "point_cloud.ply",
        )

    def save(self, iteration):
        self.diffuse.save_ply(self._ply_path("diffuse", iteration))
        self.reflection.save_ply(self._ply_path("reflection", iteration))
        exposure_dict = {
            name: self.diffuse.get_exposure_from_name(name).detach().cpu().numpy().tolist()
            for name in self.diffuse.exposure_mapping
        }
        with open(os.path.join(self.model_path, "exposure.json"), "w", encoding="utf-8") as handle:
            json.dump(exposure_dict, handle, indent=2)

    def getTrainCameras(self, scale=1.0):
        return self.base.getTrainCameras(scale)

    def getTestCameras(self, scale=1.0):
        return self.base.getTestCameras(scale)
