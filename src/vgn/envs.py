from collections import deque
from pathlib import Path
import numpy as np

from vgn.perception import UniformTSDFVolume
from vgn.simulation import GraspSim, get_quality_fn
from vgn.utils import view_on_sphere, find_urdfs
from robot_helpers.spatial import Transform


class ClutterRemovalEnv:
    def __init__(self, cfg, rng):
        self.rng = rng
        self.urdfs = find_urdfs(Path(cfg["urdf_root"]))
        self.target_object_count = cfg["object_count"]
        self.origin = Transform.t([0.0, 0.0, 0.05])
        self.sim = GraspSim(cfg["sim"], self.rng)
        self.quality_fn = get_quality_fn(cfg["metric"], self.sim)

    @property
    def object_count(self):
        return self.sim.scene.object_count

    def reset(self):
        self.sim.scene.clear()
        urdfs = self.rng.choice(self.urdfs, self.target_object_count)
        self.sim.scene.generate(self.origin, urdfs, 1.0)
        self.outcomes = deque(maxlen=2)
        return self.get_observation()

    def step(self, grasp):
        # Execute grasp
        grasp.width = self.sim.gripper.max_width  # TODO
        quality, info = self.quality_fn(grasp)
        self.outcomes.append(quality)

        # Cleanup
        if quality == 1.0:
            self.sim.scene.remove_object(info["object_uid"])
        self.sim.gripper.reset(Transform.t(np.full(3, 100)), self.sim.gripper.max_width)
        self.sim.scene.remove_outside_objects()

        # Check stopping criteria
        done = self.sim.scene.object_count == 0 or (
            len(self.outcomes) == 2 and sum(self.outcomes) == 0
        )

        observation = (None, None) if done else self.get_observation()
        return observation, quality, done, {}

    def get_observation(self):
        tsdf = UniformTSDFVolume(self.sim.scene.size, 40)
        r = 2.0 * self.sim.scene.size
        theta = np.pi / 6.0
        phis = np.linspace(0.0, 2.0 * np.pi, 5)
        views = [view_on_sphere(self.sim.scene.center, r, theta, phi) for phi in phis]
        for view in views:
            depth_img = self.sim.camera.get_image(view)[1]
            tsdf.integrate(depth_img, self.sim.camera.intrinsic, view.inv())
        return tsdf.voxel_size, tsdf.get_grid()
