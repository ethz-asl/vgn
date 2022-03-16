from collections import deque
from pathlib import Path
import numpy as np

from vgn.perception import UniformTSDFVolume
from vgn.simulation import GraspSim, get_scene, get_metric
from vgn.utils import find_urdfs, view_on_sphere
import vgn.visualizer as vis
from robot_helpers.spatial import Transform


class ClutterRemovalEnv:
    def __init__(self, cfg, rng):
        self.rng = rng
        self.urdfs = find_urdfs(Path(cfg["object_urdfs"]))
        self.object_count = cfg["object_count"]
        self.scaling = cfg["scaling"]
        self.generate_scene = get_scene(cfg["scene"])

        self.origin = Transform.t_[0.0, 0.0, 0.05]
        self.size = 0.3
        self.center = self.origin * Transform.t_[0.5 * self.size, 0.5 * self.size, 0.0]

        self.sim = GraspSim(cfg["sim"], self.rng)
        self.score_fn = get_metric(cfg["metric"])(self.sim)

    def reset(self):
        self.sim.clear()
        urdfs = self.rng.choice(self.urdfs, self.object_count)
        scales = self.rng.uniform(self.scaling["low"], self.scaling["high"], len(urdfs))
        self.generate_scene(self.sim, self.origin, self.size, urdfs, scales)
        self.outcomes = deque(maxlen=2)
        return {"object_count": self.sim.object_count}

    def step(self, grasp):
        grasp.width = self.sim.robot.max_width
        score, info = self.score_fn(grasp)
        self.outcomes.append(score)
        if score == 1.0:
            self.sim.remove_object(info["object_uid"])
        self.sim.robot.reset(Transform.t_[0, 0, 10], self.sim.robot.max_width)
        done = self.sim.object_count == 0 or (
            len(self.outcomes) == 2 and sum(self.outcomes) == 0
        )
        return score, done

    def get_observation(self, view_count=6):
        tsdf = UniformTSDFVolume(self.size, 40)
        r = 2.0 * self.size
        theta = np.pi / 6.0
        phis = 2.0 * np.pi * np.arange(view_count) / view_count
        views = [view_on_sphere(self.center, r, theta, phi) for phi in phis]
        for view in views:
            depth_img = self.sim.camera.get_image(view)[1]
            tsdf.integrate(depth_img, self.sim.camera.intrinsic, view.inv())
        # vis.scene_cloud(tsdf.voxel_size, np.asarray(tsdf.get_scene_cloud().points))
        return tsdf.voxel_size, tsdf.get_grid()
