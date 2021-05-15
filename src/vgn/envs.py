import numpy as np

from vgn.simulation import GraspSim
from vgn.utils import camera_on_sphere
from robot_tools.perception import UniformTSDFVolume
from robot_tools.spatial import Rotation, Transform
from robot_tools.utils import map_cloud_to_grid


class ClutterRemovalEnv:
    def __init__(self, scene, object_count, gui):
        self.scene = scene
        self.object_count = object_count
        self.sim = GraspSim(gui)

    def reset(self):
        self.sim.reset(self.scene, self.object_count)
        self.consecutive_failures = 1
        self.last_score = None
        return self.get_tsdf()

    def step(self, grasp):
        score, _ = self.sim.execute_grasp(grasp)

        if score == 0.0 and self.last_score == 0.0:
            self.consecutive_failures += 1
        else:
            self.consecutive_failures = 1

        done = self.sim.num_objects == 0 or self.consecutive_failures == 2

        if done:
            return (None, None), score, done, {}

        tsdf_grid, voxel_size = self.get_tsdf()
        self.last_score = score

        return (tsdf_grid, voxel_size), score, done, {}

    def get_tsdf(self):
        tsdf = UniformTSDFVolume(self.sim.size, 40)
        origin = Transform(Rotation.identity(), self.sim.origin)
        r = 2.0 * self.sim.size
        theta = np.pi / 6.0
        angles = np.linspace(0.0, 2.0 * np.pi, 5)
        extrinsics = [camera_on_sphere(origin, r, theta, phi) for phi in angles]
        for extrinsic in extrinsics:
            _, depth = self.sim.camera.render(extrinsic.inv())
            tsdf.integrate(depth, self.sim.camera.intrinsic, extrinsic)
        map_cloud = tsdf.get_map_cloud()
        points = np.asarray(map_cloud.points)
        distances = np.asarray(map_cloud.colors)[:, 0]
        tsdf_grid = map_cloud_to_grid(tsdf.voxel_size, points, distances)
        return tsdf_grid, tsdf.voxel_size
