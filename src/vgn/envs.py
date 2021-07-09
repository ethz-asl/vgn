import numpy as np

from vgn.perception import UniformTSDFVolume
from vgn.simulation import GraspSim
from vgn.utils import camera_on_sphere
from robot_utils.spatial import Rotation, Transform


class ClutterRemovalEnv:
    def __init__(self, scene, object_count, seed, gui):
        self.scene = scene
        self.object_count = object_count
        self.sim = GraspSim(gui, np.random.RandomState(seed))

    def reset(self):
        self.sim.reset(self.scene, self.object_count)
        self.consecutive_failures = 1
        self.last_score = None
        return self.get_tsdf()

    def step(self, grasp):
        grasp_pose = grasp.pose
        init_pose = grasp_pose * Transform(Rotation.identity(), [0.0, 0.0, -0.05])
        self.sim.gripper.spawn(init_pose)
        self.sim.gripper.move_linear(grasp_pose)
        score = self.sim.execute_grasp()
        self.sim.gripper.remove()
        self.sim.remove_objects_that_rolled_away()

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
        phis = np.linspace(0.0, 2.0 * np.pi, 5)
        extrinsics = [camera_on_sphere(origin, r, theta, phi) for phi in phis]
        for extrinsic in extrinsics:
            img = self.sim.camera.get_image(extrinsic.inv())
            tsdf.integrate(img, self.sim.camera.intrinsic, extrinsic)
        return tsdf.get_grid(), tsdf.voxel_size
