import time

import numpy as np
import pybullet
import scipy.stats as stats

from vgn.grasp import Label
from vgn.perception.camera import PinholeCameraIntrinsic
from vgn.utils import btsim
from vgn.utils.transform import Rotation, Transform


class GraspExperiment(object):
    """Simulation of a grasping experiment.

    In this simulation, world, task and robot base frames are identical.

    Attributes:
        object_set: The grasping object set. Available options are
            * debug: a single cuboid at a fixed pose
            * cuboid
            * ycb
    """

    def __init__(self, urdf_root, object_set, hand, size, gui=True, rtf=-1.0):
        self.urdf_root = urdf_root
        self.object_set = object_set
        self.size = size

        self.world = btsim.BtWorld(gui, rtf)
        self.robot = Robot(self.world, hand)

    def setup(self, test=False):
        self.world.reset()
        self.world.set_gravity([0.0, 0.0, -9.81])

        # Load support surface
        plane = self.world.load_urdf(self.urdf_root / "plane/plane.urdf")
        plane.set_pose(Transform(Rotation.identity(), [0.0, 0.0, 0.0]))

        # Load robot
        pose = Transform(Rotation.identity(), np.r_[0.0, 0.0, 1.0])
        self.robot.reset(pose)
        self.robot.move_gripper(1.0)

        # Load camera
        intrinsic = PinholeCameraIntrinsic(640, 480, 540.0, 540.0, 320.0, 240.0)
        self.camera = self.world.add_camera(intrinsic, 0.1, 2.0)

        # Load objects
        if self.object_set == "debug":
            self.spawn_debug_object()
        if self.object_set == "cuboid":
            self.spawn_cuboid()
        elif self.object_set == "kappler":
            self.spawn_kappler(test)

    def pause(self):
        self.world.pause()

    def resume(self):
        self.world.resume()

    def save_state(self):
        self.snapshot_id = self.world.save_state()

    def restore_state(self):
        self.world.restore_state(self.snapshot_id)

    def test_grasp(self, T_base_grasp):
        """Open-loop grasp execution.
        
        Args:
            T_base_grasp: The grasp pose w.r.t. to the robot base frame.

        Return:
            A tuple (label, width) with the grasp outcome and grasp width.
        """
        epsilon = 0.2  # threshold for detecting grasps

        T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.05])
        T_base_pregrasp = T_base_grasp * T_grasp_pregrasp

        # Place the gripper at the grasp pose
        self.robot.set_tcp(T_base_grasp, override_dynamics=True)
        if self.robot.detect_collision():
            return Label.COLLISION, 0.0

        # Close the gripper
        if not self.robot.grasp(0.0, epsilon):
            return Label.EMPTY, 0.0

        # Retrieve the object
        self.robot.move_tcp_xyz(T_base_pregrasp, check_collisions=False)

        if not self.robot.grasp(0.0, epsilon):
            return Label.SLIPPED, 0.0

        # TODO shake test

        width = self.robot.read_gripper() * self.robot.max_gripper_width

        return Label.SUCCESS, width

    def spawn_object(self, urdf, pose, scale=1.0):
        obj = self.world.load_urdf(urdf, scale=scale)
        obj.set_pose(pose)
        for _ in range(240):
            self.world.step()

    def sample_pose(self):
        l, u = 0.0, self.size
        mu, sigma = self.size / 2.0, self.size / 4.0
        X = stats.truncnorm((l - mu) / sigma, (u - mu) / sigma, loc=mu, scale=sigma)
        position = np.r_[X.rvs(2), 0.15]
        orientation = Rotation.random()
        return Transform(orientation, position)

    def spawn_debug_object(self):
        urdf = self.urdf_root / "toy_blocks/cuboid/cuboid.urdf"
        position = np.r_[0.5 * self.size, 0.5 * self.size, 0.15]
        orientation = Rotation.identity()
        self.spawn_object(urdf, Transform(orientation, position))

    def spawn_cuboid(self):
        urdf = self.urdf_root / "toy_blocks/cuboid/cuboid.urdf"
        pose = self.sample_pose()
        self.spawn_object(urdf, pose)

    def spawn_kappler(self, test):
        expected_num_of_objects = 4
        num_objects = np.random.poisson(expected_num_of_objects - 1) + 1

        urdf_dir = self.urdf_root / "kappler" / "urdfs"
        names = [d.name for d in urdf_dir.iterdir() if d.is_dir()]

        for name in np.random.choice(names, size=num_objects):
            urdf = urdf_dir / name / (name + ".urdf")
            pose = self.sample_pose()
            self.spawn_object(urdf, pose, scale=0.6)


class Robot(object):
    """Simulated robot arm with a simple parallel-jaw gripper."""

    def __init__(self, world, hand):
        self.world = world
        self.urdf_path = hand.urdf_path

        self.T_tool0_tcp = hand.T_tool0_tcp
        self.T_tcp_tool0 = self.T_tool0_tcp.inverse()
        self.max_gripper_width = hand.max_gripper_width

    def reset(self, pose):
        self.body = self.world.load_urdf(str(self.urdf_path))
        self.body.set_pose(pose)
        self.constraint = self.world.add_constraint(
            self.body,
            None,
            None,
            None,
            pybullet.JOINT_FIXED,
            [0.0, 0.0, 0.0],
            Transform.identity(),
            pose,
        )

    def get_tcp(self):
        """get the pose of TCP w.r.t. world frame."""
        return self.body.get_pose() * self.T_tool0_tcp

    def set_tcp(self, pose, override_dynamics=False):
        """Set pose of TCP w.r.t. to world frame."""
        T_world_tool0 = pose * self.T_tcp_tool0
        if override_dynamics:
            self.body.set_pose(T_world_tool0)
        self.constraint.change(T_world_tool0, max_force=300)
        self.world.step()
        return self.detect_collision()

    def move_tcp_xyz(
        self, target_pose, eef_step=0.002, check_collisions=True, vel=0.10
    ):
        """Move the TCP linearly between two poses."""
        pose = self.get_tcp()
        pos_diff = target_pose.translation - pose.translation
        n_steps = int(np.linalg.norm(pos_diff) / eef_step)

        dist_step = pos_diff / n_steps
        dur_step = np.linalg.norm(dist_step) / vel

        for _ in range(n_steps):
            pose.translation += dist_step
            self.set_tcp(pose)
            for _ in range(int(dur_step / self.world.dt)):
                self.world.step()
            if check_collisions and self.detect_collision():
                return False

        return True

    def read_gripper(self):
        """Return current opening width of the gripper (scaled to [0, 1])."""
        pos_l = self.body.joints["finger_l"].get_position()
        pos_r = self.body.joints["finger_r"].get_position()
        width = pos_l + pos_r
        return width / self.max_gripper_width

    def move_gripper(self, width):
        """Move gripper to desired opening width (scaled to [0, 1])."""
        width *= 0.5 * self.max_gripper_width
        self.body.joints["finger_l"].set_position(width)
        self.body.joints["finger_r"].set_position(width)
        for _ in range(int(0.5 / self.world.dt)):
            self.world.step()

    def grasp(self, width, epsilon):
        self.move_gripper(width)
        return self.read_gripper() > epsilon

    def detect_collision(self):
        return self.world.check_collisions(self.body)
