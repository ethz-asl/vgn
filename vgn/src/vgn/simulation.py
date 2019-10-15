from __future__ import division

import numpy as np
import pybullet

import vgn.config as cfg
from vgn import grasp
from vgn.perception import camera
from vgn.utils import sim
from vgn.utils.transform import Rotation, Transform


class GraspingExperiment(object):
    """Simulation of a grasping experiment.

    In this simulation, world, task and robot base frames are identical.

    Attributes:
        world: Reference to the simulated world.
    """

    def __init__(self, gui, real_time_factor=-1.0):
        self.world = sim.BtWorld(gui, real_time_factor)

    def setup(self, object_set):
        """Setup a grasping experiment.

        Args:
            object_set: The grasping object set. Available options are
                * debug : a single cuboid at a fixed pose
                * cuboid: a single cuboid
                * cuboids: multiple cuboids
        """
        self.world.reset()
        self.world.set_gravity([0.0, 0.0, -9.81])

        # Load support surface
        plane = self.world.load_urdf("data/urdfs/plane/plane.urdf")
        plane.set_pose(Transform(Rotation.identity(), [0.0, 0.0, 0.0]))

        # Load robot
        self.robot = RobotArm(self.world)

        # Load camera
        intrinsic = camera.PinholeCameraIntrinsic(640, 480, 540.0, 540.0, 320.0, 240.0)
        self.camera = self.world.add_camera(intrinsic, 0.1, 2.0)

        # Load objects
        if object_set == "debug":
            self.spawn_debug_object()
        if object_set == "cuboid":
            self.spawn_cuboid()
        elif object_set == "cuboids":
            self.spawn_cuboids()

    def save_state(self):
        self.snapshot_id = self.world.save_state()

    def restore_state(self):
        self.world.restore_state(self.snapshot_id)

    def test_grasp(self, T_base_grasp):
        """Open-loop grasp execution.
        
        Args:
            T_base_grasp: The grasp pose w.r.t. to the robot base frame.
        """
        epsilon = 0.2  # threshold for detecting grasps

        # Place the gripper at the pre-grasp pose
        T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.05])
        T_base_pregrasp = T_base_grasp * T_grasp_pregrasp
        self.robot.set_tcp(T_base_pregrasp, override_dynamics=True)
        if self.robot.detect_collision():
            return grasp.Outcome.COLLISION

        # Open the gripper
        self.robot.move_gripper(1.0)

        # Approach the grasp pose
        if not self.robot.move_tcp_xyz(T_base_grasp):
            return grasp.Outcome.COLLISION

        # Close the gripper
        if not self.robot.grasp(0.0, epsilon):
            return grasp.Outcome.EMPTY

        # Retrieve the object
        self.robot.move_tcp_xyz(T_base_pregrasp, check_collisions=False)

        if not self.robot.grasp(0.0, epsilon):
            return grasp.Outcome.SLIPPED

        # TODO shake test

        return grasp.Outcome.SUCCESS

    def spawn_object(self, urdf, pose):
        obj = self.world.load_urdf(urdf)
        obj.set_pose(pose)
        for _ in range(240):
            self.world.step()

    def spawn_debug_object(self):
        urdf = "data/urdfs/wooden_blocks/cuboid0.urdf"
        position = np.r_[0.5 * cfg.size, 0.5 * cfg.size, 0.12]
        orientation = Rotation.from_quat([0.0, 0.0, 0.0, 1.0])
        self.spawn_object(urdf, Transform(orientation, position))

    def spawn_cuboid(self):
        urdf = "data/urdfs/wooden_blocks/cuboid0.urdf"
        position = np.r_[np.random.uniform(0.06, cfg.size - 0.06, 2), 0.12]
        orientation = Rotation.random()
        self.spawn_object(urdf, Transform(orientation, position))

    def spawn_cuboids(self):
        for _ in range(1 + np.random.randint(4)):
            self.spawn_cuboid()


class RobotArm(object):
    """Simulated robot arm with a simple parallel-jaw gripper."""

    T_tool0_tcp = Transform(Rotation.identity(), np.r_[0.0, 0.0, 0.02])
    T_tcp_tool0 = T_tool0_tcp.inverse()
    max_opening_width = 0.06

    def __init__(self, world):
        self.world = world
        self.body = world.load_urdf("data/urdfs/hand/hand.urdf")
        pose = Transform(Rotation.identity(), np.r_[0.0, 0.0, 1.0])
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
        T_world_tool0 = pose * RobotArm.T_tcp_tool0
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
        """Return current opening width of the gripper."""
        pos_l = self.body.joints["finger_l"].get_position()
        pos_r = self.body.joints["finger_r"].get_position()
        width = pos_l + pos_r
        return width / RobotArm.max_opening_width

    def move_gripper(self, width):
        """Move gripper to desired opening width."""
        width *= 0.5 * RobotArm.max_opening_width
        self.body.joints["finger_l"].set_position(width)
        self.body.joints["finger_r"].set_position(width)
        for _ in range(int(0.5 / self.world.dt)):
            self.world.step()

    def grasp(self, width, epsilon):
        self.move_gripper(width)
        return self.read_gripper() > epsilon

    def detect_collision(self):
        return self.world.check_collisions(self.body)
