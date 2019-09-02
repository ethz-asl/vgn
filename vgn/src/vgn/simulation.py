from __future__ import division

import time

import numpy as np

import pybullet
import vgn.config as cfg
from pybullet_utils import bullet_client
from vgn import robot
from vgn.perception.camera import PinholeCameraIntrinsic
from vgn.utils import sim_utils
from vgn.utils.transform import Rotation, Transform


class Simulation(object):
    """Simulation of a grasping experiment.

    In this simulation, world, task and body frames are identical.
    """
    def __init__(self, gui, real_time_factor=-1.):
        connection_mode = pybullet.GUI if gui else pybullet.DIRECT
        self.p = bullet_client.BulletClient(connection_mode)

        intrinsic = PinholeCameraIntrinsic(640, 480, 540., 540., 320., 240.)

        self.engine = sim_utils.Engine(self.p, real_time_factor)
        self.camera = sim_utils.Camera(self.p, intrinsic, 0.1, 2.0)
        self.hand = None  # call spawn_hand to create object

    def save_state(self):
        self.snapshot_id = self.engine.save_state()

    def restore_state(self):
        assert self.snapshot_id is not None, 'save_state must be called first'
        self.engine.restore_state(self.snapshot_id)

    def spawn_plane(self):
        pose = Transform(Rotation.identity(), [0.0, 0.0, 0.0])
        sim_utils.Body(self.p, 'data/urdfs/plane/plane.urdf', pose)

    def spawn_object(self, urdf, pose):
        sim_utils.Body(self.p, urdf, pose)
        for _ in range(self.engine.hz // 2):
            self.engine.step()

    def spawn_cuboid(self):
        position = np.r_[0.5 * cfg.size, 0.5 * cfg.size, 0.12]
        orientation = Rotation.from_quat([0., 0., 0., 1.])
        urdf = 'data/urdfs/wooden_blocks/cuboid0.urdf'
        self.spawn_object(urdf, Transform(orientation, position))

    def spawn_cuboid_random(self):
        position = np.r_[np.random.uniform(0.06, cfg.size - 0.06, 2), 0.12]
        orientation = Rotation.random()
        urdf = 'data/urdfs/wooden_blocks/cuboid0.urdf'
        self.spawn_object(urdf, Transform(orientation, position))

    def spawn_cuboids(self):
        for _ in range(1 + np.random.randint(4)):
            self.spawn_cuboid_random()

    def spawn_hand(self):
        self.hand = Hand(self.p, self.engine)


class Hand(robot.Manipulator):
    T_tool0_tcp = Transform(Rotation.identity(), [0.0, 0.0, 0.02])
    T_tcp_tool0 = T_tool0_tcp.inverse()
    max_opening_width = 0.06

    def __init__(self, physics_client, engine):
        self.p = physics_client
        self.engine = engine

        pose = Transform(Rotation.identity(), np.r_[0., 0., 1.])
        self.body = sim_utils.Body(self.p, 'data/urdfs/hand/hand.urdf', pose)
        self.constraint = sim_utils.Constraint(self.p, self.body, None, None,
                                               None, pybullet.JOINT_FIXED,
                                               [0., 0., 0.],
                                               Transform.identity(), pose)

    def get_tcp_pose(self):
        return self.body.get_pose() * self.T_tool0_tcp

    def set_tcp_pose(self, pose, override_dynamics=False):
        T_world_tool0 = pose * Hand.T_tcp_tool0
        if override_dynamics:
            self.body.set_pose(T_world_tool0)
        self.constraint.change(T_world_tool0, max_force=300)
        self.engine.step()
        return len(self._detect_collision()) == 0

    def move_tcp_xyz(self,
                     target_pose,
                     eef_step=0.002,
                     check_collisions=False,
                     vel=0.10):
        pose = self.get_tcp_pose()
        pos_diff = target_pose.translation - pose.translation
        n_steps = int(np.linalg.norm(pos_diff) / eef_step)

        dist_step = pos_diff / n_steps
        dur_step = np.linalg.norm(dist_step) / vel

        for _ in range(n_steps):
            pose.translation += dist_step
            self.set_tcp_pose(pose)
            for _ in range(int(dur_step * self.engine.hz)):
                self.engine.step()
            if check_collisions and self._detect_collision():
                return False

        return True

    def _detect_collision(self):
        return self.engine.get_contacts(self.body)

    def get_gripper_opening_width(self):
        """Return the gripper opening width scaled to the range [0., 1.]."""
        pos_l = self.body.joints['finger_l'].get_position()
        pos_r = self.body.joints['finger_r'].get_position()
        return (pos_l + pos_r) / Hand.max_opening_width

    def set_gripper_opening_width(self, pos):
        pos *= 0.5 * Hand.max_opening_width
        self.body.joints['finger_l'].set_position(pos)
        self.body.joints['finger_r'].set_position(pos)
        for _ in range(self.engine.hz // 2):
            self.engine.step()

    def open_gripper(self):
        self.set_gripper_opening_width(1.0)

    def close_gripper(self):
        self.set_gripper_opening_width(0.0)
