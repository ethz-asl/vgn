import numpy as np

from vgn.utils.transform import Rotation, Transform


class Robot(object):

    def set_tcp_pose(self, pose):
        """Move the tool center point to the given pose.

        Args:
            pose: The pose of the TCP in body frame, T_body_tcp.
        """
        raise NotImplementedError

    def move_tcp_xyz(self, pose, eef_step):
        """Linearly move the EE in cartesian space to the new pose.

        Args:
            pose: The new pose of the tcp in body frame.
            eef_step: Path interpolation resolution [m].
        """
        raise NotImplementedError

    def open_gripper(self):
        """Open the gripper."""
        raise NotImplementedError

    def close_gripper(self):
        """Close the gripper."""
        raise NotImplementedError


class Grasper(object):

    def __init__(self, robot):
        """Open-loop grasp execution."""
        self.robot = robot
        self.T_grasp_pregrasp = Transform(Rotation.identity(),
                                          [0., 0., -0.05])

    def grasp(self, T_body_grasp):
        T_body_pregrasp = T_body_grasp * self.T_grasp_pregrasp
        self.robot.set_tcp_pose(T_body_pregrasp)
        self.robot.open_gripper()
        self.robot.move_tcp_xyz(T_body_grasp, eef_step=0.01)
        self.robot.close_gripper()
        self.robot.move_tcp_xyz(T_body_pregrasp, eef_step=0.01)

        return 1.0
