from enum import Enum

from vgn.utils.transform import Rotation, Transform


class Outcome(Enum):
    """Possible outcomes of a grasp attempt."""
    SUCCESS = 0
    COLLISION = 1
    EMPTY = 2
    SLIPPED = 3


class Grasper(object):
    """Open-loop grasp execution.

    First, the TCP is positioned to a pre-grasp pose, from which the grasp pose
    is approached linearly. If the grasp pose is reached without any collisions,
    the gripper is closed and the object retrieved.
    """

    def __init__(self, robot):
        self.robot = robot
        self.T_grasp_pregrasp = Transform(Rotation.identity(),
                                          [0., 0., -0.05])

    def grasp(self, T_body_grasp):
        """Execute the given grasp and report the outcome."""
        threshold = 0.2
        T_body_pregrasp = T_body_grasp * self.T_grasp_pregrasp

        if not self.robot.set_tcp_pose(T_body_pregrasp):
            return Outcome.COLLISION

        self.robot.open_gripper()

        if not self.robot.move_tcp_xyz(T_body_grasp, check_collisions=True):
            return Outcome.COLLISION

        self.robot.close_gripper()
        if self.robot.get_gripper_opening_width() < threshold:
            return Outcome.EMPTY

        self.robot.move_tcp_xyz(T_body_pregrasp)

        if self.robot.get_gripper_opening_width() < threshold:
            return Outcome.SLIPPED

        return Outcome.SUCCESS
