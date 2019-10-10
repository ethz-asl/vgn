from __future__ import print_function

import enum

from vgn.utils.transform import Rotation, Transform


class Outcome(enum.Enum):
    """Possible outcomes of a grasp attempt."""

    SUCCESS = 0
    COLLISION = 1
    EMPTY = 2
    SLIPPED = 3


def execute(robot, T_base_grasp):
    """Open-loop grasp execution.

    First, the tool is positioned to a pre-grasp pose, from which the grasp pose
    is approached linearly. If the grasp pose is reached without any collisions,
    the gripper is closed and the object retrieved.

    Args:
        robot: Reference to the manipulator which will execute the grasp.
        T_base_grasp: The pose of the grasp w.r.t. manipulator base frame.
    """
    grasp_detection_threshold = 0.2
    T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.05])

    T_base_pregrasp = T_base_grasp * T_grasp_pregrasp

    robot.set_tool_pose(T_base_pregrasp, override_dynamics=True)
    robot.open_gripper()

    if not robot.move_tool_xyz(T_base_grasp):
        return Outcome.COLLISION

    robot.close_gripper()

    if robot.get_gripper_opening_width() < grasp_detection_threshold:
        return Outcome.EMPTY

    robot.move_tool_xyz(T_base_pregrasp, check_collisions=False)

    if robot.get_gripper_opening_width() < grasp_detection_threshold:
        return Outcome.SLIPPED

    return Outcome.SUCCESS
