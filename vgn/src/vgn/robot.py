import copy


class Robot(object):

    def set_ee_pose(self, pose):
        """Move the TCP to the given pose.

        Args:
            pose: The EE pose w.r.t. body frame, T_body_tcp.
        """
        raise NotImplementedError

    def move_ee_xyz(self, displacement, eef_step):
        """Linearly move the EE in cartesian space.

        Args:
            displacement: The 3D displacement of the EE.
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
        self._robot = robot
        self._approach_dist = 0.05

    def grasp(self, grasp_pose):
        """
        Args:
            grasp_pose: T_task_grasp

        TODO:
            * Score grasp
        """
        pregrasp_pose = copy.copy(grasp_pose)
        pregrasp_pose.translation[2] -= self._approach_dist

        self._robot.set_ee_pose(pregrasp_pose)
        self._robot.move_ee_xyz([0., 0., self._approach_dist], eef_step=0.01)
        self._robot.close_gripper()
        self._robot.move_ee_xyz([0., 0., self._approach_dist], eef_step=0.01)

        return 1.0
