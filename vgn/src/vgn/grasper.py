from vgn.utils.transform import Rotation, Transform


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
        self.robot.move_tcp_xyz(T_body_grasp)
        self.robot.close_gripper()
        self.robot.move_tcp_xyz(T_body_pregrasp)

        return 1.0
