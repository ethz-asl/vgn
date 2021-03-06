import actionlib
import franka_gripper.msg
import moveit_commander
import rospy

from vgn.utils import ros_utils


class PandaCommander(object):
    def __init__(self):
        self.name = "panda_arm"
        self._connect_to_move_group()
        self._connect_to_gripper()
        rospy.loginfo("PandaCommander ready")

    def _connect_to_move_group(self):
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.move_group = moveit_commander.MoveGroupCommander(self.name)

    def _connect_to_gripper(self):
        self.grasp_client = actionlib.SimpleActionClient(
            "/franka_gripper/grasp", franka_gripper.msg.GraspAction
        )
        self.grasp_client.wait_for_server()
        rospy.loginfo("Connected to grasp action server")
        self.move_client = actionlib.SimpleActionClient(
            "/franka_gripper/move", franka_gripper.msg.MoveAction
        )
        self.move_client.wait_for_server()
        rospy.loginfo("Connected to move action server")

    def home(self):
        self.goto_joints([0, -0.785, 0, -2.356, 0, 1.57, 0.785], 0.2, 0.2)

    def goto_joints(self, joints, velocity_scaling=0.1, acceleration_scaling=0.1):
        self.move_group.set_max_velocity_scaling_factor(velocity_scaling)
        self.move_group.set_max_acceleration_scaling_factor(acceleration_scaling)
        self.move_group.set_joint_value_target(joints)
        plan = self.move_group.plan()[1]
        success = self.move_group.execute(plan, wait=True)
        self.move_group.stop()
        return success

    def goto_pose(self, pose, velocity_scaling=0.1, acceleration_scaling=0.1):
        pose_msg = ros_utils.to_pose_msg(pose)
        self.move_group.set_max_velocity_scaling_factor(velocity_scaling)
        self.move_group.set_max_acceleration_scaling_factor(acceleration_scaling)
        self.move_group.set_pose_target(pose_msg)
        plan = self.move_group.plan()[1]
        success = self.move_group.execute(plan, wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()
        return success

    def grasp(self, width=0.0, e_inner=0.1, e_outer=0.1, speed=0.1, force=10.0):
        epsilon = franka_gripper.msg.GraspEpsilon(e_inner, e_outer)
        goal = franka_gripper.msg.GraspGoal(width, epsilon, speed, force)
        self.grasp_client.send_goal(goal)
        return self.grasp_client.wait_for_result(rospy.Duration(2.0))

    def move_gripper(self, width, speed=0.1):
        goal = franka_gripper.msg.MoveGoal(width, speed)
        self.move_client.send_goal(goal)
        return self.move_client.wait_for_result(rospy.Duration(2.0))
