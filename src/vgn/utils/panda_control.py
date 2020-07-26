import actionlib
from control_msgs.msg import GripperCommand, GripperCommandAction, GripperCommandGoal
import moveit_commander
from moveit_commander.conversions import list_to_pose
from moveit_msgs.msg import MoveGroupAction
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
        name = "franka_gripper/gripper_action"
        self.gripper_client = actionlib.SimpleActionClient(name, GripperCommandAction)
        self.gripper_client.wait_for_server()

    def home(self):
        self.goto_joints([0, -0.785, 0, -2.356, 0, 1.57, 0.785], 0.2, 0.2)

    def goto_joints(self, joints, velocity_scaling=0.1, acceleration_scaling=0.1):
        self.move_group.set_max_velocity_scaling_factor(velocity_scaling)
        self.move_group.set_max_acceleration_scaling_factor(acceleration_scaling)
        self.move_group.set_joint_value_target(joints)
        plan = self.move_group.plan()
        success = self.move_group.execute(plan, wait=True)
        self.move_group.stop()
        return success

    def goto_pose(self, pose, velocity_scaling=0.1, acceleration_scaling=0.1):
        pose_msg = ros_utils.to_pose_msg(pose)
        self.move_group.set_max_velocity_scaling_factor(velocity_scaling)
        self.move_group.set_max_acceleration_scaling_factor(acceleration_scaling)
        self.move_group.set_pose_target(pose_msg)
        plan = self.move_group.plan()
        success = self.move_group.execute(plan, wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()
        return success

    def move_gripper(self, width, max_effort=10):
        command = GripperCommand(width, max_effort)
        goal = GripperCommandGoal(command)
        self.gripper_client.send_goal(goal)
        return self.gripper_client.wait_for_result(timeout=rospy.Duration(1.0))
