#!/usr/bin/env python

# Open-loop grasp execution using a Panda arm and wrist-mounted RealSense camera.


import geometry_msgs.msg
import numpy as np
import rospy
from std_srvs.srv import SetBool, Trigger

from robot_utils.ros.conversions import *
from robot_utils.ros.panda import PandaArmClient, PandaGripperClient
from robot_utils.ros.moveit import MoveItClient
from robot_utils.spatial import Rotation, Transform
from robot_utils.utils import map_cloud_to_grid
from vgn import vis
from vgn.grasp import Grasp
from vgn.srv import GetMapCloud, GetSceneCloud, PredictGrasps, PredictGraspsRequest


class PandaGraspController(object):
    def __init__(self, args):
        self.task_frame_id = rospy.get_param("~task_frame_id")
        self.base_frame_id = rospy.get_param("~base_frame_id")
        self.ee_frame_id = rospy.get_param("~ee_frame_id")
        self.ee_grasp_offset = Transform.from_list(rospy.get_param("~ee_grasp_offset"))
        self.finger_depth = rospy.get_param("~finger_depth")
        self.size = 6.0 * self.finger_depth
        self.scan_joints = rospy.get_param("~scan_joints")

        self.arm = PandaArmClient()
        self.gripper = PandaGripperClient()
        self.moveit = MoveItClient("panda_arm")
        self.configure_moveit()

        # TSDF clients
        self.reset_map = rospy.ServiceProxy("reset_map", Trigger)
        self.toggle_integration = rospy.ServiceProxy("toggle_integration", SetBool)
        self.get_scene_cloud = rospy.ServiceProxy("get_scene_cloud", GetSceneCloud)
        self.get_map_cloud = rospy.ServiceProxy("get_map_cloud", GetMapCloud)

        # VGN client
        self.predict_grasps = rospy.ServiceProxy("predict_grasps", PredictGrasps)

        rospy.loginfo("Ready to take action")

    def configure_moveit(self):
        self.moveit.move_group.set_end_effector_link(self.ee_frame_id)
        # Add a box to the planning scene to avoid collisions with the table.
        msg = geometry_msgs.msg.PoseStamped()
        msg.header.frame_id = self.base_frame_id
        msg.pose = to_pose_msg(Transform(Rotation.identity(), [0.4, 0.0, 0.2]))
        self.moveit.scene.add_box("table", msg, size=(0.6, 0.6, 0.02))
        rospy.sleep(1.0)  # wait for the scene to be updated

    def run(self):
        vis.clear()
        vis.draw_workspace(self.size)
        self.gripper.move(0.08)
        self.moveit.goto("ready")

        rospy.loginfo("Reconstructing scene")
        self.scan_scene()
        self.get_scene_cloud()
        res = self.get_map_cloud()

        rospy.loginfo("Planning grasps")
        req = PredictGraspsRequest(res.voxel_size, res.map_cloud)
        res = self.predict_grasps(req)

        if len(res.grasps) == 0:
            rospy.loginfo("No grasps detected")
            return

        # Deserialize best grasp.
        msg = res.grasps[0]
        grasp = Grasp(from_pose_msg(msg[0].pose), msg.width, msg.quality)
        vis.draw_grasp(grasp, self.finger_depth)

        # Ensure that the camera is pointing forward.
        rot = grasp.pose.rotation
        axis = rot.as_matrix()[:, 0]
        if axis[0] < 0:
            grasp.pose.rotation = rot * Rotation.from_euler("z", np.pi)

        # Execute grasp
        rospy.loginfo("Executing grasp")
        self.moveit.goto("ready")
        success = self.execute_grasp(grasp)

        # Drop object
        if success:
            rospy.loginfo("Dropping object")
            self.moveit.goto([0.678, 0.097, 0.237, -1.63, -0.031, 1.756, 0.931])
            self.gripper.move(0.08)

        self.moveit.goto("ready")

    def scan_scene(self):
        self.moveit.goto(self.scan_joints[0])
        self.reset_map()
        self.toggle_integration(True)
        for joint_target in self.scan_joints[1:]:
            self.moveit.goto(joint_target)
        self.toggle_integration(False)

    def execute_grasp(self, grasp):
        T_task_grasp = grasp.pose
        T_base_grasp = self.T_base_task * T_task_grasp

        T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.05])
        T_grasp_retreat = Transform(Rotation.identity(), [0.0, 0.0, -0.05])
        T_base_pregrasp = T_base_grasp * T_grasp_pregrasp
        T_base_retreat = T_base_grasp * T_grasp_retreat

        self.moveit.goto(T_base_pregrasp * self.T_tcp_tool0, velocity_scaling=0.2)
        self.moveit.goto(T_base_grasp * self.T_tcp_tool0)
        self.gripper.grasp(width=0.0, force=20.0)
        self.moveit.goto(T_base_retreat * self.T_tcp_tool0)

        T_retreat_lift_base = Transform(Rotation.identity(), [0.0, 0.0, 0.1])
        T_base_lift = T_retreat_lift_base * T_base_retreat
        self.moveit.goto(T_base_lift * self.T_tcp_tool0)

        return self.gripper.read() > 0.004


if __name__ == "__main__":
    rospy.init_node("panda_grasp")
    panda_grasp = PandaGraspController()
    panda_grasp.run()
