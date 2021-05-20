#!/usr/bin/env python

# Open-loop grasp execution using a Panda arm and wrist-mounted RealSense camera.


import geometry_msgs.msg
import numpy as np
import rospy
from std_srvs.srv import SetBool, Trigger

from robot_tools.ros.conversions import *
from robot_tools.ros.panda import PandaGripperClient
from robot_tools.ros.moveit import MoveItClient
from robot_tools.ros.tf import TransformTree
from robot_tools.spatial import Rotation, Transform
from vgn import vis
from vgn.grasp import Grasp
from vgn.srv import GetMapCloud, GetSceneCloud, PredictGrasps, PredictGraspsRequest


class PandaGraspController(object):
    def __init__(self):
        self.task_frame_id = rospy.get_param("~task_frame_id")
        self.base_frame_id = rospy.get_param("~base_frame_id")
        self.ee_frame_id = rospy.get_param("~ee_frame_id")
        self.ee_grasp_offset = Transform.from_list(rospy.get_param("~ee_grasp_offset"))
        self.grasp_ee_offset = self.ee_grasp_offset.inv()
        self.finger_depth = rospy.get_param("~finger_depth")
        self.size = 6.0 * self.finger_depth
        self.scan_joints = rospy.get_param("~scan_joints")

        self.gripper = PandaGripperClient()
        self.moveit = MoveItClient("panda_arm")
        self.configure_moveit()

        tf = TransformTree()
        self.T_base_task = tf.lookup(
            self.base_frame_id,
            self.task_frame_id,
            rospy.Time.now(),
            rospy.Duration(1.0),
        )

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
        msg.pose.position.x = 0.4
        msg.pose.position.z = 0.0
        self.moveit.scene.add_box("table", msg, size=(0.6, 0.6, 0.02))
        rospy.sleep(1.0)  # wait for the scene to be updated

    def run(self):
        vis.clear()
        self.gripper.move(0.08)

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
        grasp = Grasp(from_pose_msg(msg.pose), msg.width, msg.quality)
        vis.draw_grasp(grasp, self.finger_depth)

        # Ensure that the camera is pointing forward.
        rot = grasp.pose.rotation
        axis = rot.as_dcm()[:, 0]
        if axis[0] < 0:
            grasp.pose.rotation = rot * Rotation.from_euler("z", np.pi)

        # Execute grasp
        rospy.loginfo("Executing grasp")
        self.moveit.move("ready")
        success = self.execute_grasp(grasp)

        # Drop object
        if success:
            rospy.loginfo("Dropping object")
            self.moveit.move([0.678, 0.097, 0.237, -1.63, -0.031, 1.756, 0.931])
            self.gripper.move(0.08)

        self.moveit.move("ready")

    def scan_scene(self):
        self.moveit.move("ready")
        self.reset_map()
        rospy.loginfo("calling toggle integration")
        self.toggle_integration(True)
        for joint_target in self.scan_joints:
            self.moveit.move(joint_target)
        self.toggle_integration(False)

    def execute_grasp(self, grasp):
        T_task_grasp = grasp.pose
        T_base_grasp = self.T_base_task * T_task_grasp

        T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.05])
        T_grasp_retreat = Transform(Rotation.identity(), [0.0, 0.0, -0.05])
        T_base_pregrasp = T_base_grasp * T_grasp_pregrasp
        T_base_retreat = T_base_grasp * T_grasp_retreat

        self.moveit.move(T_base_pregrasp * self.grasp_ee_offset, velocity_scaling=0.2)
        self.moveit.move(T_base_grasp * self.grasp_ee_offset)
        self.gripper.grasp(width=0.0, force=20.0)
        self.moveit.move(T_base_retreat * self.grasp_ee_offset)

        T_retreat_lift_base = Transform(Rotation.identity(), [0.0, 0.0, 0.1])
        T_base_lift = T_retreat_lift_base * T_base_retreat
        self.moveit.move(T_base_lift * self.grasp_ee_offset)

        return self.gripper.read() > 0.004


if __name__ == "__main__":
    rospy.init_node("panda_grasp")
    panda_grasp = PandaGraspController()
    panda_grasp.run()
