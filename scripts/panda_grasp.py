#!/usr/bin/env python

from __future__ import division, print_function

import argparse
from pathlib2 import Path
import pickle


import cv_bridge
import franka_msgs.msg
import geometry_msgs.msg
import numpy as np
import rospy
import sensor_msgs.msg
import std_srvs.srv


from vgn import vis
from vgn.baselines import GPD
from vgn.benchmark import Logger, State
from vgn.detection import VGN
from vgn.perception import *
from vgn.utils import ros_utils
from vgn.utils.transform import Rotation, Transform
from vgn.utils.panda_control import PandaCommander


vis.set_size(0.3)
# tag lies on the table in the center of the workspace
T_base_tag = Transform(Rotation.identity(), [0.42, 0.02, 0.21])
round_id = 0


class PandaGraspController(object):
    def __init__(self, args):
        self.base_frame_id = rospy.get_param("~base_frame_id")
        self.tool0_frame_id = rospy.get_param("~tool0_frame_id")
        self.T_tool0_tcp = Transform.from_dict(rospy.get_param("~T_tool0_tcp"))  # TODO
        self.T_tcp_tool0 = self.T_tool0_tcp.inverse()
        self.finger_depth = rospy.get_param("~finger_depth")
        self.size = 6.0 * self.finger_depth
        self.scan_joints = rospy.get_param("~scan_joints")

        self.setup_panda_control()
        self.tf_tree = ros_utils.TransformTree()
        self.define_workspace()
        self.create_planning_scene()
        self.tsdf_server = TSDFServer()
        self.plan_grasps = self.select_grasp_planner(args.method)
        self.logger = Logger(args.logdir)

        rospy.loginfo("Ready to take action")

    def setup_panda_control(self):
        rospy.Subscriber(
            "/franka_state_controller/franka_states",
            franka_msgs.msg.FrankaState,
            self.robot_state_cb,
            queue_size=1,
        )
        rospy.Subscriber(
            "/joint_states", sensor_msgs.msg.JointState, self.joints_cb, queue_size=1
        )
        self.pc = PandaCommander()
        self.pc.move_group.set_end_effector_link(self.tool0_frame_id)

    def define_workspace(self):
        z_offset = -0.06
        t_tag_task = np.r_[[-0.5 * self.size, -0.5 * self.size, z_offset]]
        T_tag_task = Transform(Rotation.identity(), t_tag_task)
        self.T_base_task = T_base_tag * T_tag_task

        self.tf_tree.broadcast_static(self.T_base_task, self.base_frame_id, "task")
        rospy.sleep(1.0)  # wait for the TF to be broadcasted

    def create_planning_scene(self):
        # collision box for table
        msg = geometry_msgs.msg.PoseStamped()
        msg.header.frame_id = self.base_frame_id
        msg.pose = ros_utils.to_pose_msg(T_base_tag)
        msg.pose.position.z -= 0.01
        self.pc.scene.add_box("table", msg, size=(0.6, 0.6, 0.02))

        # collision box for camera
        msg = geometry_msgs.msg.PoseStamped()
        msg.header.frame_id = "panda_hand"
        msg.pose.position.x = 0.06
        msg.pose.position.z = 0.03
        self.pc.scene.add_box("camera", msg, size=(0.04, 0.10, 0.04))
        touch_links = self.pc.robot.get_link_names(group=self.pc.name)
        self.pc.scene.attach_box("panda_link8", "camera", touch_links=touch_links)

        rospy.sleep(1.0)  # wait for the scene to be updated

    def select_grasp_planner(self, method):
        if method == "vgn":
            return VGN(args.model)
        elif method == "gpd":
            return GPD()
        else:
            raise ValueError

    def robot_state_cb(self, msg):
        if np.any(msg.cartesian_collision):
            self.robot_error = True

    def joints_cb(self, msg):
        self.gripper_width = msg.position[7] + msg.position[8]

    def run(self):
        vis.clear()
        vis.draw_workspace()
        self.pc.move_gripper(0.08)
        self.pc.home()

        tsdf, pc = self.acquire_tsdf()
        vis.draw_tsdf(tsdf.get_volume().squeeze())
        vis.draw_points(np.asarray(pc.points))
        rospy.loginfo("Reconstructed scene")

        state = State(tsdf, pc)
        grasps, scores, planning_time = self.plan_grasps(state)
        vis.draw_grasps(grasps, scores)
        rospy.loginfo("Planned grasps")

        if len(grasps) == 0:
            rospy.loginfo("No grasps detected")
            return

        grasp, score = self.select_grasp(grasps, scores)
        vis.draw_grasp(grasp, score)
        rospy.loginfo("Selected grasp")

        self.pc.home()
        label = self.execute_grasp(grasp)
        rospy.loginfo("Grasp execution")

        self.logger.log_grasp(round_id, state, planning_time, grasp, score, label)

        if label:
            self.drop()

        self.pc.home()

    def acquire_tsdf(self):
        self.pc.goto_joints(self.scan_joints[0])

        self.tsdf_server.reset()
        self.tsdf_server.integrate = True

        for joint_target in self.scan_joints[1:]:
            self.pc.goto_joints(joint_target)

        self.tsdf_server.integrate = False
        tsdf = self.tsdf_server.low_res_tsdf
        pc = self.tsdf_server.high_res_tsdf.extract_point_cloud()

        return tsdf, pc

    def select_grasp(self, grasps, scores):
        # select the highest grasp
        heights = np.empty(len(grasps))
        for i, grasp in enumerate(grasps):
            heights[i] = grasp.pose.translation[2]
        idx = np.argmax(heights)
        grasp, score = grasps[idx], scores[idx]

        # make sure camera is pointing forward
        rot = grasp.pose.rotation
        axis = rot.as_dcm()[:, 0]
        if axis[0] < 0:
            grasp.pose.rotation = rot * Rotation.from_euler("z", np.pi)

        return grasp, score

    def execute_grasp(self, grasp):
        T_task_grasp = grasp.pose
        T_base_grasp = self.T_base_task * T_task_grasp

        T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.05])
        T_grasp_retreat = Transform(Rotation.identity(), [0.0, 0.0, -0.05])
        T_base_pregrasp = T_base_grasp * T_grasp_pregrasp
        T_base_retreat = T_base_grasp * T_grasp_retreat

        self.pc.goto_pose(T_base_pregrasp * self.T_tcp_tool0)
        self.approach_grasp(T_base_grasp)
        self.pc.grasp(force=5.0)
        self.pc.goto_pose(T_base_retreat * self.T_tcp_tool0)

        if self.gripper_width > 0.004:
            return True
        else:
            return False

    def approach_grasp(self, T_base_grasp):
        self.pc.goto_pose(T_base_grasp * self.T_tcp_tool0)

    def drop(self):
        self.pc.goto_joints([0, -0.785, 0, -2.356, 0, 1.57, 0.785], 0.2, 0.2)
        self.pc.goto_joints(
            [0.678, 0.097, 0.237, -1.63, -0.031, 1.756, 0.931], 0.2, 0.2
        )
        self.pc.move_gripper(0.08)


class TSDFServer(object):
    def __init__(self):
        self.cam_frame_id = rospy.get_param("~cam/frame_id")
        self.cam_topic_name = rospy.get_param("~cam/topic_name")
        self.intrinsic = CameraIntrinsic.from_dict(rospy.get_param("~cam/intrinsic"))
        self.size = 6.0 * rospy.get_param("~finger_depth")

        self.cv_bridge = cv_bridge.CvBridge()
        self.tf_tree = ros_utils.TransformTree()
        self.integrate = False
        rospy.Subscriber(self.cam_topic_name, sensor_msgs.msg.Image, self.sensor_cb)

    def reset(self):
        self.low_res_tsdf = TSDFVolume(self.size, 40)
        self.high_res_tsdf = TSDFVolume(self.size, 120)

    def sensor_cb(self, msg):
        if not self.integrate:
            return

        img = self.cv_bridge.imgmsg_to_cv2(msg).astype(np.float32) * 0.001
        T_cam_task = self.tf_tree.lookup(
            self.cam_frame_id, "task", msg.header.stamp, rospy.Duration(0.1)
        )

        self.low_res_tsdf.integrate(img, self.intrinsic, T_cam_task)
        self.high_res_tsdf.integrate(img, self.intrinsic, T_cam_task)


def main(args):
    rospy.init_node("panda_grasp")
    panda_grasp = PandaGraspController(args)
    panda_grasp.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="real-world grasp trials")
    parser.add_argument("--method", choices=["vgn", "gpd"], required=True)
    parser.add_argument("--logdir", type=Path, required=True)

    # vgn specific args
    parser.add_argument("--model", type=Path)

    args = parser.parse_args()
    main(args)
