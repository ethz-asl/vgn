#!/usr/bin/env python

from __future__ import division, print_function

import argparse
from pathlib2 import Path
import pickle


import cv_bridge
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

scan_joints = [
    [
        0.010523236721068734,
        -1.4790833239639014,
        0.10027132190633238,
        -2.416246868493765,
        0.08994288870361117,
        1.4022499498261343,
        0.8516519819506339,
    ],
    [
        -0.5204305512655648,
        -0.7962560020246003,
        0.9121961832112832,
        -1.6720657872605567,
        0.0914094145960278,
        1.2810500415696036,
        0.8710614908889162,
    ],
    [
        0.03202341903301707,
        0.45900370514601985,
        0.0743635250858064,
        -0.8394780465249851,
        0.01546591704007652,
        0.7776030993991428,
        0.8335337665490805,
    ],
    [
        0.0927063864391347,
        -0.6268411712966925,
        -0.5511789947129442,
        -1.8378490985067266,
        0.1656381993108548,
        1.2387742138438755,
        0.9008788375077649,
    ],
]
# tag lies on the table in the center of the workspace
T_base_tag = Transform(Rotation.identity(), [0.42, 0.02, 0.21])


class PandaGraspController(object):
    def __init__(self, args):
        self.base_frame_id = rospy.get_param("~base_frame_id")
        self.tool0_frame_id = rospy.get_param("~tool0_frame_id")
        self.T_tool0_tcp = Transform.from_dict(rospy.get_param("~T_tool0_tcp"))  # TODO
        self.T_tcp_tool0 = self.T_tool0_tcp.inverse()
        self.finger_depth = rospy.get_param("~finger_depth")
        self.size = 6.0 * self.finger_depth

        self.robot = PandaCommander()
        self.robot.move_group.set_end_effector_link(self.tool0_frame_id)
        self.tf_tree = ros_utils.TransformTree()
        self.define_workspace()
        self.create_planning_scene()
        self.tsdf_server = TSDFServer()
        self.plan_grasps = self.select_grasp_planner(args.method)
        self.logger = Logger(args.logdir)

        rospy.loginfo("Ready to take action")

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
        self.robot.scene.add_box("table", msg, size=(0.6, 0.6, 0.02))

        # collision box for camera
        msg = geometry_msgs.msg.PoseStamped()
        msg.header.frame_id = "panda_hand"
        msg.pose.position.x = 0.06
        msg.pose.position.z = 0.03
        self.robot.scene.add_box("camera", msg, size=(0.04, 0.10, 0.04))
        touch_links = self.robot.robot.get_link_names(group=self.robot.name)
        self.robot.scene.attach_box("panda_link8", "camera", touch_links=touch_links)

        rospy.sleep(1.0)  # wait for the scene to be updated

    def select_grasp_planner(self, method):
        if method == "vgn":
            return VGN(args.model)
        elif method == "gpd":
            return GPD()
        else:
            raise ValueError

    def run(self):
        vis.clear()
        vis.draw_workspace()
        self.robot.move_gripper(0.04)
        self.robot.home()

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

        self.robot.home()
        label = self.execute_grasp(grasp)
        rospy.loginfo("Grasp execution")

        self.logger.log_grasp(state, planning_time, grasp, score, label)

    def acquire_tsdf(self):
        self.robot.goto_joints(scan_joints[0])

        self.tsdf_server.reset()
        self.tsdf_server.integrate = True

        for joint_target in scan_joints[1:]:
            self.robot.goto_joints(joint_target)

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

        self.robot.goto_pose(T_base_pregrasp * self.T_tcp_tool0)
        self.approach_grasp(T_base_grasp)
        self.robot.move_gripper(0.0, max_effort=40.0)
        self.robot.goto_pose(T_base_retreat * self.T_tcp_tool0)
        self.drop()

        return True

    def approach_grasp(self, T_base_grasp):
        self.robot.goto_pose(T_base_grasp * self.T_tcp_tool0)

    def drop(self):
        self.robot.goto_joints([0, -0.785, 0, -2.356, 0, 1.57, 0.785], 0.2, 0.2)
        self.robot.goto_joints([-0.9, -0.5, 1.6, -2.1, 0.55, 2.1, 0.75], 0.2, 0.2)
        self.robot.move_gripper(0.04)


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
    rospy.spin()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="real-world grasp trials")
    parser.add_argument("--method", choices=["vgn", "gpd"], required=True)
    parser.add_argument("--logdir", type=Path, required=True)

    # vgn specific args
    parser.add_argument("--model", type=Path)

    args = parser.parse_args()
    main(args)
