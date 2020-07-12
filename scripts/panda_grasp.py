#!/usr/bin/env python

"""
Dualshock 4 interface:
  x  starts a grasp trial,
  ○  cancles a grasp trial,
  △  triggers a new round (for logging),
  □  triggers workspace calibration.
"""

from __future__ import division, print_function

import argparse
from pathlib2 import Path
import copy


import cv_bridge
import geometry_msgs.msg
import numpy as np
from panda_control.panda_commander import PandaCommander
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


scan_joint_positions = [
    [
        -0.43108035333952316,
        -0.9870793052137942,
        0.6965837768545206,
        -2.0120709856722176,
        0.12508303126065481,
        1.3463392878108555,
        0.8366453268933625,
    ],
    [
        -0.1351051290959735,
        0.11679895606962587,
        0.3614310921786124,
        -1.175866540273047,
        0.10791277298602031,
        0.946452884727054,
        1.0446157227800703,
    ],
    [
        -0.7980986830856897,
        0.0072509909150631794,
        0.3052539200105668,
        -1.4125830988298362,
        0.35131070071250914,
        0.9115777040304364,
        0.8701020664779676,
    ],
]


class PandaGraspController(object):
    def __init__(self, args):
        self.base_frame_id = rospy.get_param("~base_frame_id")
        self.tool0_frame_id = rospy.get_param("~tool0_frame_id")
        self.T_tool0_tcp = Transform.from_dict(rospy.get_param("~T_tool0_tcp"))  # check
        self.size = 6.0 * rospy.get_param("~finger_depth")

        self.robot = PandaCommander()
        self.tf_tree = ros_utils.TransformTree()
        self.tsdf_server = TSDFServer()
        self.grasp_planner = self.select_grasp_planner(args.method)
        self.logger = Logger(args.logdir)

        rospy.loginfo("Ready to take action")

    def select_grasp_planner(self, method):
        if method == "vgn":
            return VGN(rospy.get_param("~vgn/model_path"))
        elif method == "gpd":
            return GPD()
        else:
            raise ValueError

    def calibrate_workspace(self):
        vis.clear_workspace()
        self.robot.home()

        # T_base_tag = self.tf_tree.lookup(self.base_frame_id, "tag_0", rospy.Time(0))
        R_base_tag = Rotation.from_quat([-0.00889, -0.01100, -0.71420, 0.69979])
        t_base_tag = [0.38075931, 0.00874077, 0.03670042]
        T_base_tag = Transform(R_base_tag, t_base_tag)

        z_offset = -0.05
        t_tag_task = np.r_[[-0.5 * self.size, -0.5 * self.size, z_offset]]
        T_tag_task = Transform(Rotation.identity(), t_tag_task)
        self.T_base_task = T_base_tag * T_tag_task

        self.tf_tree.broadcast_static(self.T_base_task, self.base_frame_id, "task")
        rospy.sleep(1.0)  # wait for the TF to be broadcasted

        msg = geometry_msgs.msg.PoseStamped()
        msg.header.frame_id = "task"
        msg.pose.position.x = 0.15
        msg.pose.position.y = 0.15
        msg.pose.position.z = -z_offset - 0.01
        self.robot.scene.add_box("table", msg, size=(0.5, 0.5, 0.02))
        rospy.sleep(1.0)  # wait for the scene to be updated

        vis.workspace(self.size)
        rospy.loginfo("Workspace calibrated")

    def run(self):
        vis.clear()
        self.robot.home()

        tsdf, pc = self.acquire_tsdf()
        state = State(tsdf, pc)
        grasps, scores, planning_time = self.plan_grasps(state)
        if len(grasps) == 0:
            return
        grasp, score = self.select_grasp(grasps, scores)
        label = self.execute_grasp(grasp)

        self.logger.log_grasp(state, planning_time, grasp, score, label)

    def acquire_tsdf(self):
        self.tsdf_server.reset()
        self.tsdf_server.integrate = True

        for joint_target in scan_joint_positions:
            self.robot.goto_joint_target(joint_target)

        self.tsdf_server.integrate = False
        tsdf = self.tsdf_server.low_res_tsdf
        pc = self.tsdf_server.high_res_tsdf.extract_point_cloud()

        vis.tsdf(tsdf.get_volume().squeeze(), tsdf.voxel_size)
        vis.points(np.asarray(pc.points))

        return tsdf, pc

    def plan_grasps(self, state):
        grasps, scores, planning_time = self.grasp_planner.plan(state)
        vis.grasps(grasps, scores, self.finger_depth)
        return grasps, scores, planning_time

    def select_grasp(self, grasps, scores):
        grasp, score = grasps[0], scores[0]
        return grasp, score

    def execute_grasp(self, grasp):
        T_task_grasp = grasp.pose
        T_base_grasp = self.T_base_task * T_task_grasp
        T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.05])
        T_grasp_retreat = Transform(Rotation.identity(), [0.0, 0.0, -0.1])
        T_base_pregrasp = T_base_grasp * T_grasp_pregrasp
        T_base_retreat = T_base_grasp * T_grasp_retreat

        self.robot.goto_pose_target(T_base_pregrasp)
        self.approach_grasp(T_base_grasp)
        self.robot.grasp()
        self.robot.goto_pose_target(T_base_retreat)
        self.drop()

        # check success

        return True


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
        T_cam_task = self.tf_tree.lookup(self.cam_frame_id, "task", msg.header.stamp)

        self.low_res_tsdf.integrate(img, self.intrinsic, T_cam_task)
        self.high_res_tsdf.integrate(img, self.intrinsic, T_cam_task)


def main(args):
    rospy.init_node("panda_grasp")
    panda_grasp = PandaGraspController(args)

    panda_grasp.calibrate_workspace()
    panda_grasp.run()

    def joy_cb(msg):
        if msg.buttons[0]:  #   x
            panda_grasp.run()
        elif msg.buttons[1]:  # ○
            pass
        elif msg.buttons[2]:  # △
            pass
        elif msg.buttons[3]:  # □
            pass

    rospy.Subscriber("/joy", sensor_msgs.msg.Joy, joy_cb)
    rospy.spin()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="real-world grasp trials")
    parser.add_argument("--method", choices=["vgn", "gpd"], required=True)
    parser.add_argument("--logdir", type=Path, required=True)
    args = parser.parse_args()
    main(args)
