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


class PandaGraspController(object):
    def __init__(self, args):
        self.finger_depth = rospy.get_param("~finger_depth")
        self.T_tool0_tcp = Transform.from_dict(rospy.get_param("~T_tool0_tcp"))
        self.T_tool0_cam = Transform.from_dict(rospy.get_param("~T_tool0_cam"))
        self.intrinsic = CameraIntrinsic.from_dict(rospy.get_param("~cam_intrinsic"))
        self.size = 6.0 * self.finger_depth
        self.logger = Logger(args.logdir)
        self.tf_tree = ros_utils.TransformTree()
        self.cv_bridge = cv_bridge.CvBridge()
        self.robot = PandaCommander()
        self.setup_grasp_planner(args.method)
        rospy.loginfo("Ready to take action")

    def setup_grasp_planner(self, method):
        if method == "vgn":
            self.grasp_planner = VGN(rospy.get_param("~vgn/model_path"))
        elif method == "gpd":
            self.grasp_planner = GPD()
        else:
            raise ValueError

    def calibrate_workspace(self):
        self.robot.home()
        # self.T_base_task = self.tf_tree.lookup("panda_link0", "tag_0", rospy.Time(0))
        self.T_base_task = Transform(Rotation.identity(), np.r_[0.3, -0.15, 0.2])
        self.tf_tree.broadcast_static(self.T_base_task, "panda_link0", "task")
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
        tsdf = TSDFVolume(self.size, 40)
        high_res_tsdf = TSDFVolume(self.size, 120)

        def cb(msg):
            img = self.cv_bridge.imgmsg_to_cv2(msg).astype(np.float32) * 0.001
            stamp = msg.header.stamp
            T_base_tool0 = self.tf_tree.lookup("panda_link0", "panda_link8", stamp)
            T_cam_task = (T_base_tool0 * self.T_tool0_cam).inverse()
            tsdf.integrate(img, self.intrinsic, T_cam_task)
            high_res_tsdf = TSDFVolume(self.size, 120)

        waypoints = self.generate_scan_trajectory()
        self.robot.goto_pose_target(waypoints[0], 0.4, 0.4)

        sub = rospy.Subscriber("/camera", sensor_msgs.msg.Image, cb)
        rospy.sleep(1.0)
        self.robot.follow_cartesian_waypoints(waypoints, 0.1, 0.1)
        sub.unregister()
        rospy.sleep(1.0)

        pc = high_res_tsdf.extract_point_cloud()

        vis.workspace(self.size)
        vis.tsdf(tsdf.get_volume().squeeze(), tsdf.voxel_size)
        vis.points(np.asarray(pc.points))

        return tsdf, pc

    def generate_scan_trajectory(self):
        identity = Rotation.identity()
        T_task_origin = Transform(identity, [self.size / 2.0, self.size / 2.0, 0.2])
        origin = self.T_base_task * T_task_origin
        radius = 0.05
        theta = np.pi / 8.0

        waypoints = []
        for phi in np.linspace(-3.0 * np.pi / 5.0, 3 * np.pi / 5.0, 6):
            T_cam_base = camera_on_sphere(origin, radius, theta, phi)
            T_base_tool0 = (self.T_tool0_cam * T_cam_base).inverse()
            waypoints.append(ros_utils.to_pose_msg(T_base_tool0))

        return waypoints

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


def main(args):
    rospy.init_node("panda_grasp")
    panda_grasp = PandaGraspController(args)

    def joy_cb(msg):
        if msg.buttons[0]:  #   x
            panda_grasp.run()
        elif msg.buttons[1]:  # ○
            pass
        elif msg.buttons[2]:  # △
            pass
        elif msg.buttons[3]:  # □
            panda_grasp.calibrate_workspace()

    rospy.Subscriber("/joy", sensor_msgs.msg.Joy, joy_cb)
    rospy.spin()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="real-world grasp trials")
    parser.add_argument("--method", choices=["vgn", "gpd"], required=True)
    parser.add_argument("--logdir", type=Path, required=True)
    args = parser.parse_args()
    main(args)
