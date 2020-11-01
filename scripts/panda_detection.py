#!/usr/bin/env python

"""
Real-time grasp detection.
"""

import argparse
from pathlib import Path
import time

import cv_bridge
import numpy as np
import rospy
import sensor_msgs.msg
import torch

from vgn import vis
from vgn.detection import *
from vgn.perception import *
from vgn.utils import ros_utils
from vgn.utils.transform import Rotation, Transform


class GraspDetectionServer(object):
    def __init__(self, model_path):
        # define frames
        self.task_frame_id = "task"
        self.cam_frame_id = "camera_depth_optical_frame"
        self.T_cam_task = Transform(
            Rotation.from_quat([-0.679, 0.726, -0.074, -0.081]), [0.166, 0.101, 0.515]
        )

        # broadcast the tf tree (for visualization)
        self.tf_tree = ros_utils.TransformTree()
        self.tf_tree.broadcast_static(
            self.T_cam_task, self.cam_frame_id, self.task_frame_id
        )

        # define camera parameters
        self.cam_topic_name = "/camera/depth/image_rect_raw"
        self.intrinsic = CameraIntrinsic(640, 480, 383.265, 383.26, 319.39, 242.43)

        # setup a CV bridge
        self.cv_bridge = cv_bridge.CvBridge()

        # construct the grasp planner object
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = load_network(model_path, self.device)

        # initialize the visualization
        vis.clear()
        vis.draw_workspace(0.3)

        # subscribe to the camera
        self.img = None
        rospy.Subscriber(self.cam_topic_name, sensor_msgs.msg.Image, self.sensor_cb)

        # setup cb to detect grasps
        rospy.Timer(rospy.Duration(0.1), self.detect_grasps)

    def sensor_cb(self, msg):
        self.img = self.cv_bridge.imgmsg_to_cv2(msg).astype(np.float32) * 0.001

    def detect_grasps(self, _):
        if self.img is None:
            return

        tic = time.time()
        self.tsdf = TSDFVolume(0.3, 40)
        self.tsdf.integrate(self.img, self.intrinsic, self.T_cam_task)
        print("Construct tsdf ", time.time() - tic)

        tic = time.time()
        tsdf_vol = self.tsdf.get_grid()
        voxel_size = self.tsdf.voxel_size
        print("Extract grid  ", time.time() - tic)

        tic = time.time()
        qual_vol, rot_vol, width_vol = predict(tsdf_vol, self.net, self.device)
        print("Forward pass   ", time.time() - tic)

        tic = time.time()
        qual_vol, rot_vol, width_vol = process(tsdf_vol, qual_vol, rot_vol, width_vol)
        print("Filter         ", time.time() - tic)

        vis.draw_quality(qual_vol, voxel_size, threshold=0.01)

        tic = time.time()
        grasps, scores = select(qual_vol, rot_vol, width_vol, 0.90, 1)
        num_grasps = len(grasps)
        if num_grasps > 0:
            idx = np.random.choice(num_grasps, size=min(5, num_grasps), replace=False)
            grasps, scores = np.array(grasps)[idx], np.array(scores)[idx]
        grasps = [from_voxel_coordinates(g, voxel_size) for g in grasps]
        print("Select grasps  ", time.time() - tic)

        vis.clear_grasps()
        rospy.sleep(0.01)
        tic = time.time()
        vis.draw_grasps(grasps, scores, 0.05)
        print("Visualize      ", time.time() - tic)

        self.img = None
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()

    rospy.init_node("panda_detection")
    GraspDetectionServer(args.model)
    rospy.spin()
