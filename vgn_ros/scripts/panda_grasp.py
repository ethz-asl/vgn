#!/usr/bin/env python

from __future__ import division, print_function

from pathlib import Path

import numpy as np
from mayavi import mlab
import rospy
import std_srvs.srv
import time
import torch

from vgn.utils.transform import Rotation, Transform
from vgn.grasp_detector import GraspDetector
import vgn_ros.msg
from vgn_ros.srv import GetVolume, GetVolumeRequest, GetVolumeResponse
from vgn_ros.ros_utils import TransformBroadcaster, TransformListener


class PandaGraspController(object):
    def __init__(self):
        self._init()
        self._configure_grasp_detector()
        self._create_service_proxies()

    def run(self):
        self._calibrate_task_frame()
        tsdf_vol = self._scan_scene()
        grasps = self._detect_grasps(tsdf_vol)
        self._select_grasp()
        self._execute_grasp()

    def _init(self):
        self._base_frame_id = "base_link"
        self._task_frame_id = rospy.get_param("tsdf_node/frame_id")
        self._tag_frame_id = "tag_0"
        self._tf_listener = TransformListener()
        self._tf_broadcaster = TransformBroadcaster()

    def _configure_grasp_detector(self):
        config = rospy.get_param("grasp_detector")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        network_path = Path(config["model_path"])
        self._grasp_detector = GraspDetector(
            device, network_path, show_tsdf=True, show_qual=True, show_detections=True,
        )

    def _create_service_proxies(self):
        reset_srv = "/tsdf_node/reset"
        rospy.wait_for_service(reset_srv)
        self._reset_tsdf = rospy.ServiceProxy(reset_srv, std_srvs.srv.Trigger)

        toggle_srv = "/tsdf_node/toggle"
        rospy.wait_for_service(toggle_srv)
        self._toggle_tsdf = rospy.ServiceProxy(toggle_srv, std_srvs.srv.SetBool)

        get_volume_srv = "/tsdf_node/get_volume"
        rospy.wait_for_service(get_volume_srv)
        self._get_volume = rospy.ServiceProxy(get_volume_srv, GetVolume)

    def _calibrate_task_frame(self):
        tf = self._tf_listener.lookup_transform(self._base_frame_id, self._tag_frame_id)
        self._tf_broadcaster.send_static_transform(
            tf, self._base_frame_id, self._task_frame_id
        )

    def _scan_scene(self):
        self._reset_tsdf()
        self._toggle_tsdf(True)
        rospy.sleep(2.0)
        self._toggle_tsdf(False)

        tsdf_data = self._get_volume().data
        tsdf_vol = np.reshape(tsdf_data, (1, 40, 40, 40)).astype(np.float32)
        return tsdf_vol

    def _detect_grasps(self, tsdf_vol):
        grasps = self._grasp_detector.detect_grasps(tsdf_vol)
        mlab.show()
        return grasps

    def _select_grasp(self):
        pass

    def _execute_grasp(self):
        pass


if __name__ == "__main__":
    rospy.init_node("panda_grasp_controller")
    controller = PandaGraspController()
    controller.run()
