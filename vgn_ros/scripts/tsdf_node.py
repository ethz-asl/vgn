#!/usr/bin/env python

from __future__ import print_function, division

import cv_bridge
import numpy as np
import rospy
from sensor_msgs.msg import Image, PointCloud2
import std_msgs.msg
from std_srvs.srv import SetBool, SetBoolResponse, Trigger, TriggerResponse
import open3d as o3d
from rospy.numpy_msg import numpy_msg

from vgn.perception.camera import PinholeCameraIntrinsic
from vgn.perception.integration import TSDFVolume
from vgn.utils.transform import Transform
from vgn_ros import ros_utils
from vgn_ros.srv import GetVolume, GetVolumeResponse


class TSDFNode(object):
    def __init__(self):
        rospy.init_node("tsdf_node")

        self._init()
        self._subscribe_to_camera()
        self._advertise_services()
        self._setup_publishers()

        rospy.sleep(0.5)  # wait for everything to connect
        rospy.spin()

    def toggle(self, req):
        self._integrate = req.data
        return SetBoolResponse(success=True)

    def reset(self, req):
        self._tsdf = TSDFVolume(self._size, 40)
        return TriggerResponse(success=True)

    def get_volume(self, req):
        vol = self._tsdf.get_volume()
        res = GetVolumeResponse(data=vol.flatten())
        return res

    def _init(self):
        tsdf_config = rospy.get_param("tsdf_node")
        cam_config = rospy.get_param("cam")

        self._task_frame_id = tsdf_config["frame_id"]
        self._size = tsdf_config["size"]
        self._publish_rate = tsdf_config["publish_rate"]

        self._intrinsic = PinholeCameraIntrinsic.from_dict(cam_config)
        self._cam_topic_name = cam_config["topic_name"]
        self._cam_frame_id = cam_config["frame_id"]

        self._tsdf = TSDFVolume(self._size, 40)
        self._cv_bridge = cv_bridge.CvBridge()
        self._tf_listener = ros_utils.TransformListener()
        self._integrate = False

    def _subscribe_to_camera(self):
        rospy.Subscriber(self._cam_topic_name, Image, self._image_cb, queue_size=1)

    def _advertise_services(self):
        rospy.Service("~toggle", SetBool, self.toggle)
        rospy.Service("~reset", Trigger, self.reset)
        rospy.Service("~get_volume", GetVolume, self.get_volume)

    def _setup_publishers(self):
        topic_name = "~vol"
        self._vol_pub = rospy.Publisher(topic_name, PointCloud2, queue_size=10)
        rospy.Timer(rospy.Duration(1.0 / self._publish_rate), self._publish_vol)

    def _image_cb(self, img_msg):
        if not self._integrate:
            return
        depth_img = self._cv_bridge.imgmsg_to_cv2(img_msg)
        depth_img = depth_img.astype(np.float32) * 0.001
        extrinsic = self._tf_listener.lookup_transform(
            self._cam_frame_id, self._task_frame_id
        )
        self._tsdf.integrate(depth_img, self._intrinsic, extrinsic)
        rospy.logdebug("Integrated image")

    def _publish_vol(self, _):
        if self._tsdf.extract_point_cloud().is_empty():
            rospy.logdebug("Empty point cloud")
            return
        vol = self._tsdf.get_volume().squeeze()
        points, scalars = vol_to_points(vol, self._tsdf.voxel_size)
        msg = ros_utils.to_point_cloud_msg(points, scalars, frame=self._task_frame_id)
        self._vol_pub.publish(msg)


def vol_to_points(vol, voxel_size=1.0, tol=1e-3):
    points = np.argwhere(vol > tol).astype(np.float32)
    points *= voxel_size
    num_points = points.shape[0]
    scalars = vol[vol > tol].reshape((num_points, 1))
    return points, scalars


if __name__ == "__main__":
    TSDFNode()

