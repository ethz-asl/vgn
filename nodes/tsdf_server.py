#!/usr/bin/env python3

import cv_bridge
import rospy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
import std_srvs.srv

from robot_helpers.ros import tf
from robot_helpers.ros.conversions import *
from vgn.perception import UniformTSDFVolume
import vgn.srv
from vgn.utils import *


class UniformTSDFServer:
    def __init__(self):
        self.load_parameters()
        tf.init()
        self.init_topics()
        self.advertise_services()
        self.cv_bridge = cv_bridge.CvBridge()
        self.integrate = False
        rospy.loginfo("TSDF server ready")

    def load_parameters(self):
        self.frame_id = rospy.get_param("~frame_id")
        self.length = rospy.get_param("~length")
        self.resolution = rospy.get_param("~resolution")
        self.depth_scaling = rospy.get_param("~depth_scaling")
        self.cam_frame_id = rospy.get_param("~camera/frame_id")
        info_topic = rospy.get_param("~camera/info_topic")
        self.depth_topic = rospy.get_param("~camera/depth_topic")
        msg = rospy.wait_for_message(info_topic, CameraInfo)
        self.intrinsic = from_camera_info_msg(msg)

    def init_topics(self):
        self.scene_cloud_pub = rospy.Publisher("scene_cloud", PointCloud2, queue_size=1)
        self.map_cloud_pub = rospy.Publisher("map_cloud", PointCloud2, queue_size=1)
        rospy.Subscriber(self.depth_topic, Image, self.sensor_cb)

    def advertise_services(self):
        rospy.Service("reset_map", std_srvs.srv.Trigger, self.reset)
        rospy.Service("toggle_integration", std_srvs.srv.SetBool, self.toggle)
        rospy.Service("get_scene_cloud", vgn.srv.GetSceneCloud, self.get_scene_cloud)
        rospy.Service("get_map_cloud", vgn.srv.GetMapCloud, self.get_map_cloud)

    def reset(self, req):
        self.tsdf = UniformTSDFVolume(self.length, self.resolution)
        return std_srvs.srv.TriggerResponse(success=True)

    def toggle(self, req):
        self.integrate = req.data
        return std_srvs.srv.SetBoolResponse(success=True)

    def sensor_cb(self, msg):
        if self.integrate:
            depth = (
                self.cv_bridge.imgmsg_to_cv2(msg).astype(np.float32)
                * self.depth_scaling
            )
            extrinsic = tf.lookup(
                self.cam_frame_id, self.frame_id, msg.header.stamp, rospy.Duration(0.1)
            )
            self.tsdf.integrate(depth, self.intrinsic, extrinsic)

    def get_scene_cloud(self, req):
        scene_cloud = self.tsdf.get_scene_cloud()
        points = np.asarray(scene_cloud.points)
        msg = to_cloud_msg(self.frame_id, points)
        self.scene_cloud_pub.publish(msg)
        res = vgn.srv.GetSceneCloudResponse()
        res.scene_cloud = msg
        return res

    def get_map_cloud(self, req):
        map_cloud = self.tsdf.get_map_cloud()
        points = np.asarray(map_cloud.points)
        distances = np.asarray(map_cloud.colors)[:, [0]]
        msg = to_cloud_msg(self.frame_id, points, distances=distances)
        self.map_cloud_pub.publish(msg)
        res = vgn.srv.GetMapCloudResponse()
        res.voxel_size = self.tsdf.voxel_size
        res.map_cloud = msg
        return res


if __name__ == "__main__":
    rospy.init_node("tsdf_server")
    UniformTSDFServer()
    rospy.spin()
