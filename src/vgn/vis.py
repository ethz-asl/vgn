"""Render volumes, point clouds, and grasp detections in rviz."""

from __future__ import division


import colorsys
import time

import matplotlib.colors
import numpy as np
from sensor_msgs.msg import PointCloud2
import rospy
from rospy import Publisher
from visualization_msgs.msg import Marker, MarkerArray

from vgn.grasp import Grasp, from_voxel_coordinates
from vgn.utils import ros_utils, workspace_lines
from vgn.utils.transform import Transform, Rotation


cmap = matplotlib.colors.LinearSegmentedColormap.from_list("RedGreen", ["r", "g"])
DELETEALL_MSG = Marker(action=Marker.DELETEALL)
SIZE, VOXEL_SIZE, FINGER_DEPTH = 0.0, 0.0, 0.0


def set_size(size):
    global SIZE, VOXEL_SIZE, FINGER_DEPTH
    SIZE = size
    VOXEL_SIZE = SIZE / 40.0
    FINGER_DEPTH = SIZE / 6.0


def draw_workspace():
    scale = SIZE * 0.005
    pose = Transform.identity()
    scale = [scale, 0.0, 0.0]
    color = [0.5, 0.5, 0.5]
    msg = _create_marker_msg(Marker.LINE_LIST, "task", pose, scale, color)
    msg.points = [ros_utils.to_point_msg(point) for point in workspace_lines(SIZE)]
    pubs["workspace"].publish(msg)


def draw_tsdf(vol, threshold=0.01):
    msg = _create_vol_msg(vol, threshold)
    pubs["tsdf"].publish(msg)


def draw_points(points):
    msg = ros_utils.to_cloud_msg(points, frame="task")
    pubs["points"].publish(msg)


def draw_quality(vol, threshold=0.01):
    msg = _create_vol_msg(vol, threshold)
    pubs["quality"].publish(msg)


def draw_volume(vol, threshold=0.01):
    msg = _create_vol_msg(vol, threshold)
    pubs["debug"].publish(msg)


def draw_grasp(grasp, score):
    msg = _create_grasp_marker_msg(grasp, score)
    pubs["grasp"].publish(msg)


def draw_grasps(grasps, scores):
    markers = []
    for i, (grasp, score) in enumerate(zip(grasps, scores)):
        msg = _create_grasp_marker_msg(grasp, score)
        msg.id = i
        markers.append(msg)
    msg = MarkerArray(markers=markers)
    pubs["grasps"].publish(msg)


def clear():
    pubs["workspace"].publish(DELETEALL_MSG)
    pubs["tsdf"].publish(ros_utils.to_cloud_msg(np.array([]), frame="task"))
    pubs["points"].publish(ros_utils.to_cloud_msg(np.array([]), frame="task"))
    pubs["quality"].publish(ros_utils.to_cloud_msg(np.array([]), frame="task"))
    pubs["grasp"].publish(DELETEALL_MSG)
    pubs["grasps"].publish(MarkerArray(markers=[DELETEALL_MSG]))
    pubs["debug"].publish(ros_utils.to_cloud_msg(np.array([]), frame="task"))


def draw_sample(x, y, index):
    tsdf, (label, rotations, width), index = x, y, index
    grasp = Grasp(Transform(Rotation.from_quat(rotations[0]), index), width)
    grasp = from_voxel_coordinates(grasp, VOXEL_SIZE)

    clear()
    draw_workspace()
    draw_tsdf(tsdf.squeeze())
    draw_grasp(grasp, float(label))


def _create_publishers():
    pubs = dict()
    pubs["workspace"] = Publisher("/workspace", Marker, queue_size=1, latch=True)
    pubs["tsdf"] = Publisher("/tsdf", PointCloud2, queue_size=1, latch=True)
    pubs["points"] = Publisher("/points", PointCloud2, queue_size=1, latch=True)
    pubs["quality"] = Publisher("/quality", PointCloud2, queue_size=1, latch=True)
    pubs["grasp"] = Publisher("/grasp", Marker, queue_size=1, latch=True)
    pubs["grasps"] = Publisher("/grasps", MarkerArray, queue_size=1, latch=True)
    pubs["debug"] = Publisher("/debug", PointCloud2, queue_size=1, latch=True)
    return pubs


def _create_marker_msg(marker_type, frame, pose, scale, color):
    msg = Marker()
    msg.header.frame_id = frame
    msg.header.stamp = rospy.Time()
    msg.type = marker_type
    msg.action = Marker.ADD
    msg.pose = ros_utils.to_pose_msg(pose)
    msg.scale = ros_utils.to_vector3_msg(scale)
    msg.color = ros_utils.to_color_msg(color)
    return msg


def _create_vol_msg(vol, threshold):
    points = np.argwhere(vol > threshold) * VOXEL_SIZE
    values = np.expand_dims(vol[vol > threshold], 1)
    return ros_utils.to_cloud_msg(points, values, frame="task")


def _create_grasp_marker_msg(grasp, score):
    radius = 0.1 * FINGER_DEPTH
    w, d = grasp.width, FINGER_DEPTH
    scale = [radius, 0.0, 0.0]
    color = cmap(float(score))
    msg = _create_marker_msg(Marker.LINE_LIST, "task", grasp.pose, scale, color)
    msg.points = [ros_utils.to_point_msg(point) for point in _gripper_lines(w, d)]
    return msg


def _gripper_lines(width, depth):
    return [
        [0.0, 0.0, -depth / 2.0],
        [0.0, 0.0, 0.0],
        [0.0, -width / 2.0, 0.0],
        [0.0, -width / 2.0, depth],
        [0.0, width / 2.0, 0.0],
        [0.0, width / 2.0, depth],
        [0.0, -width / 2.0, 0.0],
        [0.0, width / 2.0, 0.0],
    ]


pubs = _create_publishers()
