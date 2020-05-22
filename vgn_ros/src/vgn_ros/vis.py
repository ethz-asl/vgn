import colorsys
import time

from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from sensor_msgs.msg import PointCloud2
import rospy
from rospy import Publisher
from visualization_msgs.msg import Marker, MarkerArray

from vgn.utils import vis
from vgn.utils.transform import Transform, Rotation
from vgn_ros import utils


cmap = LinearSegmentedColormap.from_list("BluRe", ["b", "r"])


def clear():
    delete_all_msg = Marker(action=Marker.DELETEALL)
    pubs["workspace"].publish(delete_all_msg)
    pubs["points"].publish(utils.to_point_cloud_msg(np.array([]), frame="task"))
    pubs["grasps"].publish(MarkerArray(markers=[delete_all_msg]))


def workspace(size, scale=0.002):
    pose = Transform.identity()
    scale = [scale, 0.0, 0.0]
    color = [0.5, 0.5, 0.5]
    msg = _create_marker_msg(Marker.LINE_LIST, "task", pose, scale, color)
    msg.points = [utils.to_point_msg(point) for point in vis.workspace_lines(size)]
    pubs["workspace"].publish(msg)


def points(points):
    msg = utils.to_point_cloud_msg(points, frame="task")
    pubs["points"].publish(msg)


def grasps(grasps, scores, finger_depth, radius=0.005):
    markers = []
    for i, (grasp, score) in enumerate(zip(grasps, scores)):
        w, d = grasp.width, finger_depth
        scale = [radius, 0.0, 0.0]
        color = cmap(float(score))
        msg = _create_marker_msg(Marker.LINE_LIST, "task", grasp.pose, scale, color)
        msg.id = i
        msg.points = [utils.to_point_msg(point) for point in vis.gripper_lines(w, d)]
        markers.append(msg)
    msg = MarkerArray(markers=markers)
    pubs["grasps"].publish(msg)


def tsdf(vol, voxel_size, threshold=0.01):
    msg = _create_vol_msg(vol, voxel_size, threshold)
    pubs["tsdf"].publish(msg)


def quality(vol, voxel_size, threshold=0.01):
    msg = _create_vol_msg(vol, voxel_size, threshold)
    pubs["quality"].publish(msg)


def _create_marker_msg(marker_type, frame, pose, scale, color):
    msg = Marker()
    msg.header.frame_id = frame
    msg.header.stamp = rospy.Time()
    msg.type = marker_type
    msg.action = Marker.ADD
    msg.pose = utils.to_pose_msg(pose)
    msg.scale = utils.to_vector3_msg(scale)
    msg.color = utils.to_color_msg(color)
    return msg


def _create_vol_msg(vol, voxel_size, threshold):
    points = np.argwhere(vol > threshold) * voxel_size
    values = np.expand_dims(vol[vol > threshold], 1)
    return utils.to_point_cloud_msg(points, values, frame="task")


def _create_publishers():
    pubs = dict()
    pubs["workspace"] = Publisher("/workspace", Marker, queue_size=1, latch=True)
    pubs["points"] = Publisher("/points", PointCloud2, queue_size=1, latch=True)
    pubs["grasps"] = Publisher("/grasps", MarkerArray, queue_size=1, latch=True)
    pubs["tsdf"] = Publisher("/tsdf", PointCloud2, queue_size=1, latch=True)
    pubs["quality"] = Publisher("/quality", PointCloud2, queue_size=1, latch=True)
    return pubs


pubs = _create_publishers()
