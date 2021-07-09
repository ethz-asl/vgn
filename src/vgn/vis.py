from sensor_msgs.msg import PointCloud2
from rospy import Publisher
from visualization_msgs.msg import MarkerArray

from robot_utils.ros.conversions import *
from vgn.utils import *


def draw_tsdf(vol, voxel_size, threshold=0.01):
    msg = create_vol_msg(vol.squeeze(), voxel_size, threshold)
    pubs["tsdf"].publish(msg)


def draw_points(points):
    msg = to_cloud_msg(points, frame_id="task")
    pubs["points"].publish(msg)


def draw_quality(vol, voxel_size, threshold=0.01):
    msg = create_vol_msg(vol, voxel_size, threshold)
    pubs["quality"].publish(msg)


def draw_volume(vol, voxel_size, threshold=0.01):
    msg = create_vol_msg(vol, voxel_size, threshold)
    pubs["debug"].publish(msg)


def draw_grasps(grasps, finger_depth):
    msg = create_grasp_marker_array("task", grasps, finger_depth)
    pubs["grasps"].publish(msg)


def clear():
    pubs["tsdf"].publish(to_cloud_msg(np.array([]), frame_id="task"))
    pubs["points"].publish(to_cloud_msg(np.array([]), frame_id="task"))
    clear_quality()
    pubs["grasp"].publish(DELETE_MARKER_ARRAY_MSG)
    clear_grasps()
    pubs["debug"].publish(to_cloud_msg(np.array([]), frame_id="task"))


def clear_quality():
    pubs["quality"].publish(to_cloud_msg(np.array([]), frame_id="task"))


def clear_grasps():
    pubs["grasps"].publish(DELETE_MARKER_ARRAY_MSG)


def _create_publishers():
    pubs = dict()
    pubs["tsdf"] = Publisher("/tsdf", PointCloud2, queue_size=1, latch=True)
    pubs["points"] = Publisher("/points", PointCloud2, queue_size=1, latch=True)
    pubs["quality"] = Publisher("/quality", PointCloud2, queue_size=1, latch=True)
    pubs["grasp"] = Publisher("/grasp", MarkerArray, queue_size=1, latch=True)
    pubs["grasps"] = Publisher("/grasps", MarkerArray, queue_size=1, latch=True)
    pubs["debug"] = Publisher("/debug", PointCloud2, queue_size=1, latch=True)
    return pubs


pubs = _create_publishers()
