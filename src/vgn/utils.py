from math import cos, sin
import matplotlib.colors
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
import ros_numpy
from sensor_msgs.msg import PointCloud2, PointField

from robot_helpers.ros.rviz import *
from robot_helpers.spatial import Transform

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("RedGreen", ["r", "g"])


def map_cloud_to_grid(voxel_size, points, distances):
    # TODO sooooooo slow
    grid = np.zeros((40, 40, 40), dtype=np.float32)
    for idx, point in enumerate(points):
        i, j, k = np.floor(point / voxel_size).astype(int)
        grid[i, j, k] = distances[idx]
    return grid


def camera_on_sphere(origin, radius, theta, phi):
    eye = np.r_[
        radius * sin(theta) * cos(phi),
        radius * sin(theta) * sin(phi),
        radius * cos(theta),
    ]
    target = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])  # this breaks when looking straight down
    return look_at(eye, target, up) * origin.inv()


def look_at(eye, center, up):
    # Returns T_cam_ref
    eye = np.asarray(eye)
    center = np.asarray(center)
    forward = center - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up = np.asarray(up) / np.linalg.norm(up)
    up = np.cross(right, forward)
    m = np.eye(4, 4)
    m[:3, 0] = right
    m[:3, 1] = -up
    m[:3, 2] = forward
    m[:3, 3] = eye
    return Transform.from_matrix(m).inv()


def task_lines(size):
    return [
        ([0.0, 0.0, 0.0], [size, 0.0, 0.0]),
        ([size, 0.0, 0.0], [size, size, 0.0]),
        ([size, size, 0.0], [0.0, size, 0.0]),
        ([0.0, size, 0.0], [0.0, 0.0, 0.0]),
        ([0.0, 0.0, size], [size, 0.0, size]),
        ([size, 0.0, size], [size, size, size]),
        ([size, size, size], [0.0, size, size]),
        ([0.0, size, size], [0.0, 0.0, size]),
        ([0.0, 0.0, 0.0], [0.0, 0.0, size]),
        ([size, 0.0, 0.0], [size, 0.0, size]),
        ([size, size, 0.0], [size, size, size]),
        ([0.0, size, 0.0], [0.0, size, size]),
    ]


def create_vol_msg(vol, voxel_size, threshold):
    vol = vol.squeeze()
    points = np.argwhere(vol > threshold) * voxel_size
    values = np.expand_dims(vol[vol > threshold], 1)
    return to_cloud_msg(points, intensities=values, frame_id="task")


def create_grasp_marker(frame, grasp, finger_depth):
    radius = 0.1 * finger_depth
    w, d = grasp.width, finger_depth
    scale = [radius, 0.0, 0.0]
    color = cmap(float(grasp.quality))
    msg = create_marker(Marker.LINE_STRIP, frame, grasp.pose, scale, color)
    msg.points = [
        to_point_msg(p)
        for p in [[0, -w / 2, d], [0, -w / 2, 0], [0, w / 2, 0], [0, w / 2, d]]
    ]
    return msg


def create_grasp_marker_array(frame, grasps, finger_depth):
    markers = []
    for i, grasp in enumerate(grasps):
        msg = create_grasp_marker(frame, grasp, finger_depth)
        msg.id = i
        markers.append(msg)
    return MarkerArray(markers=markers)


def from_cloud_msg(msg):
    data = ros_numpy.numpify(msg)
    points = np.column_stack((data["x"], data["y"], data["z"]))
    distances = data["distance"]
    return points, distances


def to_cloud_msg(frame, points, colors=None, intensities=None, distances=None):
    msg = PointCloud2()
    msg.header.frame_id = frame

    msg.height = 1
    msg.width = points.shape[0]
    msg.is_bigendian = False
    msg.is_dense = False

    msg.fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
    ]
    msg.point_step = 12
    data = points

    if colors is not None:
        raise NotImplementedError
    elif intensities is not None:
        msg.fields.append(PointField("intensity", 12, PointField.FLOAT32, 1))
        msg.point_step += 4
        data = np.hstack([points, intensities])
    elif distances is not None:
        msg.fields.append(PointField("distance", 12, PointField.FLOAT32, 1))
        msg.point_step += 4
        data = np.hstack([points, distances])

    msg.row_step = msg.point_step * points.shape[0]
    msg.data = data.astype(np.float32).tostring()

    return msg
