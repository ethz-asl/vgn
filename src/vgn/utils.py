from math import cos, sin
import matplotlib.colors
import numpy as np
import ros_numpy
from sensor_msgs.msg import PointCloud2, PointField

from robot_helpers.ros.rviz import *
from robot_helpers.spatial import Transform

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("RedGreen", ["r", "g"])


def map_cloud_to_grid(voxel_size, points, distances):
    grid = np.zeros((40, 40, 40), dtype=np.float32)
    indices = (points // voxel_size).astype(int)
    grid[tuple(indices.T)] = distances.squeeze()
    return grid


def grid_to_map_cloud(voxel_size, grid, threshold=1e-2):
    points = np.argwhere(grid > threshold) * voxel_size
    distances = np.expand_dims(grid[grid > threshold], 1)
    return points, distances


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
