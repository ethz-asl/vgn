import geometry_msgs.msg
import numpy as np
import rospy
import std_msgs.msg
from sensor_msgs.msg import PointCloud2, PointField


def as_point_msg(position):
    """Represent a numpy array as a Point message."""
    msg = geometry_msgs.msg.Point()
    msg.x = position[0]
    msg.y = position[1]
    msg.z = position[2]
    return msg


def as_vector3_msg(vector3):
    """Represent a numpy array as a Vector3 message."""
    msg = geometry_msgs.msg.Vector3()
    msg.x = vector3[0]
    msg.y = vector3[1]
    msg.z = vector3[2]
    return msg


def as_quat_msg(orientation):
    """Represent a `Rotation` object as a Quaternion message."""
    quat = orientation.as_quat()
    msg = geometry_msgs.msg.Quaternion()
    msg.x = quat[0]
    msg.y = quat[1]
    msg.z = quat[2]
    msg.w = quat[3]
    return msg


def as_pose_msg(transform):
    """Represent a `Transform` object as a Pose message."""
    msg = geometry_msgs.msg.Pose()
    msg.position = as_point_msg(transform.translation)
    msg.orientation = as_quat_msg(transform.rotation)
    return msg


def as_color_msg(color):
    """Represent a numpy array as a ColorRGBA message."""
    msg = std_msgs.msg.ColorRGBA()
    msg.r = color[0]
    msg.g = color[1]
    msg.b = color[2]
    msg.a = color[3] if len(color) == 4 else 1.
    return msg


def as_point_cloud_msg(points, frame=None, stamp=None):
    """Represent unstructured points as a PointCloud2 message.

    Args:
        points: Point coordinates as array of shape (N,3).
        frame
        stamp
    """
    msg = PointCloud2()
    msg.height = 1
    msg.width = points.shape[0]
    msg.fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = msg.point_step * points.shape[0]
    msg.is_dense = False
    msg.data = points.astype(np.float32).tostring()

    msg.header.frame_id = frame
    msg.header.stamp = stamp or rospy.Time.now()

    return msg
