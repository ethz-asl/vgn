"""Common conversions between python objects and ROS messages."""

import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2, PointField


def as_point_cloud_msg(points, colors=None, intensities=None, frame=None, stamp=None):
    """Represent unstructured points as a PointCloud2 message.

    Args:
        points: Point coordinates as array of shape (N,3).
        colors
        intensities
        frame
        stamp

    TODO:
        * Color channel
        * Intensity channel
    """
    msg = PointCloud2()
    msg.height = 1
    msg.width = points.shape[0]
    msg.fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1)]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = msg.point_step * points.shape[0]
    msg.is_dense = False
    msg.data = points.astype(np.float32).tostring()

    msg.header.frame_id = frame
    msg.header.stamp = stamp or rospy.Time.now()

    return msg
