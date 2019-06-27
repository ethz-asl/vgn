import rospy
from sensor_msgs.msg import PointCloud2

from vgn_ros.utils import ros_conversions


def draw_point_cloud(points, colors, frame, topic):
    msg = ros_conversions.as_point_cloud_msg(points=points,
                                             colors=colors,
                                             frame=frame)
    rospy.Publisher(topic, PointCloud2, queue_size=10).publish(msg)
