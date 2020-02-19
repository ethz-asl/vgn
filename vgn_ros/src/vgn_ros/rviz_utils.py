import time

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray

from vgn_ros import ros_utils


class RViz(object):
    def __init__(self, frame="world"):
        self.frame = frame
        self.create_publishers()
        time.sleep(1.0)

    def create_publishers(self):
        self.pubs = dict()
        self.pubs["point_cloud"] = rospy.Publisher(
            "/point_cloud", PointCloud2, queue_size=1
        )

    def draw_point_cloud(self, points):
        msg = ros_utils.to_point_cloud_msg(points, frame=self.frame)
        self.pubs["point_cloud"].publish(msg)
        print("Published points")

    def draw_grasp(self, grasp, quality):
        pass
