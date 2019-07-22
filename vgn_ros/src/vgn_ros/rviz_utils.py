import time

import matplotlib.cm
import matplotlib.colors
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray

from vgn_ros import ros_utils


class RViz(object):
    def __init__(self, frame='task'):
        self._pubs = dict()
        self._pubs['point_cloud'] = rospy.Publisher('/point_cloud',
                                                    PointCloud2,
                                                    queue_size=10)
        self._pubs['grasp_pose'] = rospy.Publisher('/grasp_pose',
                                                   PoseStamped,
                                                   queue_size=10)
        self._pubs['candidates'] = rospy.Publisher('/grasp_candidates',
                                                   MarkerArray,
                                                   queue_size=10)
        time.sleep(1.0)

    def draw_point_cloud(self, points):
        msg = ros_utils.to_point_cloud_msg(points, frame='task')
        self._pubs['point_cloud'].publish(msg)

    def draw_grasp_pose(self, pose):
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'task'
        msg.pose = ros_utils.to_pose_msg(pose)
        self._pubs['grasp_pose'].publish(msg)

    def draw_candidates(self, poses, scores):
        """Draw grasp candidates as arrows colored according to their score."""
        marker = Marker(action=Marker.DELETEALL)
        self._pubs['candidates'].publish(MarkerArray(markers=[marker]))

        cnorm = matplotlib.colors.Normalize(vmin=0., vmax=1.0)
        cmap = matplotlib.cm.get_cmap('winter')
        scalar_cmap = matplotlib.cm.ScalarMappable(norm=cnorm, cmap=cmap)

        marker_array = MarkerArray()
        for i, (pose, score) in enumerate(zip(poses, scores)):
            start_point = pose.translation
            end_point = pose.transform_point(np.array([0., 0., -0.02]))
            scale = [0.002, 0.004, 0.]
            color = scalar_cmap.to_rgba(score)

            marker = Marker()
            marker.header.frame_id = 'task'
            marker.header.stamp = rospy.Time.now()

            marker.id = i
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.lifetime = rospy.Duration()

            marker.points = [
                ros_utils.to_point_msg(start_point),
                ros_utils.to_point_msg(end_point),
            ]
            marker.scale = ros_utils.to_vector3_msg(scale)
            marker.color = ros_utils.to_color_msg(color)

            marker_array.markers.append(marker)

        self._pubs['candidates'].publish(marker_array)
