import geometry_msgs.msg
import matplotlib.colors
import matplotlib.cm
import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker

from vgn_ros.utils import ros_utils


def draw_point_cloud(points, colors):
    """Draw a point cloud in rviz."""
    msg = ros_utils.as_point_cloud_msg(points, colors, frame='task')
    ros_utils.publish(msg, '/reconstruction')


def draw_candidate(pose):
    """Draw the frame of a single candidate grasp pose."""
    msg = geometry_msgs.msg.PoseStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = 'task'
    msg.pose = ros_utils.as_pose_msg(pose)
    ros_utils.publish(msg, '/grasp_pose')


def draw_candidates(poses, scores, topic='/grasp_candidates'):
    """Draw grasp candidates as arrows colored according to their score."""
    remove_all_markers(topic)

    cnorm = matplotlib.colors.Normalize(vmin=0., vmax=1.0)
    cmap = matplotlib.cm.get_cmap('winter')
    scalar_cmap = matplotlib.cm.ScalarMappable(norm=cnorm, cmap=cmap)

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
            ros_utils.as_point_msg(start_point),
            ros_utils.as_point_msg(end_point)
        ]
        marker.scale = ros_utils.as_vector3_msg(scale)
        marker.color = ros_utils.as_color_msg(color)

        ros_utils.publish(marker, topic)


def remove_all_markers(topic):
    marker = Marker()
    marker.action = Marker.DELETEALL
    ros_utils.publish(marker, topic)
