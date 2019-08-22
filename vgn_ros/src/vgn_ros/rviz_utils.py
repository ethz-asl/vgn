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
                                                    queue_size=1)
        self._pubs['tsdf'] = rospy.Publisher('/tsdf',
                                             PointCloud2,
                                             queue_size=1)
        self._pubs['grasp_pose'] = rospy.Publisher('/grasp_pose',
                                                   PoseStamped,
                                                   queue_size=1)
        self._pubs['candidates'] = rospy.Publisher('/candidates',
                                                   PointCloud2,
                                                   queue_size=1)
        self._pubs['true_false'] = rospy.Publisher('/true_false',
                                                   PointCloud2,
                                                   queue_size=1)

        time.sleep(1.0)

    def draw_point_cloud(self, points):
        msg = ros_utils.to_point_cloud_msg(points, frame='task')
        self._pubs['point_cloud'].publish(msg)

    def draw_tsdf(self, voxel_grid, idx):
        if idx is not None:
            fn = lambda voxel: voxel.grid_index[0] == idx
            voxels = filter(fn, voxel_grid.voxels)
        else:
            voxels = voxel_grid.voxels

        n_voxels = len(voxels)
        voxel_size = voxel_grid.voxel_size

        points = np.empty((n_voxels, 3))
        intensities = np.empty((n_voxels, 1))
        for i, voxel in enumerate(voxels):
            ix, iy, iz = voxel.grid_index
            points[i] = [ix * voxel_size, iy * voxel_size, iz * voxel_size]
            intensities[i] = voxel.color[0]

        msg = ros_utils.to_point_cloud_msg(points, intensities, frame='task')
        self._pubs['tsdf'].publish(msg)

    def draw_grasp_pose(self, pose):
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'task'
        msg.pose = ros_utils.to_pose_msg(pose)
        self._pubs['grasp_pose'].publish(msg)

    def draw_candidates(self, poses, scores):
        points = np.reshape([p.translation for p in poses], (len(poses), 3))
        scores = np.expand_dims(scores, 1)
        msg = ros_utils.to_point_cloud_msg(points,
                                           intensities=scores,
                                           frame='task')
        self._pubs['candidates'].publish(msg)

    def draw_true_false(self, poses, trues):
        points = np.reshape([p.translation for p in poses], (len(poses), 3))
        msg = ros_utils.to_point_cloud_msg(points,
                                           intensities=trues,
                                           frame='task')
        self._pubs['true_false'].publish(msg)
