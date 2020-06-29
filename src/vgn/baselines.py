import time

from gpd_ros.msg import GraspConfigList
import numpy as np
from sensor_msgs.msg import PointCloud2
import rospy

from vgn.grasp import Grasp
from vgn.utils import ros_utils
from vgn.utils.transform import Rotation, Transform

# gpd_cloud_pub =


class GPD(object):
    def __init__(self):
        self.input_topic = "/cloud_stitched"
        self.output_topic = "/detect_grasps/clustered_grasps"

        self.cloud_pub = rospy.Publisher(self.input_topic, PointCloud2, queue_size=1)

    def __call__(self, state):
        points = np.asarray(state.pc.points)
        msg = ros_utils.to_cloud_msg(points, frame="world")

        tic = time.time()
        self.cloud_pub.publish(msg)
        result = rospy.wait_for_message(self.output_topic, GraspConfigList)
        grasps, scores = self._to_grasp_list(result)
        toc = time.time() - tic

        return grasps, scores, toc

    def _to_grasp_list(self, grasp_configs):
        grasps, scores = [], []
        for grasp_config in grasp_configs.grasps:
            # orientation
            x_axis = ros_utils.from_vector3_msg(grasp_config.axis)
            y_axis = -ros_utils.from_vector3_msg(grasp_config.binormal)
            z_axis = ros_utils.from_vector3_msg(grasp_config.approach)
            orientation = Rotation.from_dcm(np.vstack([x_axis, y_axis, z_axis]).T)
            # position
            position = ros_utils.from_point_msg(grasp_config.position)
            # width
            width = grasp_config.width.data
            # score
            score = grasp_config.score.data

            # if score < 0.0:
            #     continue  # negative score is larger than positive score (https://github.com/atenpas/gpd/issues/32#issuecomment-387846534)

            grasps.append(Grasp(Transform(orientation, position), width))
            scores.append(score)

        return grasps, scores
