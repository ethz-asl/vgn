#!/usr/bin/env python3

from pathlib import Path

import rospy

from robot_utils.ros.conversions import to_pose_msg, from_cloud_msg
from robot_utils.utils import map_cloud_to_grid
from vgn.detection import VGN, compute_grasps
from vgn.msg import Grasp
import vgn.srv
from vgn import vis


class VGNServer:
    def __init__(self):
        self.vgn = VGN(Path(rospy.get_param("model")))
        self.finger_depth = rospy.get_param("finger_depth")
        rospy.Service("predict_grasps", vgn.srv.PredictGrasps, self.predict_grasps)

    def predict_grasps(self, req):
        # Construct the input grid
        voxel_size = req.voxel_size
        points, distances = from_cloud_msg(req.map_cloud)
        tsdf_grid = map_cloud_to_grid(voxel_size, points, distances)

        # Compute grasps
        out = vgn.predict(tsdf_grid)
        grasps = compute_grasps(
            voxel_size,
            out,
            score_fn=lambda grasp: grasp.pose.translation[2],
        )

        # Visualize detections
        vis.draw_grasps(grasps, self.finger_depth)

        # Construct the response message
        res = vgn.srv.PredictGraspsResponse()
        res.grasps = [Grasp(to_pose_msg(g.pose), g.width, g.quality) for g in grasps]
        return res


if __name__ == "__main__":
    rospy.init_node("vgn_node")
    VGNServer()
    rospy.spin()
