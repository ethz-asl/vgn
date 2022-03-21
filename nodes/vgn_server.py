#!/usr/bin/env python3

from pathlib import Path
import rospy

from vgn.detection import VGN, select_local_maxima
from vgn.rviz import Visualizer
import vgn.srv
from vgn.utils import *


class VGNServer:
    def __init__(self):
        self.frame = rospy.get_param("~frame_id")
        self.vgn = VGN(Path(rospy.get_param("~model")))
        rospy.Service("predict_grasps", vgn.srv.PredictGrasps, self.predict_grasps)
        self.vis = Visualizer()
        rospy.loginfo("VGN server ready")

    def predict_grasps(self, req):
        # Construct the input grid
        voxel_size = req.voxel_size
        points, distances = from_cloud_msg(req.map_cloud)
        tsdf_grid = map_cloud_to_grid(voxel_size, points, distances)

        # Compute grasps
        out = self.vgn.predict(tsdf_grid)
        grasps, qualities = select_local_maxima(voxel_size, out, threshold=0.9)

        # Visualize detections
        self.vis.grasps(self.frame, grasps, qualities)

        # Construct the response message
        res = vgn.srv.PredictGraspsResponse()
        res.grasps = [to_grasp_config_msg(g, q) for g, q in zip(grasps, qualities)]
        return res


if __name__ == "__main__":
    rospy.init_node("vgn_server")
    VGNServer()
    rospy.spin()
