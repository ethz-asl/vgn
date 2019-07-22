from __future__ import division, print_function

import argparse
from os import path

import numpy as np
import rospy

from vgn.data import vgn_data
from vgn.perception import integration
from vgn.utils import camera, image
from vgn_ros import rviz_utils


def visualize_scene(data_dir):
    assert path.exists(data_dir), 'Data directory does not exist'

    # Create connection to RViz
    vis = rviz_utils.RViz()

    # Load camera intrinsics
    fname = path.join(data_dir, 'intrinsic.json')
    intrinsic = camera.PinholeCameraIntrinsic.from_json(fname)

    # Load camera extrinsics
    fname = path.join(data_dir, 'extrinsics.csv')
    extrinsics = vgn_data.load_extrinsics(fname)

    # Load grasps
    fname = path.join(data_dir, 'grasps.csv')
    poses, scores = vgn_data.load_grasps(fname)

    # Reconstruct point cloud
    volume = integration.TSDFVolume(length=0.2, resolution=60)
    for i, extrinsic in enumerate(extrinsics):
        depth = image.load(path.join(data_dir, '{0:03d}.png'.format(i)))
        volume.integrate(depth, intrinsic, extrinsic)
    point_cloud = volume.get_point_cloud()
    # volume.draw_point_cloud()

    # Visualize point cloud
    points = np.asarray(point_cloud.points)
    vis.draw_point_cloud(points)

    # Visualize grasps
    vis.draw_candidates(poses, scores)


def main():
    rospy.init_node('data_visualizer')

    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str)
    args = parser.parse_args()

    visualize_scene(args.data_dir)


if __name__ == '__main__':
    main()
