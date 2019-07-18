from __future__ import division, print_function

import argparse
from os import path

import numpy as np
import rospy

from vgn.perception import integration
from vgn.utils import camera, image
from vgn.utils.transform import Rotation, Transform
from vgn_ros import rviz_utils


def main():
    rospy.init_node('data_visualizer')

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir',
                        type=str,
                        default='data/debug/tmp',
                        help='Path to data of one generated scene')
    args = parser.parse_args()

    # Create connection to RViz
    vis = rviz_utils.RViz()

    # Load the data
    intrinsic = camera.PinholeCameraIntrinsic.load(
        path.join(args.datadir, 'intrinsic.json'))
    extrinsics = _load_transforms(path.join(args.datadir, 'extrinsics.txt'))
    grasps = _load_transforms(path.join(args.datadir, 'grasps.txt'))
    scores = np.loadtxt(path.join(args.datadir, 'scores.txt'), delimiter=',')

    # Reconstruct point cloud
    volume = integration.TSDFVolume(length=0.2, resolution=60)
    for i, extrinsic in enumerate(extrinsics):
        depth = image.load(path.join(args.datadir, '{0:03d}.png'.format(i)))
        volume.integrate(depth, intrinsic, extrinsic)
    point_cloud = volume.extract_point_cloud()
    # volume.draw_point_cloud()

    # Visualize point cloud
    points = np.asarray(point_cloud.points)
    vis.draw_point_cloud(points)

    # Visualize grasps
    vis.draw_candidates(grasps, scores)


def _load_transforms(fname):
    transforms = []
    for v in np.loadtxt(fname, delimiter=','):
        transforms.append(Transform(Rotation.from_quat(v[3:7]), v[:3]))
    return transforms


if __name__ == '__main__':
    main()
