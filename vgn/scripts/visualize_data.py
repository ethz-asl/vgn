from __future__ import print_function

import argparse
import os
import time

import numpy as np
import open3d
import rospy

from vgn import dataset
from vgn.perception import integration
from vgn_ros import rviz_utils


def visualize_scene(sample_dir):
    assert os.path.exists(sample_dir), 'Directory does not exist'

    # Create connection to RViz
    vis = rviz_utils.RViz()

    # Load data
    sample = dataset.load_scene_data(sample_dir)
    intrinsic = sample['intrinsic']

    # Reconstruct point cloud
    volume = integration.TSDFVolume(size=0.2, resolution=60)
    for extrinsic, depth_img in zip(sample['extrinsics'], sample['images']):
        volume.integrate(depth_img, intrinsic, extrinsic)
    point_cloud = volume.get_point_cloud()
    # open3d.draw_geometries([point_cloud])

    # Visualize point cloud
    points = np.asarray(point_cloud.points)
    vis.draw_point_cloud(points)

    # Visualize grasps
    vis.draw_candidates(sample['poses'], sample['scores'])

    # Visialize TSDF
    voxel_grid = volume.get_voxel_grid()
    vis.draw_tsdf(voxel_grid, slice_x=30)

    # Iterate over good grasps and draw their pose
    for pose, score in zip(sample['poses'], sample['scores']):
        if not np.isclose(score, 1.):
            continue
        vis.draw_grasp_pose(pose)
        time.sleep(1.0)


def main():
    rospy.init_node('data_visualizer')

    parser = argparse.ArgumentParser()
    parser.add_argument('sample_dir', type=str, help='Directory of one sample')
    args = parser.parse_args()

    visualize_scene(args.sample_dir)


if __name__ == '__main__':
    main()
