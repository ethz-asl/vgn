from __future__ import print_function

import argparse
import os
import time

import numpy as np
import open3d

import vgn.config as cfg
from vgn import data, utils
from vgn.perception import integration
from vgn.utils import vis


def visualize(args):
    assert os.path.exists(args.scene_dir), 'Directory does not exist'

    if args.rviz:
        from vgn_ros import rviz_utils
        rviz = rviz_utils.RViz()

    scene = data.load_scene(args.scene_dir)
    point_cloud, voxel_grid = data.reconstruct_volume(scene)

    # Plot volume
    tsdf = utils.voxel_grid_to_array(voxel_grid, resolution=cfg.resolution)
    vis.plot_tsdf(tsdf)

    if args.rviz:
        rviz.draw_point_cloud(np.asarray(point_cloud.points))
        rviz.draw_tsdf(voxel_grid, idx=18)
        rviz.draw_candidates(scene['poses'], scene['scores'])

        for pose, score in zip(scene['poses'], scene['scores']):
            if not np.isclose(score, 1.):
                continue
            rviz.draw_grasp_pose(pose)
            time.sleep(1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'scene_dir',
        type=str,
        help='path to data directory of one scene',
    )
    parser.add_argument(
        '--rviz',
        action='store_true',
        help='publish point clouds and grasp poses to Rviz',
    )
    args = parser.parse_args()

    if args.rviz:
        import rospy
        rospy.init_node('data_visualizer')

    visualize(args)


if __name__ == '__main__':
    main()
