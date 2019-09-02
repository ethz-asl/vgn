from __future__ import print_function

import argparse
import os
import time

import numpy as np
import open3d
from mayavi import mlab

from vgn import data
from vgn.dataset import VGNDataset
from vgn.utils import vis


def visualize(args):
    assert os.path.exists(args.scene), 'Directory does not exist'

    if args.rviz:
        from vgn_ros import rviz_utils
        rviz = rviz_utils.RViz()

    # Load data set
    dataset = VGNDataset(os.path.dirname(args.scene), augment=False)
    index = dataset.scenes.index(os.path.basename(args.scene))

    # Draw original sample
    tsdf, indices, scores, _ = dataset[index]

    mlab.figure('Original')
    vis.draw_voxels(tsdf)
    vis.draw_candidates(indices, scores)

    # # Draw an augmented sample
    # dataset.augment = True
    # tsdf, indices, scores = dataset[index]
    # mlab.figure('Augmented')
    # vis.draw_voxels(tsdf)
    # vis.draw_candidates(indices, scores)

    mlab.show()

    # Draw point cloud and candidates in rviz
    if args.rviz:
        scene = data.load_scene(args.scene)
        point_cloud, voxel_grid = data.reconstruct_volume(scene)

        rviz.draw_point_cloud(np.asarray(point_cloud.points))
        rviz.draw_tsdf(voxel_grid, idx=18)
        rviz.draw_candidates(scene['poses'], scene['scores'])

        # for pose, score in zip(scene['poses'], scene['scores']):
        #     if not np.isclose(score, 1.):
        #         continue
        #     rviz.draw_grasp_pose(pose)
        #     time.sleep(1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--scene',
        type=str,
        required=True,
        help='path to scene',
    )
    parser.add_argument(
        '--rviz',
        action='store_true',
        help='publish point clouds and grasp poses to Rviz',
    )
    args = parser.parse_args()

    if args.rviz:
        import rospy
        rospy.init_node('data_vis')

    visualize(args)


if __name__ == '__main__':
    main()
