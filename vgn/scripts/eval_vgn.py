from __future__ import print_function

import argparse
import os

import numpy as np
import open3d
import rospy
import torch
from mayavi import mlab
from scipy import ndimage

from vgn import data
from vgn.dataset import VGNDataset
from vgn.models import get_model
from vgn.utils import vis


def _prepare_batch(batch, device):
    tsdf, idx, score = batch
    tsdf = tsdf.to(device)
    idx = idx.to(device)
    score = score.squeeze().to(device)
    return tsdf, idx, score


def main(args):
    if args.rviz:
        from vgn_ros import rviz_utils
        rviz = rviz_utils.RViz()

    # Parse description
    descr = os.path.basename(os.path.dirname(args.weights))
    strings = descr.split(',')
    model = strings[1][strings[1].find('=') + 1:]
    dataset = strings[2][strings[2].find('=') + 1:]

    # Load data
    dataset_path = os.path.join('data', 'datasets', dataset)
    dataset = VGNDataset(dataset_path, augment=False)

    # Load model
    device = torch.device('cuda')
    model = get_model(model).to(device)
    model.load_state_dict(torch.load(args.weights))

    # Visualize a random scene
    index = np.random.randint(len(dataset))
    scene = dataset.scenes[index]
    # scene = '70949ab2913a4f75b26605d92f5b8f85'
    # index = dataset.scenes.index(scene)
    print('Plotting scene', scene)

    tsdf, indices, scores = dataset[index]

    with torch.no_grad():
        out = model(torch.from_numpy(tsdf).unsqueeze(0).to(device))

    tsdf = tsdf.squeeze()
    out = out.squeeze().cpu().numpy()

    if args.rviz:
        scene = data.load_scene(os.path.join(dataset_path, scene))
        point_cloud, _ = data.reconstruct_volume(scene)

        rviz.draw_point_cloud(np.asarray(point_cloud.points))
        rviz.draw_candidates(scene['poses'], scene['scores'])

    # Mask
    grasp_map = out.copy()
    grasp_map[tsdf == 0.0] = 0.0
    grasp_map[grasp_map < 0.8] = 0.0

    # Smooth
    grasp_map = ndimage.gaussian_filter(grasp_map, sigma=1.)

    # Non-maximum suppression
    max_map = ndimage.filters.maximum_filter(grasp_map, size=5)
    grasp_map = np.where(grasp_map == max_map, max_map, 0.)

    # Select candidates
    nonzero = np.nonzero(grasp_map)
    scores = grasp_map[nonzero]
    indices = np.transpose(nonzero)

    # Draw
    mlab.figure('TSDF')
    vis.draw_voxels(tsdf)
    vis.draw_candidates(indices, scores)

    mlab.figure('Network output')
    vis.draw_voxels(out, tol=0.6)

    mlab.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--weights',
        type=str,
        required=True,
        help='path to model',
    )
    parser.add_argument(
        '--rviz',
        action='store_true',
        help='publish point clouds and grasp poses to Rviz',
    )
    args = parser.parse_args()

    if args.rviz:
        rospy.init_node('eval_model')

    main(args)
