from __future__ import print_function

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import open3d
import rospy
import torch
from mayavi import mlab

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
    from vgn_ros import rviz_utils
    rviz = rviz_utils.RViz()

    descr = os.path.basename(os.path.dirname(args.weights))
    strings = descr.split(',')
    model = strings[1][strings[1].find('=') + 1:]
    dataset = strings[2][strings[2].find('=') + 1:]

    device = torch.device('cuda')
    model = get_model(model).to(device)
    model.load_state_dict(torch.load(args.weights))

    dataset_path = os.path.join('data', 'datasets', dataset)
    dataset = VGNDataset(dataset_path)

    # Visualize a random scene
    index = np.random.randint(len(dataset))
    scene = dataset.scenes[index]

    # scene = 'a4899970dc4c4a1bb76ae9c644ec5d92'
    # index = dataset.scenes.index(scene)

    print('Plotting scene', scene)
    tsdf, indices, scores = dataset[index]

    scene = data.load_scene(os.path.join(dataset_path, scene))
    point_cloud, _ = data.reconstruct_volume(scene)

    rviz.draw_point_cloud(np.asarray(point_cloud.points))

    with torch.no_grad():
        out = model(torch.from_numpy(tsdf).unsqueeze(0).to(device))

    tsdf = tsdf.squeeze()
    grasp_map = out.squeeze().cpu().numpy()

    trues = np.empty(len(indices))
    for i in range(len(indices)):
        xx, yy, zz = indices[i]
        score_pred = grasp_map[xx, yy, zz]
        trues[i] = 1. if np.isclose(np.round(score_pred), scores[i]) else 0.
    rviz.draw_candidates(scene['poses'], scene['scores'])
    rviz.draw_true_false(scene['poses'], trues)

    vis.draw_voxels(tsdf, 'TSDF')
    vis.draw_voxels(grasp_map, 'Grasp map')
    mlab.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--weights',
        type=str,
        required=True,
        help='path to model',
    )
    args = parser.parse_args()

    rospy.init_node('eval_model')

    main(args)
