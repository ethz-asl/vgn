import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import open3d
import rospy
import torch
from torch.utils.data.dataloader import DataLoader

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
    tsdf, indices, scores = dataset[index]
    scene_dir = os.path.join(dataset_path, dataset.scenes[index])

    scene = data.load_scene(scene_dir)
    point_cloud, _ = data.reconstruct_volume(scene)

    rviz.draw_point_cloud(np.asarray(point_cloud.points))

    with torch.no_grad():
        tsdf = torch.from_numpy(tsdf).unsqueeze(0).to(device)
        out = model(tsdf)

    grasp_map = out.squeeze().cpu().numpy()

    trues = np.empty((40, 1))
    for i in range(40):
        xx, yy, zz = indices[i]
        score_pred = grasp_map[xx, yy, zz]
        label = np.isclose(np.round(score_pred), scores[i])
        trues[i] = label
    rviz.draw_candidates(scene['poses'], scene['scores'])
    rviz.draw_true_false(scene['poses'], trues)

    vis.plot_tsdf(tsdf.squeeze().cpu().numpy())
    vis.plot_vgn(grasp_map)
    plt.show()


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
