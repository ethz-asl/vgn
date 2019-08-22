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


def main():

    from vgn_ros import rviz_utils
    rviz = rviz_utils.RViz()

    model_path = 'data/runs/Aug21_17-21-18,model=conv,data=cube,batch_size=32,lr=1e-03/model_model_90.pth'
    model_name = 'conv'
    data_path = 'data/datasets/cube'

    device = torch.device('cuda')
    model = get_model(model_name).to(device)
    model.load_state_dict(torch.load(model_path))

    dataset = VGNDataset(data_path)

    # select a random scene
    idx = np.random.randint(len(dataset))
    tsdf, indices, scores = dataset[idx]
    scene_dir = os.path.join(data_path, dataset.scenes[idx])

    scene = data.load_scene(scene_dir)
    point_cloud, _ = data.reconstruct_volume(scene)

    rviz.draw_point_cloud(np.asarray(point_cloud.points))

    with torch.no_grad():
        tsdf = torch.from_numpy(tsdf).unsqueeze(0).to(device)
        out = model(tsdf)

    grasp_map = out.squeeze().cpu().numpy()

    points = np.empty((40, 3))
    trues = np.empty((40, 1))
    for i in range(40):
        points[i] = scene['poses'][i].translation
        xx, yy, zz = indices[i]
        score_pred = grasp_map[xx, yy, zz]
        label = np.isclose(np.round(score_pred), scores[i])
        trues[i] = label
    rviz.draw_true_false(points, trues)

    #     vis.plot_tsdf(tsdf.squeeze().cpu().numpy())
    #     vis.plot_vgn(out.squeeze().cpu().numpy())
    #     plt.show()


if __name__ == '__main__':
    rospy.init_node('eval_model')
    main()
