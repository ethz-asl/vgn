from __future__ import print_function

import argparse
import os

import numpy as np
import open3d
import torch
from mayavi import mlab

from vgn import data, grasp
from vgn.dataset import VGNDataset
from vgn.models import get_model
from vgn.utils import vis
from vgn.utils.transform import Rotation, Transform


def main(args):
    # Parse inputs
    descr = os.path.basename(os.path.dirname(args.weights))
    strings = descr.split(",")
    model = strings[1][strings[1].find("=") + 1 :]
    dataset = strings[2][strings[2].find("=") + 1 :]

    # Load network
    device = torch.device("cuda")
    model = get_model(model).to(device)

    # Load dataset
    dataset_path = os.path.join("data", "datasets", dataset)
    dataset = VGNDataset(dataset_path)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # Evaluate
    with torch.no_grad():
        for tsdf, indices, qualities, quats in loader:

            if args.vis:
                mlab.figure("Network output")
                vis.draw_voxels(quality_out)

                mlab.figure("Grasp map")

                vis.draw_voxels(grasp_map)
                vis.draw_candidates(indices, qualities)

    # indices, qualities = grasp.select_best_grasps(grasp_mask)

    # Draw
    mlab.figure("Scene")
    vis.draw_voxels(tsdf)
    vis.draw_candidates(indices, qualities)

    for index in indices:
        i, j, k = index
        quat = quat_out[:, i, j, k]
        pose = Transform(Rotation.from_quat(quat), index)
        vis.draw_pose(pose, scale=4.0)
        print("TSDF at grasp point", tsdf[i, j, k])

    mlab.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="path to model")
    parser.add_argument("--data", type=str, required=True, help="name of dataset")
    parser.add_argument("--vis", action="store_true", help="visualize network output")
    args = parser.parse_args()

    main(args)
