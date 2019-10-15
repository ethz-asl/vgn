import argparse
import os

import numpy as np
import open3d
from mayavi import mlab
import torch

from vgn.dataset import VGNDataset
from vgn.utils import vis
from vgn.networks import get_network


def main(args):
    # Parse input
    descr = os.path.basename(os.path.dirname(args.model))
    strings = descr.split(",")
    network_name = strings[1][strings[1].find("=") + 1 :]
    dataset = strings[2][strings[2].find("=") + 1 :]

    # Load dataset
    root = os.path.join("data", "datasets", dataset)
    dataset = VGNDataset(root)

    # Load model
    device = torch.device("cuda")
    net = get_network(network_name).to(device)
    net.load_state_dict(torch.load(args.model))

    # Select a random scene
    index = np.random.randint(len(dataset))
    scene = dataset.scenes[index]
    tsdf, indices, quats, qualities = dataset[index]

    # Predict the grasp qualities and poses
    with torch.no_grad():
        quality_out, quat_out = net(torch.from_numpy(tsdf).unsqueeze(0).to(device))

    # Plot the input TSDF
    tsdf = tsdf.squeeze()
    mlab.figure()
    vis.draw_voxels(tsdf)

    # Plot the output grasp quality map
    quality_out = quality_out.squeeze().cpu().numpy()
    mlab.figure()
    vis.draw_voxels(quality_out)

    mlab.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, help="saved model ending with .pth"
    )
    args = parser.parse_args()
    main(args)
