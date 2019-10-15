import argparse
import os

import numpy as np
import open3d
from mayavi import mlab
import torch

from vgn.perception import integration
from vgn.dataset import VGNDataset
from vgn.utils import data, vis
from vgn.networks import get_network
from vgn import config as cfg


def main(args):
    # Load model
    descr = os.path.basename(os.path.dirname(args.model))
    strings = descr.split(",")
    network_name = strings[1][strings[1].find("=") + 1 :]
    device = torch.device("cuda")
    net = get_network(network_name).to(device)
    net.load_state_dict(torch.load(args.model))

    # Load data
    dataset = VGNDataset(os.path.dirname(args.scene))
    index = dataset.scenes.index(os.path.basename(args.scene))
    tsdf, indices, quats, qualities = dataset[index]

    # Predict the grasp qualities and poses
    with torch.no_grad():
        quality_out, quat_out = net(torch.from_numpy(tsdf).unsqueeze(0).to(device))

    # Plot the input TSDF
    tsdf = tsdf.squeeze()
    mlab.figure()
    vis.draw_voxels(tsdf)

    # Plot the output grasp quality map on top of reconstructed point cloud
    # TODO ugly, improve this part
    quality_out = quality_out.squeeze().cpu().numpy()

    scene = data.load_scene(args.scene)
    point_cloud, _ = integration.reconstruct_scene(
        scene["intrinsic"], scene["extrinsics"], scene["depth_imgs"], resolution=80
    )

    mlab.figure()
    vis.draw_voxels(quality_out)
    vis.draw_points(np.asarray(point_cloud.points) / cfg.size * cfg.resolution)

    mlab.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, help="saved model ending with .pth"
    )
    parser.add_argument("--scene", type=str, required=True, help="scene directory")
    args = parser.parse_args()
    main(args)
