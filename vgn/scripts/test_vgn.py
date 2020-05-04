import argparse
from pathlib import Path

import numpy as np
import open3d
from mayavi import mlab
import torch

from vgn.dataset import VgnDataset, RandomAffine, Rescale
from vgn.networks import load_network
from vgn.utils.training import *
from vgn.utils.vis import draw_sample, draw_volume


def main(data_dir, network_path, batch_size, vis):
    device = torch.device("cuda")
    kwargs = kwargs = {"num_workers": 4, "pin_memory": True}

    loader = create_test_loader(data_dir, batch_size, kwargs)
    net = load_network(network_path, device)

    for batch in loader:
        with torch.no_grad():
            tsdf, (qual, rot, width), mask = batch

            x = tsdf.to(device)
            y_pred = net(x)

        tsdf = tsdf.squeeze().numpy()
        qual = qual.squeeze().numpy()
        rot = rot.squeeze().numpy()
        width = width.squeeze().numpy()
        mask = mask.squeeze().numpy()

        qual_pred = y_pred[0].cpu().squeeze().numpy()

        # visualize wrong prediction
        if vis:
            mlab.figure("False prediction")
            mask = np.not_equal(np.round(qual_pred), qual) * mask
            print("Number of wrong prediction", mask.sum())
            draw_sample(tsdf, qual, rot[0], 10 * width, mask)
            mlab.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate a trained vgn model")
    parser.add_argument(
        "--data-dir", required=True, type=str, help="root directory of the dataset"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="saved model ending with .pth"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--vis", action="store_true", help="enable visualizations")
    args = parser.parse_args()

    main(Path(args.data_dir), Path(args.model), args.batch_size, args.vis)
