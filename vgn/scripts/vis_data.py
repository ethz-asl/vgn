import argparse
import os

import numpy as np
import open3d
from mayavi import mlab

from vgn.dataset import VGNDataset
from vgn.utils import vis


def main(args):
    # Load dataset
    dataset = VGNDataset(os.path.dirname(args.scene))
    index = dataset.scenes.index(os.path.basename(args.scene))

    tsdf, indices, quats, outcomes = dataset[index]
    quats = np.swapaxes(quats, 0, 1)

    # Visualize TSDF
    mlab.figure()
    vis.draw_voxels(tsdf)
    vis.draw_candidates(indices, quats, scores, draw_frames=True)
    mlab.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, required=True, help="scene directory")
    args = parser.parse_args()
    main(args)
