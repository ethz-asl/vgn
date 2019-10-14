import argparse
import os

import numpy as np
import open3d
from mayavi import mlab

from vgn.dataset import VGNDataset
from vgn.utils import vis


def main(args):
    # Load dataset
    dataset = VGNDataset(os.path.dirname(args.scene), rebuild_cache=args.rebuild_cache)
    index = dataset.scenes.index(os.path.basename(args.scene))

    tsdf, indices, quats, qualities = dataset[index]
    quats = np.swapaxes(quats, 0, 1)

    # Visualize TSDF
    mlab.figure()
    vis.draw_voxels(tsdf)
    vis.draw_candidates(indices, quats, qualities, draw_frames=False)
    mlab.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, required=True, help="scene directory")
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()
    main(args)
