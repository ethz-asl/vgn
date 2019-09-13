from __future__ import print_function

import argparse
import os

import numpy as np
import open3d
from mayavi import mlab

from vgn.dataset import VGNDataset
from vgn.utils import vis


def visualize(args):
    assert os.path.exists(args.scene), 'Directory does not exist'

    dataset = VGNDataset(os.path.dirname(args.scene))
    index = dataset.scenes.index(os.path.basename(args.scene))

    tsdf, indices, scores, quats = dataset[index]
    quats = np.swapaxes(quats, 0, 1)

    mlab.figure()
    vis.draw_voxels(tsdf)
    vis.draw_candidates(indices, quats, scores, draw_frames=True)

    mlab.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--scene',
        type=str,
        required=True,
        help='path to scene',
    )
    args = parser.parse_args()

    visualize(args)


if __name__ == '__main__':
    main()
