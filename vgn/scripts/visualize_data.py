import argparse
from pathlib import Path

import numpy as np
import open3d
from mayavi import mlab

from vgn.dataset import VGNDataset
from vgn.utils import vis
from vgn.utils.data import SceneData


def main(args):
    scene_dir = Path(args.scene)

    # Load data point
    dataset = VGNDataset(scene_dir.parent, rebuild_cache=args.rebuild_cache)
    index = dataset.scenes.index(scene_dir)
    tsdf_vol, indices, quats, qualities = dataset[index]
    quats = np.swapaxes(quats, 0, 1)

    # Visualize TSDF grid and reconstructed point cloud
    mlab.figure()
    vis.draw_volume(tsdf_vol.squeeze())
    vis.draw_candidates(indices, quats, qualities, draw_frames=False)
    mlab.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="visualize data from a scene")
    parser.add_argument("--scene", type=str, required=True, help="scene directory")
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()
    main(args)
