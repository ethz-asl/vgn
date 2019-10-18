import argparse
from pathlib import Path

import numpy as np
import open3d
from mayavi import mlab

from vgn.dataset import VGNDataset
from vgn.utils import vis
from vgn.perception import integration
import vgn.config as cfg
from vgn.utils.data import SceneData


def main(args):
    scene_dir = Path(args.scene)

    # Load dataset
    dataset = VGNDataset(scene_dir.parent, rebuild_cache=args.rebuild_cache)
    index = dataset.scenes.index(scene_dir)

    tsdf, indices, quats, qualities = dataset[index]
    quats = np.swapaxes(quats, 0, 1)

    # Visualize TSDF
    mlab.figure()

    scene = SceneData.load(scene_dir)
    point_cloud, _ = integration.reconstruct_scene(
        scene.intrinsic, scene.extrinsics, scene.depth_imgs, resolution=80
    )

    vis.draw_voxels(tsdf)
    vis.draw_candidates(indices, quats, qualities, draw_frames=False)
    vis.draw_points(np.asarray(point_cloud.points) / cfg.size * cfg.resolution)
    mlab.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="visualize data from a scene")
    parser.add_argument("--scene", type=str, required=True, help="scene directory")
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()
    main(args)
