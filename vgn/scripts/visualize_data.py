import argparse
from pathlib import Path

import numpy as np
import open3d
from mayavi import mlab

from vgn import config as cfg
from vgn import grasp
from vgn.dataset import VGNDataset
from vgn.utils import vis, data
from vgn.utils.data import SceneData
from vgn.perception.integration import TSDFVolume


def main(args):
    scene_dir = Path(args.scene)

    # Load scene data
    scene = SceneData.load(scene_dir)
    tsdf = TSDFVolume(cfg.size, cfg.resolution)
    tsdf.integrate_images(scene.depth_imgs, scene.intrinsic, scene.extrinsics)
    tsdf_vol = tsdf.get_volume()
    point_cloud = tsdf.extract_point_cloud()
    qualities = [0.0 if label < grasp.Label.SUCCESS else 1.0 for label in scene.labels]

    # Visualize TSDF volume, reconstructed point cloud, and grasps
    mlab.figure("Scene {}".format(scene_dir.name))
    vis.draw_volume(tsdf_vol.squeeze(), tsdf.voxel_size)
    vis.draw_point_cloud(point_cloud)
    vis.draw_grasps(scene.grasps, qualities, draw_frames=True)
    mlab.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="visualize data from a scene")
    parser.add_argument("--scene", type=str, required=True, help="scene directory")
    args = parser.parse_args()
    main(args)
