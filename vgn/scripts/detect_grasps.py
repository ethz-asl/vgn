import argparse
from pathlib import Path

import open3d
from mayavi import mlab
import torch

from vgn.utils import vis
import vgn.config as cfg
from vgn.perception import integration
from vgn import utils
from vgn.utils.data import SceneData
from vgn.grasp_detector import GraspDetector


def main(args):

    # Load scene data
    scene_dir = Path(args.scene)
    scene_data = SceneData.load(scene_dir)

    point_cloud, voxel_grid = integration.reconstruct_scene(
        scene_data.intrinsic,
        scene_data.extrinsics,
        scene_data.depth_imgs,
        resolution=40,
    )
    tsdf = utils.voxel_grid_to_array(voxel_grid, cfg.resolution)

    # Detect grasps
    detector = GraspDetector(Path(args.model))
    grasps, info = detector.detect_grasps(tsdf)

    # Plot the network output
    mlab.figure()
    vis.draw_voxels(tsdf)

    mlab.figure()
    vis.draw_voxels(info["quality_out"])

    mlab.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="detect grasps in a scene")
    parser.add_argument(
        "--model", type=str, required=True, help="saved model ending with .pth"
    )
    parser.add_argument("--scene", type=str, required=True, help="scene directory")
    args = parser.parse_args()
    main(args)
