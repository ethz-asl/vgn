import argparse
import numpy as np
from pathlib import Path


from robot_helpers.perception import CameraIntrinsic
from vgn.database import GraspDatabase
from vgn.perception import create_tsdf
import vgn.visualizer as vis


def main():
    parser = create_parser()
    args = parser.parse_args()

    db = GraspDatabase(args.root)
    scene_id = args.scene_id if args.scene_id else np.random.choice(db.scenes)
    print("Showing scene", scene_id)

    size = 0.3
    resolution = 80
    intrinsic = CameraIntrinsic(320, 240, 207.893, 207.893, 160, 120)

    imgs, views, grasps, qualities = db.read(scene_id)
    tsdf = create_tsdf(size, resolution, imgs, intrinsic, views)

    vis.scene_cloud(tsdf.voxel_size, tsdf.get_scene_cloud())
    # vis.map_cloud(tsdf.voxel_size, tsdf.get_map_cloud())
    vis.grasps(grasps, qualities, 0.05, max_grasps=5)
    vis.show()


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--scene-id", type=str)
    return parser


if __name__ == "__main__":
    main()
