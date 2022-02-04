import argparse
import numpy as np
from pathlib import Path

from robot_helpers.perception import CameraIntrinsic
from vgn.database import GraspDatabase
from vgn.perception import create_tsdf
import vgn.visualizer as vis


def main():
    args = parse_args()
    db = GraspDatabase(args.root)

    def show(scene_id):
        print("Showing scene", scene_id)
        size = 0.3
        resolution = 80
        intrinsic = CameraIntrinsic(320, 240, 207.893, 207.893, 160, 120)
        imgs, views, grasps, qualities = db.read(scene_id)
        tsdf = create_tsdf(size, resolution, imgs, intrinsic, views)
        vis.scene_cloud(tsdf.voxel_size, np.asarray(tsdf.get_scene_cloud().points))
        vis.grasps(grasps, qualities, 0.05, max_grasps=20)
        vis.show()

    if args.scene_id:
        show(args.scene_id)
    else:
        while True:
            show(np.random.choice(db.scenes))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scene-id", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    main()
