import argparse
import numpy as np
from pathlib import Path
import pandas as pd

from robot_helpers.perception import CameraIntrinsic
from vgn.data import read
from vgn.perception import create_tsdf
import vgn.visualizer as vis


def main():
    args = parse_args()
    df = pd.read_csv(args.root / "grasps.csv")

    def show(scene_id):
        print("Showing scene", scene_id)
        size = 0.3
        resolution = 80
        intrinsic = CameraIntrinsic(320, 240, 207.893, 207.893, 160, 120)
        imgs, views, grasps, qualities = read(args.root, df, scene_id)
        tsdf = create_tsdf(size, resolution, imgs, intrinsic, views)
        vis.scene_cloud(tsdf.voxel_size, np.asarray(tsdf.get_scene_cloud().points))
        vis.grasps(grasps, qualities, max_grasps=20)
        vis.show()

    if args.scene_id:
        show(args.scene_id)
    else:
        while True:
            show(np.random.choice(df.scene_id.unique()))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scene-id", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    main()
