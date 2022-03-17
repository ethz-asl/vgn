import argparse
import numpy as np
from pathlib import Path
import pandas as pd

from robot_helpers.perception import CameraIntrinsic
from vgn.data import read
from vgn.perception import create_tsdf
import vgn.visualizer as vis


def main():
    parser = create_parser()
    args = parser.parse_args()
    df = pd.read_csv(args.root / "grasps.csv")

    while True:
        scene_id = np.random.choice(df.scene_id.unique())
        print("Showing scene", scene_id)
        size = 0.3
        resolution = 80
        intrinsic = CameraIntrinsic(320, 240, 207.893, 207.893, 160, 120)
        imgs, views, grasps, scores = read(args.root, df, scene_id)
        tsdf = create_tsdf(size, resolution, imgs, intrinsic, views)
        vis.scene_cloud(tsdf.voxel_size, np.asarray(tsdf.get_scene_cloud().points))
        vis.grasps(grasps, scores, max_grasps=20)
        vis.show()


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    return parser


if __name__ == "__main__":
    main()
