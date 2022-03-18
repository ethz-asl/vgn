import argparse
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from robot_helpers.perception import CameraIntrinsic
from vgn.data import read_sensor_data, write_grid
from vgn.perception import create_tsdf


def main():
    parser = create_parser()
    args = parser.parse_args()
    df = pd.read_csv(args.grasps_folder / "grasps.csv")
    args.dataset_folder.mkdir(parents=True)

    size = 0.3
    resolution = 40
    voxel_size = size / resolution
    intrinsic = CameraIntrinsic(320, 240, 207.893, 207.893, 160, 120)

    df = create_df(voxel_size, df)
    df.to_csv(args.dataset_folder / "grasps.csv", float_format="%.4f", index=False)

    for scene_id in tqdm(df.scene_id.unique()):
        imgs, views = read_sensor_data(args.grasps_folder, scene_id)
        tsdf = create_tsdf(size, resolution, imgs, intrinsic, views)
        grid = np.expand_dims(tsdf.get_grid(), 0)  # Add channel dimension
        write_grid(grid, args.dataset_folder, scene_id)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("grasps_folder", type=Path)
    parser.add_argument("dataset_folder", type=Path)
    return parser


def create_df(voxel_size, df):
    # Snap grasps to closest grid indices
    df = df.copy()
    df.x = (df.x / voxel_size).round().astype(int)
    df.y = (df.y / voxel_size).round().astype(int)
    df.z = (df.z / voxel_size).round().astype(int)
    df.width /= voxel_size
    df = df.rename(columns={"x": "i", "y": "j", "z": "k", "score": "label"})
    return df


if __name__ == "__main__":
    main()
