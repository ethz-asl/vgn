import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

from robot_helpers.perception import CameraIntrinsic
from vgn.database import GraspDatabase
from vgn.dataset import write_grid
from vgn.perception import create_tsdf


def main():
    args = parse_args()
    db = GraspDatabase(args.grasps)
    args.root.mkdir(parents=True)

    size = 0.3
    resolution = 40
    voxel_size = size / resolution
    intrinsic = CameraIntrinsic(320, 240, 207.893, 207.893, 160, 120)

    df = create_df(voxel_size, db)
    df.to_csv(args.root / "grasps.csv", float_format="%.4f", index=False)
    write_grids(args.root, size, resolution, db, intrinsic)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("grasps", type=Path)
    parser.add_argument("root", type=Path)
    return parser.parse_args()


def create_df(voxel_size, db):
    # Snap grasps to closest grid indices
    df = db.df.copy()
    df.x = (df.x / voxel_size).round().astype(int)
    df.y = (df.y / voxel_size).round().astype(int)
    df.z = (df.z / voxel_size).round().astype(int)
    df.width /= voxel_size
    df = df.rename(columns={"x": "i", "y": "j", "z": "k", "quality": "label"})
    return df


def write_grids(root, size, resolution, db, intrinsic):
    for scene_id in tqdm(db.scenes):
        imgs, views = db.read_scene(scene_id)
        tsdf = create_tsdf(size, resolution, imgs, intrinsic, views)
        grid = np.expand_dims(tsdf.get_grid(), 0)  # Add channel dimension
        write_grid(root, scene_id, grid)


if __name__ == "__main__":
    main()
