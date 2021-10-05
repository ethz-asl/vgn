import argparse
from pathlib import Path
from tqdm import tqdm

from robot_helpers.perception import CameraIntrinsic
from vgn.database import GraspDatabase
from vgn.dataset import write_grid
from vgn.perception import create_tsdf


def main():
    parser = create_parser()
    args = parser.parse_args()

    db = GraspDatabase(args.grasps)
    args.root.mkdir()

    size = 0.3
    resolution = 40
    voxel_size = size / resolution
    intrinsic = CameraIntrinsic(320, 240, 207.893, 207.893, 160, 120)

    df = create_df(voxel_size, db)
    df.to_csv(args.root / "grasps.csv", float_format="%.4f", index=False)
    write_grids(args.root, size, resolution, db, intrinsic)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("grasps", type=Path)
    parser.add_argument("root", type=Path)
    return parser


def create_df(voxel_size, db):
    # Snap grasps to closest grid indices
    df = db.df.copy()
    df.x = (df.x / voxel_size).round()
    df.y = (df.y / voxel_size).round()
    df.z = (df.z / voxel_size).round()
    df.width /= voxel_size
    df = df.rename(columns={"x": "i", "y": "j", "z": "k"})
    return df


def write_grids(root, size, resolution, db, intrinsic):
    for scene_id in tqdm(db.scenes):
        imgs, views = db.read_scene(scene_id)
        tsdf = create_tsdf(size, resolution, imgs, intrinsic, views)
        grid = tsdf.get_grid()
        write_grid(root, scene_id, grid)


if __name__ == "__main__":
    main()
