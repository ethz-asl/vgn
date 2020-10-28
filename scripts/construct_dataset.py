import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from tqdm import tqdm

from vgn.io import *
from vgn.perception import *

RESOLUTION = 40


def main(args):
    # create directory of new dataset
    (args.dataset / "scenes").mkdir(parents=True)

    # load setup information
    size, intrinsic, _, finger_depth = read_setup(args.raw)
    assert np.isclose(size, 6.0 * finger_depth)
    voxel_size = size / RESOLUTION

    # create df
    df = read_df(args.raw)
    df["x"] /= voxel_size
    df["y"] /= voxel_size
    df["z"] /= voxel_size
    df["width"] /= voxel_size
    df = df.rename(columns={"x": "i", "y": "j", "z": "k"})
    write_df(df, args.dataset)

    # create tsdfs
    for f in tqdm(list((args.raw / "scenes").iterdir())):
        if f.suffix != ".npz":
            continue
        depth_imgs, extrinsics = read_sensor_data(args.raw, f.stem)

        if args.grid == "tsdf":
            tsdf = create_tsdf(size, RESOLUTION, depth_imgs, intrinsic, extrinsics)
            grid = tsdf.get_grid()
        elif args.grid == "occupancy":
            grid = create_occupancy_grid(size, depth_imgs, intrinsic, extrinsics)
        else:
            raise ValueError

        write_voxel_grid(args.dataset, f.stem, grid)


def create_occupancy_grid(size, depth_imgs, intrinsic, extrinsics):
    cloud = create_tsdf(size, 120, depth_imgs, intrinsic, extrinsics).get_cloud()
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
        cloud, size / RESOLUTION, [0.0, 0.0, 0.0], [size, size, size]
    )
    grid = np.zeros((1, RESOLUTION, RESOLUTION, RESOLUTION), dtype=np.float32)
    for voxel in voxel_grid.voxels:
        i, j, k = voxel.grid_index
        grid[0, i, j, k] = 1.0
    return grid


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--grid", choices=["tsdf", "occupancy"], default="tsdf")
    args = parser.parse_args()
    main(args)
