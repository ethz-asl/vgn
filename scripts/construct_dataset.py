from __future__ import division

import argparse
from pathlib2 import Path

import numpy as np
from tqdm import tqdm

from vgn.io import *
from vgn.perception import *

RESOLUTION = 40


def main(args):
    # create directory of new dataset
    (args.dataset / "scenes").mkdir(parents=True, exist_ok=True)

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
        tsdf = create_tsdf(size, RESOLUTION, depth_imgs, intrinsic, extrinsics)
        write_voxel_grid(args.dataset, f.stem, tsdf.get_grid())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    main(args)
