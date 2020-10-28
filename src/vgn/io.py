import json
import uuid

import numpy as np
import pandas as pd

from vgn.grasp import Grasp
from vgn.perception import *
from vgn.utils.transform import Rotation, Transform


def write_setup(root, size, intrinsic, max_opening_width, finger_depth):
    data = {
        "size": size,
        "intrinsic": intrinsic.to_dict(),
        "max_opening_width": max_opening_width,
        "finger_depth": finger_depth,
    }
    write_json(data, root / "setup.json")


def read_setup(root):
    data = read_json(root / "setup.json")
    size = data["size"]
    intrinsic = CameraIntrinsic.from_dict(data["intrinsic"])
    max_opening_width = data["max_opening_width"]
    finger_depth = data["finger_depth"]
    return size, intrinsic, max_opening_width, finger_depth


def write_sensor_data(root, depth_imgs, extrinsics):
    scene_id = uuid.uuid4().hex
    path = root / "scenes" / (scene_id + ".npz")
    np.savez_compressed(path, depth_imgs=depth_imgs, extrinsics=extrinsics)
    return scene_id


def read_sensor_data(root, scene_id):
    data = np.load(root / "scenes" / (scene_id + ".npz"))
    return data["depth_imgs"], data["extrinsics"]


def write_grasp(root, scene_id, grasp, label):
    # TODO concurrent writes could be an issue
    csv_path = root / "grasps.csv"
    if not csv_path.exists():
        create_csv(
            csv_path,
            ["scene_id", "qx", "qy", "qz", "qw", "x", "y", "z", "width", "label"],
        )
    qx, qy, qz, qw = grasp.pose.rotation.as_quat()
    x, y, z = grasp.pose.translation
    width = grasp.width
    append_csv(csv_path, scene_id, qx, qy, qz, qw, x, y, z, width, label)


def read_grasp(df, i):
    scene_id = df.loc[i, "scene_id"]
    orientation = Rotation.from_quat(df.loc[i, "qx":"qw"].to_numpy(np.double))
    position = df.loc[i, "x":"z"].to_numpy(np.double)
    width = df.loc[i, "width"]
    label = df.loc[i, "label"]
    grasp = Grasp(Transform(orientation, position), width)
    return scene_id, grasp, label


def read_df(root):
    return pd.read_csv(root / "grasps.csv")


def write_df(df, root):
    df.to_csv(root / "grasps.csv", index=False)


def write_voxel_grid(root, scene_id, voxel_grid):
    path = root / "scenes" / (scene_id + ".npz")
    np.savez_compressed(path, grid=voxel_grid)


def read_voxel_grid(root, scene_id):
    path = root / "scenes" / (scene_id + ".npz")
    return np.load(path)["grid"]


def read_json(path):
    with path.open("r") as f:
        data = json.load(f)
    return data


def write_json(data, path):
    with path.open("w") as f:
        json.dump(data, f, indent=4)


def create_csv(path, columns):
    with path.open("w") as f:
        f.write(",".join(columns))
        f.write("\n")


def append_csv(path, *args):
    row = ",".join([str(arg) for arg in args])
    with path.open("a") as f:
        f.write(row)
        f.write("\n")
