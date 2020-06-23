from __future__ import division, print_function

import uuid

import numpy as np
import pandas as pd


from vgn.grasp import to_voxel_coordinates
from vgn.utils import io


def metrics(log_dir):
    rounds = pd.read_csv(log_dir / "rounds.csv")
    trials = pd.read_csv(log_dir / "grasps.csv")

    # success rate
    success_rate = trials["label"].mean() * 100

    # percent cleared
    df = (
        trials[["round_id", "label"]]
        .groupby("round_id")
        .sum()
        .rename(columns={"label": "cleared_count"})
        .merge(rounds, on="round_id")
    )
    percent_cleared = df["cleared_count"].sum() / df["object_count"].sum() * 100

    # planning time
    planning_time = trials["planning_time"].mean()

    return success_rate, percent_cleared, planning_time


class Logger(object):
    def __init__(self, log_dir):
        self._root = log_dir
        self._root.mkdir(parents=True)
        self._tsdfs_dir = self._root / "tsdfs"
        self._tsdfs_dir.mkdir()
        self._clouds_dir = self._root / "clouds"
        self._clouds_dir.mkdir()

    def add_round(self, round_id, object_count):
        csv_path = self._root / "rounds.csv"
        if not csv_path.exists():
            io.create_csv(csv_path, "round_id,object_count")
        io.append_csv(csv_path, round_id, object_count)

    def log_grasp(self, round_id, tsdf, points, planning_time, grasp, score, label):
        csv_path = self._root / "grasps.csv"

        if not csv_path.exists():
            header = (
                "round_id,scene_id,planning_time,i,j,k,qx,qy,qz,qw,width,score,label"
            )
            io.create_csv(csv_path, header)

        scene_id = uuid.uuid4().hex
        tsdf_path = self._tsdfs_dir / (scene_id + ".npz")
        np.savez_compressed(str(tsdf_path), tsdf=tsdf.get_volume())
        cloud_path = self._clouds_dir / (scene_id + ".npz")
        np.savez_compressed(str(cloud_path), points=points)
        grasp = to_voxel_coordinates(grasp, tsdf.voxel_size)
        qx, qy, qz, qw = grasp.pose.rotation.as_quat()
        i, j, k = np.round(grasp.pose.translation).astype(np.int)
        width = grasp.width
        label = int(label)

        io.append_csv(
            csv_path,
            round_id,
            scene_id,
            planning_time,
            i,
            j,
            k,
            qx,
            qy,
            qz,
            qw,
            width,
            score,
            label,
        )
