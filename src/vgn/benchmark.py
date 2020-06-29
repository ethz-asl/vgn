from __future__ import division

import collections
import uuid

import numpy as np
import pandas as pd

from vgn.grasp import *
from vgn.utils import io


State = collections.namedtuple("State", ["tsdf", "pc"])


class Logger(object):
    def __init__(self, log_dir):
        self.root = log_dir
        self.tsdfs_dir = self.root / "tsdfs"
        self.clouds_dir = self.root / "clouds"
        self.round_csv_path = self.root / "rounds.csv"
        self.grasps_csv_path = self.root / "grasps.csv"

        self._create_dirs()
        self._create_csv_files()

    def new_round(self, object_count):
        round_id = self.round_id + 1
        io.append_csv(self.round_csv_path, round_id, object_count)

    def log_grasp(self, state, planning_time, grasp, score, label):
        tsdf, points = state.tsdf, np.asarray(state.pc.points)
        scene_id = uuid.uuid4().hex
        tsdf_path = self.tsdfs_dir / (scene_id + ".npz")
        np.savez_compressed(str(tsdf_path), tsdf=tsdf.get_volume())
        cloud_path = self.clouds_dir / (scene_id + ".npz")
        np.savez_compressed(str(cloud_path), points=points)
        grasp = to_voxel_coordinates(grasp, tsdf.voxel_size)
        qx, qy, qz, qw = grasp.pose.rotation.as_quat()
        i, j, k = np.round(grasp.pose.translation).astype(np.int)
        width = grasp.width
        label = int(label)

        io.append_csv(
            self.grasps_csv_path,
            self.round_id,
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

    @property
    def round_id(self):
        df = pd.read_csv(self.round_csv_path)
        return -1 if df.empty else df["round_id"].max()

    def _create_dirs(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.tsdfs_dir.mkdir(exist_ok=True)
        self.clouds_dir.mkdir(exist_ok=True)

    def _create_csv_files(self):
        if not self.round_csv_path.exists():
            io.create_csv(self.round_csv_path, ["round_id", "object_count"])

        if not self.grasps_csv_path.exists():
            columns = [
                "round_id",
                "scene_id",
                "planning_time",
                "i",
                "j",
                "k",
                "qx",
                "qy",
                "qz",
                "qw",
                "width",
                "score",
                "label",
            ]
            io.create_csv(self.grasps_csv_path, columns)


def compute_metrics(log_dir):
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
    planning_time = {
        "mean": trials["planning_time"].mean(),
        "std": trials["planning_time"].std(),
    }

    return success_rate, percent_cleared, planning_time
