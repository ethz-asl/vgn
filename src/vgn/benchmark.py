"""Simulated grasping in clutter.

Each round, N objects are randomly placed in a tray. Then, the system is run until
(a) no objects remain, (b) the planner failed to find a grasp hypothesis, or (c)
three consecutive failed grasp attempts.

Measured metrics are
  * grasp success rate
  * percent cleared
  * planning time
"""

from __future__ import division

import collections
from pathlib2 import Path
import uuid

import numpy as np
import pandas as pd
import tqdm

from vgn import vis
from vgn.grasp import *
from vgn.simulation import GraspSimulation
from vgn.utils import io


State = collections.namedtuple("State", ["tsdf", "pc"])


def run(
    grasp_plan_fn,
    log_dir,
    object_set="test",
    object_count=5,
    rounds=40,
    no_contact=False,
    sim_gui=False,
    seed=1,
    n=5,
    N=None,
):
    config = Path("config/sim.yaml")
    sim = GraspSimulation(object_set, config, gui=sim_gui, seed=seed)
    logger = Logger(log_dir)

    for _ in tqdm.tqdm(range(rounds)):
        sim.reset(object_count)
        logger.new_round(sim.num_objects)
        consecutive_failures = 1
        last_label = None

        while sim.num_objects > 0 and consecutive_failures < 3:
            # scan the scene
            tsdf, pc = sim.acquire_tsdf(n=n, N=N)

            if pc.is_empty():
                break  # empty point cloud, abort this round TODO how is it possible to get here ?

            # visualize
            vis.clear()
            vis.workspace(sim.size)
            vis.tsdf(tsdf.get_volume().squeeze(), tsdf.voxel_size)
            vis.points(np.asarray(pc.points))

            # plan grasps
            state = State(tsdf, pc)
            grasps, scores, planning_time = grasp_plan_fn(state)

            if len(grasps) == 0:
                break  # no detections found, abort this round

            # select grasp
            grasp, score = grasps[0], scores[0]

            # visualize
            vis.grasps(grasps, scores, sim.config["finger_depth"])

            # execute grasp
            label, _ = sim.execute_grasp(grasp.pose, abort_on_contact=no_contact)

            # log the grasp
            logger.log_grasp(state, planning_time, grasp, score, label)

            if last_label == Label.FAILURE and label == Label.FAILURE:
                consecutive_failures += 1
            else:
                consecutive_failures = 1
            last_label = label


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
    grasps = pd.read_csv(log_dir / "grasps.csv")

    # number of grasps
    n_grasps = len(grasps.index)

    # success rate
    success_rate = grasps["label"].mean() * 100

    # percent cleared
    df = (
        grasps[["round_id", "label"]]
        .groupby("round_id")
        .sum()
        .rename(columns={"label": "cleared_count"})
        .merge(rounds, on="round_id")
    )
    percent_cleared = df["cleared_count"].sum() / df["object_count"].sum() * 100

    # planning time
    planning_time = {
        "mean": grasps["planning_time"].mean(),
        "std": grasps["planning_time"].std(),
    }

    return n_grasps, success_rate, percent_cleared, planning_time
