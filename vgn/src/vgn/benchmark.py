"""Clutter removal benchmark.

Each round, N objects are randomly placed in a tray. Then, the system is run until
(a) no objects remain, (b) the planner failed to find a grasp hypothesis, or (c)
three consecutive failed grasp attempts.

Measured metrics are
  * grasp success rate
  * percent cleared
  * planning time
"""

from __future__ import division

import uuid

import numpy as np
import pandas as pd
import tqdm

from vgn.grasp import *
from vgn.simulation import GraspSimulation
from vgn.utils import io
from vgn_ros import vis


def run(planner, object_set, object_count, rounds, log_dir, sim_gui, seed):
    sim = GraspSimulation(object_set, "config/default.yaml", gui=sim_gui, seed=seed)
    logger = Logger(log_dir)

    for round_id in tqdm.tqdm(range(rounds)):
        sim.reset(object_count)
        logger.add_round(round_id, sim.num_objects)
        consecutive_failures = 1
        last_label = None

        while sim.num_objects > 0 and consecutive_failures < 3:
            # scan the scene
            tsdf, pc = sim.acquire_tsdf(num_viewpoints=5)
            tsdf_vol = tsdf.get_volume()
            points = np.asarray(pc.points, dtype=np.float32)

            # visualize
            vis.clear()
            vis.workspace(sim.size)
            vis.tsdf(tsdf_vol.squeeze(), tsdf.voxel_size)
            vis.points(points)

            # plan grasps
            grasps, scores, time = planner(tsdf, pc)
            if len(grasps) == 0:
                break  # no detections found, abort this round

            # execute a random grasp candidate
            i = np.random.randint(len(grasps))
            grasp, score = grasps[i], scores[i]

            # visualize
            vis.grasps(grasps, scores, sim.config["finger_depth"])

            # execute grasp
            label, _ = sim.execute_grasp(grasp.pose)

            # log the grasp
            logger.log_grasp(round_id, tsdf, points, time, grasp, score, label)

            if last_label == Label.FAILURE and label == Label.FAILURE:
                consecutive_failures += 1
            else:
                consecutive_failures = 1
            last_label = label


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
