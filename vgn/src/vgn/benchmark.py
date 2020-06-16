from __future__ import division, print_function

from datetime import datetime
import time
import uuid

import numpy as np
import pandas as pd
import torch
import tqdm


from vgn import *
from vgn.networks import load_network
from vgn.simulation import GraspSimulation
from vgn.utils import io
from vgn_ros import vis


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_network(args.model, device)

    rng = np.random.RandomState(args.seed)
    sim = GraspSimulation(
        args.object_set, "config/default.yaml", random_state=rng, gui=args.sim_gui
    )
    logger = Logger(args.logdir)

    for round_id in tqdm.tqdm(range(args.rounds)):
        sim.reset(args.object_count)
        logger.add_round(round_id, sim.num_objects)
        consecutive_failures = 1
        last_label = None

        while sim.num_objects > 0 and consecutive_failures < 3:
            # scan the scene
            tsdf, pc = sim.acquire_tsdf(num_viewpoints=3)
            tsdf_vol = tsdf.get_volume()
            points = np.asarray(pc.points, dtype=np.float32)

            if args.rviz:
                vis.clear()
                vis.workspace(sim.size)
                vis.tsdf(tsdf_vol.squeeze(), tsdf.voxel_size)
                vis.points(points)

            # plan grasps
            tic = time.time()
            out = predict(tsdf_vol, net, device)
            out = process(out)
            grasps, scores = select(out)
            grasps = [from_voxel_coordinates(g, tsdf.voxel_size) for g in grasps]
            toc = time.time() - tic

            if len(grasps) == 0:
                break  # no detections found, abort this round

            # select highest scored grasp
            grasp, score = grasps[0], scores[0]

            # visualize
            if args.rviz:
                vis.grasps(grasps, scores, sim.config["finger_depth"])
                vis.quality(out[0], tsdf.voxel_size)

            # execute grasp
            label, _ = sim.execute_grasp(grasp.pose)

            # log the grasp
            logger.log_grasp(round_id, tsdf, points, toc, grasp, score, label)

            if last_label == Label.FAILURE and label == Label.FAILURE:
                consecutive_failures += 1
            else:
                consecutive_failures = 1
            last_label = label


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
