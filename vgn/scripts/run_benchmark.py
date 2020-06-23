"""Clutter removal benchmark.

Each round, N objects are randomly placed in a tray. Then, the system is run until
(a) no objects remain, (b) VGN failed to find a grasp hypothesis, or (c) three
consecutive failed grasp attempts.
"""

import argparse
from pathlib2 import Path
import time

import numpy as np
import rospy
import torch
import tqdm

from vgn.benchmark import Logger
from vgn.grasp import *
from vgn.detection import *
from vgn.networks import load_network
from vgn.simulation import GraspSimulation
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

            # execute a random grasp candidate
            i = np.random.randint(len(grasps))
            grasp, score = grasps[i], scores[i]

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="simulated clutter removal benchmark")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--object-set", type=str, default="adversarial")
    parser.add_argument("--object-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sim-gui", action="store_true")
    parser.add_argument("--rviz", action="store_true")
    args = parser.parse_args()

    if args.rviz:
        rospy.init_node("benchmark", anonymous=True)

    main(args)
