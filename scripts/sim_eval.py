"""Simulated grasping in clutter.

Each round, N objects are randomly placed in a tray. Then, the system is run until
(a) no objects remain, (b) the planner failed to find a grasp hypothesis, or (c)
three consecutive failed grasp attempts.

Measured metrics are
  * grasp success rate
  * percent cleared
  * planning time
"""

import argparse
from pathlib2 import Path
import time

import numpy as np
import rospy
import tqdm

from vgn import vis
from vgn.baselines import GPD
from vgn.benchmark import Logger, State
from vgn.detection import VGN
from vgn.grasp import Label
from vgn.simulation import GraspSimulation


def main(args):
    rospy.init_node("sim_eval")

    config = Path("config/sim.yaml")
    abort_on_contact = not args.allow_contact
    sim = GraspSimulation(args.object_set, config, gui=args.sim_gui, seed=args.seed)
    logger = Logger(args.logdir)

    if args.method == "vgn":
        plan_grasps = VGN(args.model)
    elif args.method == "gpd":
        plan_grasps = GPD()
    else:
        raise ValueError

    for _ in tqdm.tqdm(range(args.rounds)):
        sim.reset(args.object_count)
        logger.new_round(sim.num_objects)
        consecutive_failures = 1
        last_label = None

        while sim.num_objects > 0 and consecutive_failures < 3:
            # scan the scene
            tsdf, pc = sim.acquire_tsdf(num_viewpoints=5)

            # visualize
            vis.clear()
            vis.workspace(sim.size)
            vis.tsdf(tsdf.get_volume().squeeze(), tsdf.voxel_size)
            vis.points(np.asarray(pc.points))

            # plan grasps
            state = State(tsdf, pc)
            grasps, scores, planning_time = plan_grasps(state)

            if len(grasps) == 0:
                break  # no detections found, abort this round

            # select grasp
            grasp, score = grasps[0], scores[0]

            # visualize
            vis.grasps(grasps, scores, sim.config["finger_depth"])

            # execute grasp
            label, _ = sim.execute_grasp(grasp.pose, abort_on_contact=abort_on_contact)

            # log the grasp
            logger.log_grasp(state, planning_time, grasp, score, label)

            if last_label == Label.FAILURE and label == Label.FAILURE:
                consecutive_failures += 1
            else:
                consecutive_failures = 1
            last_label = label


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="simulated grasp trials")

    parser.add_argument("--method", choices=["vgn", "gpd"], required=True)
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--object-set", type=str, default="test")
    parser.add_argument("--object-count", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--allow-contact", action="store_true")
    parser.add_argument("--sim-gui", action="store_true")
    parser.add_argument("--seed", type=int, default=1)

    # vgn specific args
    parser.add_argument("--model", type=Path)

    args = parser.parse_args()
    main(args)
