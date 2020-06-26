import argparse
from pathlib2 import Path

import rospy
import time
import torch

from vgn import benchmark
from vgn.grasp import *
from vgn.detection import *
from vgn.networks import load_network
from vgn_ros import vis


def main(args):
    rospy.init_node("eval_vgn", anonymous=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_network(args.model, device)

    def grasp_planning_fn(tsdf, pc):
        tsdf_vol = tsdf.get_volume()

        tic = time.time()
        out = predict(tsdf_vol, net, device)
        out = process(out)
        grasps, scores = select(out)
        grasps = [from_voxel_coordinates(g, tsdf.voxel_size) for g in grasps]
        toc = time.time() - tic

        vis.quality(out[0], tsdf.voxel_size)

        return grasps, scores, toc

    benchmark.run(
        grasp_planning_fn,
        args.object_set,
        args.object_count,
        args.rounds,
        args.logdir,
        args.sim_gui,
        args.seed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="simulated clutter removal benchmark")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--object-set", type=str, default="adversarial")
    parser.add_argument("--object-count", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--sim-gui", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    main(args)
