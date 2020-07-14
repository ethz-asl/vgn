import argparse
from pathlib2 import Path

import rospy

from vgn.baselines import GPD
from vgn.benchmark import run
from vgn.detection import VGN


def main(args):
    rospy.init_node("sim_eval")

    # select grasp planner
    if args.method == "vgn":
        grasp_planner = VGN(args.model)
    elif args.method == "gpd":
        grasp_planner = GPD()
    else:
        raise ValueError

    # run the benchmark
    run(
        grasp_planner,
        args.logdir,
        args.object_set,
        args.object_count,
        args.rounds,
        args.no_contact,
        args.sim_gui,
        args.seed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="simulated grasp trials")

    parser.add_argument("--method", choices=["vgn", "gpd"], required=True)
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--object-set", type=str, default="test")
    parser.add_argument("--object-count", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--no-contact", action="store_true")
    parser.add_argument("--sim-gui", action="store_true")
    parser.add_argument("--seed", type=int, default=1)

    # vgn specific args
    parser.add_argument("--model", type=Path)

    args = parser.parse_args()
    main(args)
