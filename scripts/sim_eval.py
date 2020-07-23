import argparse
from pathlib2 import Path

import rospy

from vgn.baselines import GPD
from vgn.benchmark import run
from vgn.detection import VGN


def main(args):
    rospy.init_node("sim_eval")

    if args.method == "vgn":
        grasp_planner = VGN(args.model)
    elif args.method == "gpd":
        grasp_planner = GPD()
    else:
        raise ValueError

    run(
        grasp_plan_fn=grasp_planner,
        log_dir=args.logdir,
        scene=args.scene,
        object_set=args.object_set,
        object_count=args.object_count,
        rounds=args.rounds,
        sim_gui=args.sim_gui,
        seed=args.seed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="simulated grasp trials")

    parser.add_argument("--method", choices=["vgn", "gpd"], required=True)
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--object-set", type=str, required=True)
    parser.add_argument("--object-count", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--sim-gui", action="store_true")
    parser.add_argument("--seed", type=int, default=1)

    # vgn specific args
    parser.add_argument("--model", type=Path)

    args = parser.parse_args()
    main(args)
