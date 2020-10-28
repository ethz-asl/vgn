import argparse
from pathlib import Path

from vgn.detection import VGN
from vgn.experiments import clutter_removal


def main(args):

    if args.rviz or str(args.model) == "gpd":
        import rospy

        rospy.init_node("sim_grasp", anonymous=True)

    if str(args.model) == "gpd":
        from vgn.baselines import GPD

        grasp_planner = GPD()
    else:
        grasp_planner = VGN(args.model, rviz=args.rviz)

    clutter_removal.run(
        grasp_plan_fn=grasp_planner,
        logdir=args.logdir,
        description=args.description,
        scene=args.scene,
        object_set=args.object_set,
        num_objects=args.num_objects,
        num_rounds=args.num_rounds,
        seed=args.seed,
        sim_gui=args.sim_gui,
        rviz=args.rviz,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--logdir", type=Path, default="data/experiments")
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--scene", type=str, choices=["pile", "packed"], default="pile")
    parser.add_argument("--object-set", type=str, default="blocks")
    parser.add_argument("--num-objects", type=int, default=5)
    parser.add_argument("--num-rounds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sim-gui", action="store_true")
    parser.add_argument("--rviz", action="store_true")
    args = parser.parse_args()
    main(args)
