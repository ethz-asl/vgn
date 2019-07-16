""""Script to generate a synthetic grasp dataset using physical simulation."""
import argparse

from vgn.data_generator import generate_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "basedir",
        type=str,
        help="The base directory in which data is stored",
    )
    parser.add_argument(
        "--n-scenes",
        type=int,
        default=1000,
        help="Number of generated virtual scenes",
    )
    parser.add_argument(
        "--n-candidates-per-scene",
        type=int,
        default=100,
        help="Number of grasps candidates per scene",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="Number of processes used for data collection",
    )
    parser.add_argument(
        "--no-sim-gui",
        action="store_true",
        help="Run simulation in headless mode",
    )
    args = parser.parse_args()

    generate_dataset(
        basedir=args.basedir,
        n_scenes=args.n_scenes,
        n_candidates_per_scene=args.n_candidates_per_scene,
        n_workers=args.n_workers,
        sim_gui=not args.no_sim_gui,
    )


if __name__ == "__main__":
    main()
