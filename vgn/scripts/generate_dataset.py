import argparse
from pathlib import Path

from mpi4py import MPI

from vgn.data_generation import generate_samples
from vgn.utils.io import load_dict


def main(args):
    num_workers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()

    if rank == 0:
        print("Generating grasp samples using {} processes.".format(num_workers))

    config = load_dict(Path(args.config))
    urdf_root = Path(config["urdf_root"])
    data_dir = Path(args.data_dir)

    generate_samples(
        urdf_root=urdf_root,
        hand_config=config,
        object_set=args.object_set,
        num_scenes=args.num_scenes // num_workers,
        num_grasps=args.num_grasps,
        max_num_trials=args.max_num_trials,
        data_dir=data_dir,
        sim_gui=args.sim_gui,
        rtf=args.rtf,
        rank=rank,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="generate synthetic grasping experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", type=str, required=True, help="root directory of the dataset"
    )
    parser.add_argument(
        "--object-set",
        type=str,
        choices=["debug", "cuboid"],
        default="debug",
        help="object set to be used",
    )
    parser.add_argument(
        "--num-scenes", type=int, default=2, help="number of scenes to be generated"
    )
    parser.add_argument(
        "--num-grasps",
        type=int,
        default=10,
        help="number of grasps to be generated per scene",
    )
    parser.add_argument(
        "--max-num-trials",
        type=int,
        default=200,
        help="max number of grasp trials before the scene is skipped",
    )
    parser.add_argument(
        "--config", type=str, default="config/default.yaml", help="configuration file"
    )
    parser.add_argument("--sim-gui", action="store_true", help="show simulation GUI")
    parser.add_argument(
        "--rtf", type=float, default=-1.0, help="real time factor of the simulation"
    )
    args = parser.parse_args()
    main(args)
