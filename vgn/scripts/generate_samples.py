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

    generate_samples(
        urdf_root=Path(config["urdf_root"]),
        hand_config=load_dict(Path(config["hand_config"])),
        object_set=config["object_set"],
        num_scenes=config["num_scenes"] // num_workers,
        num_grasps_per_scene=config["num_grasps_per_scene"],
        root_dir=Path(args.root_dir),
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
        "--root_dir", type=str, required=True, help="root directory of the dataset"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="data generation configuration file",
    )
    parser.add_argument("--sim-gui", action="store_true", help="disable headless mode")
    parser.add_argument(
        "--rtf", type=float, default=-1.0, help="real time factor of the simulation"
    )
    args = parser.parse_args()
    main(args)
