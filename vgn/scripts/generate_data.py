import argparse
from pathlib import Path

from mpi4py import MPI

from vgn.data_generation import generate_data
from vgn.utils.io import load_dict


def main(args):
    n_workers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()

    if rank == 0:
        print("Generating data using {} processes.".format(n_workers))

    config = load_dict(Path(args.config))

    generate_data(
        urdf_root=Path(config["urdf_root"]),
        hand_config=load_dict(Path(config["hand_config"])),
        object_set=config["object_set"],
        num_scenes=config["num_scenes"],
        num_grasps_per_scene=config["num_grasps_per_scene"],
        root_dir=Path(args.root),
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
        "--root", type=str, required=True, help="root directory of the dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/data_generation.yaml",
        help="path to data generation configuration",
    )
    parser.add_argument("--sim-gui", action="store_true", help="disable headless mode")
    parser.add_argument(
        "--rtf", type=float, default=-1.0, help="real time factor of the simulation"
    )
    args = parser.parse_args()
    main(args)
