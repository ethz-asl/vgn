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

    data_gen_config = load_dict(Path(args.data_gen_config))
    sim_config = load_dict(Path(args.sim_config))

    generate_data(
        root=Path(args.root),
        object_set=data_gen_config["object_set"],
        n_scenes=data_gen_config["n_scenes"],
        n_grasps=data_gen_config["n_grasps"],
        n_viewpoints=data_gen_config["n_viewpoints"],
        vol_size=data_gen_config["vol_size"],
        vol_res=data_gen_config["vol_res"],
        urdf_root=Path(sim_config["urdf_root"]),
        sim_gui=sim_config["sim_gui"],
        rtf=sim_config["rtf"],
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
        "--data-gen-config",
        type=str,
        default="config/data_generation.yaml",
        help="path to data generation configuration",
    )
    parser.add_argument(
        "--sim-config",
        type=str,
        default="config/simulation.yaml",
        help="path to simulation configuration",
    )
    args = parser.parse_args()
    main(args)
