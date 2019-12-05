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

    root_dir = Path(args.root)
    data_gen_config = load_dict(Path(args.data_gen_config))
    sim_config = load_dict(Path(args.sim_config))

    generate_data(
        root_dir=root_dir,
        object_set=args.object_set,
        n_scenes=args.n_scenes // n_workers,
        n_grasps=data_gen_config["n_grasps"],
        n_viewpoints=data_gen_config["n_viewpoints"],
        vol_res=data_gen_config["vol_res"],
        urdf_root=Path(sim_config["urdf_root"]),
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
        "--object-set",
        choices=["debug", "cuboid", "cuboids"],
        default="debug",
        help="object set to be used",
    )
    parser.add_argument(
        "--n-scenes", type=int, default=800, help="numbers of scenes to generate"
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
    parser.add_argument("--sim-gui", action="store_true", help="disable headless mode")
    parser.add_argument(
        "--rtf", type=float, default=-1.0, help="real time factor of the simulation"
    )
    args = parser.parse_args()
    main(args)

sim_gui: False
rtf: -1.0
