import argparse
from pathlib import Path

import open3d
import numpy as np
from tqdm import tqdm

from vgn.constants import vgn_res
from vgn.grasp import Label
from vgn.grasp_detector import GraspDetector
from vgn.perception.exploration import sample_hemisphere
from vgn.perception.integration import TSDFVolume
from vgn.simulation import GraspingExperiment
from vgn.utils.io import load_dict

threshold = 0.9
n_experiments = 20
n_views_per_scene = 10


def evaluate(model_file, urdf_root, sim_gui, rtf):
    sim = GraspingExperiment(urdf_root, sim_gui, rtf)
    detector = GraspDetector(model_file, sim.size)

    outcomes = np.zeros((n_experiments,))

    for n in tqdm(range(n_experiments)):
        sim.setup(args.object_set)
        sim.pause()

        # Reconstruct scene
        extrinsics = sample_hemisphere(sim.size, n_views_per_scene)
        tsdf = TSDFVolume(sim.size, vgn_res)
        for extrinsic in extrinsics:
            _, depth_img = sim.camera.render(extrinsic)
            tsdf.integrate(depth_img, sim.camera.intrinsic, extrinsic)
        tsdf_vol = tsdf.get_volume()

        # Detect grasps
        grasps, qualities, info = detector.detect_grasps(tsdf_vol, threshold)

        # Test highest ranked grasp
        i = np.argmax(qualities)
        sim.world.resume()
        out = sim.test_grasp(grasps[i].pose)

        # Store outcome
        outcomes[n] = out

    return (outcomes >= Label.SUCCESS).sum()


def main(args):

    sim_config = load_dict(Path(args.sim_config))

    n_successes = evaluate(
        model_file=Path(args.model),
        urdf_root=Path(sim_config["urdf_root"]),
        sim_gui=args.sim_gui,
        rtf=args.rtf,
    )

    print("Success rate: {}/{}".format(n_succeses, n_experiments))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="evaluate the vgn grasp detector in simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, required=True, help="saved model ending with .pth"
    )
    parser.add_argument(
        "--object-set",
        choices=["debug", "cuboid", "cuboids"],
        default="debug",
        help="object set to be used",
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
