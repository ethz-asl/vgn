import argparse
from pathlib import Path

import open3d
import numpy as np
from tqdm import tqdm

from vgn.grasp import Label
from vgn.grasp_detector import GraspDetector
from vgn.perception.exploration import sample_hemisphere
from vgn.perception.integration import TSDFVolume
from vgn.simulation import GraspingExperiment


def main(args):
    sim = GraspingExperiment(args.sim_gui, args.rtf)
    detector = GraspDetector(Path(args.model))

    n_experiments = 20
    outcomes = np.zeros((n_experiments,))

    for n in tqdm(range(n_experiments)):
        sim.setup(args.object_set)
        sim.pause()

        # Reconstruct scene
        n_views_per_scene = 16
        extrinsics = sample_hemisphere(n_views_per_scene)
        tsdf = TSDFVolume(0.24, 40)
        for extrinsic in extrinsics:
            _, depth_img = sim.camera.render(extrinsic)
            tsdf.integrate(depth_img, sim.camera.intrinsic, extrinsic)
        tsdf_vol = tsdf.get_volume()

        # Detect grasps
        grasps, qualities, info = detector.detect_grasps(tsdf_vol, tsdf.voxel_size, 0.8)

        # Test highest ranked grasp
        i = np.argmax(qualities)
        sim.world.resume()
        out = sim.test_grasp(grasps[i].pose)

        # Store outcome
        outcomes[n] = out

    print("Results")
    print("=======")
    print("COLLISION", (outcomes == Label.COLLISION).sum())
    print("EMPTY", (outcomes == Label.EMPTY).sum())
    print("SLIPPED", (outcomes == Label.SLIPPED).sum())
    print("SUCCESS", (outcomes == Label.SUCCESS).sum())
    print("ROBUST", (outcomes == Label.ROBUST).sum())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="generate synthetic grasping experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, required=True, help="root directory of the dataset"
    )
    parser.add_argument(
        "--object-set",
        choices=["debug", "cuboid", "cuboids"],
        default="debug",
        help="object set to be used",
    )
    parser.add_argument("--sim-gui", action="store_true", help="disable headless mode")
    parser.add_argument(
        "--rtf", type=float, default=-1.0, help="real time factor of the simulation"
    )
    args = parser.parse_args()
    main(args)
