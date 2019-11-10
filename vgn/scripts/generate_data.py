"""This script generates a dataset of synthetic grasping experiments.

For each experiment, a virtual scene is generated, then depth images
from multiple viewpoints are rendered and a specified number of grasp
candidates are sampled and evaluated. The data of each scene is stored
in a unique folder within the root directory.
"""

from __future__ import print_function, division

import argparse
from pathlib import Path
import uuid

import numpy as np
import scipy.signal as signal
import tqdm
from mpi4py import MPI
import open3d


from vgn.config import Config
from vgn.simulation import GraspingExperiment
from vgn.grasp import Grasp, Label
from vgn.utils.data import SceneData
from vgn.perception.integration import TSDFVolume
from vgn.perception.exploration import sample_hemisphere
from vgn.utils.transform import Rotation, Transform


def sample_grasp_point(gripper_depth, point_cloud):
    epsilon = 0.2

    points = np.asarray(point_cloud.points)
    normals = np.asarray(point_cloud.normals)

    idx = np.random.randint(len(points))
    point, normal = points[idx], normals[idx]
    z_offset = np.random.uniform(
        (0.0 - epsilon) * gripper_depth, (1.0 + epsilon) * gripper_depth
    )
    point = point - normal * (z_offset - gripper_depth)

    return point, normal


def evaluate_grasp_point(sim, pos, normal, n_rotations=9):
    # Define initial grasp frame on object surface
    z_axis = -normal
    x_axis = np.r_[1.0, 0.0, 0.0]
    if np.isclose(np.abs(np.dot(x_axis, z_axis)), 1.0, 1e-4):
        x_axis = np.r_[0.0, 1.0, 0.0]
    y_axis = np.cross(z_axis, x_axis)
    x_axis = np.cross(y_axis, z_axis)
    R = Rotation.from_dcm(np.vstack((x_axis, y_axis, z_axis)).T)

    # Try to grasp with different yaw angles
    yaws = np.linspace(-0.5 * np.pi, 0.5 * np.pi, n_rotations)
    outcomes = []
    for yaw in yaws:
        ori = R * Rotation.from_euler("z", yaw)
        sim.restore_state()
        outcomes.append(sim.test_grasp(Transform(ori, pos)))

    # Detect mid-point of widest peak of successful yaw angles
    successes = (np.asarray(outcomes) == Label.SUCCESS).astype(float)
    if np.sum(successes):
        peaks, properties = signal.find_peaks(
            x=np.r_[0, successes, 0], height=1, width=1
        )
        idx_of_widest_peak = peaks[np.argmax(properties["widths"])] - 1
        ori = R * Rotation.from_euler("z", yaws[idx_of_widest_peak])
    else:
        ori = Rotation.identity()

    return Grasp(Transform(ori, pos)), int(np.max(outcomes))


def main(args):
    n_workers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()

    if rank == 0:
        print("Generating data using {} processes.".format(n_workers))

    # Load the configuration
    config = Config(Path(args.config))

    tsdf_res = config["data_generation"]["tsdf_res"]
    tsdf_size = config["scene"]["size"]

    urdf_root = Path(config["simulation"]["urdf_root"])

    n_scenes = config["data_generation"]["n_scenes"]
    n_viewpoints = config["data_generation"]["n_viewpoints"]
    n_grasps = config["data_generation"]["n_grasps"]

    gripper_depth = 0.5 * config["hand"]["max_width"]  # TODO(mbreyer) parameterize this

    # Setup simulation
    sim = GraspingExperiment(urdf_root, tsdf_size, args.sim_gui, args.rtf)

    # Create the root directory if it does not exist yet
    root = Path(args.root)
    if not root.exists() and rank == 0:
        root.mkdir()

    for _ in tqdm.tqdm(range(n_scenes), disable=rank is not 0):
        # Setup experiment
        sim.setup(args.object_set)
        sim.save_state()

        # Reconstruct scene
        intrinsic = sim.camera.intrinsic
        extrinsics = sample_hemisphere(tsdf_size, n_viewpoints)
        depth_imgs = [sim.camera.render(e)[1] for e in extrinsics]

        volume = TSDFVolume(tsdf_size, tsdf_res)
        for depth_img, extrinsic in zip(depth_imgs, extrinsics):
            volume.integrate(depth_img, intrinsic, extrinsic)
        point_cloud = volume.extract_point_cloud()

        # Sample and evaluate grasp candidates
        grasps, labels = [], []

        is_positive = lambda o: o == Label.SUCCESS
        n_negatives = 0

        while len(grasps) < n_grasps:
            point, normal = sample_grasp_point(gripper_depth, point_cloud)
            grasp, label = evaluate_grasp_point(sim, point, normal)
            if is_positive(label) or n_negatives < n_grasps // 2:
                grasps.append(grasp)
                labels.append(label)
                n_negatives += not is_positive(label)

        data = SceneData(depth_imgs, intrinsic, extrinsics, grasps, labels)
        data.save(root / str(uuid.uuid4().hex))


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
        "--config",
        type=str,
        default="config/default.yaml",
        help="path to configuration file",
    )
    parser.add_argument("--sim-gui", action="store_true", help="disable headless mode")
    parser.add_argument(
        "--rtf", type=float, default=-1.0, help="real time factor of the simulation"
    )
    args = parser.parse_args()
    main(args)
