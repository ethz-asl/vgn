"""Script to generate a synthetic grasp dataset using physical simulation."""
from __future__ import print_function, division

import argparse
import os
import uuid

import numpy as np
import scipy.signal as signal
import tqdm
from mpi4py import MPI
import open3d

import vgn.config as cfg
from vgn import grasp, simulation
import vgn.utils.data
from vgn.perception import integration, exploration
from vgn.utils.transform import Rotation, Transform


def collect_dataset(data, n_scenes, n_grasps_per_scene, sim_gui, rtf, rank):
    """Generate a dataset of synthetic grasps.

    This script will generate multiple virtual scenes, and for each scene
    render depth images from multiple viewpoints and sample a specified number
    of grasps. It also ensures that the number of positive and  negative
    samples are balanced.

    For each scene, it will create a unique folder within root_dir, and store

        * the rendered depth images as PNGs,
        * the camera intrinsic parameters,
        * the extrinsic parameters corresponding to each image,
        * pose and outcome for each sampled grasp.

    Args:
        data: Name of the dataset.
        n_scenes: Number of generated virtual scenes.
        n_grasps_per_scene: Number of grasp candidates sampled per scene.
        sim_gui: Run the simulation in a GUI or in headless mode.
        rtf: Real time factor of the simulation.
        rank: MPI rank.
    """
    s = simulation.GraspingExperiment(sim_gui, rtf)

    # Create the root directory if it does not exist
    root_dir = os.path.join("data", "datasets", data)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    for _ in tqdm.tqdm(range(n_scenes), disable=rank is not 0):
        scene_data = {}

        # Setup experiment
        s.setup(data)
        s.save_state()

        # Reconstruct scene
        n_views_per_scene = 16
        extrinsics = exploration.sample_hemisphere(n_views_per_scene)
        depth_imgs = [s.camera.render(e)[1] for e in extrinsics]
        point_cloud, _ = integration.reconstruct_scene(
            s.camera.intrinsic, extrinsics, depth_imgs
        )

        scene_data["intrinsic"] = s.camera.intrinsic
        scene_data["extrinsics"] = extrinsics
        scene_data["depth_imgs"] = depth_imgs

        # Sample and evaluate grasps candidates
        scene_data["poses"] = []
        scene_data["outcomes"] = []

        is_positive = lambda outcome: outcome == grasp.Outcome.SUCCESS.value
        n_negatives = 0

        while len(scene_data["poses"]) < n_grasps_per_scene:
            point, normal = sample_point(point_cloud)
            pose, outcome = evaluate_point(s, point, normal)
            if is_positive(outcome) or n_negatives < n_grasps_per_scene // 2:
                scene_data["poses"].append(pose)
                scene_data["outcomes"].append(outcome)
                n_negatives += not is_positive(outcome)

        dirname = os.path.join(root_dir, str(uuid.uuid4().hex))
        vgn.utils.data.store_scene(dirname, scene_data)


def sample_point(point_cloud):
    """Uniformly sample a grasp point from a point cloud with a random offset
    along the negative surface normal.
    """
    gripper_depth = 0.5 * cfg.max_width
    epsilon = 0.2

    points = np.asarray(point_cloud.points)
    normals = np.asarray(point_cloud.normals)
    selection = np.random.randint(len(points))
    point, normal = points[selection], normals[selection]
    z_offset = np.random.uniform(
        (0.0 - epsilon) * gripper_depth, (1.0 + epsilon) * gripper_depth
    )
    point = point - normal * (z_offset - gripper_depth)

    return point, normal


def evaluate_point(sim, pos, normal, n_rotations=9):
    """Try to grasp the point using different yaws."""
    # Define initial grasp frame on object surface
    z_axis = -normal
    x_axis = np.r_[1.0, 0.0, 0.0]
    if np.isclose(np.abs(np.dot(x_axis, z_axis)), 1.0, 1e-4):
        x_axis = np.r_[0.0, 1.0, 0.0]
    y_axis = np.cross(z_axis, x_axis)
    x_axis = np.cross(y_axis, z_axis)
    R = Rotation.from_dcm(np.vstack((x_axis, y_axis, z_axis)).T)

    yaws = np.linspace(-0.5 * np.pi, 0.5 * np.pi, n_rotations)
    outcomes = []

    for yaw in yaws:
        ori = R * Rotation.from_euler("z", yaw)
        sim.restore_state()
        outcome = sim.test_grasp(Transform(ori, pos))
        outcomes.append(outcome.value)

    # Detect mid-point of widest peak of successful yaw angles
    successes = (np.asarray(outcomes) == grasp.Outcome.SUCCESS.value).astype(float)
    ori = Rotation.identity()

    if np.sum(successes):
        peaks, properties = signal.find_peaks(
            x=np.r_[0, successes, 0], height=1, width=1
        )
        idx_of_widest_peak = peaks[np.argmax(properties["widths"])] - 1
        yaw = yaws[idx_of_widest_peak]
        ori = R * Rotation.from_euler("z", yaw)

    # Due to the symmetric geometry of a parallel-jaw gripper, make sure
    # the y-axis always points upwards.
    y_axis = ori.as_dcm()[:, 1]
    if np.dot(y_axis, np.r_[0.0, 0.0, 1.0]) < 0.0:
        ori *= Rotation.from_euler("z", np.pi)

    return Transform(ori, pos), np.max(outcomes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="name of dataset")
    parser.add_argument(
        "--n-scenes",
        type=int,
        default=1000,
        help="number of generated virtual scenes (default: 1000)",
    )
    parser.add_argument(
        "--n-grasps-per-scene",
        type=int,
        default=40,
        help="number of grasp candidates per scene (default: 40)",
    )
    parser.add_argument("--sim-gui", action="store_true", help="disable headless mode")
    parser.add_argument("--rtf", type=float, default=-1.0, help="real time factor")
    args = parser.parse_args()

    n_workers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()

    if rank == 0:
        print("Generating data using {} processes.".format(n_workers))

    collect_dataset(
        data=args.data,
        n_scenes=args.n_scenes // n_workers,
        n_grasps_per_scene=args.n_grasps_per_scene,
        sim_gui=args.sim_gui,
        rtf=args.rtf,
        rank=rank,
    )


if __name__ == "__main__":
    main()
