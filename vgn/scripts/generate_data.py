"""Script to generate a synthetic grasp dataset using physical simulation."""
from __future__ import division

import argparse
import logging
import os
import uuid

import numpy as np
import scipy.signal as signal
import tqdm
from mpi4py import MPI

import vgn.config as cfg
from vgn import data_utils, grasp, simulation
from vgn.perception import integration, viewpoint
from vgn.utils.transform import Rotation, Transform


def generate_dataset(data, n_scenes, n_grasps_per_scene, sim_gui, rank):
    """Generate a dataset of synthetic grasps.

    This script will generate multiple virtual scenes, and for each scene
    render depth images from multiple viewpoints and sample a specified number
    of grasps. It also ensures that the number of positive and  negative
    samples are balanced.

    For each scene, it will create a unique folder within root_dir, and store

        * the rendered depth images as PNGs,
        * the camera intrinsic parameters,
        * the extrinsic parameters corresponding to each image,
        * pose and score for each sampled grasp.

    Args:
        data: Name of the dataset.
        n_scenes: Number of generated virtual scenes.
        n_grasps_per_scene: Number of grasp candidates sampled per scene.
        sim_gui: Run the simulation in a GUI or in headless mode.
        rank: MPI rank.
    """
    n_views_per_scene = 16
    rtf = -1.0

    s = simulation.Simulation(sim_gui, rtf)

    # Create the root directory if it does not exist
    root_dir = os.path.join("data", "datasets", data)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    for _ in tqdm.tqdm(range(n_scenes), disable=rank is not 0):
        scene_data = {}

        # Generate scene
        s.generate_scene(data)
        s.save_state()

        # Reconstruct scene
        intrinsic = s.camera.intrinsic
        extrinsics = viewpoint.sample_hemisphere(n_views_per_scene)
        depth_imgs = [s.camera.get_rgb_depth(e)[1] for e in extrinsics]
        point_cloud, _ = integration.reconstruct_scene(
            intrinsic, extrinsics, depth_imgs
        )

        scene_data["intrinsic"] = s.camera.intrinsic
        scene_data["extrinsics"] = extrinsics
        scene_data["depth_imgs"] = depth_imgs

        # Sample and evaluate grasps candidates
        scene_data["poses"] = []
        scene_data["scores"] = []

        is_positive = lambda score: np.isclose(score, 1.0)
        n_negatives = 0

        while len(scene_data["poses"]) < n_grasps_per_scene:
            point, normal = sample_point(point_cloud)
            pose, score = evaluate_point(s, point, normal)
            if is_positive(score) or n_negatives < n_grasps_per_scene // 2:
                scene_data["poses"].append(pose)
                scene_data["scores"].append(score)
                n_negatives += not is_positive(score)

        dirname = os.path.join(root_dir, str(uuid.uuid4().hex))
        data_utils.store_scene(dirname, scene_data)


def sample_point(point_cloud):
    """Uniformly sample a grasp point from a point cloud with a random offset
    along the negative surface normal.
    """
    gripper_depth = 0.5 * cfg.max_width
    thresh = 0.2

    points = np.asarray(point_cloud.points)
    normals = np.asarray(point_cloud.normals)
    selection = np.random.randint(len(points))
    point, normal = points[selection], normals[selection]
    z_offset = np.random.uniform(
        (0.0 - thresh) * gripper_depth, (1.0 + thresh) * gripper_depth
    )
    point = point - normal * (z_offset - gripper_depth)

    return point, normal


def evaluate_point(sim, pos, normal, n_rotations=9):
    # Define initial grasp frame on object surface
    z_axis = -normal
    x_axis = np.r_[1.0, 0.0, 0.0]
    if np.isclose(np.abs(np.dot(x_axis, z_axis)), 1.0, 1e-4):
        x_axis = np.r_[0.0, 1.0, 0.0]
    y_axis = np.cross(z_axis, x_axis)
    x_axis = np.cross(y_axis, z_axis)
    R = Rotation.from_dcm(np.vstack((x_axis, y_axis, z_axis)).T)

    yaws = np.linspace(-0.5 * np.pi, 0.5 * np.pi, n_rotations)
    scores = []

    for yaw in yaws:
        ori = R * Rotation.from_euler("z", yaw)
        sim.restore_state()
        outcome = grasp.execute(sim.robot, Transform(ori, pos))
        scores.append(outcome == grasp.Outcome.SUCCESS)

    if np.sum(scores):
        # Detect the peak over yaw orientations
        peaks, properties = signal.find_peaks(x=np.r_[0, scores, 0], height=1, width=1)
        idx_of_widest_peak = peaks[np.argmax(properties["widths"])] - 1
        yaw = yaws[idx_of_widest_peak]
        ori, score = R * Rotation.from_euler("z", yaw), 1.0
    else:
        ori, score = R, 0.0

    # Due to the symmetric geometry of a parallel-jaw gripper, make sure
    # the y-axis always points upwards.
    y_axis = ori.as_dcm()[:, 1]
    if np.dot(y_axis, np.r_[0.0, 0.0, 1.0]) < 0.0:
        ori *= Rotation.from_euler("z", np.pi)

    return Transform(ori, pos), score


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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    n_workers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()

    if rank == 0:
        logging.info("Generating data using %d processes.", n_workers)

    generate_dataset(
        data=args.data,
        n_scenes=args.n_scenes // n_workers,
        n_grasps_per_scene=args.n_grasps_per_scene,
        sim_gui=args.sim_gui,
        rank=rank,
    )


if __name__ == "__main__":
    main()
