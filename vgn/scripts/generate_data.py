"""This script generates a dataset of synthetic grasping experiments.

For each experiment, a virtual scene is generated, then depth images
from multiple viewpoints are rendered and a specified number of grasp
candidates are sampled and evaluated. The data of each scene is stored
in a unique folder within the root directory.
"""

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
from vgn.simulation import GraspingExperiment
from vgn.grasp import Grasp, Label
from vgn.utils.data import SceneData
from vgn.perception import integration, exploration
from vgn.utils.transform import Rotation, Transform


def sample_grasp_point(point_cloud):
    gripper_depth = 0.5 * cfg.max_width
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

    # Ensure that the y-axis of the parallel-jaw gripper points upwards
    y_axis = ori.as_dcm()[:, 1]
    if np.dot(y_axis, np.r_[0.0, 0.0, 1.0]) < 0.0:
        ori *= Rotation.from_euler("z", np.pi)
    pose = Transform(ori, pos)

    return Grasp(pose), int(np.max(outcomes))


def main(args):
    n_workers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()

    if rank == 0:
        print("Generating data using {} processes.".format(n_workers))

    resolution = 80  # TODO make this transparent
    s = GraspingExperiment(args.sim_gui, args.rtf)

    # Create the root directory if it does not exist yet
    if not os.path.exists(args.root) and rank == 0:
        os.makedirs(args.root)

    for _ in tqdm.tqdm(range(args.scenes), disable=rank is not 0):
        # Setup experiment
        s.setup(args.object_set)
        s.save_state()

        # Reconstruct scene
        n_views_per_scene = 16  # TODO(mbreyer): move to config
        intrinsic = s.camera.intrinsic
        extrinsics = exploration.sample_hemisphere(n_views_per_scene)
        depth_imgs = [s.camera.render(e)[1] for e in extrinsics]
        point_cloud, _ = integration.reconstruct_scene(
            intrinsic, extrinsics, depth_imgs, resolution
        )

        # Sample and evaluate grasp candidates
        grasps, labels = [], []

        is_positive = lambda o: o == Label.SUCCESS
        n_negatives = 0

        while len(grasps) < args.grasps:
            point, normal = sample_grasp_point(point_cloud)
            grasp, label = evaluate_grasp_point(s, point, normal)
            if is_positive(label) or n_negatives < args.grasps // 2:
                grasps.append(grasp)
                labels.append(label)
                n_negatives += not is_positive(label)

        data = SceneData(intrinsic, extrinsics, depth_imgs, grasps, labels)
        data.save(os.path.join(args.root, str(uuid.uuid4().hex)))


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
        "--scenes", type=int, default=1000, help="number of generated virtual scenes"
    )
    parser.add_argument(
        "--grasps", type=int, default=40, help="number of grasp candidates per scene"
    )
    parser.add_argument("--sim-gui", action="store_true", help="disable headless mode")
    parser.add_argument(
        "--rtf", type=float, default=-1.0, help="real time factor of the simulation"
    )
    args = parser.parse_args()
    main(args)
