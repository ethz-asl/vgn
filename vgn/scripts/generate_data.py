"""Script to generate a synthetic grasp dataset using physical simulation."""
from __future__ import division

import argparse
import logging
import os
import uuid

import numpy as np
import tqdm
from mpi4py import MPI

import vgn.config as cfg
from vgn import data, grasp, simulation
from vgn.perception import integration
from vgn.perception.viewpoints import sample_hemisphere


def generate_dataset(root_dir, n_scenes, n_grasps_per_scene, sim_gui, rank):
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
        root_dir: Root directory of the dataset.
        n_scenes: Number of generated virtual scenes.
        n_grasps_per_scene: Number of grasp candidates sampled per scene.
        sim_gui: Run the simulation in a GUI or in headless mode.
        rank: MPI rank.
    """
    n_views_per_scene = 16
    real_time = False

    s = simulation.Simulation(sim_gui, real_time)
    g = grasp.Grasper(robot=s)

    # Create the root directory if it does not exist
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    for _ in tqdm.tqdm(range(n_scenes), disable=rank is not 0):
        scene = {
            'intrinsic': s.camera.intrinsic,
            'extrinsics': [],
            'depth_imgs': [],
            'poses': [],
            'scores': [],
        }

        # Generate a new scene
        s.reset()
        s.spawn_plane()
        s.spawn_debug_cuboid()
        # s.spawn_debug_cylinder()
        s.spawn_robot()
        s.save_state()

        # Reconstruct the volume
        volume = integration.TSDFVolume(cfg.size, 100)
        extrinsics = sample_hemisphere(n_views_per_scene, cfg.size)
        for i, extrinsic in enumerate(extrinsics):
            _, depth = s.camera.get_rgb_depth(extrinsic)
            volume.integrate(depth, s.camera.intrinsic, extrinsic)
            scene['extrinsics'].append(extrinsic)
            scene['depth_imgs'].append(depth)
        point_cloud = volume.get_point_cloud()
        # import open3d as o3d
        # o3d.visualization.draw_geometries([volume.get_point_cloud()])

        # Sample and evaluate candidate grasp points
        is_positive = lambda score: np.isclose(score, 1.)
        n_negatives = 0

        while len(scene['poses']) < n_grasps_per_scene:
            point, normal = grasp.sample_uniform(point_cloud, 0.005, 0.02)
            pose, score = grasp.evaluate(s, g, point, normal)
            if is_positive(score) or n_negatives < n_grasps_per_scene // 2:
                scene['poses'].append(pose)
                scene['scores'].append(score)
                n_negatives += not is_positive(score)

        dirname = os.path.join(root_dir, str(uuid.uuid4().hex))
        data.store_scene(dirname, scene)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'root_dir',
        type=str,
        help='root directory of the dataset',
    )
    parser.add_argument(
        '--n-scenes',
        type=int,
        default=1000,
        help='number of generated virtual scenes',
    )
    parser.add_argument(
        '--n-grasps-per-scene',
        type=int,
        default=10,
        help='number of grasp candidates per scene',
    )
    parser.add_argument(
        '--sim-gui',
        action='store_true',
        help='disable headless mode',
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    n_workers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()

    if rank == 0:
        logging.info('Generating data using %d processes.', n_workers)

    generate_dataset(
        root_dir=args.root_dir,
        n_scenes=args.n_scenes // n_workers,
        n_grasps_per_scene=args.n_grasps_per_scene,
        sim_gui=args.sim_gui,
        rank=rank,
    )


if __name__ == '__main__':
    main()
