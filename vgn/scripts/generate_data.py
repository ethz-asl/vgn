"""Script to generate a synthetic grasp dataset using physical simulation."""
import argparse
import uuid
from collections import OrderedDict
from os import makedirs, path

import numpy as np

from vgn import candidate, grasp, samplers, simulation
from vgn.data_generator import generate_dataset
from vgn.perception import integration, viewpoints
from vgn.utils import camera, image
from vgn.utils.transform import Rotation, Transform


def generate_dataset(base_dir, n_scenes, n_grasps_per_scene, n_workers,
                     sim_gui):
    """Generate a dataset of synthetic grasps.

    This script will generate multiple virtual scenes, and for each scene
    render depth images from multiple viewpoints and sample a specified number
    of grasps.

    For each scene, it will create a unique folder within base_dir, and store

        - the rendered depth images as PNGs,
        - intrinsic.json with the camera intrinsic parameters,
        - extrinsics.csv with the extrinsic parameters corresponding to each image,
        - grasps.csv containing grasp pose and score for each sampled grasp.

    Args:
        base_dir: Base directory of the dataset.
        n_scenes: Number of generated virtual scenes.
        n_grasps_per_scene: Number of grasp candidates sampled per scene.
        n_workers: Number of processes used for the data generation.
        sim_gui: Run the simulation in a GUI or in headless mode.

    TODO:
        * Distribute data collection.
    """
    n_views_per_scene = 16
    real_time = False

    s = simulation.Simulation(sim_gui, real_time)
    g = grasp.Grasper(robot=s)

    # Create the base dir if not existing
    if not path.exists(base_dir):
        makedirs(base_dir)

    # Pre-allocate memory
    image_names = []
    extrinsics_pos = np.empty((n_views_per_scene, 3))
    extrinsics_ori = np.empty((n_views_per_scene, 4))
    grasps_pos = np.empty((n_grasps_per_scene, 3))
    grasps_ori = np.empty((n_grasps_per_scene, 4))
    scores = np.empty((n_grasps_per_scene, 1))

    for _ in range(n_scenes):
        # Create a unique folder for storing data from this scene
        dirname = path.join(base_dir, str(uuid.uuid4().hex))
        makedirs(dirname)

        # Generate a new scene
        s.reset()
        s.spawn_plane()
        s.spawn_debug_cuboid()
        # s.spawn_debug_cylinder()
        s.spawn_robot()
        s.save_state()

        # Reconstruct the volume
        size = 0.2
        volume = integration.TSDFVolume(size, resolution=60)
        extrinsics = viewpoints.sample_hemisphere(n_views_per_scene, size)
        for i, extrinsic in enumerate(extrinsics):
            _, depth = s.camera.get_rgb_depth(extrinsic)
            volume.integrate(depth, s.camera.intrinsic, extrinsic)
            extrinsics_pos[i] = extrinsic.translation
            extrinsics_ori[i] = extrinsic.rotation.as_quat()

            # Write the image to disk
            image_name = '{0:03d}.png'.format(i)
            image.save(path.join(dirname, image_name), depth)

        # Sample candidate grasp points
        points, normals = samplers.uniform(n_grasps_per_scene,
                                           volume,
                                           min_z_offset=0.005,
                                           max_z_offset=0.02)

        # Score the candidates
        for i, (point, normal) in enumerate(zip(points, normals)):
            score, orientation = candidate.evaluate(s, g, point, normal)
            grasps_pos[i] = point
            grasps_ori[i] = orientation.as_quat()
            scores[i] = score

        # Write intrinsics to disk
        s.camera.intrinsic.to_json(path.join(dirname, 'intrinsic.json'))

        # Write extrinsics to disk
        data = np.hstack((extrinsics_pos, extrinsics_ori))
        fname = path.join(dirname, 'extrinsics.csv')
        np.savetxt(fname, data, '%.3f', ',', header='x,y,z,qx,qy,qz,qw')

        # Write grasps to disk
        data = np.hstack((grasps_pos, grasps_ori, scores))
        fname = path.join(dirname, 'grasps.csv')
        np.savetxt(fname, data, '%.3f', ',', header='x,y,z,qx,qy,qz,qw,score')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'base_dir',
        type=str,
        help='The base directory in which data is stored',
    )
    parser.add_argument(
        '--n-scenes',
        type=int,
        default=1000,
        help='Number of generated virtual scenes',
    )
    parser.add_argument(
        '--n-grasps-per-scene',
        type=int,
        default=100,
        help='Number of grasp candidates per scene',
    )
    parser.add_argument(
        '--n-workers',
        type=int,
        default=1,
        help='Number of processes used for data collection',
    )
    parser.add_argument(
        '--sim-gui',
        action='store_true',
        help='Disable headless mode',
    )
    args = parser.parse_args()

    generate_dataset(
        base_dir=args.base_dir,
        n_scenes=args.n_scenes,
        n_grasps_per_scene=args.n_grasps_per_scene,
        n_workers=args.n_workers,
        sim_gui=args.sim_gui,
    )


if __name__ == '__main__':
    main()
