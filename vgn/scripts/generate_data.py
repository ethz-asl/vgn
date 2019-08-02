"""Script to generate a synthetic grasp dataset using physical simulation."""
import argparse
import json
import os
import uuid

import numpy as np

from vgn import candidate, grasp, samplers, simulation
from vgn.perception import integration
from vgn.perception.viewpoints import sample_hemisphere
from vgn.utils import camera, image
from vgn.utils.transform import Rotation, Transform


def generate_dataset(root_dir, n_scenes, n_grasps_per_scene, n_workers,
                     sim_gui):
    """Generate a dataset of synthetic grasps.

    This script will generate multiple virtual scenes, and for each scene
    render depth images from multiple viewpoints and sample a specified number
    of grasps.

    For each scene, it will create a unique folder within root_dir, and store

        * the rendered depth images as PNGs,
        * the camera intrinsic parameters,
        * the extrinsic parameters corresponding to each image,
        * pose and score for each sampled grasp.

    Args:
        root_dir: Root directory of the dataset.
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
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    for _ in range(n_scenes):
        viewpoints = []
        grasps = []

        # Create a unique folder for storing data from this scene
        dirname = os.path.join(root_dir, str(uuid.uuid4().hex))
        os.makedirs(dirname)

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
        extrinsics = sample_hemisphere(n_views_per_scene, size)
        for i, extrinsic in enumerate(extrinsics):
            _, depth = s.camera.get_rgb_depth(extrinsic)
            volume.integrate(depth, s.camera.intrinsic, extrinsic)

            # Write the image to disk
            image_name = '{0:03d}.png'.format(i)
            image.save(os.path.join(dirname, image_name), depth)
            viewpoints.append({
                'image_name': image_name,
                'extrinsic': extrinsic.to_dict(),
            })

        # Sample candidate grasp points
        points, normals = samplers.uniform(n_grasps_per_scene,
                                           volume,
                                           min_z_offset=0.005,
                                           max_z_offset=0.02)

        # Score the candidates
        for i, (point, normal) in enumerate(zip(points, normals)):
            pose, score = candidate.evaluate(s, g, point, normal)
            grasps.append({'pose': pose.to_dict(), 'score': score})

        # Write intrinsics to disk
        s.camera.intrinsic.to_json(os.path.join(dirname, 'intrinsic.json'))

        # Write extrinsics to disk
        fname = os.path.join(dirname, 'viewpoints.json')
        with open(fname, 'wb') as fp:
            json.dump(viewpoints, fp, indent=4)

        # Write grasps to disk
        fname = os.path.join(dirname, 'grasps.json')
        with open(fname, 'wb') as fp:
            json.dump(grasps, fp, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'root_dir',
        type=str,
        help='The root directory of the dataset',
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
        root_dir=args.root_dir,
        n_scenes=args.n_scenes,
        n_grasps_per_scene=args.n_grasps_per_scene,
        n_workers=args.n_workers,
        sim_gui=args.sim_gui,
    )


if __name__ == '__main__':
    main()
