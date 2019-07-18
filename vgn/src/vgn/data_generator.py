import uuid
from os import makedirs, path

import numpy as np

from vgn import candidate, grasp, samplers, simulation
from vgn.perception import integration, viewpoints
from vgn.utils import camera, image
from vgn.utils.transform import Rotation, Transform


def generate_dataset(basedir, n_scenes, n_candidates_per_scene, n_workers,
                     sim_gui):
    """Generate a dataset of synthetic grasps.

    Args:
        basedir: Root directory of the dataset.
        n_scenes: Number of generated virtual scenes.
        n_candidates_per_scene: Number of candidates sampled per scene.
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
    if not path.exists(basedir):
        makedirs(basedir)

    # Allocate memory for storing the data
    extrinsics = np.empty((n_views_per_scene, 7))
    grasps = np.empty((n_candidates_per_scene, 7))
    scores = np.empty((n_candidates_per_scene, ))

    for _ in range(n_scenes):
        # Create a unique folder for storing data from this scene
        dirname = path.join(basedir, str(uuid.uuid4().hex))
        makedirs(dirname)

        # Generate a new scene
        s.reset()
        s.spawn_plane()
        s.spawn_debug_cuboid()
        # s.spawn_debug_cylinder()
        s.spawn_robot()
        s.save_state()

        # Reconstruct the volume
        length = 0.2
        volume = integration.TSDFVolume(length, resolution=60)
        camera_poses = viewpoints.sample_hemisphere(n_views_per_scene, length)
        for i, extrinsic in enumerate(camera_poses):
            _, depth = s.camera.get_rgb_depth(extrinsic)
            volume.integrate(depth, s.camera.intrinsic, extrinsic)

            # Write the image to disk
            fname = path.join(dirname, "{0:03d}.png".format(i))
            image.save(fname, depth)
            extrinsics[i] = np.r_[extrinsic.translation,
                                  extrinsic.rotation.as_quat()]

        # Sample candidate grasp points
        points, normals = samplers.uniform(n_candidates_per_scene,
                                           volume,
                                           min_z_offset=0.005,
                                           max_z_offset=0.02)

        # Score the candidates
        for i, (point, normal) in enumerate(zip(points, normals)):
            score, orientation = candidate.evaluate(s, g, point, normal)
            scores[i] = score
            grasps[i] = np.r_[point, orientation.as_quat()]

        # Write the intrinsic, extrinsics, grasp poses, and scores to disk
        fmt = "%.3f"
        s.camera.intrinsic.save(path.join(dirname, "intrinsic.json"))
        np.savetxt(path.join(dirname, "extrinsics.txt"), extrinsics, fmt, ",")
        np.savetxt(path.join(dirname, "grasps.txt"), grasps, fmt, ",")
        np.savetxt(path.join(dirname, "scores.txt"), scores, fmt, ",")
