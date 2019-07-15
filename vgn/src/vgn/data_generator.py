import numpy as np

from vgn import grasper
from vgn import simulation
from vgn.candidates import samplers
from vgn.perception import integration, viewpoints
from vgn.utils import camera_intrinsics


def generate_dataset(dataset_path, n_scenes, n_candidates_per_scene, n_workers,
                     sim_gui, rviz):
    """Generate a dataset of synthetic grasps.

    Args:
        dataset_path: Path to which the HDF5 dataset is written to.
        n_scenes: Number of generated virtual scenes.
        n_candidates_per_scene: Number of candidates sampled per scene.
        n_workers: Number of processes used for the data generation.
        sim_gui: Run the simulation in a GUI or in headless mode.
        rviz: Publish point clouds and grasp candidates to rviz.

    TODO:
        * Distribute data collection.
    """
    length = 0.3
    n_views_per_scene = 10
    n_grasps_per_scene = 20

    s = simulation.Simulation(sim_gui)
    g = grasper.Grasper(robot=s)

    if rviz:
        from vgn_ros.utils import rviz_utils

    for _ in range(n_scenes):
        # Generate a new scene
        s.reset()
        s.spawn_plane()
        s.spawn_debug_cuboid()
        s.spawn_robot()
        s.save_state()

        # Reconstruct the volume
        volume = integration.TSDFVolume(length=length, resolution=100)
        camera_poses = viewpoints.sample_hemisphere(n_views_per_scene, length)
        for T_eye_world in camera_poses:
            rgb, depth = s.camera.get_rgb_depth(T_eye_world)
            volume.integrate(rgb, depth, s.camera.intrinsic, T_eye_world)
        point_cloud = volume.extract_point_cloud()
        # volume.draw_point_cloud()

        if rviz:
            points = np.asarray(point_cloud.points)
            colors = np.asarray(point_cloud.colors)
            rviz_utils.draw_point_cloud(points, colors)

        # Sample grasp candidates
        poses = samplers.uniform(point_cloud, n_candidates_per_scene)

        # Score the grasps
        scores = np.zeros(shape=(n_candidates_per_scene, ))
        for i, pose in enumerate(poses):

            if rviz:
                rviz_utils.draw_candidate(pose)

            s.restore_state()
            outcome = g.grasp(pose)
            scores[i] = 1. if outcome == grasper.Outcome.SUCCESS else 0.
            print(outcome)

        if rviz:
            rviz_utils.draw_candidates(poses, scores)
