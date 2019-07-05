import numpy as np

from vgn import grasper
from vgn import simulation
from vgn.candidates import samplers
from vgn.perception import integration, viewpoints
from vgn.utils import camera_intrinsics


def generate_dataset(dataset_path, n_scenes, n_grasps_per_scene, n_workers, sim_gui, rviz):
    """Generate a dataset of synthetic grasps.

    Args:
        dataset_path: Path to which the HDF5 dataset is written to.
        n_scenes: Number of generated virtual scenes.
        n_grasps_per_scene: Number of grasps sampled per scene.
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

    for n_scene in range(n_scenes):
        # Generate a new scene
        s.reset()
        s.spawn_plane()
        s.spawn_cuboid()
        s.spawn_robot()
        s.save_state()

        # Reconstruct the volume
        volume = integration.TSDFVolume(length=length, resolution=100)
        camera_poses = viewpoints.sample_hemisphere(n_views_per_scene, length)

        for T_eye_world in camera_poses:
            rgb, depth = s.camera.get_rgb_depth(T_eye_world)
            volume.integrate(rgb, depth, s.camera.intrinsic, T_eye_world)

        if rviz:
            points, colors, _ = volume.get_point_cloud()
            rviz_utils.draw_point_cloud(points, colors)
        # volume.draw_point_cloud()

        # Sample grasps
        grasp_poses = samplers.uniform(n_grasps_per_scene, volume)

        # Score the grasps
        scores = np.ones(shape=(n_grasps_per_scene,))
        for i, grasp_pose in enumerate(grasp_poses):

            if rviz:
                rviz_utils.draw_grasp_pose(grasp_pose)

            s.restore_state()
            outcome = g.grasp(grasp_pose)
            scores[i] = 1. if outcome == grasper.Outcome.SUCCESS else 0.
            print(outcome)

        # if rviz:
        #     rviz_utils.draw_grasp_candidates(grasp_poses,
        #                                      scores, '/grasp_candidates')
