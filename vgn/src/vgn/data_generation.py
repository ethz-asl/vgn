import uuid

import numpy as np
import scipy.signal as signal
from tqdm import tqdm

from vgn.grasp import Label, Grasp
from vgn.perception.exploration import sample_hemisphere
from vgn.perception.integration import TSDFVolume
from vgn.simulation import GraspingExperiment
from vgn.utils.io import save_dict
from vgn.utils.data import SceneData
from vgn.utils.transform import Rotation, Transform


def generate_data(
    root_dir,
    object_set,
    n_scenes,
    n_grasps,
    n_viewpoints,
    vol_res,
    urdf_root,
    sim_gui,
    rtf,
    rank,
):
    # Setup simulation
    sim = GraspingExperiment(urdf_root, sim_gui, rtf)
    gripper_depth = 0.5 * sim.robot.max_opening_width

    if rank == 0:
        root_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "object_set": object_set,
            "n_scenes": n_scenes,
            "n_grasps": n_grasps,
            "n_viewpoints": n_viewpoints,
            "vol_size": sim.size,
            "vol_res": vol_res,
        }
        save_dict(config, root_dir / "config.yaml")

    for _ in tqdm(range(n_scenes), disable=rank is not 0):
        # Setup experiment
        sim.setup(object_set)
        sim.save_state()

        # Reconstruct scene
        intrinsic = sim.camera.intrinsic
        extrinsics = sample_hemisphere(sim.size, n_viewpoints)
        depth_imgs = [sim.camera.render(e)[1] for e in extrinsics]

        tsdf = TSDFVolume(sim.size, vol_res)
        tsdf.integrate_images(depth_imgs, intrinsic, extrinsics)
        point_cloud = tsdf.extract_point_cloud()

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
        data.save(root_dir / str(uuid.uuid4().hex))


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
