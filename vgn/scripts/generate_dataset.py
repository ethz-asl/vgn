from __future__ import print_function, division

import argparse
from pathlib2 import Path
import uuid

from mpi4py import MPI
import numpy as np
import open3d as o3d
import scipy.signal as signal
from tqdm import tqdm

from vgn import Label, to_voxel_coordinates
from vgn.grasp import Grasp
from vgn.simulation import GraspSimulation
from vgn.utils import io
from vgn.utils.transform import Rotation, Transform

OBJECT_COUNT_LAMBDA = 4
VIEWPOINT_COUNT = 5
GRASPS_PER_SCENE = 40


def main(args):
    workers, rank = setup_mpi()
    create_dataset_dir(args.dataset, rank)
    sim = GraspSimulation(args.object_set, "config/default.yaml", gui=args.sim_gui)
    finger_depth = sim.config["finger_depth"]
    grasps_per_worker = args.grasps // workers
    pbar = tqdm(total=grasps_per_worker, disable=rank is not 0)

    for _ in range(grasps_per_worker // GRASPS_PER_SCENE):
        # generate heap
        object_count = np.random.poisson(OBJECT_COUNT_LAMBDA) + 1
        sim.reset(object_count)
        sim.save_state()

        # reconstruct and crop surface from point cloud
        tsdf, pc = sim.acquire_tsdf(num_viewpoints=VIEWPOINT_COUNT)
        l, u = 1.2 * finger_depth, sim.size - 1.2 * finger_depth
        z = sim.world.bodies[0].get_pose().translation[2] + 0.005
        pc = pc.crop(np.r_[l, l, z], np.r_[u, u, sim.size])

        if pc.is_empty():
            print("Point cloud empty, skipping scene")
            continue

        # store the tsdf
        tsdf_path = store_tsdf(args.dataset, tsdf)

        for _ in range(GRASPS_PER_SCENE):
            # sample and evaluate a grasp point
            point, normal = sample_grasp_point(pc, finger_depth)
            grasp, label = evaluate_grasp_point(sim, point, normal)

            # store the sample in voxel coordinates
            grasp = to_voxel_coordinates(grasp, tsdf.voxel_size)
            store_sample(args.dataset, tsdf_path, grasp, label)
            pbar.update()

    pbar.close()


def setup_mpi():
    workers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    return workers, rank


def create_dataset_dir(dataset_dir, rank):
    if rank != 0:
        return
    dataset_dir.mkdir(exist_ok=True)
    csv_path = dataset_dir / "grasps.csv"
    if not csv_path.exists():
        io.create_csv(csv_path, "tsdf,i,j,k,qx,qy,qz,qw,width,label")


def sample_grasp_point(point_cloud, finger_depth, eps=0.1):
    points = np.asarray(point_cloud.points)
    normals = np.asarray(point_cloud.normals)
    idx = np.random.randint(len(points))
    point, normal = points[idx], normals[idx]
    grasp_depth = np.random.uniform(-eps * finger_depth, (1.0 + eps) * finger_depth)
    point = point + normal * grasp_depth
    return point, normal


def evaluate_grasp_point(sim, pos, normal, num_rotations=6):
    # define initial grasp frame on object surface
    z_axis = -normal
    x_axis = np.r_[1.0, 0.0, 0.0]
    if np.isclose(np.abs(np.dot(x_axis, z_axis)), 1.0, 1e-4):
        x_axis = np.r_[0.0, 1.0, 0.0]
    y_axis = np.cross(z_axis, x_axis)
    x_axis = np.cross(y_axis, z_axis)
    R = Rotation.from_dcm(np.vstack((x_axis, y_axis, z_axis)).T)

    # try to grasp with different yaw angles
    yaws = np.linspace(0.0, np.pi, num_rotations)
    outcomes, widths = [], []
    for yaw in yaws:
        ori = R * Rotation.from_euler("z", yaw)
        sim.restore_state()
        outcome, width = sim.execute_grasp(Transform(ori, pos), remove=False)
        outcomes.append(outcome)
        widths.append(width)

    # detect mid-point of widest peak of successful yaw angles
    successes = (np.asarray(outcomes) == Label.SUCCESS).astype(float)
    if np.sum(successes):
        peaks, properties = signal.find_peaks(
            x=np.r_[0, successes, 0], height=1, width=1
        )
        idx_of_widest_peak = peaks[np.argmax(properties["widths"])] - 1
        ori = R * Rotation.from_euler("z", yaws[idx_of_widest_peak])
        width = widths[idx_of_widest_peak]

    return Grasp(Transform(ori, pos), width), int(np.max(outcomes))


def store_tsdf(dataset_dir, tsdf):
    # store TSDF volume in compressed .npz format
    path = dataset_dir / (uuid.uuid4().hex + ".npz")
    np.savez_compressed(str(path), tsdf=tsdf.get_volume())
    return path


def store_sample(dataset_dir, tsdf_path, grasp, label):
    # add a row to the table (TODO concurrent writes could be an issue)
    csv_path = dataset_dir / "grasps.csv"
    qx, qy, qz, qw = grasp.pose.rotation.as_quat()
    i, j, k = np.round(grasp.pose.translation).astype(np.int)
    width = grasp.width
    io.append_csv(csv_path, tsdf_path.name, i, j, k, qx, qy, qz, qw, width, label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--object-set", type=str, required=True)
    parser.add_argument("--grasps", type=int, default=1000)
    parser.add_argument("--sim-gui", action="store_true")
    args = parser.parse_args()
    main(args)
